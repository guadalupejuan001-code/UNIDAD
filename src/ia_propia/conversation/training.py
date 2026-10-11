"""Local, resumable scratch training; test scores are computed only by finalize()."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import uuid

TRAIN_DEFAULTS = {
    "batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.01,
    "grad_clip": 1.0, "max_epochs": 100, "max_steps": 100000,
    "max_seconds": 86400, "validation_every": 100, "checkpoint_every": 100,
    "patience": 12, "min_delta": 0.0001, "seed": 42,
    "warmup_steps": 100, "torch_threads": 4,
}
MODEL_DEFAULTS = {
    "vocab_size": 2048, "d_model": 128, "n_layers": 4, "n_heads": 4,
    "context_length": 256, "dropout": 0.0, "ffn_multiplier": 4, "seed": 42,
}


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Conversation training requires the conversation dependencies") from exc


def _reject_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _birth(pid):
    try:
        # Field 22 after the process name, which can contain spaces and parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


@contextlib.contextmanager
def _run_lock(run_dir):
    import fcntl
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    fd = os.open(run_dir / "writer.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run already has an active writer") from exc
        yield
    finally:
        os.close(fd)


def _configs(model_config, training_config):
    model = dict(MODEL_DEFAULTS)
    training = dict(TRAIN_DEFAULTS)
    for supplied, defaults, result in ((model_config, MODEL_DEFAULTS, model),
                                       (training_config, TRAIN_DEFAULTS, training)):
        if supplied is not None:
            if not isinstance(supplied, dict) or set(supplied) - set(defaults):
                raise ValueError("Unknown configuration fields")
            result.update(supplied)
    ints = {"batch_size": (1, 256), "max_epochs": (1, 100000),
            "max_steps": (1, 100000000), "validation_every": (1, 1000000),
            "checkpoint_every": (1, 1000000), "patience": (1, 100000),
            "seed": (0, 2**32 - 1), "warmup_steps": (0, 10000000), "torch_threads": (1, 32)}
    for key, (low, high) in ints.items():
        value = training[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
    for key, low, high in (("learning_rate", 0, 1), ("weight_decay", -1e-30, 10),
                           ("grad_clip", 0, 1000000), ("max_seconds", 0, 86400),
                           ("min_delta", -1e-30, 100)):
        value = training[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
            raise ValueError(f"Invalid {key}")
    # Constructor validates architecture; the requested vocabulary is bounded before fitting.
    if type(model["vocab_size"]) is not int or not 262 <= model["vocab_size"] <= 16384:
        raise ValueError("Invalid vocab_size")
    return model, training


def _encoded(records, tokenizer, context_length):
    from .tokenizer import ASSISTANT_ID, dialogue_tokens
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        tokens, supervised = dialogue_tokens(record["messages"], tokenizer)
        inputs = tokens[:-1]
        labels = [token if teach else -100 for token, teach in zip(tokens[1:], supervised[1:])]
        last_assistant = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
        labels[:last_assistant] = [-100] * last_assistant
        for start in range(0, len(inputs), context_length):
            chunk_inputs = inputs[start:start + context_length]
            chunk_labels = labels[start:start + context_length]
            if any(label != -100 for label in chunk_labels):
                result[record["split"]].append((chunk_inputs, chunk_labels))
    if any(not result[split] for split in result):
        raise ValueError("Each split needs at least one encoded assistant target")
    return result


def _batch(examples, indices, torch):
    selected = [examples[index] for index in indices]
    length = max(len(inputs) for inputs, _ in selected)
    inputs = torch.zeros((len(selected), length), dtype=torch.long)
    labels = torch.full((len(selected), length), -100, dtype=torch.long)
    for row, (tokens, targets) in enumerate(selected):
        inputs[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        labels[row, :len(targets)] = torch.tensor(targets, dtype=torch.long)
    return inputs, labels


def _evaluate(model, examples, batch_size):
    torch = _torch()
    import torch.nn.functional as functional
    total_loss, targets = 0.0, 0
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                inputs, labels = _batch(examples, range(start, min(start + batch_size, len(examples))), torch)
                logits = model(inputs)
                count = int((labels != -100).sum())
                loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                labels.reshape(-1), ignore_index=-100,
                                                reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite evaluation loss")
                total_loss += float(loss)
                targets += count
    finally:
        model.train(previous)
    return {"nll": total_loss / targets, "targetTokens": targets,
            "perplexity": math.exp(min(total_loss / targets, 700)), "examples": len(examples)}


def _baseline(examples, vocab_size):
    counts = [1] * vocab_size  # Train-only Laplace smoothing includes unseen validation/test tokens.
    for _, labels in examples:
        for token in labels:
            if token != -100:
                counts[token] += 1
    total = sum(counts)
    return [math.log(total / count) for count in counts]


def _baseline_score(examples, losses):
    values = [losses[token] for _, labels in examples for token in labels if token != -100]
    return {"nll": sum(values) / len(values), "targetTokens": len(values)}


def _checkpoint(run_dir, model, tokenizer, optimizer, scheduler, state, *, selected=False):
    from safetensors.torch import save_file
    torch = _torch()
    checkpoints = Path(run_dir) / "checkpoints"
    checkpoints.mkdir(exist_ok=True, mode=0o700)
    name = f"step-{state['step']:09d}-{uuid.uuid4().hex}"
    temporary = checkpoints / ("." + name)
    temporary.mkdir(mode=0o700)
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                  str(temporary / "weights.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state()}, temporary / "resume.pt")
        (temporary / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")
        checkpoint_state = copy.deepcopy(state)
        checkpoint_state["last_checkpoint"] = name
        if selected:
            checkpoint_state["best_checkpoint"] = name
        checkpoint_state["python_rng"] = random.getstate()
        _atomic_json(temporary / "state.json", checkpoint_state)
        hashes = {path.name: _sha(path) for path in temporary.iterdir() if path.is_file()}
        _atomic_json(temporary / "manifest.json", {"schemaVersion": 1, "files": hashes})
        for path in temporary.iterdir():
            os.chmod(path, 0o600)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(temporary, checkpoints / name)
        directory = os.open(checkpoints, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        state["last_checkpoint"] = name
        if selected:
            state["best_checkpoint"] = name
        return name
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)


def _checkpoint_files(run_dir, name):
    if not isinstance(name, str) or "/" in name or name.startswith("."):
        raise ValueError("Invalid checkpoint pointer")
    directory = Path(run_dir) / "checkpoints" / name
    manifest = _json_read(directory / "manifest.json")
    expected = {"weights.safetensors", "resume.pt", "tokenizer.json", "state.json"}
    if set(manifest) != {"schemaVersion", "files"} or type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] != 1 or set(manifest["files"]) != expected:
        raise ValueError("Invalid checkpoint manifest")
    for file, digest in manifest["files"].items():
        if not isinstance(digest, str) or _sha(directory / file) != digest:
            raise ValueError("Checkpoint integrity mismatch")
    return directory


def _tuple_tree(value):
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _load_checkpoint(run_dir, name):
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from .model import ConversationModel
    torch = _torch()
    directory = _checkpoint_files(run_dir, name)
    state = _json_read(directory / "state.json")
    model = ConversationModel(state["model_config"])
    weights = load_file(str(directory / "weights.safetensors"), device="cpu")
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Nonfinite checkpoint weights")
    reference = model.state_dict()
    if set(weights) != set(reference) or any(weights[key].shape != reference[key].shape or weights[key].dtype != reference[key].dtype for key in reference):
        raise ValueError("Checkpoint tensor keys/shapes/dtypes mismatch")
    model.load_state_dict(weights, strict=True)
    tokenizer = Tokenizer.from_str((directory / "tokenizer.json").read_text(encoding="utf-8"))
    from .tokenizer import validate_tokenizer
    validate_tokenizer(tokenizer)
    if tokenizer.get_vocab_size() != state["model_config"]["vocab_size"]:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    # Only files produced by this local trainer are resumed; no downloaded pickle is loaded.
    payload = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True)
    return model, tokenizer, state, payload



def _validate_resume_state(state, examples):
    expected = {"schemaVersion", "corpus_path", "corpus_sha256", "requested_model_config",
                "model_config", "training_config", "step", "epoch", "cursor", "order",
                "elapsed_seconds", "best_validation_nll", "bad_validations", "last_validation_step",
                "last_checkpoint", "best_checkpoint", "created_at", "assistant_targets", "window_policy",
                "python_rng"}
    if not isinstance(state, dict) or set(state) != expected or type(state["schemaVersion"]) is not int or state["schemaVersion"] != 1:
        raise ValueError("Invalid checkpoint state schema")
    requested, config = _configs(state["requested_model_config"], state["training_config"])
    if requested != state["requested_model_config"] or config != state["training_config"]:
        raise ValueError("Incomplete checkpoint configuration")
    actual = state["model_config"]
    if type(actual) is not dict or set(actual) != set(requested) or any(actual[key] != requested[key] for key in requested if key != "vocab_size") or type(actual["vocab_size"]) is not int or not 262 <= actual["vocab_size"] <= requested["vocab_size"]:
        raise ValueError("Checkpoint actual/requested model configuration disagree")
    if state["assistant_targets"] != "last_reply_only" or state["window_policy"] != "contiguous_nonoverlapping":
        raise ValueError("Unsupported checkpoint supervision policy")
    for key in ("step", "epoch", "cursor", "bad_validations"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError("Invalid checkpoint counters")
    if type(state["last_validation_step"]) is not int or not -1 <= state["last_validation_step"] <= state["step"]:
        raise ValueError("Invalid checkpoint validation step")
    for key in ("elapsed_seconds", "created_at"):
        if type(state[key]) not in (int, float) or not math.isfinite(state[key]) or state[key] < 0:
            raise ValueError("Invalid checkpoint time")
    score = state["best_validation_nll"]
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or score < 0):
        raise ValueError("Invalid checkpoint validation score")
    count = len(examples["train"])
    order, cursor = state["order"], state["cursor"]
    if type(order) is not list or any(type(index) is not int for index in order):
        raise ValueError("Invalid checkpoint order")
    if order:
        if len(order) != count or set(order) != set(range(count)) or not 0 <= cursor < count:
            raise ValueError("Invalid checkpoint permutation/cursor")
        if cursor % config["batch_size"]:
            raise ValueError("Checkpoint cursor is not a completed batch boundary")
    elif cursor != 0:
        raise ValueError("Empty checkpoint order has nonzero cursor")
    batches = math.ceil(count / config["batch_size"])
    if state["step"] != state["epoch"] * batches + cursor // config["batch_size"]:
        raise ValueError("Checkpoint step/epoch/cursor disagree")


def _validate_resume_payload(payload, optimizer, scheduler, model, state):
    torch = _torch()
    if type(payload) is not dict or set(payload) != {"optimizer", "scheduler", "torch_rng"}:
        raise ValueError("Invalid optimizer checkpoint payload")
    rng = payload["torch_rng"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.get_rng_state().shape:
        raise ValueError("Invalid torch RNG checkpoint")
    saved = payload["optimizer"]
    expected = optimizer.state_dict()
    if type(saved) is not dict or set(saved) != {"state", "param_groups"} or type(saved["state"]) is not dict:
        raise ValueError("Invalid optimizer state")
    if type(saved["param_groups"]) is not list or len(saved["param_groups"]) != 1:
        raise ValueError("Invalid optimizer groups")
    group = saved["param_groups"][0]
    reference = expected["param_groups"][0]
    if type(group) is not dict or set(group) != set(reference):
        raise ValueError("Invalid optimizer group fields")
    for key in reference:
        if key == "lr":
            expected_lr = state["training_config"]["learning_rate"] * min(1.0, (state["step"] + 1) / max(1, state["training_config"]["warmup_steps"]))
            if type(group[key]) not in (int, float) or group[key] != expected_lr:
                raise ValueError("Optimizer learning rate mismatch")
        elif group[key] != reference[key]:
            raise ValueError("Optimizer configuration mismatch")
    parameters = list(model.parameters())
    wanted = set(range(len(parameters))) if state["step"] else set()
    if set(saved["state"]) != wanted:
        raise ValueError("Optimizer parameter state mismatch")
    for index, moments in saved["state"].items():
        if type(moments) is not dict or set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid optimizer moments")
        step = moments["step"]
        if not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.shape != torch.Size([]) or not torch.isfinite(step).all() or float(step) != state["step"]:
            raise ValueError("Optimizer step mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = moments[key]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != parameters[index].dtype or tensor.shape != parameters[index].shape or not torch.isfinite(tensor).all():
                raise ValueError("Invalid optimizer moment tensor")
        if torch.any(moments["exp_avg_sq"] < 0):
            raise ValueError("Negative optimizer variance")
    saved_scheduler = payload["scheduler"]
    if type(saved_scheduler) is not dict or set(saved_scheduler) != set(scheduler.state_dict()):
        raise ValueError("Invalid scheduler fields")
    if saved_scheduler["last_epoch"] != state["step"] or saved_scheduler["_step_count"] != state["step"] + 1 or saved_scheduler["base_lrs"] != [state["training_config"]["learning_rate"]] or saved_scheduler["_last_lr"] != [group["lr"]] or saved_scheduler["lr_lambdas"] != [None]:
        raise ValueError("Scheduler step/configuration mismatch")


def _status_write(run_dir, state, status, reason=None, **extra):
    value = {"schemaVersion": 1, "status": status, "reason": reason,
             "pid": os.getpid(), "processBirth": _birth(os.getpid()), "heartbeatAt": time.time(),
             "step": state.get("step", 0), "epoch": state.get("epoch", 0),
             "elapsedSeconds": state.get("elapsed_seconds", 0),
             "lastCheckpoint": state.get("last_checkpoint"), "bestCheckpoint": state.get("best_checkpoint"),
             "bestValidationNll": state.get("best_validation_nll"),
             "corpusSha256": state.get("corpus_sha256"), "experimental": True,
             "environmentLifetimeGuaranteed": False, "cooperativeSignalInstalled": False, **extra}
    _atomic_json(Path(run_dir) / "status.json", value)
    return value


def _log(run_dir, event):
    path = Path(run_dir) / "events.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(_json_bytes({"at": time.time(), **event}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def train(corpus_path, run_dir, *, model_config=None, training_config=None,
          resume=False, budget_seconds=None):
    """Train a local run; budget_seconds limits this invocation, not the persistent total budget."""
    invocation_start = time.monotonic()
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if budget_seconds is not None and (type(budget_seconds) not in (int, float) or
            not math.isfinite(budget_seconds) or not 0 < budget_seconds <= 86400):
        raise ValueError("Invalid invocation time budget")
    requested_model, requested_training = _configs(model_config, training_config)
    torch = _torch()
    from .data import load_corpus
    from .model import ConversationModel, assistant_token_loss, validate_config
    from .tokenizer import fit_tokenizer
    corpus_path = Path(corpus_path).resolve()
    run_dir = Path(run_dir).resolve()
    corpus_hash = _sha(corpus_path)
    records = load_corpus(corpus_path)
    if _sha(corpus_path) != corpus_hash:
        raise ValueError("Corpus changed while loading")
    validate_config(requested_model)
    with _run_lock(run_dir):
        if resume:
            if (run_dir / "evaluation.json").exists():
                raise ValueError("Finalized runs cannot resume; start a new run")
            prior = _json_read(run_dir / "status.json")
            if prior.get("status") == "planned" and (prior.get("pid") != os.getpid() or prior.get("processBirth") != _birth(os.getpid())):
                raise RuntimeError("This run is reserved for its planned worker")
            if not prior.get("lastCheckpoint"):
                raise ValueError("Run has no resumable checkpoint")
            model, tokenizer, state, payload = _load_checkpoint(run_dir, prior["lastCheckpoint"])
            if corpus_hash != state["corpus_sha256"]:
                raise ValueError("Resume corpus hash mismatch")
            if model_config is not None and requested_model != state["requested_model_config"]:
                raise ValueError("Resume model configuration mismatch")
            if training_config is not None and requested_training != state["training_config"]:
                raise ValueError("Resume training configuration mismatch")
            config = state["training_config"]
        else:
            if (run_dir / "status.json").exists():
                existing = _json_read(run_dir / "status.json")
                if existing.get("status") != "planned" or existing.get("pid") != os.getpid() or existing.get("processBirth") != _birth(os.getpid()):
                    raise ValueError("Run already exists; resume it or choose a new directory")
            config = requested_training
            torch.set_num_threads(config["torch_threads"])
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            tokenizer = fit_tokenizer([record for record in records if record["split"] == "train"],
                                      vocab_size=requested_model["vocab_size"])
            actual_model = dict(requested_model, vocab_size=tokenizer.get_vocab_size())
            model = ConversationModel(actual_model)
            state = {"schemaVersion": 1, "corpus_path": str(corpus_path), "corpus_sha256": corpus_hash,
                     "requested_model_config": requested_model, "model_config": actual_model,
                     "training_config": config, "step": 0, "epoch": 0, "cursor": 0,
                     "order": [], "elapsed_seconds": 0.0, "best_validation_nll": None,
                     "bad_validations": 0, "last_validation_step": -1,
                     "last_checkpoint": None, "best_checkpoint": None,
                     "created_at": time.time(), "assistant_targets": "last_reply_only",
                     "window_policy": "contiguous_nonoverlapping"}
            payload = None
        torch.set_num_threads(config["torch_threads"])
        examples = _encoded(records, tokenizer, state["model_config"]["context_length"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        warmup = config["warmup_steps"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
        if payload is not None:
            _validate_resume_state(state, examples)
            _validate_resume_payload(payload, optimizer, scheduler, model, state)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(_tuple_tree(state.pop("python_rng")))
        start = invocation_start
        elapsed_before = state["elapsed_seconds"]
        stop_requested = []
        stop_file = run_dir / "stop-request.json"

        def pending_stop():
            if stop_requested:
                return True
            if stop_file.exists():
                request = _json_read(stop_file)
                return request.get("pid") == os.getpid() and request.get("processBirth") == _birth(os.getpid())
            return False
        previous_handlers = {}
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, lambda signum, frame: stop_requested.append(signum))
        except ValueError:
            # Non-main-thread callers retain their process-level signal handlers.
            pass

        def heartbeat():
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - start
            _status_write(run_dir, state, "running", cooperativeSignalInstalled=len(previous_handlers) == 2)

        def save(*, selected=False):
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - start
            _checkpoint(run_dir, model, tokenizer, optimizer, scheduler, state, selected=selected)
            heartbeat()
            # Retain only the durable last and validation-selected snapshots.
            import shutil
            keep = {state["last_checkpoint"], state["best_checkpoint"]}
            for directory in (run_dir / "checkpoints").iterdir():
                if directory.is_dir() and not directory.name.startswith(".") and directory.name not in keep:
                    shutil.rmtree(directory)

        def validate():
            score = _evaluate(model, examples["validation"], config["batch_size"])
            state["last_validation_step"] = state["step"]
            improved = (state["best_validation_nll"] is None or
                        score["nll"] < state["best_validation_nll"] - config["min_delta"])
            if improved:
                state["best_validation_nll"] = score["nll"]
                state["bad_validations"] = 0
                save(selected=True)
            else:
                state["bad_validations"] += 1
            _log(run_dir, {"event": "validation", "step": state["step"], "score": score,
                           "selected": improved})
            return improved

        try:
            heartbeat()
            if state["last_validation_step"] < 0:
                validate()  # Random initialization may remain best if training hurts validation.
            model.train()
            reason, status = None, "completed"
            last_heartbeat = time.monotonic()
            while True:
                elapsed_now = elapsed_before + time.monotonic() - start
                if pending_stop():
                    reason, status = "signal", "stopped"
                    break
                if state["bad_validations"] >= config["patience"]:
                    reason = "validation_patience"
                    break
                if elapsed_now >= config["max_seconds"]:
                    reason = "total_time_budget"
                    break
                if budget_seconds is not None and time.monotonic() - start >= budget_seconds:
                    reason, status = "invocation_time_budget", "stopped"
                    break
                if state["step"] >= config["max_steps"] or state["epoch"] >= config["max_epochs"]:
                    reason = "training_limit"
                    break
                if not state["order"]:
                    state["order"] = list(range(len(examples["train"])))
                    random.shuffle(state["order"])
                    state["cursor"] = 0
                end = min(state["cursor"] + config["batch_size"], len(state["order"]))
                inputs, labels = _batch(examples["train"], state["order"][state["cursor"]:end], torch)
                optimizer.zero_grad(set_to_none=True)
                loss = assistant_token_loss(model(inputs), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                state["step"] += 1
                state["cursor"] = end
                if end == len(state["order"]):
                    state["epoch"] += 1
                    state["order"] = []
                    state["cursor"] = 0
                checkpoint_saved = False
                if state["step"] % config["validation_every"] == 0:
                    checkpoint_saved = validate()
                if state["step"] % config["checkpoint_every"] == 0 and not checkpoint_saved:
                    save()
                if time.monotonic() - last_heartbeat >= 10:
                    heartbeat()
                    last_heartbeat = time.monotonic()
                _log(run_dir, {"event": "train", "step": state["step"], "nll": float(loss.detach()),
                               "gradientNorm": float(norm), "learningRate": optimizer.param_groups[0]["lr"]})
            # A segment pause preserves the validation cadence; final limits get one final validation.
            if status == "completed" and state["last_validation_step"] != state["step"]:
                validate()
            save()
            result = _status_write(run_dir, state, status, reason)
            _log(run_dir, {"event": status, "step": state["step"], "reason": reason})
            return result
        except BaseException as exc:
            # Keep the last coherent optimizer state when an interrupted step did not finish.
            # An exception can occur between backward and the optimizer boundary.
            # Preserve the preceding complete checkpoint rather than serialize partial updates.
            _status_write(run_dir, state, "failed", type(exc).__name__, error=str(exc)[:2000])
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)



def export_best(run_dir, export_dir):
    """Export a validation-selected immutable snapshot without reading the test split.

    This may run while the trainer writes. If retention removes a previously
    selected directory during loading, retry the new durable pointer.
    """
    from .artifacts import save_artifact
    run_dir = Path(run_dir).resolve()
    for attempt in range(3):
        status = _json_read(run_dir / "status.json")
        name = status.get("bestCheckpoint")
        if not name:
            raise ValueError("No validation-selected checkpoint")
        try:
            model, tokenizer, state, _ = _load_checkpoint(run_dir, name)
            break
        except FileNotFoundError:
            if attempt == 2:
                raise RuntimeError("Selected checkpoint changed during export; retry")
    metadata = {"corpusSha256": state["corpus_sha256"], "selectedStep": state["step"],
                "trainedFrom": "random_weights", "selectionSplit": "validation",
                "selectedCheckpoint": name, "validationNll": state["best_validation_nll"],
                "assistantTargets": "last_reply_only", "windowPolicy": "contiguous_nonoverlapping",
                "testEvaluated": False, "config": state["training_config"]}
    return save_artifact(export_dir, model, tokenizer, training=metadata, evaluation=None)


def finalize(run_dir, *, corpus_path=None, export_dir=None):
    """Evaluate the validation-selected snapshot on test, explicitly and outside the loop."""
    from .data import load_corpus
    from .artifacts import save_artifact
    run_dir = Path(run_dir).resolve()
    with _run_lock(run_dir):
        status = _json_read(run_dir / "status.json")
        if not status.get("bestCheckpoint"):
            raise ValueError("No validation-selected checkpoint")
        model, tokenizer, state, _ = _load_checkpoint(run_dir, status["bestCheckpoint"])
        path = Path(corpus_path or state["corpus_path"]).resolve()
        if _sha(path) != state["corpus_sha256"]:
            raise ValueError("Evaluation corpus hash mismatch")
        records = load_corpus(path)
        if _sha(path) != state["corpus_sha256"]:
            raise ValueError("Corpus changed while loading")
        examples = _encoded(records, tokenizer, state["model_config"]["context_length"])
        _torch().set_num_threads(state["training_config"]["torch_threads"])
        losses = _baseline(examples["train"], tokenizer.get_vocab_size())
        evaluation = {"schemaVersion": 1, "corpusSha256": state["corpus_sha256"],
                      "selectedCheckpoint": status["bestCheckpoint"], "selectedStep": state["step"],
                      "selectionSplit": "validation", "testUsedForSelection": False,
                      "assistantTargets": "last_reply_only", "windowPolicy": "contiguous_nonoverlapping",
                      "train": _evaluate(model, examples["train"], state["training_config"]["batch_size"]),
                      "validation": _evaluate(model, examples["validation"], state["training_config"]["batch_size"]),
                      "test": _evaluate(model, examples["test"], state["training_config"]["batch_size"]),
                      "unigramBaseline": {split: _baseline_score(examples[split], losses) for split in examples},
                      "experimental": True, "instructionFollowingValidated": False}
        _atomic_json(run_dir / "evaluation.json", evaluation)
        if export_dir is not None:
            save_artifact(export_dir, model, tokenizer,
                          training={"corpusSha256": state["corpus_sha256"], "selectedStep": state["step"],
                                    "trainedFrom": "random_weights", "selectionSplit": "validation",
                                    "config": state["training_config"]}, evaluation=evaluation)
        _log(run_dir, {"event": "explicit_finalization", "selectedStep": state["step"],
                       "testNll": evaluation["test"]["nll"]})
        return evaluation


def _main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--request", required=True)
    args = parser.parse_args()
    request = _json_read(args.request)
    allowed = {"corpus_path", "run_dir", "model_config", "training_config", "resume", "budget_seconds"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("Invalid worker request")
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                train(**request)
                break
            except RuntimeError as exc:
                if str(exc) != "This run already has an active writer" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    except BaseException as exc:
        run_dir = Path(request["run_dir"])
        status_path = run_dir / "status.json"
        prior = _json_read(status_path) if status_path.exists() else {}
        if prior.get("pid") == os.getpid() and prior.get("status") == "planned":
            _atomic_json(status_path, {**prior, "status": "failed", "reason": type(exc).__name__,
                                     "error": str(exc)[:2000], "heartbeatAt": time.time()})
        raise


if __name__ == "__main__":
    _main()
