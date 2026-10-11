"""Local, resumable scratch training; test scores are computed only by finalize()."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
            allowed = set(defaults) | ({"window_policy", "loss_reduction", "sampling_weights", "projection_mode", "adaptive_sampling", "objective", "train_precision", "partition_policy"} if result is training else set())
            if not isinstance(supplied, dict) or set(supplied) - allowed:
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
    if training.get("window_policy", "contiguous_nonoverlapping") not in ("contiguous_nonoverlapping", "sliding_context"):
        raise ValueError("Invalid window_policy")
    if training.get("loss_reduction", "token_mean") not in ("token_mean", "sequence_mean"):
        raise ValueError("Invalid loss_reduction")
    if training.get("projection_mode", "full") not in ("full", "supervised"):
        raise ValueError("Invalid projection_mode")
    if "train_precision" in training and (type(training["train_precision"]) is not str or training["train_precision"] != "cpu_bfloat16"):
        raise ValueError("train_precision requires cpu_bfloat16")
    if "partition_policy" in training and (type(training["partition_policy"]) is not str or training["partition_policy"] != "train_validation_only"):
        raise ValueError("partition_policy requires train_validation_only")
    if "objective" in training and training["objective"] not in ("assistant_last_reply", "causal_lm"):
        raise ValueError("Invalid training objective")
    if "sampling_weights" in training:
        _validate_sampling_rules(training["sampling_weights"])
        training["sampling_weights"] = copy.deepcopy(training["sampling_weights"])
    if "adaptive_sampling" in training:
        from .adaptive import validate_config
        training["adaptive_sampling"] = validate_config(training["adaptive_sampling"])
        if training["adaptive_sampling"] is None:
            raise ValueError("adaptive_sampling requires an explicit configuration")
    # Constructor validates architecture; the requested vocabulary is bounded before fitting.
    if type(model["vocab_size"]) is not int or not 262 <= model["vocab_size"] <= 16384:
        raise ValueError("Invalid vocab_size")
    return model, training


def _validate_sampling_rules(rules):
    """Explicit provenance selectors; matched factors multiply from base weight 1."""
    if type(rules) is not list or not 1 <= len(rules) <= 64:
        raise ValueError("sampling_weights requires 1 to 64 rules")
    seen = set()
    for rule in rules:
        if type(rule) is not dict or not {"dataset", "factor"} <= set(rule) or set(rule) - {"dataset", "prefixPairs", "factor"}:
            raise ValueError("Sampling rules require dataset/factor and optional prefixPairs")
        dataset = rule["dataset"]
        if type(dataset) is not str or not dataset.strip() or len(dataset) > 256 or any(ord(char) < 32 for char in dataset):
            raise ValueError("Invalid sampling dataset")
        try:
            dataset.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Invalid sampling dataset UTF-8") from exc
        if "prefixPairs" in rule and (type(rule["prefixPairs"]) is not int or not 1 <= rule["prefixPairs"] <= 32):
            raise ValueError("Invalid sampling prefixPairs")
        factor = rule["factor"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or not 0 < factor <= 1_000_000:
            raise ValueError("Invalid sampling factor")
        selector = (dataset, rule.get("prefixPairs"))
        if selector in seen:
            raise ValueError("Duplicate sampling selector")
        seen.add(selector)


def _source_weight(source, rules):
    matched, weight = [], 1.0
    for index, rule in enumerate(rules):
        if source["dataset"] != rule["dataset"]:
            continue
        if "prefixPairs" in rule and (type(source.get("prefixPairs")) is not int or source["prefixPairs"] != rule["prefixPairs"]):
            continue
        weight *= rule["factor"]
        matched.append(index)
    if not math.isfinite(weight) or not 0 < weight <= 1_000_000_000_000:
        raise ValueError("Combined sampling weight must be positive, finite and at most 1e12")
    return weight, matched


def _window_ranges(input_length, context_length, window_policy):
    if window_policy not in ("contiguous_nonoverlapping", "sliding_context"):
        raise ValueError("Invalid window_policy")
    stride = context_length if window_policy == "contiguous_nonoverlapping" else max(1, context_length // 2)
    previous_end = 0
    for nominal_start in range(0, input_length, stride):
        end = min(nominal_start + context_length, input_length)
        start = nominal_start if window_policy == "contiguous_nonoverlapping" else max(0, end - context_length)
        yield start, end, previous_end
        previous_end = end
        if end == input_length:
            break


def _encoding_kwargs(config):
    result = {"objective": config["objective"]} if "objective" in config else {}
    if "partition_policy" in config:
        result["partition_policy"] = config["partition_policy"]
    return result


def _supervision_policy(config):
    return "all_text_tokens" if config.get("objective") == "causal_lm" else "last_reply_only"


def _load_training_corpus(path, config):
    if config.get("objective") == "causal_lm":
        from .language_data import load_text_corpus
        records = load_text_corpus(path)
    else:
        from .data import load_corpus
        records = load_corpus(path)
    if config.get("partition_policy") == "train_validation_only" and any(record["split"] == "test" for record in records):
        raise ValueError("train_validation_only corpus must not contain TEST records")
    return records


def _encoded(records, tokenizer, context_length, window_policy="contiguous_nonoverlapping", *, objective=None, partition_policy=None):
    if partition_policy is not None:
        return _encoded_data(records, tokenizer, context_length, window_policy,
                             objective=objective, partition_policy=partition_policy)[0]
    if objective is None:
        return _encoded_data(records, tokenizer, context_length, window_policy)[0]
    return _encoded_data(records, tokenizer, context_length, window_policy, objective=objective)[0]


def _encoded_data(records, tokenizer, context_length, window_policy, sampling_rules=None, *, objective=None, partition_policy=None):
    """Retain causal overlap; supervise each objective's target exactly once."""
    from .tokenizer import ASSISTANT_ID, dialogue_tokens
    if objective not in (None, "assistant_last_reply", "causal_lm"):
        raise ValueError("Invalid encoding objective")
    if partition_policy is not None and (type(partition_policy) is not str or partition_policy != "train_validation_only"):
        raise ValueError("Invalid encoding partition_policy")
    if partition_policy == "train_validation_only" and any(record["split"] == "test" for record in records):
        raise ValueError("train_validation_only corpus must not contain TEST records")
    result = {"train": [], "validation": [], "test": []}
    weights, record_ids, strata = [], [], {}
    rule_matches = [0] * len(sampling_rules or [])
    for record in records:
        if objective == "causal_lm":
            from .tokenizer import text_tokens
            tokens = text_tokens(record["text"], tokenizer)
            inputs, labels = tokens[:-1], tokens[1:]
        else:
            tokens, supervised = dialogue_tokens(record["messages"], tokenizer)
            inputs = tokens[:-1]
            labels = [token if teach else -100 for token, teach in zip(tokens[1:], supervised[1:])]
            last_assistant = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
            labels[:last_assistant] = [-100] * last_assistant
        for start, end, previous_end in _window_ranges(len(inputs), context_length, window_policy):
            chunk_inputs = inputs[start:end]
            chunk_labels = labels[start:end]
            overlap = max(0, previous_end - start)
            chunk_labels[:overlap] = [-100] * overlap
            if any(label != -100 for label in chunk_labels):
                result[record["split"]].append((chunk_inputs, chunk_labels))
                if sampling_rules is not None and record["split"] == "train":
                    source = record["source"]
                    weight, matched = _source_weight(source, sampling_rules)
                    weights.append(weight)
                    record_ids.append(record["id"])
                    for index in matched:
                        rule_matches[index] += 1
                    prefix = source.get("prefixPairs") if type(source.get("prefixPairs")) is int else None
                    key = (source["dataset"], prefix)
                    value = strata.setdefault(key, {"windows": 0, "ids": set(), "totalWeight": 0.0})
                    value["windows"] += 1
                    value["ids"].add(record["id"])
                    value["totalWeight"] += weight
    required = ("train", "validation") if partition_policy == "train_validation_only" else result
    if any(not result[split] for split in required):
        raise ValueError("Each split needs at least one encoded text target" if objective == "causal_lm"
                         else "Each split needs at least one encoded assistant target")
    summary = None
    if sampling_rules is not None:
        if any(count == 0 for count in rule_matches):
            raise ValueError("Every sampling rule must match an encoded training window")
        total = sum(weights)
        summary = {"policy": "provenance_weighted_with_replacement", "unit": "encoded_window",
                   "baseWeight": 1.0, "factorsMultiply": True, "epochDraws": len(weights),
                   "uniqueWindows": len(weights), "uniqueRecords": len(set(record_ids)), "totalWeight": total,
                   "ruleMatchedTrainingWindows": rule_matches, "validationSampled": False, "testSampled": False,
                   "rules": copy.deepcopy(sampling_rules),
                   "strata": [{"dataset": dataset, "prefixPairs": prefix, "windows": value["windows"],
                               "uniqueRecords": len(value["ids"]), "totalWeight": value["totalWeight"],
                               "expectedShare": value["totalWeight"] / total,
                               "expectedDrawsPerEpoch": len(weights) * value["totalWeight"] / total}
                              for (dataset, prefix), value in sorted(strata.items(), key=lambda item: (item[0][0], str(item[0][1])))]}
    return result, weights, record_ids, summary


def _sampling_metadata(state):
    result = {"trainingSampling": copy.deepcopy(state["sampling"])} if "sampling" in state else {}
    config = state.get("training_config", {})
    if "adaptive_sampling" in config:
        if "trainingSampling" in result:
            result["trainingSampling"]["purpose"] = "fixed_provenance_baseline_for_adaptive_sampling"
        result["trainingAdaptiveSampling"] = {
            "policy": "train_loss_ema", "unit": "encoded_window",
            "config": copy.deepcopy(config["adaptive_sampling"]),
            "validationSampled": False, "testSampled": False,
            "refresh": "after draining the current saved chunk",
            "maxDrawsBeforeRefresh": config["batch_size"] * config["adaptive_sampling"]["refresh_every_steps"],
        }
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


def _evaluate(model, examples, batch_size, loss_reduction="token_mean", projection_mode="full"):
    if loss_reduction not in ("token_mean", "sequence_mean"):
        raise ValueError("Invalid loss_reduction")
    if projection_mode not in ("full", "supervised"):
        raise ValueError("Invalid projection_mode")
    torch = _torch()
    import torch.nn.functional as functional
    if projection_mode == "supervised":
        from .model import assistant_token_losses
    total_loss, targets = 0.0, 0
    sequence_loss, sequences = 0.0, 0
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                inputs, labels = _batch(examples, range(start, min(start + batch_size, len(examples))), torch)
                selected = labels != -100
                count = int(selected.sum())
                if projection_mode == "supervised":
                    logits = model.selected_logits(inputs, selected)
                    token_losses = assistant_token_losses(logits, labels, projection_mode="supervised")
                    loss = token_losses.sum()
                else:
                    logits = model(inputs)
                    loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                    labels.reshape(-1), ignore_index=-100,
                                                    reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite evaluation loss")
                total_loss += float(loss)
                targets += count
                if loss_reduction == "sequence_mean":
                    if projection_mode == "full":
                        token_losses = functional.cross_entropy(logits[selected], labels[selected], reduction="none")
                    rows = selected.nonzero(as_tuple=False)[:, 0]
                    sums = token_losses.new_zeros(labels.shape[0]).scatter_add(0, rows, token_losses)
                    counts = selected.sum(dim=1)
                    valid = counts > 0
                    sequence_loss += float((sums[valid] / counts[valid]).sum())
                    sequences += int(valid.sum())
    finally:
        model.train(previous)
    token_nll = total_loss / targets
    nll = sequence_loss / sequences if loss_reduction == "sequence_mean" else token_nll
    result = {"nll": nll, "targetTokens": targets,
              "perplexity": math.exp(min(nll, 700)), "examples": len(examples)}
    if loss_reduction == "sequence_mean":
        result.update(tokenWeightedNll=token_nll, sequenceMeanNll=nll,
                      lossReduction="sequence_mean", sequenceUnit="encoded_window")
    return result


def _baseline(examples, vocab_size):
    counts = [1] * vocab_size  # Train-only Laplace smoothing includes unseen validation/test tokens.
    for _, labels in examples:
        for token in labels:
            if token != -100:
                counts[token] += 1
    total = sum(counts)
    return [math.log(total / count) for count in counts]


def _baseline_score(examples, losses, loss_reduction="token_mean"):
    if loss_reduction not in ("token_mean", "sequence_mean"):
        raise ValueError("Invalid loss_reduction")
    values = [losses[token] for _, labels in examples for token in labels if token != -100]
    token_nll = sum(values) / len(values)
    if loss_reduction == "token_mean":
        return {"nll": token_nll, "targetTokens": len(values)}
    per_sequence = [[losses[token] for token in labels if token != -100] for _, labels in examples]
    per_sequence = [sum(values) / len(values) for values in per_sequence if values]
    sequence_nll = sum(per_sequence) / len(per_sequence)
    return {"nll": sequence_nll, "targetTokens": len(values),
            "tokenWeightedNll": token_nll, "sequenceMeanNll": sequence_nll,
            "lossReduction": "sequence_mean", "sequenceUnit": "encoded_window"}


def _loss_metadata(config):
    # Missing fields remain missing for legacy checkpoints and artifact schemas.
    result = {}
    if "partition_policy" in config:
        result["partitionPolicy"] = config["partition_policy"]
        result["testPartitionPresent"] = False
    if "train_precision" in config:
        result["trainPrecision"] = config["train_precision"]
    if "projection_mode" in config:
        result["projectionMode"] = config["projection_mode"]
    if "loss_reduction" in config:
        result["lossReduction"] = config["loss_reduction"]
    if config.get("loss_reduction") == "sequence_mean":
        result["sequenceUnit"] = "encoded_window"
    if "objective" in config:
        result["trainingObjective"] = config["objective"]
        result["assistantTargets"] = _supervision_policy(config)
        result["corpusFormat"] = "text" if config["objective"] == "causal_lm" else "messages"
    return result


def _precision_runtime(config):
    """Pin same-mode resume to the observed local Torch/CPU math environment."""
    torch = _torch()
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("cpu_bfloat16 training requires no enclosing CPU autocast")
    flags = next((line.split(":", 1)[1].strip() for line in
                  Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")), "")
    return {"trainPrecision": "cpu_bfloat16", "torchVersion": str(torch.__version__),
            "torchBuildSha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "cpuFlagsSha256": hashlib.sha256(flags.encode()).hexdigest(),
            "cpuCapability": torch.backends.cpu.get_cpu_capability(),
            "torchThreads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads(),
            "mkldnnEnabled": torch.backends.mkldnn.enabled,
            "mkldnnDeterministic": torch.backends.mkldnn.deterministic,
            "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministicWarnOnly": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32MatmulPrecision": torch.get_float32_matmul_precision()}


def _bf16_train_forward(model, inputs, selected=None):
    torch = _torch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(inputs) if selected is None else model.selected_logits(inputs, selected)
    if logits.dtype != torch.bfloat16:
        raise ValueError("cpu_bfloat16 forward did not produce BF16 logits")
    # Cross entropy and reductions run on F32 logits outside autocast.
    return logits.float()


def _precision_storage_guard(model, optimizer):
    torch = _torch()
    tensors = list(model.state_dict().values())
    tensors += [value for moments in optimizer.state.values() for value in moments.values()
                if isinstance(value, torch.Tensor)]
    for tensor in tensors:
        if tensor.is_floating_point() and (tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError("cpu_bfloat16 requires finite F32 master parameters/Adam state")


def _evaluate_configured(model, examples, config):
    if config.get("train_precision") == "cpu_bfloat16":
        with _torch().autocast("cpu", enabled=False):
            return _evaluate_configured_fp32(model, examples, config)
    return _evaluate_configured_fp32(model, examples, config)


def _evaluate_configured_fp32(model, examples, config):
    if config.get("projection_mode", "full") == "supervised":
        return _evaluate(model, examples, config["batch_size"],
                         loss_reduction=config.get("loss_reduction", "token_mean"),
                         projection_mode="supervised")
    if config.get("loss_reduction", "token_mean") == "token_mean":
        return _evaluate(model, examples, config["batch_size"])
    return _evaluate(model, examples, config["batch_size"], loss_reduction="sequence_mean")


def _checkpoint(run_dir, model, tokenizer, optimizer, scheduler, state, *, selected=False):
    from safetensors.torch import save_file
    torch = _torch()
    if state["training_config"].get("train_precision") == "cpu_bfloat16":
        _precision_storage_guard(model, optimizer)
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




def _validate_lineage(lineage):
    import re
    fields = {"parentModelId", "parentWeightsSha256", "parentCorpusSha256", "rootOrigin"}
    if type(lineage) is not dict or set(lineage) != fields or lineage["rootOrigin"] != "random_weights":
        raise ValueError("Invalid own-model lineage")
    if type(lineage["parentModelId"]) is not str or re.fullmatch(r"unidad-conversation-[0-9a-f]{64}", lineage["parentModelId"]) is None:
        raise ValueError("Invalid lineage parent model identity")
    for key in ("parentWeightsSha256", "parentCorpusSha256"):
        if type(lineage[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", lineage[key]) is None:
            raise ValueError("Invalid lineage SHA-256")


def _initial_own_model(path, model_config):
    from .artifacts import load_artifact
    from .model import validate_config
    model, tokenizer, manifest = load_artifact(path)
    config = model.config
    if model_config is not None and validate_config({**config, **model_config}) != config:
        raise ValueError("Initial model configuration mismatch")
    metadata = manifest["training"]
    origin = metadata.get("trainedFrom")
    if origin == "own_checkpoint":
        _validate_lineage(metadata.get("lineage"))
    elif origin != "random_weights" or "lineage" in metadata:
        raise ValueError("Initial artifact must originate from Unidad's own random-weight training")
    lineage = {"parentModelId": manifest["modelId"],
               "parentWeightsSha256": manifest["files"]["weights.safetensors"]["sha256"],
               "parentCorpusSha256": metadata.get("corpusSha256"), "rootOrigin": "random_weights"}
    _validate_lineage(lineage)
    return model, tokenizer, config, lineage


def _origin_metadata(state):
    if "lineage" not in state:
        return {"trainedFrom": "random_weights"}
    _validate_lineage(state["lineage"])
    return {"trainedFrom": "own_checkpoint", "lineage": copy.deepcopy(state["lineage"]),
            "optimizerReset": True, "schedulerReset": True}


def _validate_resume_state(state, examples):
    expected = {"schemaVersion", "corpus_path", "corpus_sha256", "requested_model_config",
                "model_config", "training_config", "step", "epoch", "cursor", "order",
                "elapsed_seconds", "best_validation_nll", "bad_validations", "last_validation_step",
                "last_checkpoint", "best_checkpoint", "created_at", "assistant_targets", "window_policy",
                "python_rng"}
    if (not isinstance(state, dict) or not expected <= set(state) or set(state) - expected - {"lineage", "sampling", "adaptive_sampling", "epoch_cursor", "train_precision_runtime"}
            or type(state["schemaVersion"]) is not int or state["schemaVersion"] != 1):
        raise ValueError("Invalid checkpoint state schema")
    if "lineage" in state:
        _validate_lineage(state["lineage"])
    requested, config = _configs(state["requested_model_config"], state["training_config"])
    if requested != state["requested_model_config"] or config != state["training_config"]:
        raise ValueError("Incomplete checkpoint configuration")
    if config.get("partition_policy") == "train_validation_only" and (examples["test"] or not examples["train"] or not examples["validation"]):
        raise ValueError("Checkpoint train_validation_only partition mismatch")
    if ("train_precision_runtime" in state) != (config.get("train_precision") == "cpu_bfloat16"):
        raise ValueError("Checkpoint precision runtime policy mismatch")
    weighted = "sampling_weights" in config
    adaptive = "adaptive_sampling" in config
    if ("adaptive_sampling" in state) != adaptive or ("epoch_cursor" in state) != adaptive:
        raise ValueError("Checkpoint adaptive sampling policy mismatch")
    if ("sampling" in state) != weighted:
        raise ValueError("Checkpoint sampling policy mismatch")
    if weighted and (type(state["sampling"]) is not dict or state["sampling"].get("rules") != config["sampling_weights"]):
        raise ValueError("Checkpoint sampling rules mismatch")
    actual = state["model_config"]
    if type(actual) is not dict or set(actual) != set(requested) or any(actual[key] != requested[key] for key in requested if key != "vocab_size") or type(actual["vocab_size"]) is not int or not 262 <= actual["vocab_size"] <= requested["vocab_size"]:
        raise ValueError("Checkpoint actual/requested model configuration disagree")
    if state["assistant_targets"] != _supervision_policy(config) or state["window_policy"] != config.get("window_policy", "contiguous_nonoverlapping"):
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
        if (any(not 0 <= index < count for index in order) or not 0 <= cursor < len(order)
                or (not adaptive and len(order) != count)
                or (not weighted and not adaptive and set(order) != set(range(count)))):
            raise ValueError("Invalid checkpoint permutation/cursor")
        if cursor % config["batch_size"]:
            raise ValueError("Checkpoint cursor is not a completed batch boundary")
    elif cursor != 0:
        raise ValueError("Empty checkpoint order has nonzero cursor")
    batches = math.ceil(count / config["batch_size"])
    epoch_cursor = cursor
    if adaptive:
        epoch_cursor = state["epoch_cursor"]
        if (type(epoch_cursor) is not int or not 0 <= epoch_cursor < count
                or epoch_cursor % config["batch_size"] or epoch_cursor < cursor):
            raise ValueError("Invalid adaptive epoch cursor")
        chunk_capacity = config["batch_size"] * config["adaptive_sampling"]["refresh_every_steps"]
        chunk_start = epoch_cursor - cursor
        if chunk_start % chunk_capacity:
            raise ValueError("Invalid adaptive refresh boundary")
        if order:
            expected_draws = min(chunk_capacity, count - chunk_start)
            if len(order) != expected_draws:
                raise ValueError("Invalid adaptive sampling chunk")
        cache = state["adaptive_sampling"]
        counts = cache.get("counts") if type(cache) is dict else None
        if (type(counts) is not list or len(counts) != count
                or any(type(value) is not int or value < 0 for value in counts)
                or sum(counts) != state["epoch"] * count + epoch_cursor):
            raise ValueError("Adaptive observations disagree with completed training draws")
        if any(counts[index] < consumed for index, consumed in Counter(order[:cursor]).items()):
            raise ValueError("Adaptive observations omit consumed chunk draws")
    if state["step"] != state["epoch"] * batches + epoch_cursor // config["batch_size"]:
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
             "environmentLifetimeGuaranteed": False, "cooperativeSignalInstalled": False,
             **_origin_metadata(state), **_loss_metadata(state.get("training_config", {})),
             **_sampling_metadata(state), **extra}
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
          resume=False, budget_seconds=None, initial_model=None):
    """Train a local run; budget_seconds limits this invocation, not the persistent total budget."""
    invocation_start = time.monotonic()
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if resume and initial_model is not None:
        raise ValueError("initial_model is only allowed for a new stage, not resume")
    if budget_seconds is not None and (type(budget_seconds) not in (int, float) or
            not math.isfinite(budget_seconds) or not 0 < budget_seconds <= 86400):
        raise ValueError("Invalid invocation time budget")
    requested_model, requested_training = _configs(model_config, training_config)
    torch = _torch()
    from .model import ConversationModel, assistant_token_loss, validate_config
    from .tokenizer import fit_tokenizer
    corpus_path = Path(corpus_path).resolve()
    run_dir = Path(run_dir).resolve()
    corpus_hash = _sha(corpus_path)
    if initial_model is None and not resume:
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
            if model_config is not None:
                candidate = validate_config({**state["requested_model_config"], **model_config}) if "lineage" in state else validate_config(requested_model)
                if candidate != state["requested_model_config"]:
                    raise ValueError("Resume model configuration mismatch")
            if training_config is not None and requested_training != state["training_config"]:
                raise ValueError("Resume training configuration mismatch")
            config = state["training_config"]
            records = _load_training_corpus(corpus_path, config)
            if _sha(corpus_path) != corpus_hash:
                raise ValueError("Corpus changed while loading")
        else:
            if (run_dir / "status.json").exists():
                existing = _json_read(run_dir / "status.json")
                if existing.get("status") != "planned" or existing.get("pid") != os.getpid() or existing.get("processBirth") != _birth(os.getpid()):
                    raise ValueError("Run already exists; resume it or choose a new directory")
            config = requested_training
            records = _load_training_corpus(corpus_path, config)
            if _sha(corpus_path) != corpus_hash:
                raise ValueError("Corpus changed while loading")
            torch.set_num_threads(config["torch_threads"])
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            lineage = None
            if initial_model is not None:
                model, tokenizer, actual_model, lineage = _initial_own_model(initial_model, model_config)
                requested_model = dict(actual_model)
            else:
                fit = fit_tokenizer
                if config.get("objective") == "causal_lm":
                    from .tokenizer import fit_text_tokenizer
                    fit = fit_text_tokenizer
                tokenizer = fit([record for record in records if record["split"] == "train"],
                                vocab_size=requested_model["vocab_size"])
                actual_model = dict(requested_model, vocab_size=tokenizer.get_vocab_size())
                model = ConversationModel(actual_model)
            state = {"schemaVersion": 1, "corpus_path": str(corpus_path), "corpus_sha256": corpus_hash,
                     "requested_model_config": requested_model, "model_config": actual_model,
                     "training_config": config, "step": 0, "epoch": 0, "cursor": 0,
                     "order": [], "elapsed_seconds": 0.0, "best_validation_nll": None,
                     "bad_validations": 0, "last_validation_step": -1,
                     "last_checkpoint": None, "best_checkpoint": None,
                     "created_at": time.time(), "assistant_targets": _supervision_policy(config),
                     "window_policy": config.get("window_policy", "contiguous_nonoverlapping")}
            if lineage is not None:
                state["lineage"] = lineage
            payload = None
        torch.set_num_threads(config["torch_threads"])
        if config.get("train_precision") == "cpu_bfloat16":
            runtime = _precision_runtime(config)
            if payload is not None:
                saved_runtime = state.get("train_precision_runtime")
                if (type(saved_runtime) is not dict or set(saved_runtime) != set(runtime)
                        or any(type(saved_runtime[key]) is not type(value) or saved_runtime[key] != value
                               for key, value in runtime.items())):
                    raise ValueError("Resume precision runtime mismatch")
            if payload is None:
                state["train_precision_runtime"] = runtime
        if "sampling_weights" in config:
            examples, sampling_weights, sampling_ids, sampling_summary = _encoded_data(
                records, tokenizer, state["model_config"]["context_length"], state["window_policy"], config["sampling_weights"], **_encoding_kwargs(config))
            if payload is not None:
                if state.get("sampling") != sampling_summary:
                    raise ValueError("Checkpoint sampling distribution differs from its corpus/configuration")
            else:
                state["sampling"] = sampling_summary
        else:
            if "adaptive_sampling" in config:
                examples, sampling_weights, sampling_ids, _ = _encoded_data(
                    records, tokenizer, state["model_config"]["context_length"], state["window_policy"], [], **_encoding_kwargs(config))
            else:
                examples = _encoded(records, tokenizer, state["model_config"]["context_length"], state["window_policy"], **_encoding_kwargs(config))
                sampling_weights, sampling_ids = None, None
        if payload is not None:
            _validate_resume_state(state, examples)
        adaptive_sampler = None
        adaptive_sources = None
        if "adaptive_sampling" in config:
            from .adaptive import AdaptiveSampler
            adaptive_sampler = AdaptiveSampler(
                [f"train-window:{index}" for index in range(len(examples["train"]))],
                sampling_weights, config["adaptive_sampling"],
                state=state.get("adaptive_sampling") if payload is not None else None)
            source_by_id = {record["id"]: record["source"]["dataset"] for record in records if record["split"] == "train"}
            adaptive_sources = [source_by_id[record_id] for record_id in sampling_ids]
            if payload is None:
                state["adaptive_sampling"] = adaptive_sampler.state_dict()
                state["epoch_cursor"] = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        warmup = config["warmup_steps"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
        if payload is not None:
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
            if adaptive_sampler is not None:
                state["adaptive_sampling"] = adaptive_sampler.state_dict()
            _checkpoint(run_dir, model, tokenizer, optimizer, scheduler, state, selected=selected)
            heartbeat()
            # Retain only the durable last and validation-selected snapshots.
            import shutil
            keep = {state["last_checkpoint"], state["best_checkpoint"]}
            for directory in (run_dir / "checkpoints").iterdir():
                if directory.is_dir() and not directory.name.startswith(".") and directory.name not in keep:
                    shutil.rmtree(directory)

        def validate():
            score = _evaluate_configured(model, examples["validation"], config)
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
                    if adaptive_sampler is not None:
                        weights = adaptive_sampler.weights()
                        draws = min(config["batch_size"] * config["adaptive_sampling"]["refresh_every_steps"],
                                    len(examples["train"]) - state["epoch_cursor"])
                        state["order"] = random.choices(range(len(examples["train"])), weights=weights, k=draws)
                        ratios = [weight / base for weight, base in zip(weights, sampling_weights)]
                        source_mass, base_mass = defaultdict(float), defaultdict(float)
                        for source, weight, base in zip(adaptive_sources, weights, sampling_weights):
                            source_mass[source] += weight
                            base_mass[source] += base
                        total, base_total = math.fsum(weights), math.fsum(sampling_weights)
                        actual_draws = Counter(adaptive_sources[index] for index in state["order"])
                        _log(run_dir, {"event": "adaptive_sampling_chunk", "step": state["step"],
                                       "epoch": state["epoch"], "epochCursor": state["epoch_cursor"],
                                       "draws": draws, "uniqueWindowsDrawn": len(set(state["order"])),
                                       "uniqueRecordsDrawn": len({sampling_ids[index] for index in state["order"]}),
                                       "minWeightRatio": min(ratios), "maxWeightRatio": max(ratios),
                                       "sources": [{"dataset": source, "baseShare": base_mass[source] / base_total,
                                                    "adaptiveShare": source_mass[source] / total,
                                                    "sampledDraws": actual_draws[source]}
                                                   for source in sorted(base_mass)]})
                    elif sampling_weights is None:
                        state["order"] = list(range(len(examples["train"])))
                        random.shuffle(state["order"])
                    else:
                        # The full sampled order and RNG are checkpointed. N draws
                        # define an epoch; they are not N distinct training records.
                        state["order"] = random.choices(range(len(examples["train"])),
                                                        weights=sampling_weights, k=len(examples["train"]))
                        unique = len(set(state["order"]))
                        _log(run_dir, {"event": "sampling_epoch", "epoch": state["epoch"],
                                       "draws": len(state["order"]), "uniqueWindowsDrawn": unique,
                                       "uniqueRecordsDrawn": len({sampling_ids[index] for index in state["order"]}),
                                       "repeatedWindowDraws": len(state["order"]) - unique})
                    state["cursor"] = 0
                end = min(state["cursor"] + config["batch_size"], len(state["order"]))
                batch_indices = state["order"][state["cursor"]:end]
                inputs, labels = _batch(examples["train"], batch_indices, torch)
                optimizer.zero_grad(set_to_none=True)
                if adaptive_sampler is not None:
                    # Reuse one CE computation for both the gradient objective
                    # and detached TRAIN-only difficulty observations.
                    from .model import assistant_token_losses
                    projection = config.get("projection_mode", "full")
                    selected = labels != -100
                    if config.get("train_precision") == "cpu_bfloat16":
                        logits = _bf16_train_forward(model, inputs, selected if projection == "supervised" else None)
                    else:
                        logits = model.selected_logits(inputs, selected) if projection == "supervised" else model(inputs)
                    token_losses = assistant_token_losses(logits, labels, projection_mode=projection)
                    rows = selected.nonzero(as_tuple=False)[:, 0]
                    sums = token_losses.new_zeros(labels.shape[0]).scatter_add(0, rows, token_losses)
                    sequence_losses = sums / selected.sum(dim=1)
                    loss = (sequence_losses.mean() if config.get("loss_reduction", "token_mean") == "sequence_mean"
                            else token_losses.mean())
                    difficulty = sequence_losses.detach().tolist()
                    adaptive_sampler.validate_update(batch_indices, difficulty)
                elif config.get("projection_mode", "full") == "supervised":
                    logits = (_bf16_train_forward(model, inputs, labels != -100)
                              if config.get("train_precision") == "cpu_bfloat16"
                              else model.selected_logits(inputs, labels != -100))
                    loss = assistant_token_loss(logits, labels,
                                                loss_reduction=config.get("loss_reduction", "token_mean"),
                                                projection_mode="supervised")
                elif config.get("loss_reduction", "token_mean") == "token_mean":
                    logits = (_bf16_train_forward(model, inputs)
                              if config.get("train_precision") == "cpu_bfloat16" else model(inputs))
                    loss = assistant_token_loss(logits, labels)
                else:
                    logits = (_bf16_train_forward(model, inputs)
                              if config.get("train_precision") == "cpu_bfloat16" else model(inputs))
                    loss = assistant_token_loss(logits, labels, loss_reduction="sequence_mean")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                if adaptive_sampler is not None:
                    adaptive_sampler.update(batch_indices, difficulty)
                state["step"] += 1
                state["cursor"] = end
                if adaptive_sampler is not None:
                    state["epoch_cursor"] += len(batch_indices)
                if end == len(state["order"]):
                    if adaptive_sampler is None or state["epoch_cursor"] == len(examples["train"]):
                        state["epoch"] += 1
                        if adaptive_sampler is not None:
                            state["epoch_cursor"] = 0
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
                **_origin_metadata(state), **_loss_metadata(state["training_config"]), **_sampling_metadata(state), "selectionSplit": "validation",
                "selectedCheckpoint": name, "validationNll": state["best_validation_nll"],
                "assistantTargets": state["assistant_targets"], "windowPolicy": state["window_policy"],
                "testEvaluated": False, "config": state["training_config"]}
    return save_artifact(export_dir, model, tokenizer, training=metadata, evaluation=None)


def finalize(run_dir, *, corpus_path=None, export_dir=None):
    """Evaluate the validation-selected snapshot on test, explicitly and outside the loop."""
    from .artifacts import save_artifact
    run_dir = Path(run_dir).resolve()
    with _run_lock(run_dir):
        status = _json_read(run_dir / "status.json")
        if not status.get("bestCheckpoint"):
            raise ValueError("No validation-selected checkpoint")
        model, tokenizer, state, _ = _load_checkpoint(run_dir, status["bestCheckpoint"])
        if state["training_config"].get("partition_policy") == "train_validation_only":
            raise ValueError("This run has no TEST partition; use export_best instead of finalize")
        path = Path(corpus_path or state["corpus_path"]).resolve()
        if _sha(path) != state["corpus_sha256"]:
            raise ValueError("Evaluation corpus hash mismatch")
        records = _load_training_corpus(path, state["training_config"])
        if _sha(path) != state["corpus_sha256"]:
            raise ValueError("Corpus changed while loading")
        examples = _encoded(records, tokenizer, state["model_config"]["context_length"], state["window_policy"], **_encoding_kwargs(state["training_config"]))
        _torch().set_num_threads(state["training_config"]["torch_threads"])
        losses = _baseline(examples["train"], tokenizer.get_vocab_size())
        loss_reduction = state["training_config"].get("loss_reduction", "token_mean")
        evaluation = {"schemaVersion": 1, "corpusSha256": state["corpus_sha256"],
                      "selectedCheckpoint": status["bestCheckpoint"], "selectedStep": state["step"],
                      "selectionSplit": "validation", "testUsedForSelection": False, **_origin_metadata(state),
                      **_loss_metadata(state["training_config"]), **_sampling_metadata(state),
                      "assistantTargets": state["assistant_targets"], "windowPolicy": state["window_policy"],
                      "train": _evaluate_configured(model, examples["train"], state["training_config"]),
                      "validation": _evaluate_configured(model, examples["validation"], state["training_config"]),
                      "test": _evaluate_configured(model, examples["test"], state["training_config"]),
                      "unigramBaseline": {split: _baseline_score(examples[split], losses, loss_reduction) for split in examples},
                      "experimental": True, "instructionFollowingValidated": False}
        _atomic_json(run_dir / "evaluation.json", evaluation)
        if export_dir is not None:
            save_artifact(export_dir, model, tokenizer,
                          training={"corpusSha256": state["corpus_sha256"], "selectedStep": state["step"],
                                    **_origin_metadata(state), **_loss_metadata(state["training_config"]),
                                    **_sampling_metadata(state),
                                    "selectionSplit": "validation",
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
    allowed = {"corpus_path", "run_dir", "model_config", "training_config", "resume", "budget_seconds", "initial_model"}
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
