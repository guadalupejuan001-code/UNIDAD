"""Run bounded segments of one native, resumable conversational training stage.

The step boundary only requests the trainer's cooperative stop. Its persistent
configuration, optimizer, sampling, RNG and validation cadence stay unchanged.
This runner exports validation-selected weights without opening TEST or claiming
that a lower loss establishes conversation competence.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import uuid

from ia_propia.conversation import training


CONFIG = {
    "batch_size": 8, "learning_rate": 0.0001, "weight_decay": 0.01,
    "grad_clip": 1.0, "max_epochs": 10000, "max_steps": 8192,
    "max_seconds": 7200, "validation_every": 128, "checkpoint_every": 128,
    "patience": 20, "min_delta": 0.0001, "seed": 20261013,
    "warmup_steps": 16, "torch_threads": 4, "window_policy": "sliding_context",
    "loss_reduction": "token_mean", "projection_mode": "supervised",
    "objective": "assistant_last_reply", "partition_policy": "train_validation_only",
    "train_precision": "cpu_bfloat16", "trainable_parameters": "all",
    "data_loading": "mmap",
}
OWNER = "unidad-conversation-pilot-v1"
PARENT_FILES = ("manifest.json", "config.json", "tokenizer.json", "weights.safetensors")


def source_hashes():
    package = Path(training.__file__).resolve().parent
    return {"runner": training._sha(Path(__file__)),
            **{path.name: training._sha(path) for path in sorted(package.glob("*.py"))}}


@contextmanager
def runner_lock(run_dir):
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(run_dir / "pilot-runner.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This pilot already has an active runner") from exc
        yield
    finally:
        os.close(descriptor)


def _clear_own_stop(run_dir, design):
    """Caller holds the native writer lock; never clear another worker's request."""
    path = run_dir / "stop-request.json"
    if not path.exists():
        return
    request = training._json_read(path)
    if request.get("owner") != OWNER or request.get("runnerId") != design["runnerId"]:
        raise RuntimeError("An unrelated stop request is present; it was preserved")
    pid = request.get("pid")
    if type(pid) is not int or pid < 1 or not isinstance(request.get("processBirth"), str):
        raise ValueError("Malformed owned stop request")
    # A new CLI process may clear only a dead previous process. Same-process
    # library callers are safe here because the native writer lock is held.
    if pid != os.getpid() and training._birth(pid) == request["processBirth"]:
        raise RuntimeError("The previous stop-request owner is still alive")
    path.unlink()


def run_pilot(corpus, expected_corpus_sha256, parent_model, run_dir, *,
              stop_after_step=256, budget_seconds=600, resume=False):
    if (type(expected_corpus_sha256) is not str
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_corpus_sha256) is None):
        raise ValueError("Expected corpus SHA-256 must be a 64-character digest")
    if type(stop_after_step) is not int or stop_after_step < 1:
        raise ValueError("stop_after_step must be a positive absolute step")
    if (type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds)
            or not 0 < budget_seconds <= 86400):
        raise ValueError("Invalid invocation budget")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    corpus, parent, run_dir = map(lambda path: Path(path).resolve(), (corpus, parent_model, run_dir))
    expected_corpus_sha256 = expected_corpus_sha256.lower()
    if training._sha(corpus) != expected_corpus_sha256:
        raise ValueError("Corpus SHA-256 differs from the declared identity")
    from ia_propia.conversation.disk_data import StreamingCorpus
    validated = StreamingCorpus(corpus, CONFIG)
    if not validated.counts["train"] or not validated.counts["validation"]:
        raise ValueError("Pilot requires TRAIN and validation, with no TEST")
    model, tokenizer, model_config, lineage = training._initial_own_model(parent, None)
    del model, tokenizer
    parent_manifest = training._json_read(parent / "manifest.json")
    identity = {"corpusPath": str(corpus), "corpusSha256": expected_corpus_sha256,
                "parentPath": str(parent), "parentModelId": parent_manifest["modelId"],
                "parentFiles": {name: training._sha(parent / name) for name in PARENT_FILES},
                "modelConfig": model_config, "lineage": lineage,
                "sourceSha256": source_hashes()}
    invocation_id = uuid.uuid4().hex
    with runner_lock(run_dir):
        design_path = run_dir / "pilot-design.json"
        # The native lock makes stale-stop cleanup safe even when a caller uses
        # the native trainer directly rather than this runner.
        with training._run_lock(run_dir):
            if resume:
                design = training._json_read(design_path)
                if any(design.get(key) != value for key, value in identity.items()):
                    raise ValueError("Frozen pilot corpus, parent or source code changed")
                status = training._json_read(run_dir / "status.json")
                _, _, state, _ = training._load_checkpoint(run_dir, status["lastCheckpoint"])
                if state["training_config"] != design["trainingConfig"]:
                    raise ValueError("Checkpoint differs from the frozen training configuration")
                if state.get("lineage") != design["lineage"]:
                    raise ValueError("Checkpoint differs from the frozen parent lineage")
                if stop_after_step <= state["step"]:
                    raise ValueError("Requested boundary must exceed the saved step")
                _clear_own_stop(run_dir, design)
            else:
                if design_path.exists() or (run_dir / "status.json").exists():
                    raise ValueError("Pilot already exists; resume it or choose a new directory")
                if (run_dir / "stop-request.json").exists():
                    raise RuntimeError("Existing stop request was preserved")
                _, frozen_config = training._configs(None, CONFIG)
                design = {"schemaVersion": 1, "owner": OWNER, "runnerId": uuid.uuid4().hex,
                          "createdAtUtc": datetime.now(timezone.utc).isoformat(), **identity,
                          "trainingConfig": frozen_config, "recordSplits": validated.counts,
                          "testEvaluated": False, "conversationAccepted": False}
                training._atomic_json(design_path, design)
            if stop_after_step > design["trainingConfig"]["max_steps"]:
                raise ValueError("Step boundary exceeds the persistent stage limit")

        original_log = training._log
        stop_written = False

        def log(path, event):
            nonlocal stop_written
            result = original_log(path, event)
            if event.get("event") == "train" and event.get("step") == stop_after_step:
                stop_path = run_dir / "stop-request.json"
                if stop_path.exists():
                    raise RuntimeError("A stop request appeared during training; it was preserved")
                training._atomic_json(stop_path, {"pid": os.getpid(),
                    "processBirth": training._birth(os.getpid()), "owner": OWNER,
                    "runnerId": design["runnerId"], "invocationId": invocation_id,
                    "requestedStep": stop_after_step})
                stop_written = True
            return result

        try:
            training._log = log
            result = training.train(corpus, run_dir, resume=resume,
                initial_model=None if resume else parent,
                training_config=None if resume else design["trainingConfig"],
                expected_corpus_sha256=expected_corpus_sha256,
                expected_model_config=model_config, budget_seconds=budget_seconds)
        finally:
            training._log = original_log

        unchanged = (source_hashes() == identity["sourceSha256"]
                     and training._sha(corpus) == expected_corpus_sha256
                     and all(training._sha(parent / name) == digest
                             for name, digest in identity["parentFiles"].items()))
        receipt = {"schemaVersion": 1, "invocationId": invocation_id,
                   "runnerId": design["runnerId"], "resume": resume,
                   "requestedStopAfterStep": stop_after_step, "budgetSeconds": budget_seconds,
                   "cooperativeStepStopWritten": stop_written, "nativeStatus": result,
                   "inputsAndSourceUnchanged": unchanged, "identity": identity,
                   "testEvaluated": False, "conversationAccepted": False}
        if unchanged:
            _, _, best_state, _ = training._load_checkpoint(run_dir, result["bestCheckpoint"])
            selected_step = best_state["step"]
            snapshot = run_dir / "snapshots" / f"best-step{selected_step:06d}"
            if snapshot.exists():
                from ia_propia.conversation.artifacts import load_artifact
                _, _, manifest = load_artifact(snapshot)
                if manifest["training"].get("selectedCheckpoint") != result["bestCheckpoint"]:
                    raise ValueError("Existing snapshot has a different selected checkpoint")
            else:
                manifest = training.export_best(run_dir, snapshot)
            receipt.update({"snapshot": str(snapshot), "modelId": manifest["modelId"],
                "selectedStep": selected_step, "validationImprovedFromParent": selected_step > 0,
                "snapshotFiles": {name: training._sha(snapshot / name) for name in PARENT_FILES}})
        receipt_path = run_dir / "pilot-receipts" / f"step{result['step']:06d}-{invocation_id}.json"
        training._atomic_json(receipt_path, receipt)
        training._atomic_json(run_dir / "pilot-latest.json", receipt)
        if not unchanged:
            raise ValueError("Inputs or source code changed during training; checkpoints and receipt preserved")
        return receipt


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stop-after-step", type=int, default=256)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    receipt = run_pilot(args.corpus, args.expected_corpus_sha256, args.parent_model,
        args.run_dir, stop_after_step=args.stop_after_step, budget_seconds=args.budget_seconds,
        resume=args.resume)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
