"""Bounded native TEXT/SFT stages with exact resume and cooperative resource stops.

This runner uses the same native trainer and artifact format. It never evaluates
TEST, invents replies or promotes a model from loss. Resource ceilings request
a checkpoint at a complete optimizer boundary; they are not hard OS limits.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import resource
import threading
import time
import uuid

from ia_propia.conversation import training
from run_conversation_pilot import OWNER, PARENT_FILES, _clear_own_stop, runner_lock


def source_hashes():
    package = Path(training.__file__).resolve().parent
    return {str(p.relative_to(package)): training._sha(p)
            for p in sorted(package.glob("*.py"))} | {
        "runner": training._sha(Path(__file__)),
        "shared_pilot_helpers": training._sha(Path(__file__).with_name("run_conversation_pilot.py")),
    }


def resource_snapshot(run_dir):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    disk = 0
    for directory, _, files in os.walk(run_dir):
        for name in files:
            try:
                disk += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                pass  # Native checkpoints publish by rename.
    return {"peakRssBytes": rss, "runDiskBytes": disk}


def run_stage(corpus, expected_sha, run_dir, *, training_config=None,
              model_config=None, parent_model=None, resume=False,
              stop_after_step=1024, budget_seconds=900,
              max_rss_mib=6144, max_disk_mib=2048):
    if type(expected_sha) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha) is None:
        raise ValueError("Expected corpus SHA must be a 64-character digest")
    if type(resume) is not bool or type(stop_after_step) is not int or stop_after_step < 1:
        raise ValueError("Invalid resume or absolute step boundary")
    if (type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds)
            or not 0 < budget_seconds <= 3600):
        raise ValueError("Invocation budget must be finite and at most 3600 seconds")
    for value in (max_rss_mib, max_disk_mib):
        if type(value) is not int or not 256 <= value <= 16384:
            raise ValueError("Resource ceilings must be integer MiB in [256, 16384]")
    corpus, run_dir = Path(corpus).resolve(), Path(run_dir).resolve()
    expected_sha = expected_sha.lower()
    if training._sha(corpus) != expected_sha:
        raise ValueError("Corpus differs from its declared SHA")
    if not resume and ((model_config is None) == (parent_model is None)):
        raise ValueError("New stage requires exactly one random model config or own parent")
    if parent_model is not None and model_config is not None:
        raise ValueError("Parent and random architecture cannot be combined")
    parent = Path(parent_model).resolve() if parent_model is not None else None
    if parent is not None and (run_dir == parent or run_dir.is_relative_to(parent)
                               or parent.is_relative_to(run_dir)):
        raise ValueError("Run directory must not overlap the own parent artifact")
    invocation_id = uuid.uuid4().hex
    with runner_lock(run_dir):
        with training._run_lock(run_dir):
            design_path = run_dir / "stage-design.json"
            if resume:
                design = training._json_read(design_path)
                if design["corpusPath"] != str(corpus) or design["corpusSha256"] != expected_sha:
                    raise ValueError("Resume corpus identity changed")
                if design["sourceSha256"] != source_hashes():
                    raise ValueError("Frozen stage source changed")
                _, config = training._configs(None, design["trainingConfig"])
                if training_config is not None and training._configs(None, training_config)[1] != config:
                    raise ValueError("Resume training configuration changed")
                if model_config is not None and model_config != design["randomModelConfig"]:
                    raise ValueError("Resume model configuration changed")
                saved_parent = design["parentPath"]
                if parent is not None and str(parent) != saved_parent:
                    raise ValueError("Resume own parent path changed")
                parent = Path(saved_parent) if saved_parent is not None else None
                if parent is not None and any(training._sha(parent / f) != h
                                              for f, h in design["parentFiles"].items()):
                    raise ValueError("Own parent artifact changed")
                status = training._json_read(run_dir / "status.json")
                _, _, state, _ = training._load_checkpoint(run_dir, status["lastCheckpoint"])
                if state["training_config"] != config or stop_after_step <= state["step"]:
                    raise ValueError("Invalid frozen configuration or resumed step boundary")
                _clear_own_stop(run_dir, design)
            else:
                if design_path.exists() or (run_dir / "status.json").exists():
                    raise ValueError("Stage already exists; resume or choose a new directory")
                if (run_dir / "stop-request.json").exists():
                    raise RuntimeError("Existing stop request was preserved")
                requested, config = training._configs(model_config, training_config)
                if (config.get("objective") not in ("causal_lm", "assistant_last_reply")
                        or config.get("partition_policy") != "train_validation_only"
                        or config.get("data_loading") != "mmap"
                        or config.get("trainable_parameters") != "all"):
                    raise ValueError("Stage requires explicit objective, TRAIN/validation mmap and all parameters")
                from ia_propia.conversation.disk_data import StreamingCorpus
                counts = StreamingCorpus(corpus, config).counts
                if not counts["train"] or not counts["validation"] or counts["test"]:
                    raise ValueError("Stage requires TRAIN and validation, with no TEST")
                parent_files, actual_config = {}, None
                if parent is not None:
                    model, _, actual_config, _ = training._initial_own_model(parent, None)
                    del model
                    parent_files = {f: training._sha(parent / f) for f in PARENT_FILES}
                design = {"schemaVersion": 1, "owner": OWNER, "runnerId": uuid.uuid4().hex,
                          "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                          "corpusPath": str(corpus), "corpusSha256": expected_sha,
                          "parentPath": str(parent) if parent else None,
                          "parentFiles": parent_files, "parentModelConfig": actual_config,
                          "randomModelConfig": requested if parent is None else None,
                          "trainingConfig": config, "sourceSha256": source_hashes(),
                          "recordSplits": counts, "testEvaluated": False,
                          "conversationAccepted": False,
                          "resourceCeilings": {"rssMiB": max_rss_mib, "runDiskMiB": max_disk_mib,
                              "policy": "cooperative stop, finite OS environment remains the hard bound"}}
                training._atomic_json(design_path, design)
            if stop_after_step > config["max_steps"]:
                raise ValueError("Boundary exceeds the persistent stage step limit")
        original_log = training._log
        finished = threading.Event()
        request_lock = threading.Lock()
        stop_reason = []
        observations = {"peakRssBytes": 0, "peakRunDiskBytes": 0}
        monitor_errors = []

        def request_stop(reason):
            with request_lock:
                path = run_dir / "stop-request.json"
                if path.exists():
                    previous = training._json_read(path)
                    if previous.get("owner") != OWNER or previous.get("runnerId") != design["runnerId"]:
                        raise RuntimeError("Unrelated stop request appeared and was preserved")
                    return
                training._atomic_json(path, {"pid": os.getpid(), "processBirth": training._birth(os.getpid()),
                    "owner": OWNER, "runnerId": design["runnerId"], "invocationId": invocation_id,
                    "reason": reason, "requestedStep": stop_after_step})
                stop_reason.append(reason)

        def observe():
            values = resource_snapshot(run_dir)
            observations["peakRssBytes"] = max(observations["peakRssBytes"], values["peakRssBytes"])
            observations["peakRunDiskBytes"] = max(observations["peakRunDiskBytes"], values["runDiskBytes"])
            if values["peakRssBytes"] > max_rss_mib * 1024**2:
                request_stop("rss_ceiling")
            elif values["runDiskBytes"] > max_disk_mib * 1024**2:
                request_stop("run_disk_ceiling")

        def monitor():
            try:
                while not finished.wait(0.5):
                    observe()
            except BaseException as error:
                monitor_errors.append(str(error))
                # Let the native trainer checkpoint its next coherent boundary.
                try:
                    request_stop("resource_monitor_failed")
                except RuntimeError:
                    pass

        def log(path, event):
            original_log(path, event)
            if event.get("event") == "train" and event.get("step") == stop_after_step:
                request_stop("absolute_step_boundary")

        observe()
        watcher = threading.Thread(target=monitor, name="unidad-stage-resources", daemon=True)
        watcher.start()
        try:
            training._log = log
            result = training.train(corpus, run_dir, resume=resume,
                model_config=design["randomModelConfig"] if not resume and parent is None else None,
                training_config=config if not resume else None,
                initial_model=parent if not resume else None,
                expected_model_config=design["parentModelConfig"],
                expected_corpus_sha256=expected_sha, budget_seconds=budget_seconds)
        finally:
            training._log = original_log
            finished.set()
            watcher.join(timeout=5)
        observe()
        unchanged = (source_hashes() == design["sourceSha256"] and training._sha(corpus) == expected_sha
                     and (parent is None or all(training._sha(parent / f) == h
                                               for f, h in design["parentFiles"].items())))
        receipt = {"schemaVersion": 1, "invocationId": invocation_id, "resume": resume,
                   "nativeStatus": result, "resourceObservations": observations,
                   "cooperativeStopReasons": stop_reason, "resourceMonitorErrors": monitor_errors,
                   "requestedStopAfterStep": stop_after_step, "budgetSeconds": budget_seconds,
                   "sourceAndInputsUnchanged": unchanged, "testEvaluated": False,
                   "conversationAccepted": False}
        if unchanged and not monitor_errors:
            _, _, state, _ = training._load_checkpoint(run_dir, result["bestCheckpoint"])
            snapshot = run_dir / "snapshots" / f"best-step{state['step']:06d}"
            if snapshot.exists():
                from ia_propia.conversation.artifacts import load_artifact
                _, _, manifest = load_artifact(snapshot)
                if manifest["training"]["selectedCheckpoint"] != result["bestCheckpoint"]:
                    raise ValueError("Snapshot identity changed")
            else:
                manifest = training.export_best(run_dir, snapshot)
            receipt.update({"snapshot": str(snapshot), "selectedStep": state["step"],
                "modelId": manifest["modelId"],
                "snapshotFiles": {f: training._sha(snapshot / f) for f in PARENT_FILES}})
        receipt["resourceObservations"]["processPeakRssIncludingExportBytes"] = resource_snapshot(run_dir)["peakRssBytes"]
        training._atomic_json(run_dir / "stage-latest.json", receipt)
        training._atomic_json(run_dir / "stage-receipts" / f"step{result['step']:06d}-{invocation_id}.json", receipt)
        if not unchanged or monitor_errors:
            raise ValueError("Source/input or monitoring failure; native checkpoints preserved")
        return receipt


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--parent-model", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int, default=1024)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--max-rss-mib", type=int, default=6144)
    parser.add_argument("--max-disk-mib", type=int, default=2048)
    args = parser.parse_args()
    receipt = run_stage(args.corpus, args.expected_corpus_sha256, args.run_dir,
        training_config=training._json_read(args.training_config) if args.training_config else None,
        model_config=training._json_read(args.model_config) if args.model_config else None,
        parent_model=args.parent_model, resume=args.resume, stop_after_step=args.stop_after_step,
        budget_seconds=args.budget_seconds, max_rss_mib=args.max_rss_mib, max_disk_mib=args.max_disk_mib)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
