"""Package a closed action-copy-v2 arm's exact BEST and native recovery bytes.

Stdlib only. No Torch, pickle interpretation, generation, training, signals or
Git operations. Each invocation has one CPU, nice19 and a 120-second ceiling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ("weights.safetensors", "resume.pt", "tokenizer.json", "state.json", "manifest.json")
ARTIFACT = ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")
PARENT_ID = "unidad-conversation-e202eafc8f11bcda7587db496c6e7ac1670efbd612685c465658ae2f07bd54fa"


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def rel(path):
    return str(path.relative_to(ROOT))


def info(folder, names):
    return {name: {"bytes": (folder / name).stat().st_size, "sha256": sha(folder / name)} for name in names}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return sha(path)


def deliver(arm, preflight=False):
    started = time.monotonic()
    run = ROOT / "private/learning-repair-20261013/runs/autonomy-action-copy-v2" / arm
    stem = "autonomy-action-copy-v2-" + arm + "-final"
    docs = ROOT / "docs/learning-repair-20261013"
    model = ROOT / "models/learning-repair-20261013/autonomy" / ("action-copy-v2-" + arm + "-final-best256")
    packages = ROOT / "checkpoints/learning-repair-20261013/autonomy-action-copy-v2" / (arm + "-final")
    utility = load_helper("closed_delivery_utils", docs / "prepare_control_final_delivery.py")
    transport = load_helper("native_parts_transport", ROOT / "examples/package_native_checkpoint.py")
    exposure = load_helper("native_target_exposure", ROOT / "examples/audit_learning_exposure.py")
    status_raw, stage_raw = (run / "status.json").read_bytes(), (run / "stage-latest.json").read_bytes()
    status, stage, design = json.loads(status_raw), json.loads(stage_raw), read(run / "stage-design.json")
    if status["status"] not in ("stopped", "completed") or utility.alive(status["pid"], status["processBirth"]):
        raise RuntimeError("Arm remains live or nonterminal; nothing exported")
    if (status["step"] != 256 or stage["nativeStatus"] != status or stage["requestedStopAfterStep"] != 256
            or stage["cooperativeStopReasons"] != ["absolute_step_boundary"]
            or not stage["sourceAndInputsUnchanged"] or stage["resourceMonitorErrors"]
            or stage["testEvaluated"] or stage["conversationAccepted"]):
        raise ValueError("Final native closure receipt incomplete or inconsistent")
    if (status["lineage"]["parentModelId"] != PARENT_ID or design["corpusSha256"] != status["corpusSha256"]
            or design["trainingConfig"]["max_steps"] != 256 or design["trainingConfig"]["max_seconds"] != 600):
        raise ValueError("Matched E202 lineage or finite config mismatch")
    sources = utility.frozen_sources(design)
    roles = {"best": status["bestCheckpoint"], "last": status["lastCheckpoint"]}
    native = {role: info(run / "checkpoints" / name, NATIVE) for role, name in roles.items()}
    for role, name in roles.items():
        checkpoint = run / "checkpoints" / name
        if read(checkpoint / "manifest.json")["files"] != {n: item["sha256"] for n, item in native[role].items() if n != "manifest.json"}:
            raise ValueError("Native manifest mismatch")
        if read(checkpoint / "state.json")["training_config"] != design["trainingConfig"]:
            raise ValueError("Native frozen configuration mismatch")
    snapshot = Path(stage["snapshot"])
    snapshot_info = info(snapshot, ARTIFACT)
    manifest = read(snapshot / "manifest.json")
    if ({name: item["sha256"] for name, item in snapshot_info.items()} != stage["snapshotFiles"]
            or manifest["modelId"] != stage["modelId"]
            or manifest["training"]["selectedCheckpoint"] != roles["best"]
            or manifest["training"]["selectedStep"] != 256
            or manifest["evaluation"]["productionAccepted"]
            or any(snapshot_info[n] != native["best"][n] for n in ("weights.safetensors", "tokenizer.json"))):
        raise ValueError("Selected BEST snapshot is inconsistent")
    ledgers = {role: exposure.audit(run / "tokenized-cache", run / "checkpoints" / name / "state.json", 30)
               for role, name in roles.items()}
    shared = all(native["best"][n] == native["last"][n] for n in ("weights.safetensors", "resume.pt", "tokenizer.json"))
    if preflight:
        return {"arm": arm, "ready": True, "modelId": stage["modelId"], "sharedBestLastPayload": shared,
                "snapshot": rel(snapshot), "bestStep": manifest["training"]["selectedStep"]}
    frozen = docs / (stem + "-inputs")
    receipt_path, infer_path = docs / (stem + "-delivery.json"), docs / (stem + "-inference-export.json")
    if any(p.exists() for p in (model, packages, frozen, receipt_path, infer_path)):
        raise FileExistsError("Final output exists; refusing replacement")
    model.parent.mkdir(parents=True, exist_ok=True)
    model_temp = model.parent / (".copy-best-" + uuid.uuid4().hex)
    model_temp.mkdir(mode=0o700)
    try:
        for name in ARTIFACT:
            shutil.copyfile(snapshot / name, model_temp / name)
        if info(model_temp, ARTIFACT) != snapshot_info:
            raise ValueError("Copied inference snapshot mismatch")
        fd = os.open(model.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            transport._rename_new(fd, model_temp.name, model.name)
        finally:
            os.close(fd)
    except BaseException:
        shutil.rmtree(model_temp)
        raise
    packages.mkdir(parents=True)
    best_parent = packages / "BEST"
    best_parent.mkdir()
    packed_best = best_parent / roles["best"]
    package_result = transport.pack(run / "checkpoints" / roles["best"], packed_best)
    best_package = utility.verify_package(packed_best)
    if roles["best"] == roles["last"]:
        last_delivery = {"kind": "same-native-checkpoint", "path": rel(packed_best)}
    elif shared:
        last = packages / "LAST-reference" / roles["last"]
        last.mkdir(parents=True)
        for name in ("state.json", "manifest.json"):
            shutil.copyfile(run / "checkpoints" / roles["last"] / name, last / name)
        file_sources = {}
        package_files = {item["name"]: item for item in best_package["files"]}
        for name in NATIVE:
            if name in ("state.json", "manifest.json"):
                components = [{"path": rel(last / name), **native["last"][name]}]
            else:
                components = [{"path": rel(packed_best / part["name"]), "bytes": part["bytes"], "sha256": part["sha256"]}
                              for part in package_files[name]["parts"]]
            file_sources[name] = {**native["last"][name], "sources": components}
        reference = {"schemaVersion": 1, "format": "unidad-native-files-by-reference-v1",
                     "checkpointName": roles["last"], "files": file_sources,
                     "limitation": "LAST metadata plus byte-identical BEST payload parts. Restore with restore_action_native_reference.py; not directly loadable."}
        reference_sha = write(last / "native-reference.json", reference)
        last_delivery = {"kind": "native-files-by-reference", "path": rel(last), "referenceSha256": reference_sha}
    else:
        last_parent = packages / "LAST"
        last_parent.mkdir()
        packed_last = last_parent / roles["last"]
        last_delivery = {"kind": "native-parts", "path": rel(packed_last),
                         "packageResult": transport.pack(run / "checkpoints" / roles["last"], packed_last)}
        utility.verify_package(packed_last)
    frozen.mkdir()
    copies = {}
    for name, source in {"status.json": run / "status.json", "stage-latest.json": run / "stage-latest.json",
                         "stage-design.json": run / "stage-design.json", "cache-manifest.json": run / "tokenized-cache/manifest.json"}.items():
        target = frozen / name
        shutil.copyfile(source, target)
        if target.read_bytes() != source.read_bytes():
            raise ValueError("Frozen metadata changed during copy")
        copies[name] = {"path": rel(target), "sha256": sha(target), "bytes": target.stat().st_size}
    exposure_paths = {}
    for role, ledger in ledgers.items():
        path = docs / (stem + "-" + role + "-target-exposure.json")
        exposure_paths[role] = {"path": rel(path), "sha256": write(path, ledger), "counts": ledger["counts"]}
    if ((run / "status.json").read_bytes() != status_raw or (run / "stage-latest.json").read_bytes() != stage_raw
            or info(snapshot, ARTIFACT) != snapshot_info or info(model, ARTIFACT) != snapshot_info
            or utility.frozen_sources(design) != sources
            or any(info(run / "checkpoints" / name, NATIVE) != native[role] for role, name in roles.items())):
        raise ValueError("Closed original bytes changed during export")
    inference = {"schemaVersion": 1, "modelId": stage["modelId"], "artifactPath": rel(model),
                 "selectedStep": 256, "selection": "Exact byte copy of native validation-selected BEST snapshot; not selected by replies",
                 "artifactFiles": snapshot_info, "parentModelId": PARENT_ID, "lineage": status["lineage"],
                 "corpusSha256": status["corpusSha256"], "contextLength": 2048,
                 "conversationAccepted": False, "productionAccepted": False, "promotionPerformed": False,
                 "generationExecuted": False, "testOrReservedEvaluationRead": False}
    inference_sha = write(infer_path, inference)
    receipt = {"schemaVersion": 1, "kind": "closed-action-copy-v2-arm-final-delivery", "arm": arm,
               "createdAtUtc": datetime.now(timezone.utc).isoformat(), "preparedBy": "project-AI", "humanReview": False,
               "modelId": stage["modelId"], "status": status, "statusSha256": hashlib.sha256(status_raw).hexdigest(),
               "stageLatestSha256": hashlib.sha256(stage_raw).hexdigest(), "checkpointRoles": roles,
               "nativeFiles": native, "sharedBestLastWeightsOptimizerAndTokenizer": shared,
               "bestNativeDelivery": {"path": rel(packed_best), "packageResult": package_result},
               "lastNativeDelivery": last_delivery, "inferenceExport": {"path": rel(infer_path), "sha256": inference_sha},
               "frozenMetadata": copies, "trainerSources": sources, "targetExposure": exposure_paths,
               "integrity": {"originalNativeAndSnapshotHashesUnchanged": True, "inferenceCopyByteExact": True,
                             "nativePartsAndRecombinedFileHashesVerified": True, "sourceHashesUnchanged": True},
               "resourceLimits": {"cpuCount": 1, "nice": 19, "cpuSeconds": 120, "wallSeconds": 120, "addressSpaceMiB": 512, "maxNativePartMiB": 32},
               "elapsedWallSeconds": time.monotonic() - started, "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "helperSha256": sha(Path(__file__)), "method": {"torchImported": "torch" in sys.modules,
                   "modelWeightsLoaded": False, "optimizerPickleLoaded": False, "trainingExecuted": False,
                   "modelGenerationExecuted": False, "signalsSent": False, "testOrReservedRead": False},
               "publication": {"localArtifactsPrepared": True, "gitHubUploadVerified": False},
               "conversationAccepted": False, "productionAccepted": False, "promotionPerformed": False,
               "limitation": "TRAIN target exposure and selected bytes do not establish authorized argument copying or completion. Actual model task evaluation remains separate; arm validation NLLs use different corpora."}
    receipt_sha = write(receipt_path, receipt)
    return {"arm": arm, "modelId": stage["modelId"], "modelPath": rel(model), "receiptPath": rel(receipt_path),
            "receiptSha256": receipt_sha, "inferenceSha256": inference_sha,
            "bestWeightsSha256": snapshot_info["weights.safetensors"]["sha256"],
            "exposure": ledgers["best"]["counts"], "elapsedWallSeconds": receipt["elapsedWallSeconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("treatment", "control"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Final export wall ceiling")))
    signal.alarm(120)
    print(json.dumps(deliver(args.arm, args.preflight), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
