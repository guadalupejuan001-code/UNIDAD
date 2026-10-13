"""Prepare terminal matched-v4 CONTROL bytes and exact TRAIN exposure, without Torch.

Default/preflight is read-only. `deliver` refuses live or incomplete workers before
creating any output. BEST0 reuses published parent weights through an explicit
reference; `restore-best-reference` reconstructs that native checkpoint exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import shutil
import struct
import time
from datetime import datetime, timezone
import uuid


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "private/learning-repair-20261013/runs/natural-latam-matched-v4/control"
DOCS = ROOT / "docs/learning-repair-20261013"
STEM = "natural-latam-v4-control-final"
PACKAGES = ROOT / "checkpoints/learning-repair-20261013/natural-latam-v4/control-final"
MODEL_REFERENCE = ROOT / "models/learning-repair-20261013/natural-latam-v4/control-final-BEST-reference"
PARENT = ROOT / "models/learning-repair-20261013/conversation-15m/intermediate-step004096"
NATIVE = ("weights.safetensors", "resume.pt", "tokenizer.json", "state.json", "manifest.json")
ARTIFACT = ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")
REF_FORMAT = "unidad-native-best-with-existing-weights-v1"
IO_BYTES = 1024 * 1024


def limits():
    os.nice(10)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(IO_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def infos(folder, names):
    return {n: {"bytes": (folder / n).stat().st_size, "sha256": sha(folder / n)} for n in names}


def read(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return sha(path)


def alive(pid, birth):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
    except FileNotFoundError:
        return False
    return parts[19] == str(birth) and parts[0] != "Z"


def frozen_sources(design):
    paths = {p.name: p for p in (ROOT / "src/ia_propia/conversation").glob("*.py")}
    paths["runner"] = ROOT / "examples/run_learning_stage.py"
    paths["shared_pilot_helpers"] = ROOT / "examples/run_conversation_pilot.py"
    result = {n: {"path": rel(p), "sha256": sha(p), "bytes": p.stat().st_size}
              for n, p in paths.items()}
    if {n: v["sha256"] for n, v in result.items()} != design["sourceSha256"]:
        raise ValueError("Pinned trainer sources changed")
    return result


def terminal():
    raw = (RUN / "status.json").read_bytes()
    status = json.loads(raw)
    if status["status"] not in ("stopped", "completed") or alive(status["pid"], status["processBirth"]):
        raise RuntimeError("CONTROL remains live or nonterminal; no output created")
    if status["step"] != 4096 or status["reason"] != "signal":
        raise RuntimeError("This delivery requires the configured terminal 4096-step boundary")
    stage_raw = (RUN / "stage-latest.json").read_bytes()
    stage = json.loads(stage_raw)
    if (stage["nativeStatus"] != status or stage["requestedStopAfterStep"] != 4096
            or stage["cooperativeStopReasons"] != ["absolute_step_boundary"]
            or not stage["sourceAndInputsUnchanged"] or stage["resourceMonitorErrors"]
            or stage["testEvaluated"] or stage["conversationAccepted"]):
        raise ValueError("Native terminal receipt is absent, incomplete or inconsistent")
    design = read(RUN / "stage-design.json")
    if design["corpusSha256"] != status["corpusSha256"] or design["trainingConfig"]["max_steps"] != 4096:
        raise ValueError("Frozen control design mismatch")
    if (status["lineage"]["parentWeightsSha256"] != design["parentFiles"]["weights.safetensors"]
            or status["lineage"]["parentModelId"] != "unidad-conversation-072eedd276160f7581c5a70b57e66ae3000d2e664ed32f6b5a4669d4d12dcf57"):
        raise ValueError("Matched SFT072 BEST4096 parent mismatch")
    return raw, status, stage_raw, stage, design


def transport():
    path = ROOT / "examples/package_native_checkpoint.py"
    spec = importlib.util.spec_from_file_location("native_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_package(folder):
    package = read(folder / "native-checkpoint-package.json")
    pointer = read(folder / "native-checkpoint-pointer.json")
    if sha(folder / "native-checkpoint-package.json") != pointer["manifestSha256"]:
        raise ValueError("Package pointer mismatch")
    for entry in package["files"]:
        whole = hashlib.sha256()
        total = 0
        for part in entry["parts"]:
            path = folder / part["name"]
            if path.stat().st_size != part["bytes"] or sha(path) != part["sha256"] or part["bytes"] > 32 * 1024**2:
                raise ValueError("Native part size/hash mismatch")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(IO_BYTES), b""):
                    whole.update(block)
                    total += len(block)
        if total != entry["bytes"] or whole.hexdigest() != entry["sha256"]:
            raise ValueError("Recombined native byte stream mismatch")
    return package


def order_hash(order):
    return hashlib.sha256(b"".join(struct.pack("<I", n) for n in order)).hexdigest()


def tuples(value):
    return tuple(tuples(n) for n in value) if isinstance(value, list) else value


def exposure(status):
    cache_dir = RUN / "tokenized-cache"
    cache_raw = (cache_dir / "manifest.json").read_bytes()
    cache = json.loads(cache_raw)
    index_raw = (cache_dir / "train.index.bin").read_bytes()
    if hashlib.sha256(index_raw).hexdigest() != cache["splits"]["train"]["index"]["sha256"]:
        raise ValueError("TRAIN index mismatch")
    windows = list(struct.iter_unpack("<QQ", index_raw))
    count = len(windows)
    if count != cache["splits"]["train"]["count"] or count != 9951:
        raise ValueError("Unexpected control TRAIN window count")
    lengths, targets, eos = [], [], []
    whole = hashlib.sha256()
    offset = 0
    with (cache_dir / "train.windows.bin").open("rb") as stream:
        for start, length in windows:
            if start != offset or not 1 <= length <= 512:
                raise ValueError("TRAIN window index bounds mismatch")
            raw = stream.read(length * 8)
            if len(raw) != length * 8:
                raise ValueError("TRAIN cache truncated")
            whole.update(raw)
            labels = struct.unpack("<" + "i" * length, raw[length * 4:])
            if any(n != -100 and not 0 <= n < 2048 for n in labels) or labels.count(2) != 1:
                raise ValueError("Invalid supervised TRAIN labels/EOS")
            lengths.append(length)
            targets.append(sum(n >= 0 for n in labels))
            eos.append(sum(n == 2 for n in labels))
            offset += length * 8
        if stream.read(1):
            raise ValueError("TRAIN cache trailing bytes")
    if whole.hexdigest() != cache["splits"]["train"]["data"]["sha256"] or offset != cache["splits"]["train"]["data"]["bytes"]:
        raise ValueError("TRAIN cache hash/size mismatch")
    ledgers, draws = {}, {}
    for role, name in {"best": status["bestCheckpoint"], "last": status["lastCheckpoint"]}.items():
        state_path = RUN / "checkpoints" / name / "state.json"
        state_raw = state_path.read_bytes()
        state = json.loads(state_raw)
        epoch, cursor, order = state["epoch"], state["cursor"], state["order"]
        batch = state["training_config"]["batch_size"]
        if not 0 <= cursor <= count or state["step"] != epoch * ((count + batch - 1) // batch) + (cursor + batch - 1) // batch:
            raise ValueError("Native epoch/cursor/optimizer accounting mismatch")
        rng = random.Random(state["training_config"]["seed"])
        hashes = {}
        if order:
            if sorted(order) != list(range(count)):
                raise ValueError("Native TRAIN order is not a permutation")
            for e in range(epoch + 1):
                replay = list(range(count))
                rng.shuffle(replay)
                hashes[str(e)] = order_hash(replay)
            if replay != order:
                raise ValueError("Native sampler order replay mismatch")
        elif (epoch, cursor, state["step"]) != (0, 0, 0):
            raise ValueError("Only the untrained step-zero checkpoint may have empty order")
        if rng.getstate() != tuples(state["python_rng"]):
            raise ValueError("Native sampler Python RNG replay mismatch")
        prefix = set(order[:cursor])
        draws[role] = [epoch + int(i in prefix) for i in range(count)]
        weights = draws[role]
        actual = {"windowDraws": sum(weights),
                  "unpaddedInputPositions": sum(weights[i] * lengths[i] for i in range(count)),
                  "supervisedTargetPositionsIncludingEos": sum(weights[i] * targets[i] for i in range(count)),
                  "supervisedEosPositions": sum(weights[i] * eos[i] for i in range(count))}
        actual["supervisedPayloadPositionsExcludingEos"] = actual["supervisedTargetPositionsIncludingEos"] - actual["supervisedEosPositions"]
        ledgers[role] = {"checkpointName": name, "checkpointStateSha256": hashlib.sha256(state_raw).hexdigest(),
                         "checkpointStep": state["step"], "completedEpochs": epoch, "currentEpochCursor": cursor,
                         "optimizerLogicalUpdates": state["step"], "logicalBatchSize": batch,
                         "maxMicroBatchSize": state["training_config"]["micro_batch_size"],
                         "epochLogicalUpdates": (count + batch - 1) // batch,
                         "epochFinalBatchWindows": count % batch or batch,
                         "naiveStepTimesBatchOvercount": state["step"] * batch - actual["windowDraws"],
                         "actual": actual, "samplerOrderUint32LeSha256": hashes,
                         "currentOrderUint32LeSha256": order_hash(order),
                         "consumedPrefixUint32LeSha256": order_hash(order[:cursor]),
                         "samplerOrderAndPythonRngReplayed": True}
    ledger = {"schemaVersion": 1, "kind": "closed-matched-control-exact-train-exposure", "preparedBy": "project-AI",
              "humanReview": False, "identity": cache["identity"], "cacheManifestSha256": hashlib.sha256(cache_raw).hexdigest(),
              "trainIndexSha256": hashlib.sha256(index_raw).hexdigest(), "trainDataSha256": whole.hexdigest(),
              "trainIndexBytesRead": len(index_raw), "trainDataBytesRead": offset, "trainWindowsPerEpoch": count,
              "unpaddedInputPositionsPerEpoch": sum(lengths), "supervisedPositionsPerEpoch": sum(targets),
              "checkpoints": ledgers, "formula": "epoch * sum(TRAIN window count) + sum(count[order[:cursor]])",
              "eosTokenId": 2, "ignoredLabelValue": -100, "labelsCriterion": "label >= 0",
              "method": {"trainLabelsRead": True, "trainInputTokensDecoded": False, "corpusRead": False,
                         "validationWindowsRead": False, "testWindowsRead": False, "reservedEvaluationRead": False,
                         "torchImported": False, "weightsLoaded": False, "optimizerPickleLoaded": False,
                         "trainingExecuted": False, "testsExecuted": False},
              "limitation": "Only executed CONTROL exposure; excludes inherited training and padding. BEST0 reports zero new control learning, even though inherited parent weights were trained previously. Counts do not establish conversational acceptance."}
    return ledger, lengths, targets, eos, draws


def deliver(max_disk_mib):
    started = time.monotonic()
    status_raw, status, stage_raw, stage, design = terminal()
    sources = frozen_sources(design)
    helper = transport()
    roles = {"best": status["bestCheckpoint"], "last": status["lastCheckpoint"]}
    native_infos = {name: infos(RUN / "checkpoints" / name, NATIVE) for name in set(roles.values())}
    for name, info in native_infos.items():
        manifest = read(RUN / "checkpoints" / name / "manifest.json")
        if manifest["files"] != {n: v["sha256"] for n, v in info.items() if n != "manifest.json"}:
            raise ValueError("Native checkpoint manifest mismatch")
        if read(RUN / "checkpoints" / name / "state.json")["training_config"] != design["trainingConfig"]:
            raise ValueError("Native checkpoint configuration differs from frozen design")
    ledger, lengths, targets, eos, draws = exposure(status)
    snapshot = Path(stage["snapshot"])
    snapshot_info = infos(snapshot, ARTIFACT)
    if {n: v["sha256"] for n, v in snapshot_info.items()} != stage["snapshotFiles"]:
        raise ValueError("Final selected snapshot mismatch")
    if any(snapshot_info[n] != native_infos[roles["best"]][n] for n in ("weights.safetensors", "tokenizer.json")):
        raise ValueError("Final snapshot differs from selected native BEST")
    final_manifest = read(snapshot / "manifest.json")
    if final_manifest["training"]["selectedCheckpoint"] != roles["best"] or final_manifest["evaluation"]["productionAccepted"]:
        raise ValueError("Final BEST selection mismatch")
    best_state = read(RUN / "checkpoints" / roles["best"] / "state.json")
    parent_info = infos(PARENT, ARTIFACT)
    best_is_parent = best_state["step"] == 0 and snapshot_info["weights.safetensors"] == parent_info["weights.safetensors"]
    if best_is_parent and any(snapshot_info[n] != parent_info[n] for n in ("config.json", "tokenizer.json")):
        raise ValueError("BEST0 parent tokenizer/config mismatch")
    estimated_bytes = sum(v["bytes"] for v in native_infos[roles["last"]].values()) + 4 * 1024**2
    if roles["best"] != roles["last"]:
        estimated_bytes += sum(v["bytes"] for n, v in native_infos[roles["best"]].items()
                               if not best_is_parent or n != "weights.safetensors")
    if estimated_bytes > max_disk_mib * 1024**2:
        raise ValueError("Delivery would exceed the configured new-disk budget")
    frozen = DOCS / f"{STEM}-inputs"
    outputs = (PACKAGES, MODEL_REFERENCE, frozen, DOCS / f"{STEM}-window-exposure.csv",
               DOCS / f"{STEM}-target-exposure.json", DOCS / f"{STEM}-delivery.json")
    if any(p.exists() for p in outputs):
        raise FileExistsError("CONTROL final delivery output already exists; nothing overwritten")
    # All closure, identity, sampler and input checks finish before these writes.
    PACKAGES.mkdir(parents=True)
    last_parent = PACKAGES / "LAST"
    last_parent.mkdir()
    last_result = helper.pack(RUN / "checkpoints" / roles["last"], last_parent / roles["last"])
    verify_package(last_parent / roles["last"])
    if best_is_parent:
        best_dir = PACKAGES / "BEST-reference" / roles["best"]
        best_dir.mkdir(parents=True)
        for name in NATIVE:
            if name != "weights.safetensors":
                shutil.copyfile(RUN / "checkpoints" / roles["best"] / name, best_dir / name)
        copied_names = tuple(n for n in NATIVE if n != "weights.safetensors")
        if infos(best_dir, copied_names) != {n: native_infos[roles["best"]][n] for n in copied_names}:
            raise ValueError("Copied native BEST metadata hash mismatch")
        best_delivery = {"schemaVersion": 1, "format": REF_FORMAT, "checkpointName": roles["best"],
                         "nativeFiles": native_infos[roles["best"]],
                         "weightsReference": {"path": rel(PARENT / "weights.safetensors"), **parent_info["weights.safetensors"]},
                         "selectedBestStep": 0, "newControlOptimizerUpdates": 0,
                         "limitation": "Native metadata plus an existing byte-identical weight reference. Use restore-best-reference to materialize all five native files; this directory is not directly loadable."}
        write_json(best_dir / "native-best-reference.json", best_delivery)
        best_delivery = {"kind": "existing-parent-weights-reference", "path": rel(best_dir),
                         "referenceSha256": sha(best_dir / "native-best-reference.json"), "record": best_delivery}
    elif roles["best"] == roles["last"]:
        best_delivery = {"kind": "same-native-checkpoint-as-last", "path": rel(last_parent / roles["last"])}
    else:
        best_parent = PACKAGES / "BEST"
        best_parent.mkdir()
        result = helper.pack(RUN / "checkpoints" / roles["best"], best_parent / roles["best"])
        verify_package(best_parent / roles["best"])
        best_delivery = {"kind": "native-parts", "path": rel(best_parent / roles["best"]), "packageResult": result}
    frozen.mkdir()
    copies = {}
    for name, source in {"status.json": RUN / "status.json", "stage-latest.json": RUN / "stage-latest.json",
                         "stage-design.json": RUN / "stage-design.json", "cache-manifest.json": RUN / "tokenized-cache/manifest.json"}.items():
        target = frozen / name
        shutil.copyfile(source, target)
        if target.read_bytes() != source.read_bytes():
            raise ValueError("Frozen control metadata copy mismatch")
        copies[name] = {"sourcePath": rel(source), "evidencePath": rel(target), "bytes": target.stat().st_size, "sha256": sha(target)}
    csv_path = DOCS / f"{STEM}-window-exposure.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["train_window_index", "real_input_positions", "supervised_positions_including_eos",
                         "supervised_eos_positions", "best_draws", "last_draws"])
        for i in range(len(lengths)):
            writer.writerow([i, lengths[i], targets[i], eos[i], draws["best"][i], draws["last"][i]])
    ledger["perWindowLedger"] = {"path": rel(csv_path), "sha256": sha(csv_path), "rows": len(lengths)}
    ledger_path = DOCS / f"{STEM}-target-exposure.json"
    ledger_sha = write_json(ledger_path, ledger)
    MODEL_REFERENCE.mkdir()
    shutil.copyfile(snapshot / "manifest.json", MODEL_REFERENCE / "manifest.final-selection.json")
    if sha(MODEL_REFERENCE / "manifest.final-selection.json") != snapshot_info["manifest.json"]["sha256"]:
        raise ValueError("Final selection manifest copy mismatch")
    reference = {"schemaVersion": 1, "kind": "final-control-best-reference", "modelId": stage["modelId"],
                 "selectedBestStep": best_state["step"], "selectedCheckpoint": roles["best"],
                 "bestIsInheritedParentWithoutNewControlLearning": best_is_parent, "finalSnapshotFiles": snapshot_info,
                 "artifactDirectory": rel(PARENT) if best_is_parent else None,
                 "finalSelectionManifest": "manifest.final-selection.json", "newDuplicateWeights": False,
                 "limitation": "This reference directory is not directly loadable with ConversationEngine. When artifactDirectory is present, load that complete existing artifact. A learned BEST needs a separate complete inference export."}
    write_json(MODEL_REFERENCE / "artifact-reference.json", reference)
    if ((RUN / "status.json").read_bytes() != status_raw or (RUN / "stage-latest.json").read_bytes() != stage_raw
            or frozen_sources(design) != sources or infos(snapshot, ARTIFACT) != snapshot_info
            or infos(PARENT, ARTIFACT) != parent_info
            or any(infos(RUN / "checkpoints" / n, NATIVE) != info for n, info in native_infos.items())):
        raise ValueError("Frozen terminal source changed during delivery")
    if sha(RUN / "tokenized-cache/train.index.bin") != ledger["trainIndexSha256"] or sha(RUN / "tokenized-cache/train.windows.bin") != ledger["trainDataSha256"]:
        raise ValueError("Frozen TRAIN cache changed during delivery")
    receipt = {"schemaVersion": 1, "kind": "closed-matched-control-final-delivery", "preparedBy": "project-AI",
               "humanReview": False, "createdAtUtc": datetime.now(timezone.utc).isoformat(), "status": status,
               "statusSha256": hashlib.sha256(status_raw).hexdigest(), "stageLatestSha256": hashlib.sha256(stage_raw).hexdigest(),
               "lineage": status["lineage"], "parentSelectedSftStep": 4096, "parentNativeLast8192Used": False,
               "checkpointRoles": roles, "selectedBestStep": best_state["step"], "bestIsInheritedParentWithoutNewControlLearning": best_is_parent,
               "nativeFiles": native_infos, "bestDelivery": best_delivery, "lastDelivery": last_result,
               "finalBestReference": reference, "frozenMetadataCopies": copies, "sourceFiles": sources,
               "targetExposureReceiptPath": rel(ledger_path), "targetExposureReceiptSha256": ledger_sha,
               "targetExposure": {r: e["actual"] for r, e in ledger["checkpoints"].items()},
               "conversationAccepted": False, "testEvaluated": False, "sourceAndInputsUnchanged": True,
               "budget": {"cpuCount": 1, "addressSpaceMiB": 512, "maxCpuSeconds": 120, "maxWallSeconds": 120,
                          "maxNewDiskMiB": max_disk_mib, "maxPartMiB": 32},
               "measurement": {"elapsedWallSeconds": time.monotonic() - started,
                               "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
               "helperPath": rel(Path(__file__).resolve()), "helperSha256": sha(Path(__file__).resolve()),
               "transportHelperSha256": sha(ROOT / "examples/package_native_checkpoint.py"),
               "integrity": {"allNativeHashesUnchanged": True, "allSourceHashesUnchanged": True,
                             "allLastPartAndRecombinedFileHashesVerified": True, "frozenCopiesByteExact": True},
               "method": {"torchImported": False, "weightsLoaded": False, "optimizerPickleLoaded": False,
                          "nativeRestoreExecuted": False, "trainingExecuted": False, "modelInferenceExecuted": False,
                          "softwareTestsExecuted": False, "signalsSent": False, "testOrReservedRead": False},
               "publication": {"localFilesPrepared": True, "gitHubUploadVerified": False},
               "limitation": "Native bytes and control exposure only; BEST0 is inherited parent quality, not new learning. Actual LAST inference export and conversational evaluation are separate."}
    receipt_path = DOCS / f"{STEM}-delivery.json"
    new_files = [p for folder in (PACKAGES, MODEL_REFERENCE, frozen) for p in folder.rglob("*") if p.is_file()]
    new_files.extend((csv_path, ledger_path))
    new_bytes = sum(p.stat().st_size for p in new_files)
    if new_bytes > max_disk_mib * 1024**2 or any(p.stat().st_size >= 100 * 1024**2 for p in new_files):
        raise ValueError("Actual delivery disk or published file bound exceeded")
    receipt["newDeliveryBytesBeforeThisReceipt"] = new_bytes
    receipt["localNewFilesBeforeThisReceipt"] = {rel(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in new_files}
    receipt_sha = write_json(receipt_path, receipt)
    print(json.dumps({"receiptPath": rel(receipt_path), "receiptSha256": receipt_sha,
                      "ledgerSha256": ledger_sha, "selectedBestStep": best_state["step"],
                      "lastWeightsSha256": native_infos[roles["last"]]["weights.safetensors"]["sha256"],
                      "actualExposure": ledger["checkpoints"]["last"]["actual"]}, indent=2))


def restore_best_reference(reference_dir, destination):
    source = Path(reference_dir).resolve()
    destination = Path(destination).absolute()
    record = read(source / "native-best-reference.json")
    if record["format"] != REF_FORMAT or destination.name != record["checkpointName"]:
        raise ValueError("Reference format or native checkpoint basename mismatch")
    if set(record["nativeFiles"]) != set(NATIVE) or set(p.name for p in source.iterdir()) != (set(NATIVE) - {"weights.safetensors"}) | {"native-best-reference.json"}:
        raise ValueError("Unexpected reference metadata file set")
    weights = (ROOT / record["weightsReference"]["path"]).resolve()
    if not weights.is_relative_to(ROOT) or not weights.is_file():
        raise ValueError("Weight reference must resolve to an existing repository file")
    for name in NATIVE:
        path = weights if name == "weights.safetensors" else source / name
        info = record["nativeFiles"][name]
        if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise ValueError("Native reference input hash/size mismatch")
    if destination.exists() or not destination.parent.is_dir():
        raise FileExistsError("Restore requires a new destination and existing parent")
    temp = destination.parent / (".control-best-restore-" + uuid.uuid4().hex)
    temp.mkdir(mode=0o700)
    try:
        for name in NATIVE:
            shutil.copyfile(weights if name == "weights.safetensors" else source / name, temp / name)
        if infos(temp, NATIVE) != record["nativeFiles"]:
            raise ValueError("Restored native BEST hash mismatch")
        if read(temp / "manifest.json")["files"] != {n: v["sha256"] for n, v in record["nativeFiles"].items() if n != "manifest.json"}:
            raise ValueError("Restored native manifest mismatch")
        helper = transport()
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            helper._rename_new(fd, temp.name, destination.name)
        finally:
            os.close(fd)
    except BaseException:
        shutil.rmtree(temp)
        raise
    print(json.dumps({"restoredNativeBest": str(destination), "checkpointName": record["checkpointName"],
                      "nativeFileHashesVerified": True, "trainingExecuted": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "deliver", "restore-best-reference"), nargs="?", default="preflight")
    parser.add_argument("--max-new-disk-mib", type=int, default=256)
    parser.add_argument("--reference-dir")
    parser.add_argument("--destination")
    args = parser.parse_args()
    limits()
    if args.operation == "preflight":
        status = read(RUN / "status.json")
        sources = frozen_sources(read(RUN / "stage-design.json"))
        print(json.dumps({"operation": "read-only-preflight", "status": status["status"], "step": status["step"],
                          "workerAlive": alive(status["pid"], status["processBirth"]), "sourcesPinned": bool(sources),
                          "noOutputCreated": True, "deliveryRequiresTerminal4096": True}, indent=2))
    elif args.operation == "deliver":
        if not 256 <= args.max_new_disk_mib <= 512:
            parser.error("max-new-disk-mib must be between 256 and 512")
        deliver(args.max_new_disk_mib)
    else:
        if not args.reference_dir or not args.destination:
            parser.error("restore-best-reference needs --reference-dir and --destination")
        restore_best_reference(args.reference_dir, args.destination)


if __name__ == "__main__":
    main()
