"""Count actual supervised TRAIN positions from a native mmap checkpoint.

This utility uses only the standard library. It never loads model weights,
pickle/optimizer state, a tokenizer, validation windows, or TEST windows.
The completed-epoch formula applies only to uniform sampling without replacement;
other sampling policies are deliberately refused rather than estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import resource
import struct
import time
from datetime import datetime, timezone


INDEX = struct.Struct("<QQ")
IGNORE_INDEX = -100


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(path, deadline):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            if time.monotonic() > deadline:
                raise TimeoutError("Finite exposure-audit wall budget exceeded")
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(cache, state_path, max_seconds):
    started = time.monotonic()
    deadline = started + max_seconds
    manifest_path = cache / "manifest.json"
    index_path = cache / "train.index.bin"
    data_path = cache / "train.windows.bin"
    paths = [manifest_path, index_path, data_path, state_path]
    before = {str(path): digest(path, deadline) for path in paths}
    manifest = json.loads(manifest_path.read_bytes())
    state_bytes = state_path.read_bytes()
    require(hashlib.sha256(state_bytes).hexdigest() == before[str(state_path)],
            "Checkpoint state changed before parsing")
    state = json.loads(state_bytes)
    identity = manifest["identity"]
    require(manifest["schemaVersion"] == 1 and
            identity["windowEncoding"] == "little-endian-int32-inputs-labels-v1",
            "Unsupported native mmap format")
    require(identity["includeTest"] is False and
            identity["partitionPolicy"] == "train_validation_only",
            "Exposure audit requires a TRAIN/validation-only cache")
    require(state["data_loading"]["cache"] == manifest,
            "Checkpoint and cache manifests differ")
    require(state["corpus_sha256"] == identity["corpusSha256"],
            "Checkpoint corpus identity differs")
    config = state["training_config"]
    require(config.get("data_loading") == "mmap" and
            config.get("objective", "assistant_last_reply") == identity["objective"] and
            config["window_policy"] == identity["windowPolicy"],
            "Checkpoint training policy differs from mmap identity")
    require(not config.get("sampling_weights") and not config.get("adaptive_sampling") and
            "sampling" not in state and "adaptive_sampling" not in state,
            "Only uniform sampling without replacement has exact completed-epoch accounting here")
    train = manifest["splits"]["train"]
    count = train["count"]
    require(type(count) is int and count > 0, "No TRAIN windows")
    for key, path in (("index", index_path), ("data", data_path)):
        require(train[key]["name"] == path.name and
                train[key]["bytes"] == path.stat().st_size and
                train[key]["sha256"] == before[str(path)], "TRAIN file identity differs")
    require(index_path.stat().st_size == count * INDEX.size, "Invalid TRAIN index size")
    epoch, cursor, step, order = (state[key] for key in ("epoch", "cursor", "step", "order"))
    require(all(type(value) is int and value >= 0 for value in (epoch, cursor, step)),
            "Invalid checkpoint counters")
    require(type(order) is list and all(type(value) is int for value in order),
            "Invalid order")
    require((not order and cursor == 0) or
            (len(order) == count and set(order) == set(range(count)) and 0 <= cursor < count),
            "Checkpoint order is not a uniform epoch permutation")
    batch_size = config["batch_size"]
    require(type(batch_size) is int and batch_size > 0 and cursor % batch_size == 0,
            "Cursor is not a completed logical batch boundary")
    require(step == epoch * math.ceil(count / batch_size) + cursor // batch_size,
            "Step, epoch, and cursor disagree")
    targets, lengths = [], []
    offset = 0
    with index_path.open("rb") as indexes, data_path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for _ in range(count):
                require(time.monotonic() <= deadline, "Finite wall budget exceeded")
                start, length = INDEX.unpack(indexes.read(INDEX.size))
                require(start == offset and 1 <= length <= identity["contextLength"],
                        "Invalid TRAIN index bounds")
                offset += length * 8
                require(offset <= len(data), "Index exceeds TRAIN data")
                # Native storage is [inputs:int32[length], labels:int32[length]].
                labels = struct.unpack_from(f"<{length}i", data, start + length * 4)
                require(all(label == IGNORE_INDEX or 0 <= label < identity["vocabSize"]
                            for label in labels), "Invalid target or ignore index")
                supervised = sum(label != IGNORE_INDEX for label in labels)
                require(supervised > 0, "TRAIN window has no positive supervision")
                targets.append(supervised)
                lengths.append(length)
            require(offset == len(data), "Unindexed TRAIN bytes")
    after = {str(path): digest(path, deadline) for path in paths}
    require(before == after, "An input changed during exposure accounting")
    partial_targets = sum(targets[index] for index in order[:cursor])
    partial_inputs = sum(lengths[index] for index in order[:cursor])
    per_epoch_targets, per_epoch_inputs = sum(targets), sum(lengths)
    report = {
        "schemaVersion": 1,
        "kind": "native-checkpoint-exact-supervised-train-exposure",
        "preparedBy": "project-AI",
        "humanReview": False,
        "observedAtUtc": datetime.now(timezone.utc).isoformat(),
        "checkpointName": state_path.parent.name,
        "checkpointStateSha256": before[str(state_path)],
        "cacheManifestSha256": before[str(manifest_path)],
        "trainIndexSha256": before[str(index_path)],
        "trainDataSha256": before[str(data_path)],
        "identity": identity,
        "trainingConfig": config,
        "exposureState": {"step": step, "epoch": epoch, "cursor": cursor, "order": order},
        "counts": {
            "step": step,
            "completedEpochs": epoch,
            "currentEpochCursor": cursor,
            "trainWindowsPerEpoch": count,
            "supervisedTargetsPerEpoch": per_epoch_targets,
            "unpaddedInputPositionsPerEpoch": per_epoch_inputs,
            "completeEpochTargetPositions": epoch * per_epoch_targets,
            "partialEpochTargetPositions": partial_targets,
            "actualWindowDraws": epoch * count + cursor,
            "actualTargetPositions": epoch * per_epoch_targets + partial_targets,
            "actualUnpaddedInputPositions": epoch * per_epoch_inputs + partial_inputs,
        },
        "checkpointElapsedSeconds": state["elapsed_seconds"],
        "bestValidationNll": state["best_validation_nll"],
        "formula": "epoch * sum(count(label != -100) per TRAIN window) + "
                   "sum(target_counts[order[i]] for i in range(cursor))",
        "method": {
            "weightsLoaded": False, "resumePickleLoaded": False, "torchImported": False,
            "trainLabelsRead": True, "trainInputsDecoded": False,
            "validationWindowsRead": False, "testWindowsRead": False,
            "corpusRead": False, "evaluationRead": False, "testsExecuted": False,
            "allInputHashesUnchanged": True,
        },
        "limitation": "Positions presented to the loss, including supervised EOS; "
                      "not unique words, knowledge acquired, gradient share, or conversation quality.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()
    require(1 <= args.max_seconds <= 300, "Audit budget must be 1..300 seconds")
    # Keep bookkeeping below the active model's resources.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (args.max_seconds, args.max_seconds))
    report = audit(args.cache.resolve(), args.state.resolve(), args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        prior = json.loads(args.output.read_bytes())
        report["observedAtUtc"] = prior.get("observedAtUtc")
        require(prior == report, "Refusing to overwrite a different exposure receipt")
    else:
        with args.output.open("xb") as stream:
            stream.write(encoded(report))
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "counts": report["counts"], "allInputHashesUnchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
