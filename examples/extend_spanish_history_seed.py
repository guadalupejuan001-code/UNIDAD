#!/usr/bin/env python3
"""Extend frozen TRAIN/VALIDATION digest guards; never read TEST text.

The old JSON object remains unmodified under its existing keys, and its exact
original bytes are copied separately. Added whole-text hashes retain every
observed fold rather than choosing one fold when a collision occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import tempfile
import time

MIB = 1 << 20
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-seed", type=Path, required=True)
    parser.add_argument("--initial-seed-sha256", required=True)
    parser.add_argument("--wide-corpus", type=Path, required=True)
    parser.add_argument("--wide-corpus-sha256", required=True)
    parser.add_argument("--wide-manifest", type=Path, required=True)
    parser.add_argument("--wide-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    started = time.monotonic()
    cpu_started = time.process_time()
    read_bytes = written_bytes = 0
    limit = 128 * MIB

    def check(incoming=0, outgoing=0):
        if read_bytes + written_bytes + incoming + outgoing > limit:
            raise ValueError("128-MiB seed read/write budget exceeded")
        if time.monotonic() - started >= 590:
            raise TimeoutError("600-second seed preparation limit approached")

    def pinned_read(path, expected):
        nonlocal read_bytes
        check(incoming=path.stat().st_size)
        raw = path.read_bytes()
        read_bytes += len(raw)
        if digest(raw) != expected:
            raise ValueError("frozen history guard input differs: " + path.name)
        return raw

    if args.output_dir.exists():
        raise ValueError("output exists; frozen history guards are never replaced")
    initial_raw = pinned_read(args.initial_seed, args.initial_seed_sha256)
    initial = json.loads(initial_raw)
    # Both are public TRAIN/VALIDATION-only metadata; assert this before use.
    if any(split not in ("train", "validation") for split in initial["seenRecordTextSha256"].values()):
        raise ValueError("initial full-record guard unexpectedly includes TEST")
    if any(split not in ("train", "validation") for splits in initial["seenNormalizedParagraphSha256"].values() for split in splits):
        raise ValueError("initial paragraph guard unexpectedly includes TEST")
    wide_manifest_raw = pinned_read(args.wide_manifest, args.wide_manifest_sha256)
    wide_manifest = json.loads(wide_manifest_raw)
    if wide_manifest.get("splitCounts", {}).get("test") != 0:
        raise ValueError("wide manifest does not explicitly exclude TEST")
    wide_raw = pinned_read(args.wide_corpus, args.wide_corpus_sha256)
    raw_folds, normalized_folds, paragraph_folds = defaultdict(set), defaultdict(set), defaultdict(set)
    family_folds, counts = {}, Counter()
    for line in wide_raw.splitlines():
        check()
        record = json.loads(line)
        split, group, text = record["split"], record["groupId"], record["text"]
        if split not in ("train", "validation"):
            raise ValueError("TEST is forbidden in wide history guard")
        if group in family_folds and family_folds[group] != split:
            raise ValueError("wide family changed its previous fold")
        family_folds[group] = split
        counts[split] += 1
        raw_folds[digest(text.encode())].add(split)
        normalized_folds[digest(" ".join(text.split()).encode())].add(split)
        for part in set(text.split("\n\n") + text.splitlines()):
            if len(WORD.findall(part)) >= 12:
                paragraph_folds[digest(" ".join(part.split()).encode())].add(split)
    if counts != Counter({k: v for k, v in wide_manifest["splitCounts"].items() if k != "test"}):
        raise ValueError("wide corpus counts differ from pinned manifest")
    # No old key is overwritten, including old collision representation.
    additions = {
        "wideHistoryExtension": {
            "schemaVersion": 1,
            "initialSeedSha256": args.initial_seed_sha256,
            "wideCorpusSha256": args.wide_corpus_sha256,
            "wideManifestSha256": args.wide_manifest_sha256,
            "wideRecords": sum(counts.values()),
            "wideSplitCounts": dict(counts),
            "wideFamilyCount": len(family_folds),
            "minimumParagraphWords": 12,
            "testTextRead": False,
            "oldSeedKeysChanged": False,
            "normalization": "Unicode whitespace collapse only; not semantic paraphrase detection",
        },
        "seenWideRecordTextSha256MultiFold": {key: sorted(value) for key, value in raw_folds.items()},
        "seenWideNormalizedFullRecordTextSha256MultiFold": {key: sorted(value) for key, value in normalized_folds.items()},
        "seenWideNormalizedParagraphSha256MultiFold": {key: sorted(value) for key, value in paragraph_folds.items()},
        "seenWideFamilyFolds": family_folds,
    }
    if set(initial) & set(additions):
        raise ValueError("extension would overwrite a historical seed key")
    extended_raw = encode({**initial, **additions})
    encoded_seed = json.loads(extended_raw)
    if any(encoded_seed[key] != value for key, value in initial.items()):
        raise ValueError("historical seed object changed")
    del encoded_seed
    del wide_raw
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.output_dir.name + ".", dir=args.output_dir.parent))

    def write(name, raw):
        nonlocal written_bytes
        check(outgoing=len(raw))
        with (staging / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        written_bytes += len(raw)

    try:
        write("initial-seed.json", initial_raw)
        write("wide-history-seed.json", extended_raw)
        helper_raw = Path(__file__).read_bytes()
        check(incoming=len(helper_raw))
        read_bytes += len(helper_raw)
        report = {
            "schemaVersion": 1,
            "status": "prepared",
            "initialSeed": {"path": str(args.initial_seed), "sha256": args.initial_seed_sha256, "bytes": len(initial_raw)},
            "exactInitialSeedCopySha256": digest(initial_raw),
            "extendedSeedSha256": digest(extended_raw),
            "extendedSeedBytes": len(extended_raw),
            "oldSeedKeysChanged": False,
            "wideCorpusSha256": args.wide_corpus_sha256,
            "wideManifestSha256": args.wide_manifest_sha256,
            "wideRecords": sum(counts.values()),
            "wideSplitCounts": dict(counts),
            "wideFamilies": len(family_folds),
            "newGuardHashCounts": {"rawFull": len(raw_folds), "normalizedFull": len(normalized_folds), "normalizedParagraph": len(paragraph_folds)},
            "newGuardMultipleFoldHashCounts": {"rawFull": sum(len(value) > 1 for value in raw_folds.values()), "normalizedFull": sum(len(value) > 1 for value in normalized_folds.values()), "normalizedParagraph": sum(len(value) > 1 for value in paragraph_folds.values())},
            "historicalCrossfoldHashInformationPreserved": True,
            "helperSha256": digest(helper_raw),
            "testTextRead": False,
            "modelsOrTokenizersLoaded": False,
            "softwareTestsExecuted": False,
            "resources": {"readBytes": read_bytes, "writtenBytesBeforeReceipt": written_bytes, "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "wallSeconds": time.monotonic() - started, "cpuSeconds": time.process_time() - cpu_started},
            "limits": {"cpuCores": 1, "addressSpaceBytes": 512 * MIB, "readPlusWriteBytes": limit, "seconds": 600},
            "limitations": ["No semantic or exhaustive cross-TEST paraphrase guard.", "Initial normalized whole-record hashes unavailable; only the extension gains them.", "Old initial seed bytes and keys remain unchanged, including its historical digest-only TEST ID metadata."],
        }
        write("seed-preparation-receipt.json", encode(report))
        os.rename(staging, args.output_dir)
        print(json.dumps({"status": "prepared", "output": str(args.output_dir), "extendedSeedSha256": digest(extended_raw), "readBytes": read_bytes, "writtenBytes": written_bytes, "multipleFoldParagraphHashes": report["newGuardMultipleFoldHashCounts"]["normalizedParagraph"]}))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
