#!/usr/bin/env python3
"""Remove two wholly defective regional parents from frozen v3, preserving all other bytes."""
import argparse
import collections
import hashlib
import json
import resource
import time
from pathlib import Path


CONTROL_SHA = "48cc11848568c03a32727660968e47695d8eeb43c3302cbf7055eefe2e449491"
REGIONAL_V3_SHA = "aea0316e1fa6985b079f91a2104e9d3ba3889931c8d57162dcfaf463fd135b36"
EXCLUDED = {"wiktionary:es:regional:1005730:6394036": "algarrobilla",
            "wiktionary:es:regional:935839:6383121": "caraota"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(previous, output, design):
    started = time.monotonic()
    if output.exists():
        raise ValueError("Output already exists; frozen history is preserved")
    control = (previous / "natural-control.train-validation.jsonl").read_bytes()
    regionals = (previous / "regional-sft.train-validation.jsonl").read_bytes()
    if sha(control) != CONTROL_SHA or sha(regionals) != REGIONAL_V3_SHA:
        raise ValueError("Frozen v3 input identity mismatch")
    kept, dropped = [], []
    for raw in regionals.splitlines(keepends=True):
        row = json.loads(raw)
        if row["split"] not in ("train", "validation"):
            raise ValueError("TEST is not allowed")
        if row["source"]["parentRecordId"] in EXCLUDED:
            dropped.append(row)
        else:
            kept.append(raw)
    if len(dropped) != 6 or len(kept) != 463:
        raise ValueError("Unexpected final review delta")
    final_regional = b"".join(kept)
    combined = control + final_regional
    if len(control) + len(final_regional) + len(combined) >= 64 * 1024**2:
        raise ValueError("64 MiB output budget exceeded")
    output.mkdir(parents=True)
    files = {"natural-control.train-validation.jsonl": control,
             "regional-sft.train-validation.jsonl": final_regional,
             "natural-latam-sft.train-validation.jsonl": combined,
             "prospective-design.json": design.read_bytes(),
             "preceding-target-exclusions.json": (previous / "target-exclusions.json").read_bytes()}
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    receipt = {"schemaVersion": 1, "review": "Project AI, not human; final semantic QA removes whole defective parents",
        "inputRegionalSha256": REGIONAL_V3_SHA,
        "removedParents": EXCLUDED, "removedRows": [r["id"] for r in dropped],
        "reasons": {"algarrobilla": "Alternative sense still has genus/originaria agreement and missing standalone botanical clarity",
                    "caraota": "Alternative plant sense has malformed grammar and broken sentence; no positive targets rewritten"},
        "allOtherRegionalRowsByteExact": True, "originalControlByteExact": True,
        "oldSourceAndVersionsPreserved": True, "testRowsRead": 0,
        "modelOrTokenizerLoaded": False}
    (output / "final-review-exclusions.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    arms = {}
    for arm, name in (("natural-control", "natural-control.train-validation.jsonl"),
                      ("natural-latam", "natural-latam-sft.train-validation.jsonl")):
        raw = files[name]
        rows = [json.loads(line) for line in raw.splitlines()]
        content = hashlib.sha256()
        counts = collections.Counter()
        for row in rows:
            counts[row["split"]] += 1
            content.update((json.dumps(row["messages"], ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode())
        report = {"schemaVersion": 1, "curriculumVersion": "v4", "arm": arm,
            "status": "prepared-not-trained-not-conversation-accepted",
            "corpusSha256": sha(raw), "contentSha256": content.hexdigest(),
            "file": {"path": name, "bytes": len(raw), "sha256": sha(raw)},
            "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
            "designSha256": sha(design.read_bytes()), "helperSha256": sha(Path(__file__).read_bytes()),
            "precedingRegionalSha256": REGIONAL_V3_SHA,
            "finalReviewExclusionsSha256": sha((output / "final-review-exclusions.json").read_bytes()),
            "regionalVersionLabels": "Retained per-record v3 labels intentionally preserve all surviving bytes; final v4 is a strict subset",
            "policy": "assistant_last_reply; old targets/1704 VAL bytes and original group folds preserved",
            "comparison": "Two arms from same own parent, same 4096 logical updates/config; different internal VAL NLLs not comparable; same external DEV actual histories"}
        manifest_path = output / (arm + ".manifest.json")
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        arms[arm] = {"path": str(manifest_path), "manifestSha256": sha(manifest_path.read_bytes()),
                     "corpusSha256": sha(raw), "contentSha256": content.hexdigest(),
                     "splitCounts": report["splitCounts"]}
    receipt.update({"elapsedSeconds": time.monotonic() - started,
                    "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "regionalRows": len(kept), "arms": arms,
                    "helperSha256": sha(Path(__file__).read_bytes())})
    (output / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-v3", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    print(json.dumps(build(args.previous_v3, args.output_dir, args.design), ensure_ascii=False))


if __name__ == "__main__":
    main()
