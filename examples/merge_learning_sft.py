#!/usr/bin/env python3
"""Merge fixed source-curated SFT corpora without training or TEST access."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_human_dialogues as human
from ia_propia.conversation.contract import DEFAULT_SYSTEM_PROMPT
from ia_propia.conversation.data import load_corpus


def effective_messages(record):
    messages = human.normalized_messages(record)
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
    return messages


def merge(args):
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows, by_id, receipts, duplicate_ids = [], {}, [], 0
    for path_string, fingerprint, declared_kind in args.input:
        path = Path(path_string)
        if human.sha(path) != fingerprint:
            raise ValueError("immutable SFT input hash differs")
        current = load_corpus(path)
        for record in current:
            if record["split"] not in ("train", "validation"):
                raise ValueError("TEST is forbidden in training input")
            if record["source"]["kind"] != declared_kind:
                raise ValueError("source authorship kind differs")
            if record["id"] in by_id:
                earlier = by_id[record["id"]]
                if any(record[key] != earlier[key] for key in ("groupId", "split", "messages")):
                    raise ValueError("duplicate ID changes training content")
                duplicate_ids += 1
                continue
            by_id[record["id"]] = record
            rows.append(record)
        receipts.append({"path": str(path), "sha256": fingerprint, "records": len(current),
                         "declaredSourceKind": declared_kind})
    family_splits, by_prompt, by_dialogue = {}, defaultdict(list), defaultdict(list)
    for record in rows:
        group = record["groupId"]
        if group in family_splits and family_splits[group] != record["split"]:
            raise ValueError("source family crosses training/validation")
        family_splits[group] = record["split"]
        messages = effective_messages(record)
        by_prompt[human.fingerprint(messages[:-1])].append(record)
        by_dialogue[human.fingerprint(messages)].append(record)
    excluded, collisions = set(), {}
    for name, mapping in (("fullPrompt", by_prompt), ("fullDialogue", by_dialogue)):
        clashes = [values for values in mapping.values() if len({r["split"] for r in values}) > 1]
        groups = {r["groupId"] for values in clashes for r in values}
        excluded.update(groups)
        collisions[name] = {"classes": len(clashes), "families": len(groups)}
    retained, seen_dialogues, duplicate_dialogues = [], set(), 0
    for record in rows:
        if record["groupId"] in excluded:
            continue
        fingerprint = human.fingerprint(effective_messages(record))
        if fingerprint in seen_dialogues:
            duplicate_dialogues += 1
            continue
        seen_dialogues.add(fingerprint)
        retained.append(record)
    split_counts = Counter(r["split"] for r in retained)
    human_counts = Counter(r["split"] for r in retained if r["source"]["kind"] == "public-human-reviewed")
    ratios = {split: human_counts[split] / split_counts[split] for split in ("train", "validation")}
    if not .30 <= ratios["train"] <= .70:
        raise ValueError("human TRAIN proportion outside fixed 30–70% range")
    output = args.output / "sft.train-validation.jsonl"
    files = human.write_records(output, retained)
    content = hashlib.sha256()
    for record in retained:
        content.update((json.dumps(record["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    for receipt in receipts:
        if human.sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("SFT input changed during merge")
    exclusion_path = args.target_exclusion_receipt
    source_exclusions = json.loads(exclusion_path.read_text())
    report = {"schemaVersion": 1, "purpose": "fixed finite SFT union; not acceptance evidence",
              "corpusSha256": files["sha256"], "contentSha256": content.hexdigest(), "file": files,
              "splitCounts": {"train": split_counts["train"], "validation": split_counts["validation"], "test": 0},
              "humanRecordCounts": dict(human_counts), "humanRecordRatios": ratios,
              "sources": [{**r, "license": "Apache-2.0 upstream" if r["declaredSourceKind"] == "public-human-reviewed" else "Project AI-authored fiction; no independent data license declared", "receipt": r["path"]} for r in receipts],
              "tokenizerSha256": args.tokenizer_sha256, "contextLength": 512,
              "lengthPolicy": "Components fixed to own BPE2048 before merge; merge never changes message text",
              "supervision": "assistant_last_reply; only final assistant reply and EOS are positive targets",
              "targetExclusions": {"receiptPath": str(exclusion_path), "receiptSha256": human.sha(exclusion_path),
                                   "policy": "Author-identified erroneous prefixes omitted; unchanged erroneous history retained in later correct prefixes", "sourceReceipt": source_exclusions},
              "guards": {"crossfold": collisions, "excludedWholeFamilyIds": sorted(excluded),
                         "duplicateIdsRemoved": duplicate_ids, "samefoldExactDialoguesRemoved": duplicate_dialogues,
                         "sourceRecordsBeforeGuards": len(rows), "retainedRecords": len(retained), "foldsMoved": 0,
                         "sourceTextsOrTargetsRewritten": False},
              "sampling": "uniform shuffle without replacement, no weights, resampling or duplicated target IDs",
              "testUsedForTraining": False, "testReadOrTokenizedDuringMerge": False,
              "modelsOrTokenizersLoadedDuringMerge": False, "ownerMessagesIncluded": False,
              "helperSha256": human.sha(Path(__file__)),
              "limitations": ["Upstream human review is not exhaustive factual certification",
                              "Synthetic templates repeat task semantics; unique IDs are not proof of unlimited diversity",
                              "Source-author editorial review is by project AI assistants, not independent human evaluation"]}
    human.write_json(args.output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=3, action="append", required=True, metavar=("PATH", "SHA256", "KIND"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    args = parser.parse_args()
    report = merge(args)
    print(json.dumps({key: report[key] for key in ("corpusSha256", "contentSha256", "splitCounts", "humanRecordRatios", "file", "guards")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
