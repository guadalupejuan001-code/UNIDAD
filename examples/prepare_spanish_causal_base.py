#!/usr/bin/env python3
"""Freeze a bounded public Spanish TEXT base; no models or tokenizer calls.

Uses pinned Wikipedia rows and already hash-verified OASST importer outputs.
Selection depends only on source IDs, fixed prose filters and byte budgets.
No owner memory, model answers, losses or acceptance cases are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_public_spanish_language as public
from ia_propia.conversation.data import assign_split, load_corpus
from ia_propia.conversation.language_data import load_text_corpus, validate_text_record

MIB = 1 << 20
OASST_INPUTS = (
    ("oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b", "96dc38a0aa4434516840c67b75c5ddabbfb4c0daf1e4b46cca274bc1c1e36825"),
    ("oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75", "161b8c4bda185889d1e77b1581eccd6b24810f37292e30db05edc9a1ef07b5b9"),
)
IDENTITY_MANIFEST_SHA = "cb5d566e66c3c0a26c135348fbd9b23f25b1df7fa1e3e4f325e9fb6c3b5d6838"


def normalize(text):
    return " ".join(text.split())


def prose_paragraphs(text):
    """Select source paragraphs; preserve their words and internal punctuation."""
    kept, reasons = [], Counter()
    for line in text.splitlines():
        paragraph = line.strip()
        words = public.WORD.findall(paragraph)
        if len(words) < 25:
            reasons["short_heading_list_or_fragment"] += 1
        elif any(t in paragraph for t in public.SPECIAL_TOKENS):
            reasons["reserved_control_token"] += 1
        elif sum(ch.isalpha() for ch in paragraph) / max(1, sum(not ch.isspace() for ch in paragraph)) < .55:
            reasons["low_prose_letter_fraction"] += 1
        elif len({w.casefold() for w in words} & public.SPANISH_FUNCTIONS) < 3:
            reasons["weak_spanish_prose_signal"] += 1
        else:
            kept.append(paragraph)
    return kept, dict(reasons)


def clone_families(records, paragraphs):
    """Reject whole families sharing normalized source text across folds."""
    by_text = defaultdict(list)
    for record in records:
        for text in paragraphs.get(record["id"], [record["text"]]):
            by_text[public.sha(normalize(text).encode())].append(record)
    collisions = [rows for rows in by_text.values() if len({r["split"] for r in rows}) > 1]
    groups = {r["groupId"] for rows in collisions for r in rows}
    return groups, {"crossfoldNormalizedTextClasses": len(collisions), "wholeFamilies": len(groups)}


def select_families(records, budgets, salt):
    """Stable per-fold whole-family selection, with a soft final-family cap."""
    families = defaultdict(list)
    for record in records:
        families[(record["split"], record["groupId"])].append(record)
    selected, receipt = [], {}
    for split in ("train", "validation", "test"):
        groups = [key for key in families if key[0] == split]
        groups.sort(key=lambda key: public.sha((salt + "\0" + key[1]).encode()))
        used, count = 0, 0
        for key in groups:
            if used >= budgets[split]:
                break
            rows = families[key]
            selected.extend(rows)
            used += sum(len(r["text"].encode()) for r in rows)
            count += 1
        receipt[split] = {"requestedPayloadBytes": budgets[split], "selectedPayloadBytes": used,
                          "selectedFamilies": count, "availableFamilies": len(groups)}
    return selected, receipt


def oasst_records(directory, identity_manifest):
    if public.file_sha(identity_manifest) != IDENTITY_MANIFEST_SHA:
        raise ValueError("identity guard manifest differs from fixed reviewed v3")
    guard = json.loads(identity_manifest.read_text())
    excluded = set(guard["externalSelfIdentityGuard"]["excludedSourceFamilyIds"]) | set(guard["guards"]["excludedFamilyIds"])
    messages, receipts, duplicates = {}, [], 0
    for dataset, revision, expected in OASST_INPUTS:
        path = directory / f"{dataset}-importer-intermediate.jsonl"
        if public.file_sha(path) != expected:
            raise ValueError("pinned OASST canonical input differs")
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != expected or manifest["source"]["revision"] != revision or manifest["source"]["externalSyntheticExcluded"] is not True:
            raise ValueError("OASST provenance differs")
        receipts.append({"dataset": "OpenAssistant/" + dataset, "revision": revision,
                         "canonicalSha256": expected, "canonicalManifestSha256": public.file_sha(manifest_path),
                         "rawSourceSha256": manifest["sourceSha256"], "sourceFilters": manifest["filters"]})
        for record in load_corpus(path):
            if record["split"] != assign_split(record["groupId"]):
                raise ValueError("historical OASST fold changed")
            for mid, message in zip(record["source"]["messageIds"], record["messages"]):
                value = {"id": "oasst-text:" + mid, "groupId": record["groupId"], "split": record["split"],
                         "source": {"dataset": "OpenAssistant/" + dataset, "kind": "public-human-reviewed",
                                    "language": "es", "license": "Apache-2.0", "revision": revision,
                                    "messageId": mid, "role": message["role"], "canonicalCorpusSha256": expected,
                                    "extraction": "one original reviewed message, content only; no role wrapper",
                                    "externalSyntheticExcluded": True}, "text": message["content"]}
                if mid in messages:
                    previous = messages[mid]
                    if any(previous[k] != value[k] for k in ("groupId", "split", "text")) or previous["source"]["role"] != message["role"]:
                        raise ValueError("same public message ID changed content or fold")
                    duplicates += 1
                    continue
                messages[mid] = value
    records, reasons = [], Counter()
    for mid in sorted(messages):
        record = messages[mid]
        if record["groupId"] in excluded:
            reasons["historical_exact_clone_or_external_identity_family"] += 1
        elif len(public.WORD.findall(record["text"])) < 10:
            reasons["message_below_10_words"] += 1
        else:
            validate_text_record(record)
            records.append(record)
    return records, {"inputs": receipts, "identityManifestSha256": IDENTITY_MANIFEST_SHA,
                     "uniqueSourceMessages": len(messages), "repeatedMessageIdsIgnored": duplicates,
                     "historicalExcludedFamilyIds": sorted(excluded), "messageFilters": dict(reasons)}


def wiki_page(order):
    # Starts beyond the historical helper's first 240 pages; same immutable
    # snapshot and ID-derived folds, but a distinct deterministic article pool.
    return public.fetch_page(order + 541)


def prepare(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw_cache = args.output / "wiki-row-pages"
    raw_cache.mkdir(mode=0o700)
    deadline = started + args.budget_seconds
    check = lambda: resource_check(args.output, deadline, args.disk_cap_mib * MIB)
    human, human_receipt = oasst_records(args.oasst_directory, args.identity_manifest)
    human_budget = int(args.oasst_target_mib * MIB)
    budgets = {"train": int(human_budget * .8), "validation": int(human_budget * .1), "test": int(human_budget * .1)}
    human, human_selection = select_families(human, budgets, "unidad-causal-oasst-v1")
    wiki, paragraphs, pages, filters, ids = [], {}, [], Counter(), set()
    payload = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, args.max_pages, args.workers):
            check()
            for order, offset, raw, headers, value in pool.map(wiki_page, range(first, min(first + args.workers, args.max_pages))):
                check()
                name = f"page-{order:04d}-offset-{offset}.json"
                public.write_exclusive(raw_cache / name, raw)
                pages.append({"order": order, "offset": offset, "file": name, "sha256": public.sha(raw), "bytes": len(raw), "response": headers})
                for wrapped in value["rows"]:
                    row = wrapped.get("row", {})
                    reason = public.wiki_quality(row, wrapped.get("truncated_cells", []))
                    if reason:
                        filters[reason] += 1
                        continue
                    if row["id"] in ids:
                        raise ValueError("Wikipedia article ID repeated")
                    ids.add(row["id"])
                    kept, paragraph_filters = prose_paragraphs(row["text"])
                    filters.update(paragraph_filters)
                    cleaned = "\n\n".join(kept)
                    if len(public.WORD.findall(cleaned)) < 150:
                        filters["retained_prose_below_150_words"] += 1
                        continue
                    group = "wikipedia:es:" + row["id"]
                    raw_sha = public.sha(row["text"].encode())
                    chunks = public.split_utf8_exact(cleaned)
                    for index, chunk in enumerate(chunks):
                        record = {"id": f"wikipedia:{public.WIKI_CONFIG}:{row['id']}:prose:{index}", "groupId": group,
                                  "split": assign_split(group), "source": {"dataset": "wikimedia/wikipedia", "kind": "public-encyclopedic",
                                  "language": "es", "configuration": public.WIKI_CONFIG, "revision": public.WIKI_REVISION,
                                  "license": "CC-BY-SA-3.0", "licenseDeclaration": "pinned HF card, also declares GFDL; current Wikimedia legal declarations retained separately",
                                  "pageId": row["id"], "title": row["title"], "url": row["url"],
                                  "attributionHistoryUrl": "https://es.wikipedia.org/w/index.php?curid=" + row["id"] + "&action=history",
                                  "snapshotDate": "2023-11-01", "articleTextSha256": raw_sha, "selectedProseSha256": public.sha(cleaned.encode()),
                                  "articleRevisionIdAvailable": False, "rowIndex": wrapped["row_idx"], "rowPageOffset": offset,
                                  "chunkIndex": index, "textSelection": "source paragraphs >=25 words, letter fraction >=0.55, >=3 Spanish function-word types; trim outer whitespace and join with two newlines"}, "text": chunk}
                        validate_text_record(record)
                        wiki.append(record)
                        paragraphs[record["id"]] = kept
                        payload += len(chunk.encode())
            print(json.dumps({"stage": "wikipedia", "pages": len(pages), "payloadMiB": payload / MIB, "elapsedSeconds": time.monotonic() - started}), flush=True)
            if payload >= args.wiki_target_mib * MIB:
                break
    if payload < args.wiki_target_mib * MIB:
        raise ValueError("bounded Wikipedia download did not reach declared payload target")
    all_records = human + wiki
    clones, clone_receipt = clone_families(all_records, paragraphs)
    retained, seen, duplicate_count = [], set(), 0
    for record in all_records:
        if record["groupId"] in clones:
            continue
        digest = public.sha(record["text"].encode())
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        retained.append(record)
    check()
    license_receipt = public.licenses(args.output)
    for name in ["oasst1-LICENSE", "oasst1-LICENSE.receipt.json", "oasst1-README.md", "oasst2-README.md"]:
        source = args.oasst_raw_directory / name
        target = args.output / "source-license-documents" / name
        shutil.copyfile(source, target)
        target.chmod(0o600)
    files = {}
    for split in ("train", "validation", "test"):
        rows = [r for r in retained if r["split"] == split]
        suffix = "test-reserved" if split == "test" else split
        path = args.output / f"causal-spanish.{suffix}.jsonl"
        files[split] = public.write_jsonl(path, rows)
        load_text_corpus(path)
    training = [r for r in retained if r["split"] != "test"]
    path = args.output / "causal-spanish.train-validation.jsonl"
    files["trainValidation"] = public.write_jsonl(path, training)
    load_text_corpus(path)
    if clone_families(retained, paragraphs)[0]:
        raise ValueError("crossfold source text clone remains")
    if {r["groupId"] for r in training} & {r["groupId"] for r in retained if r["split"] == "test"}:
        raise ValueError("TEST family crosses training folds")
    for dataset, _, expected in OASST_INPUTS:
        if public.file_sha(args.oasst_directory / f"{dataset}-importer-intermediate.jsonl") != expected:
            raise ValueError("OASST input changed during preparation")
    manifest = {"schemaVersion": 1, "purpose": "richer public Spanish causal language base; not acceptance evidence",
                "createdFrom": "pinned public sources; no pretrained weights, external generation or owner messages",
                "files": files, "splits": public.summary(retained, "text"),
                "payloadUtf8Bytes": sum(len(r["text"].encode()) for r in retained),
                "uniqueTextRecords": len(retained), "helperSha256": public.file_sha(Path(__file__)),
                "sharedPublicHelperSha256": public.file_sha(Path(public.__file__)),
                "oasst": human_receipt, "oasstSelection": human_selection,
                "wikipedia": {"dataset": "wikimedia/wikipedia", "revision": public.WIKI_REVISION, "configuration": public.WIKI_CONFIG,
                              "snapshotDate": "2023-11-01", "sampling": "historical page permutation with order 541 onward; unchanged document-ID family folds",
                              "pages": pages, "paragraphAndArticleFilters": dict(filters)},
                "guards": {**clone_receipt, "excludedCrossfoldFamilyIds": sorted(clones), "exactRecordDuplicatesRemoved": duplicate_count,
                           "historicalOasstFoldsMoved": 0, "wholeEligibleFamiliesSelected": True},
                "licenseDeclarations": license_receipt, "modelCalls": False, "tokenizerCalls": False,
                "testScoringPerformed": False, "finalAcceptanceCasesRead": False, "ownerMessagesRead": False,
                "resources": {"budgetSeconds": args.budget_seconds, "diskCapMiB": args.disk_cap_mib,
                              "elapsedSeconds": time.monotonic() - started, "diskBytes": directory_bytes(args.output)},
                "limitations": ["Source review and prose filters do not certify factual accuracy or conversational ability",
                                "Wikipedia is encyclopedic and includes dated claims, citations and extraction artifacts",
                                "Exact paragraph/message guards do not detect semantic paraphrases",
                                "Paragraph selection changes separators and drops short headings/lists, not source words",
                                "Wikipedia article-level revision and rights certification are unavailable",
                                "Existing own weights may previously have encountered some historical source families"]}
    public.write_json(args.output / "manifest.json", manifest)
    check()
    return manifest


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def resource_check(path, deadline, maximum):
    if time.monotonic() >= deadline:
        raise TimeoutError("declared preparation time budget exhausted")
    if directory_bytes(path) > maximum:
        raise ValueError("declared preparation disk budget exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasst-directory", required=True, type=Path)
    parser.add_argument("--oasst-raw-directory", required=True, type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oasst-target-mib", type=float, default=8)
    parser.add_argument("--wiki-target-mib", type=float, default=12)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--disk-cap-mib", type=int, default=2048)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.workers <= 4 and 4 <= args.max_pages <= 240 and 1 <= args.oasst_target_mib <= 20 and 1 <= args.wiki_target_mib <= 30 and 60 <= args.budget_seconds <= 900 and 64 <= args.disk_cap_mib <= 2048):
        parser.error("arguments exceed fixed resource limits")
    manifest = prepare(args)
    print(json.dumps({"stage": "complete", "files": manifest["files"], "splits": manifest["splits"], "payloadMiB": manifest["payloadUtf8Bytes"] / MIB, "resources": manifest["resources"]}), flush=True)


if __name__ == "__main__":
    main()
