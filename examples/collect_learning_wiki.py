#!/usr/bin/env python3
"""Collect a finite pinned Spanish Wiki batch and queue own causal -> SFT stages.

No model, tokenizer, evaluation suite or private conversation is loaded. Run
only while the supervisor is waiting_data. All paths are recipe-configured;
relative paths resolve against the repository, not a machine-specific home.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
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
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_public_spanish_language as public
import prepare_spanish_causal_base as causal

MIB = 1 << 20
ROOT = Path(__file__).resolve().parents[1]
PIN = {"dataset": "wikimedia/wikipedia", "revision": public.WIKI_REVISION,
       "configuration": public.WIKI_CONFIG, "snapshotDate": "2023-11-01"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def portable(value):
    value = Path(value).resolve()
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def checked(spec):
    result = path(spec["path"])
    if public.file_sha(result) != spec["sha256"]:
        raise ValueError("fixed collector input differs: " + result.name)
    return result


def paragraph_sha(text):
    return public.sha(causal.normalize(text).encode())


def metadata_only_document(raw):
    """Skip encoded JSON text strings before decoding cached source metadata.

    Only byte positions and escape delimiters are inspected in omitted values;
    their Unicode content is never decoded, retained or used for selection.
    """
    marker = re.compile(rb'(?<!\\)"text"\s*:\s*"')
    pieces, spans, cursor = [], [], 0
    for match in marker.finditer(raw):
        if match.start() < cursor:
            continue
        end = match.end()
        while end < len(raw):
            if raw[end] == 92:
                end += 2
            elif raw[end] == 34:
                break
            else:
                end += 1
        if end >= len(raw):
            raise ValueError("unterminated cached source text value")
        pieces.extend((raw[cursor:match.end()], b'"'))
        spans.append((match.end() - 1, end + 1))
        cursor = end + 1
    pieces.append(raw[cursor:])
    return json.loads(b"".join(pieces)), spans


def latest_sft(supervisor):
    if supervisor.get("operationalState") != "waiting_for_data" or supervisor.get("trainingActive") or supervisor.get("nativeWorkerAlive"):
        return None
    results = [(key, value) for key, value in supervisor.get("stageResults", {}).items() if value.get("kind") == "sft"]
    if not results:
        return None
    # Dictionary insertion order is completion order in the supervisor state.
    return results[-1][0]


def seed(args):
    """Seed only IDs/metadata from all source rows; never decode TEST text."""
    initial = read(args.initial_manifest)
    if initial["wikipedia"]["revision"] != PIN["revision"] or initial["wikipedia"]["configuration"] != PIN["configuration"]:
        raise ValueError("initial seed source is outside the fixed snapshot")
    seen_ids, article_shas, prose_shas, records, paragraphs = set(), set(), set(), {}, defaultdict(set)
    page_receipts = []
    for receipt in initial["wikipedia"]["pages"]:
        if receipt["response"]["xRevision"] != PIN["revision"]:
            raise ValueError("initial source page revision differs")
        file = args.initial_directory / "wiki-row-pages" / receipt["file"]
        if public.file_sha(file) != receipt["sha256"]:
            raise ValueError("initial pinned page cache changed")
        rows = metadata_only_document(file.read_bytes())[0]["rows"]
        seen_ids.update(str(wrapped["row"]["id"]) for wrapped in rows)
        page_receipts.append({"order": receipt["order"], "offset": receipt["offset"], "sha256": receipt["sha256"]})
    for split in ("train", "validation", "test"):
        spec = initial["files"][split]
        file = args.initial_directory / spec["file"]
        if public.file_sha(file) != spec["sha256"]:
            raise ValueError("initial TEXT source identity differs")
        with file.open("rb") as stream:
            for raw in stream:
                if split == "test":
                    # Parse only the source object before the final text field;
                    # reserved text is neither decoded, scored nor tokenized.
                    prefix = raw.split(b',"text":', 1)[0].decode("utf-8")
                    source, _ = json.JSONDecoder().raw_decode(prefix.split('"source":', 1)[1])
                else:
                    record = json.loads(raw)
                    source = record["source"]
                    for paragraph in record["text"].split("\n\n"):
                        paragraphs[paragraph_sha(paragraph)].add(split)
                    records[public.sha(record["text"].encode())] = split
                if source["dataset"] == PIN["dataset"]:
                    seen_ids.add(str(source["pageId"]))
                    article_shas.add(source["articleTextSha256"])
                    prose_shas.add(source["selectedProseSha256"])
    value = {"schemaVersion": 1, "source": PIN, "nextPageOrder": max(r["order"] for r in page_receipts) + 1,
             "seenArticleIds": sorted(seen_ids), "seenArticleTextSha256": sorted(article_shas),
             "seenSelectedProseSha256": sorted(prose_shas), "seenRecordTextSha256": records,
             "seenNormalizedParagraphSha256": {key: sorted(folds) for key, folds in paragraphs.items()},
             "initialManifestSha256": public.file_sha(args.initial_manifest), "initialSourcePages": page_receipts,
             "allInitialFoldsSeeded": True, "testSeedFields": "source identifiers and metadata digests only; no text decoded",
             "lastPublishedCycle": 0, "published": []}
    atomic_json(args.output, value)
    print(json.dumps({"status": "seeded", "seenArticleIds": len(seen_ids), "nextPageOrder": value["nextPageOrder"], "sha256": public.file_sha(args.output)}), flush=True)


def anchors(args):
    """Freeze small OASST whole-family rehearsal from initial TRAIN/VAL only."""
    source = args.initial_train_validation
    if public.file_sha(source) != args.expected_sha256:
        raise ValueError("initial train/validation corpus hash differs")
    families = defaultdict(list)
    for raw in source.open("rb"):
        record = json.loads(raw)
        if record["split"] not in ("train", "validation"):
            raise ValueError("reserved record in rehearsal input")
        if record["source"]["dataset"] in ("OpenAssistant/oasst1", "OpenAssistant/oasst2"):
            families[(record["split"], record["groupId"])].append(record)
    rows, budget = [], int(args.cap_mib * MIB)
    for split, ratio in (("train", .9), ("validation", .1)):
        used = 0
        keys = sorted((key for key in families if key[0] == split), key=lambda key: public.sha(("unidad-fixed-rehearsal-v1\0" + key[1]).encode()))
        for key in keys:
            current = families[key]
            cost = sum(len(encoded(record)) for record in current)
            if used + cost > budget * ratio:
                continue
            rows.extend(current)
            used += cost
    if not all(any(r["split"] == split for r in rows) for split in ("train", "validation")):
        raise ValueError("rehearsal lacks TRAIN or VALIDATION")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = public.write_jsonl(args.output, rows)
    print(json.dumps({"status": "anchors_frozen", **result, "splits": dict(Counter(r["split"] for r in rows)), "retentionOnly": True}), flush=True)


def fetch_page(order, maximum_raw, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("collector time budget exhausted before request")
    page_count = math.ceil(public.WIKI_ROWS / 100)
    if math.gcd(7919, page_count) != 1:
        raise ValueError("pinned page stride is not coprime")
    offset = ((order * 7919 + 37) % page_count) * 100
    query = urlencode({"dataset": PIN["dataset"], "config": PIN["configuration"], "split": "train", "offset": offset, "length": min(100, public.WIKI_ROWS - offset), "revision": PIN["revision"]})
    url = "https://datasets-server.huggingface.co/rows?" + query
    raw, headers = public.download(url, revision=PIN["revision"])
    if len(raw) > maximum_raw or time.monotonic() >= deadline:
        raise ValueError("bounded source response or time budget exceeded")
    value, spans = metadata_only_document(raw)
    rows = value.get("rows")
    if value.get("partial") or value.get("num_rows_total") != public.WIKI_ROWS or not isinstance(rows, list) or len(rows) > 100 or len(spans) != len(rows):
        raise ValueError("partial, changed or unrecognized pinned row schema")
    for wrapped, (start, end) in zip(rows, spans):
        row = wrapped["row"]
        if public.assign_split("wikipedia:es:" + str(row["id"])) != "test":
            row["text"] = json.loads(raw[start:end])
    return offset, raw, headers, value, url


def collect(recipe, inbox, supervisor_path):
    supervisor = read(supervisor_path)
    parent = latest_sft(supervisor)
    if parent is None:
        return {"status": "waiting_for_completed_sft", "downloadedPages": 0}
    if recipe["source"] != PIN:
        raise ValueError("collector source is outside its fixed authorized snapshot")
    limits = recipe["limits"]
    if not (1 <= limits["maxPages"] <= 80 and 60 <= limits["budgetSeconds"] <= 900 and
            64 <= limits["maxDiskMiB"] <= 2048 and 1 <= limits["corpusCapMiB"] <= 24 and
            1 <= limits["targetNovelPayloadMiB"] <= 20 and 1 <= limits["maxNetworkMiB"] <= 512):
        raise ValueError("collector limits exceed bounded authorized recipe")
    seed_path, rehearsal_path, sft_path = checked(recipe["seed"]), checked(recipe["anchors"]), checked(recipe["sftCorpus"])
    sft_manifest = checked(recipe["sftManifest"])
    configs = {kind: checked(recipe[kind + "TrainingConfig"]) for kind in ("base", "sft")}
    for kind, config_path in configs.items():
        config = read(config_path)
        if config["objective"] != ("causal_lm" if kind == "base" else "assistant_last_reply") or config["partition_policy"] != "train_validation_only":
            raise ValueError("stage objective/fold policy differs")
    state_path, output_root = path(recipe["statePath"]), path(recipe["outputRoot"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)
    lock = state_path.with_suffix(".lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {"status": "another_collector_active", "downloadedPages": 0}
    try:
        state = read(state_path) if state_path.exists() else read(seed_path)
        if state["source"] != PIN:
            raise ValueError("persistent source identity changed")
        # Recover a staged transaction after a process stop: finalize its exact
        # existing sidecar instead of downloading/publishing another corpus.
        pending = state.pop("pendingApproval", None)
        if pending:
            bundle_path = inbox / pending["bundleFile"]
            if public.file_sha(bundle_path) != pending["manifestSha256"]:
                raise ValueError("pending collector bundle changed")
            atomic_json(inbox / pending["approvalFile"], pending["approval"])
            atomic_json(state_path, state)
            return {"status": "publication_recovered", "cycleId": pending["cycleId"], "downloadedPages": 0}
        published = state.get("published", [])
        failed_last = bool(published and supervisor.get("cycle", {}).get("cycleId") == published[-1]["cycleId"] and
                           supervisor.get("cycleFailed") and supervisor.get("stageIndex", 0) >= len(supervisor.get("cycle", {}).get("stages", [])))
        if published and not failed_last and not any(key.startswith(published[-1]["cycleId"] + ":") and value.get("kind") == "sft" for key, value in supervisor.get("stageResults", {}).items()):
            return {"status": "previous_collector_cycle_pending", "downloadedPages": 0}
        cycle_number = state.get("lastPublishedCycle", 0) + 1
        cycle_id = "wiki-cycle-" + str(cycle_number).zfill(4)
        final = output_root / cycle_id
        completed_staging = []
        for previous in output_root.glob("." + cycle_id + ".*"):
            if not previous.is_dir():
                continue
            if (previous / "publication.json").exists():
                validate_publication(previous, cycle_id)
                completed_staging.append(previous)
            else:
                # Only this collector's current-cycle unpublished scratch is
                # removed. Published corpora and training runs are preserved.
                shutil.rmtree(previous)
        if len(completed_staging) > 1:
            raise ValueError("multiple completed collector transactions require review")
        if completed_staging:
            if final.exists():
                raise ValueError("both staged and final publication exist")
            os.rename(completed_staging[0], final)
            sync_directory(output_root)
        if final.exists() and (final / "publication.json").exists():
            validate_publication(final, cycle_id)
            finish_publication(final, inbox, state_path)
            return {"status": "publication_recovered", "cycleId": cycle_id, "downloadedPages": 0}
        if final.exists() or (inbox / (cycle_id + ".json")).exists():
            raise ValueError("unrecorded collector publication exists; preserve it for review")
        if shutil.disk_usage(output_root).free < limits["maxDiskMiB"] * MIB:
            raise ValueError("collector free disk guard failed")
        staging = Path(tempfile.mkdtemp(prefix="." + cycle_id + ".", dir=output_root))
        try:
            return prepare_cycle(recipe, limits, state, state_path, staging, final, inbox, parent,
                                 cycle_id, cycle_number, rehearsal_path, sft_path, sft_manifest, configs)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        lock.close()


def prepare_cycle(recipe, limits, state, state_path, staging, final, inbox, parent,
                  cycle_id, cycle_number, rehearsal_path, sft_path, sft_manifest, configs):
    started = time.monotonic()
    deadline = started + limits["budgetSeconds"]
    cap = limits["corpusCapMiB"] * MIB
    # Check the total local collector store, not just the current temporary run.
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("collector time budget exhausted")
        size = causal.directory_bytes(staging.parent) + causal.directory_bytes(state_path.parent) + causal.directory_bytes(inbox)
        if size > limits["maxDiskMiB"] * MIB:
            raise ValueError("collector output, state and inbox disk cap exceeded")
    rows = [json.loads(raw) for raw in rehearsal_path.open("rb")]
    if any(r["split"] not in ("train", "validation") for r in rows):
        raise ValueError("TEST is forbidden in rehearsal")
    serialized_bytes = sum(len(encoded(r)) for r in rows)
    seen_ids = set(state["seenArticleIds"])
    article_digests = set(state["seenArticleTextSha256"])
    prose_digests = set(state["seenSelectedProseSha256"])
    past_texts = state["seenRecordTextSha256"]
    past_paragraphs = state["seenNormalizedParagraphSha256"]
    filters, pages, source_ids = Counter(), [], set()
    candidate_paragraphs, novel_payload, network_bytes = {}, 0, 0
    order = state["nextPageOrder"]
    page_count = math.ceil(public.WIKI_ROWS / 100)
    for _ in range(limits["maxPages"]):
        check()
        if order >= page_count:
            break
        if network_bytes + 25_000_001 > limits["maxNetworkMiB"] * MIB:
            break
        offset, raw, headers, value, url = fetch_page(order, 25_000_000, deadline)
        network_bytes += len(raw)
        if network_bytes > limits["maxNetworkMiB"] * MIB:
            raise ValueError("collector network byte budget exceeded")
        # Receipts retain raw source hashes, not duplicated large row caches.
        pages.append({"order": order, "offset": offset, "rows": len(value["rows"]), "url": url,
                      "rawSha256": public.sha(raw), "bytes": len(raw), "response": headers})
        order += 1
        for wrapped in value["rows"]:
            row = wrapped.get("row", {})
            pid = str(row.get("id", ""))
            if not pid:
                raise ValueError("missing pinned article ID")
            if pid in seen_ids:
                filters["already_seen_article_id"] += 1
                continue
            seen_ids.add(pid)
            group = "wikipedia:es:" + pid
            split = public.assign_split(group)
            if split == "test":
                filters["reserved_source_id_text_not_read"] += 1
                continue
            reason = public.wiki_quality(row, wrapped.get("truncated_cells", []))
            if reason:
                filters[reason] += 1
                continue
            raw_digest = public.sha(row["text"].encode())
            if raw_digest in article_digests:
                filters["already_seen_source_article_digest"] += 1
                continue
            article_digests.add(raw_digest)
            kept, reasons = causal.prose_paragraphs(row["text"])
            filters.update(reasons)
            clean = "\n\n".join(kept)
            clean_digest = public.sha(clean.encode())
            if len(public.WORD.findall(clean)) < 150 or clean_digest in prose_digests:
                filters["short_or_repeated_selected_article"] += 1
                continue
            prose_digests.add(clean_digest)
            hashes = {paragraph_sha(paragraph) for paragraph in kept}
            chunks = public.split_utf8_exact(clean)
            if any(digest in past_paragraphs for digest in hashes) or any(public.sha(chunk.encode()) in past_texts for chunk in chunks):
                filters["known_source_text_or_paragraph_whole_family"] += 1
                continue
            if len(chunks) > 64:
                filters["article_above_64_chunks"] += 1
                continue
            family = []
            for index, chunk in enumerate(chunks):
                record = {"id": f"wikipedia:{public.WIKI_CONFIG}:{pid}:prose:{index}", "groupId": group,
                          "split": split, "source": {**PIN, "kind": "public-encyclopedic", "language": "es",
                          "license": "CC-BY-SA-3.0", "licenseDeclaration": "pinned HF card declares CC-BY-SA-3.0/GFDL; Wikimedia legal declarations retained separately",
                          "pageId": pid, "title": row["title"], "url": row["url"],
                          "attributionHistoryUrl": "https://es.wikipedia.org/w/index.php?curid=" + pid + "&action=history",
                          "articleTextSha256": raw_digest, "selectedProseSha256": clean_digest,
                          "articleRevisionIdAvailable": False, "rowIndex": wrapped["row_idx"],
                          "rowPageOffset": offset, "chunkIndex": index,
                          "textSelection": "fixed Spanish prose paragraph filters; outer trim and two-newline join"}, "text": chunk}
                public.validate_text_record(record)
                family.append(record)
            cost = sum(len(encoded(record)) for record in family)
            if serialized_bytes + cost > cap:
                filters["whole_article_exceeds_remaining_corpus_cap"] += 1
                continue
            rows.extend(family)
            candidate_paragraphs[group] = hashes
            source_ids.add(pid)
            serialized_bytes += cost
            novel_payload += len(clean.encode())
        print(json.dumps({"stage": "collect", "cycleId": cycle_id, "pages": len(pages), "nextPageOrder": order,
                          "novelPayloadMiB": novel_payload / MIB, "networkMiB": network_bytes / MIB}), flush=True)
        if novel_payload >= limits["targetNovelPayloadMiB"] * MIB or serialized_bytes >= cap * .99:
            break
    # Current candidates: reject whole families sharing normalized paragraphs
    # between TRAIN and VALIDATION, including the fixed rehearsal corpus.
    by_paragraph = defaultdict(list)
    for record in rows:
        hashes = candidate_paragraphs.get(record["groupId"], {paragraph_sha(p) for p in record["text"].split("\n\n")})
        for digest in hashes:
            by_paragraph[digest].append(record)
    excluded = {r["groupId"] for values in by_paragraph.values() if len({r["split"] for r in values}) > 1 for r in values}
    by_exact = defaultdict(list)
    families = defaultdict(list)
    for record in rows:
        by_exact[public.sha(record["text"].encode())].append(record)
        families[record["groupId"]].append(record)
    excluded.update(r["groupId"] for values in by_exact.values() if len({r["split"] for r in values}) > 1 for r in values)
    if any(r["groupId"] in excluded and r["source"]["dataset"].startswith("OpenAssistant/") for r in rows):
        raise ValueError("new source collides crossfold with fixed rehearsal; no anchor folds may move")
    retained, exact, duplicate_groups = [], set(), set()
    for group, current in families.items():
        if group in excluded:
            continue
        hashes = [public.sha(record["text"].encode()) for record in current]
        if len(set(hashes)) != len(hashes) or any(digest in exact for digest in hashes):
            if current[0]["source"]["dataset"].startswith("OpenAssistant/"):
                raise ValueError("fixed rehearsal has exact duplicate text")
            duplicate_groups.add(group)
            continue
        exact.update(hashes)
        retained.extend(current)
    novel = [r for r in retained if r["source"]["dataset"] == PIN["dataset"]]
    counts = Counter(r["split"] for r in retained)
    if not novel or not all(any(r["split"] == split for r in novel) for split in ("train", "validation")):
        raise ValueError("bounded batch lacks new TRAIN and VALIDATION articles")
    corpus = staging / "causal-spanish.train-validation.jsonl"
    result = public.write_jsonl(corpus, retained)
    if result["bytes"] > cap:
        raise ValueError("serialized corpus exceeded cap")
    content = hashlib.sha256()
    for record in retained:
        content.update(encoded(record["text"]))
    license_path = checked(recipe["licenseReceipt"])
    license_info = read(license_path)
    manifest = {"schemaVersion": 1, "purpose": "finite new-to-stage pinned Wiki batch with identified OASST rehearsal",
                "corpusSha256": result["sha256"], "contentSha256": content.hexdigest(), "file": result,
                "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
                "testUsedForTraining": False, "testTextRead": False, "tokenizerRefit": False,
                "addedRecords": len(novel), "addedSourceIds": sorted({r["source"]["pageId"] for r in novel}),
                "newUniqueArticles": len({r["groupId"] for r in novel}), "rehearsalRecords": len(retained) - len(novel),
                "rehearsalCorpusSha256": recipe["anchors"]["sha256"], "rehearsalCountsAsNewDiversity": False,
                "sources": [{**PIN, "license": "CC-BY-SA-3.0/GFDL declarations; article-level rights not certified",
                             "receipt": "source-pages.json", "pages": pages, "licenseReceipt": recipe["licenseReceipt"]},
                            {"dataset": "OpenAssistant/oasst1+oasst2", "license": "Apache-2.0", "receipt": recipe["anchors"]["path"], "retentionOnly": True}],
                "sourceLicenseDeclarations": license_info, "sourceFilters": dict(filters),
                "guards": {"excludedWholeFamilyIds": sorted(excluded), "exactDuplicateWholeFamilyIds": sorted(duplicate_groups),
                           "foldsMoved": 0, "allInitialFoldArticleIdsSeeded": state["allInitialFoldsSeeded"]},
                "pageOrderStart": state["nextPageOrder"], "nextPageOrder": order,
                "recipeSha256": recipe["recipeSha256"], "sourceHelperSha256": public.file_sha(Path(__file__)),
                "limits": limits, "downloadedBytes": network_bytes, "elapsedSeconds": time.monotonic() - started,
                "limitations": ["Snapshot is 2023-11-01, not current news or factual certification",
                                "Exact digests do not detect semantic paraphrases; previous reserved text is never consulted",
                                "Article IDs and full-article/prose metadata digests cover all initial folds; exact overlap with individual historical TEST paragraphs/chunks is not exhaustively checked because those texts are never decoded",
                                "No model improvement or conversational readiness is inferred from these data"]}
    public.write_json(staging / "manifest.json", manifest)
    public.write_json(staging / "source-pages.json", pages)
    retention = read(sft_manifest)
    retention.update(retentionOnlyStage=True, novelDataParentStage=cycle_id + ":base",
                     addedRecords=0, addedSourceIds=[], reuseReason="fixed conversational retention after new causal source batch")
    public.write_json(staging / "sft-retention-provenance.json", retention)
    check()
    # Queue references final paths, but all bytes and a recoverable publication
    # journal are durable in staging before the atomic directory publication.
    stages = []
    for kind, corpus_path, fingerprint, provenance_path, parent_stage in (
            ("base", final / corpus.name, result["sha256"], final / "manifest.json", parent),
            ("sft", sft_path, recipe["sftCorpus"]["sha256"], final / "sft-retention-provenance.json", cycle_id + ":base")):
        config = read(configs[kind])
        stages.append({"id": kind, "kind": kind, "runDir": portable(path(recipe["runRoot"]) / cycle_id / kind),
                       "corpusPath": portable(corpus_path), "corpusSha256": fingerprint,
                       "trainingConfigPath": portable(configs[kind]), "trainingConfigSha256": public.file_sha(configs[kind]),
                       "parentStage": parent_stage, "stopAfterStep": config["max_steps"],
                       "invocationBudgetSeconds": 3600, "maxRSSMiB": 6144, "maxDiskMiB": 2048,
                       "provenanceManifestPath": portable(provenance_path), "provenanceManifestSha256": public.file_sha(staging / provenance_path.name)})
    bundle = {"schemaVersion": 1, "cycleId": cycle_id, "stages": stages,
              "purpose": "own model progressive causal new-source batch then fixed SFT retention"}
    bundle_file = cycle_id + ".json"
    approval_file = cycle_id + ".approval.json"
    bundle_sha = public.sha(encoded(bundle))
    approval = {"approved": True, "manifestSha256": bundle_sha,
                "approvedBy": "project-agent-with-owner-standing-authorization", "standingOwnerAuthorization": True,
                "independentHumanReview": False, "approvedAtUtc": datetime.now(timezone.utc).isoformat()}
    for record in retained:
        state["seenRecordTextSha256"][public.sha(record["text"].encode())] = record["split"]
        for paragraph in record["text"].split("\n\n"):
            digest = paragraph_sha(paragraph)
            state["seenNormalizedParagraphSha256"].setdefault(digest, [])
            if record["split"] not in state["seenNormalizedParagraphSha256"][digest]:
                state["seenNormalizedParagraphSha256"][digest].append(record["split"])
    state.update(seenArticleIds=sorted(seen_ids), seenArticleTextSha256=sorted(article_digests),
                 seenSelectedProseSha256=sorted(prose_digests), nextPageOrder=order, lastPublishedCycle=cycle_number)
    state["published"].append({"cycleId": cycle_id, "corpusSha256": result["sha256"], "contentSha256": content.hexdigest(),
                              "addedRecords": len(novel), "newUniqueArticles": manifest["newUniqueArticles"], "manifestSha256": bundle_sha})
    state["pendingApproval"] = {"cycleId": cycle_id, "bundleFile": bundle_file, "approvalFile": approval_file,
                                "manifestSha256": bundle_sha, "approval": approval}
    atomic_json(staging / "publication.json", {"bundle": bundle, "state": state})
    check()
    for file in staging.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(staging)
    os.rename(staging, final)
    sync_directory(final.parent)
    finish_publication(final, inbox, state_path)
    return {"status": "queued", "cycleId": cycle_id, "parentStage": parent, "bundleSha256": bundle_sha,
            "corpusSha256": result["sha256"], "splitCounts": manifest["splitCounts"], "addedRecords": len(novel),
            "newUniqueArticles": manifest["newUniqueArticles"], "nextPageOrder": order, "downloadedPages": len(pages)}


def finish_publication(final, inbox, state_path):
    journal = read(final / "publication.json")
    state = journal["state"]
    pending = state["pendingApproval"]
    atomic_json(inbox / pending["bundleFile"], journal["bundle"])
    if public.file_sha(inbox / pending["bundleFile"]) != pending["manifestSha256"]:
        raise ValueError("publication journal bundle identity differs")
    atomic_json(state_path, state)
    # Last admission-visible operation: a complete approved bundle is now safe.
    atomic_json(inbox / pending["approvalFile"], pending["approval"])
    state.pop("pendingApproval")
    atomic_json(state_path, state)


def validate_publication(directory, cycle_id):
    journal = read(directory / "publication.json")
    bundle, state = journal["bundle"], journal["state"]
    if bundle["cycleId"] != cycle_id or state["source"] != PIN or state["pendingApproval"]["cycleId"] != cycle_id:
        raise ValueError("recovered collector transaction identity differs")
    if public.sha(encoded(bundle)) != state["pendingApproval"]["manifestSha256"]:
        raise ValueError("recovered collector bundle hash differs")
    for stage in bundle["stages"]:
        if public.file_sha(directory / Path(stage["provenanceManifestPath"]).name) != stage["provenanceManifestSha256"]:
            raise ValueError("recovered collector provenance differs")
        if stage["kind"] == "base" and public.file_sha(directory / Path(stage["corpusPath"]).name) != stage["corpusSha256"]:
            raise ValueError("recovered collector corpus differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("seed")
    init.add_argument("--initial-directory", type=Path, required=True)
    init.add_argument("--initial-manifest", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("anchors")
    replay.add_argument("--initial-train-validation", type=Path, required=True)
    replay.add_argument("--expected-sha256", required=True)
    replay.add_argument("--cap-mib", type=float, default=3)
    replay.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("collect")
    run.add_argument("--recipe", type=Path, required=True)
    run.add_argument("--inbox-dir", type=Path, required=True)
    run.add_argument("--supervisor-state", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seed":
        seed(args)
    elif args.command == "anchors":
        if not 1 <= args.cap_mib <= 8:
            parser.error("anchor cap outside 1–8 MiB")
        anchors(args)
    else:
        recipe = read(args.recipe)
        recipe["recipeSha256"] = public.file_sha(args.recipe)
        print(json.dumps(collect(recipe, args.inbox_dir, args.supervisor_state), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
