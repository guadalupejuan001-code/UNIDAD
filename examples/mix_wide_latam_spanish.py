#!/usr/bin/env python3
"""Mix pinned public Spanish TEXT sources in finite whole-family shards.

Derived from the byte-pinned original and next-Tatoeba mixers. Adds pinned
lexical alias/fold registries without treating the untrained wide-v1 mixture as
already learned. Historical wide guards are used only if explicitly present. No model/tokenizer/evaluation
calls or TEST files. The original helper remains unchanged. The recipe
identifies genuinely new sources separately from fixed OASST rehearsal. Output
does not enqueue, reload or signal a training process.
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
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.language_data import validate_text_record

ROOT = Path(__file__).resolve().parents[1]
PARENT_HELPER_SHA256 = "3325d02b789e3fa7ef2a7e18a2b2799e608e9bb54f44b8b400df3dcf97791efc"
MIB = 1 << 20
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def normalized(text):
    return " ".join(text.split())


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def paragraphs(text):
    # Tatoeba packs separate original sentences with single newlines. Treat
    # those as paragraph units; short units remain covered by whole-record keys.
    return list({p for p in text.split("\n\n") + text.splitlines() if len(WORD.findall(p)) >= 12})


def full_keys(text):
    return {("raw", sha(text.encode())), ("space", sha(normalized(text).encode()))}


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.read_bytes = 0
        self.written_bytes = 0
        self.started = time.monotonic()
        self.cpu_started = time.process_time()

    def check(self, incoming=0, outgoing=0):
        if self.read_bytes + self.written_bytes + incoming + outgoing > self.limit:
            raise ValueError("configured total file read/write budget exceeded")
        if time.monotonic() - self.started >= 590:
            raise TimeoutError("600-second mixer time limit approached")

    def read(self, spec):
        source = path(spec["path"])
        self.check(incoming=source.stat().st_size)
        raw = source.read_bytes()
        self.read_bytes += len(raw)
        if sha(raw) != spec["sha256"]:
            raise ValueError("immutable mixer input differs: " + source.name)
        return raw

    def write(self, target, raw):
        self.check(outgoing=len(raw))
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.written_bytes += len(raw)


def mix(recipe, output, budget):
    cap = recipe["limits"]["corpusCapMiB"] * MIB
    max_shards = recipe["limits"]["maxShards"]
    prefix = recipe.get("outputPrefix", "wide-spanish")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", prefix):
        raise ValueError("invalid output corpus prefix")
    if cap > 24 * MIB or not 1 <= max_shards <= 3:
        raise ValueError("recipe exceeds fixed shard bounds")
    if output.exists():
        raise ValueError("output already exists; no frozen corpus is overwritten")
    specifications = [recipe["initialSeed"], recipe["humanV3"], recipe["rehearsal"]["corpus"], recipe["rehearsal"]["provenance"]]
    for source in recipe["sources"]:
        specifications.extend((source["corpus"], source["provenance"]))
    specifications.extend(recipe.get("familyRegistries", []))
    unique = {spec["path"]: spec for spec in specifications}
    if len(unique) != len(specifications):
        for spec in specifications:
            if unique[spec["path"]]["sha256"] != spec["sha256"]:
                raise ValueError("same source path has conflicting pins")
    input_bytes = sum(path(spec["path"]).stat().st_size for spec in unique.values())
    novel_bytes = sum(path(source["corpus"]["path"]).stat().st_size for source in recipe["sources"])
    anchor_bytes = path(recipe["rehearsal"]["corpus"]["path"]).stat().st_size
    # Conservative reservation includes every source byte, all configured
    # repeated anchors and 1MiB for receipts before reading any corpus.
    upper_output = novel_bytes + max_shards * anchor_bytes + MIB
    budget.check(incoming=input_bytes, outgoing=upper_output)
    cache = {}
    def checked(spec):
        key = spec["path"]
        if key not in cache:
            cache[key] = budget.read(spec)
        return cache[key]
    past_full, past_paragraphs = defaultdict(set), defaultdict(set)
    seed = json.loads(checked(recipe["initialSeed"]))
    for digest, split in seed["seenRecordTextSha256"].items():
        past_full[("raw", digest)].add(split)
    for digest, splits in seed["seenNormalizedParagraphSha256"].items():
        past_paragraphs[digest].update(splits)
    for digest, splits in seed.get("seenWideRecordTextSha256MultiFold", {}).items():
        past_full[("raw", digest)].update(splits)
    for digest, splits in seed.get("seenWideNormalizedFullRecordTextSha256MultiFold", {}).items():
        past_full[("space", digest)].update(splits)
    for digest, splits in seed.get("seenWideNormalizedParagraphSha256MultiFold", {}).items():
        past_paragraphs[digest].update(splits)
    historical_family_folds = dict(seed.get("seenWideFamilyFolds", {}))
    extension_receipt = seed.get("wideHistoryExtension")
    alias_families = {}
    registry_receipts = []
    for spec in recipe.get("familyRegistries", []):
        registry = json.loads(checked(spec))
        registry_receipts.append({**spec, "familyCount": len(registry["families"])})
        for family in registry["families"]:
            group, split = family["groupId"], family["split"]
            if split not in ("train", "validation", "test"):
                raise ValueError("invalid registered lexical fold")
            if group in historical_family_folds and historical_family_folds[group] != split:
                raise ValueError("registered lexical family has conflicting historical folds")
            historical_family_folds[group] = split
            for alias in family["aliases"]:
                previous = alias_families.setdefault(alias, (group, split))
                if previous != (group, split):
                    raise ValueError("lexical aliases require an explicit whole-family reconciliation before mixing")
        del registry
    del alias_families
    if any(split not in ("train", "validation") for splits in list(past_full.values()) + list(past_paragraphs.values()) for split in splits):
        raise ValueError("historical guard unexpectedly includes TEST text folds")
    for raw in checked(recipe["humanV3"]).splitlines():
        row = json.loads(raw)
        if row["split"] not in ("train", "validation"):
            raise ValueError("TEST is forbidden in historical guard input")
        for message in row["messages"]:
            if message["role"] == "system":
                continue
            for key in full_keys(message["content"]):
                past_full[key].add(row["split"])
            for paragraph in paragraphs(message["content"]):
                past_paragraphs[sha(normalized(paragraph).encode())].add(row["split"])
    del seed
    families, ids, source_receipts = {}, {}, []
    source_by_key = {source["key"]: source for source in recipe["sources"]}
    if len(source_by_key) != len(recipe["sources"]):
        raise ValueError("duplicate source recipe key")
    rehearsal = recipe["rehearsal"]
    all_sources = [{**rehearsal, "retentionOnly": True}] + [{**s, "retentionOnly": False} for s in recipe["sources"]]
    duplicate_ids = 0
    for source in all_sources:
        source_manifest = json.loads(checked(source["provenance"]))
        source_receipts.append({"key": source["key"], "genre": source["genre"], "dataset": source["dataset"],
                                "license": source["license"], "receipt": source["provenance"]["path"],
                                "receiptSha256": source["provenance"]["sha256"], "corpus": source["corpus"],
                                "retentionOnly": source["retentionOnly"], "sourceUrls": source.get("sourceUrls", [])})
        del source_manifest
        for raw in checked(source["corpus"]).splitlines(keepends=True):
            budget.check()
            if not raw.strip():
                continue
            record = json.loads(raw)
            validate_text_record(record)
            if record["split"] not in ("train", "validation"):
                raise ValueError("TEST is forbidden in mixed language corpus")
            identifier, group = record["id"], record["groupId"]
            if group in historical_family_folds and historical_family_folds[group] != record["split"]:
                raise ValueError("new corpus changes a previously assigned historical family fold")
            if identifier in ids:
                if ids[identifier] != record:
                    raise ValueError("same record ID changes text/fold/provenance")
                duplicate_ids += 1
                continue
            ids[identifier] = record
            family = families.setdefault(group, {"groupId": group, "split": record["split"], "retentionOnly": source["retentionOnly"], "entries": []})
            if family["split"] != record["split"] or family["retentionOnly"] != source["retentionOnly"]:
                raise ValueError("family changes fold or mixes new data with rehearsal")
            if not raw.endswith(b"\n"):
                raw += b"\n"
            family["entries"].append({"raw": raw, "record": record, "sourceKey": source["key"]})
    anchors = [family for family in families.values() if family["retentionOnly"]]
    novel = [family for family in families.values() if not family["retentionOnly"]]
    anchor_size = sum(len(entry["raw"]) for family in anchors for entry in family["entries"])
    if anchor_size >= cap:
        raise ValueError("fixed rehearsal leaves no space for new families")
    excluded, reasons, samefold_partial_history = set(), defaultdict(set), set()
    full_owners, paragraph_owners = defaultdict(set), defaultdict(set)
    for family in families.values():
        group, split = family["groupId"], family["split"]
        for entry in family["entries"]:
            text = entry["record"]["text"]
            for key in full_keys(text):
                full_owners[key].add(group)
                if not family["retentionOnly"] and key in past_full:
                    excluded.add(group)
                    reasons[group].add("known_historical_full_text_same_or_different_fold")
            for paragraph in paragraphs(text):
                digest = sha(normalized(paragraph).encode())
                paragraph_owners[digest].add(group)
                if not family["retentionOnly"] and digest in past_paragraphs:
                    if any(previous != split for previous in past_paragraphs[digest]):
                        excluded.add(group)
                        reasons[group].add("historical_crossfold_paragraph_ge12_words")
                    else:
                        samefold_partial_history.add(group)
    for label, mapping in (("full_text", full_owners), ("paragraph_ge12_words", paragraph_owners)):
        for groups in mapping.values():
            if len({families[group]["split"] for group in groups}) > 1:
                for group in groups:
                    if not families[group]["retentionOnly"]:
                        excluded.add(group)
                        reasons[group].add("new_or_rehearsal_crossfold_" + label)
    retained, full_seen, duplicate_texts = [], set(), 0
    # Anchors have priority and retain their original bytes/folds. Novel family
    # order is deterministic, independent of scores, lengths or evaluation.
    ordered = anchors + sorted(novel, key=lambda f: sha(("wide-spanish-v1\0" + f["groupId"]).encode()))
    for family in ordered:
        group = family["groupId"]
        if group in excluded:
            continue
        duplicate = False
        local = set()
        unique_entries = []
        for entry in family["entries"]:
            key = sha(normalized(entry["record"]["text"]).encode())
            if key in local and not family["retentionOnly"]:
                duplicate_texts += 1
                duplicate = True
                reasons[group].add("within_family_repeated_normalized_full_text")
            local.add(key)
            if key in full_seen and not family["retentionOnly"]:
                duplicate = True
            unique_entries.append(entry)
        if duplicate:
            excluded.add(group)
            reasons[group].add("samefold_repeated_full_text_whole_family")
            continue
        family["entries"] = unique_entries
        full_seen.update(local)
        family["bytes"] = sum(len(entry["raw"]) for entry in unique_entries)
        retained.append(family)
    anchors = [f for f in retained if f["retentionOnly"]]
    new = [f for f in retained if not f["retentionOnly"]]
    available = cap - sum(f["bytes"] for f in anchors)
    deferred = [f for f in new if f["bytes"] > available]
    new = [f for f in new if f["bytes"] <= available]
    if not new:
        raise ValueError("no bounded genuinely new family remains")
    # First-fit decreasing never slices a family. Complete-family moves or
    # swaps then ensure each shard has new TRAIN and VALIDATION examples.
    bins = []
    for family in sorted(new, key=lambda f: (-f["bytes"], f["groupId"])):
        for shard in bins:
            if shard["bytes"] + family["bytes"] <= available:
                shard["families"].append(family)
                shard["bytes"] += family["bytes"]
                break
        else:
            bins.append({"families": [family], "bytes": family["bytes"]})
    if len(bins) > max_shards:
        raise ValueError("whole-family packing exceeds recipe maxShards; keep original sources for a larger future batch")
    # If packing creates a shard without a new fold, move one complete family
    # from another shard that has >=2 of that fold, without repeating it.
    for shard in bins:
        for split in ("train", "validation"):
            if any(f["split"] == split for f in shard["families"]):
                continue
            candidates = [(f, other) for other in bins if other is not shard
                          and sum(f["split"] == split for f in other["families"]) >= 2
                          for f in other["families"] if f["split"] == split and shard["bytes"] + f["bytes"] <= available]
            if not candidates:
                swaps = [(incoming, outgoing, other) for other in bins if other is not shard
                         and sum(f["split"] == split for f in other["families"]) >= 2
                         for incoming in other["families"] if incoming["split"] == split
                         for outgoing in shard["families"] if outgoing["split"] != split
                         and sum(f["split"] == outgoing["split"] for f in shard["families"]) >= 2
                         and shard["bytes"] - outgoing["bytes"] + incoming["bytes"] <= available
                         and other["bytes"] - incoming["bytes"] + outgoing["bytes"] <= available]
                if not swaps:
                    raise ValueError("cannot preserve whole families with novel TRAIN and VALIDATION in every bounded shard")
                incoming, outgoing, other = min(swaps, key=lambda values: (abs(values[0]["bytes"] - values[1]["bytes"]), values[0]["groupId"], values[1]["groupId"]))
                shard["families"].remove(outgoing)
                shard["families"].append(incoming)
                other["families"].remove(incoming)
                other["families"].append(outgoing)
                shard["bytes"] += incoming["bytes"] - outgoing["bytes"]
                other["bytes"] += outgoing["bytes"] - incoming["bytes"]
                continue
            family, previous = min(candidates, key=lambda value: (value[0]["bytes"], value[0]["groupId"]))
            previous["families"].remove(family)
            previous["bytes"] -= family["bytes"]
            shard["families"].append(family)
            shard["bytes"] += family["bytes"]
    budget.check(outgoing=sum(shard["bytes"] + anchor_size for shard in bins) + MIB)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    shard_reports, coverage = [], {}
    try:
        for index, shard in enumerate(bins, 1):
            filename = f"{prefix}.shard-{index:03d}.train-validation.jsonl"
            entries = [entry for family in anchors + sorted(shard["families"], key=lambda f: f["groupId"]) for entry in family["entries"]]
            raw = b"".join(entry["raw"] for entry in entries)
            if len(raw) > cap:
                raise ValueError("serialized final shard cap exceeded")
            corpus_sha = sha(raw)
            budget.write(staging / filename, raw)
            content = hashlib.sha256()
            counts, costs, novelty = Counter(), {}, []
            for entry in entries:
                row, key = entry["record"], entry["sourceKey"]
                content.update(encode(row["text"]))
                counts[row["split"]] += 1
                cost = costs.setdefault(key, {"records": 0, "words": 0, "payloadUtf8Bytes": 0, "serializedBytes": 0,
                                              "sourceMetadataUtf8Bytes": 0, "splits": Counter()})
                cost["records"] += 1
                cost["words"] += len(WORD.findall(row["text"]))
                cost["payloadUtf8Bytes"] += len(row["text"].encode())
                cost["serializedBytes"] += len(entry["raw"])
                cost["sourceMetadataUtf8Bytes"] += len(encode(row["source"]))
                cost["splits"][row["split"]] += 1
                if not families[row["groupId"]]["retentionOnly"]:
                    novelty.append(row["id"])
            for family in shard["families"]:
                if family["groupId"] in coverage:
                    raise ValueError("new source family appears in multiple shards")
                coverage[family["groupId"]] = filename
            report = {"schemaVersion": 1, "corpusSha256": corpus_sha, "contentSha256": content.hexdigest(),
                      "file": filename, "bytes": len(raw), "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
                      "testUsedForTraining": False, "sources": source_receipts,
                      "addedRecords": len(novelty), "addedSourceIds": novelty,
                      "newFamilyIds": sorted(f["groupId"] for f in shard["families"]),
                      "rehearsalRecords": len(entries) - len(novelty), "rehearsalCountsAsNewDiversity": False,
                      "costSourceGenre": [{"key": key, "genre": next(s["genre"] for s in all_sources if s["key"] == key), **value} for key, value in costs.items()],
                      "recipeSha256": recipe["recipeSha256"], "helperSha256": recipe["helperSha256"], "parentHelperSha256": PARENT_HELPER_SHA256,
                      "textOrMetadataRewritten": False, "foldsMoved": 0,
                      "limitations": recipe["limitations"]}
            manifest_file = filename.replace(".train-validation.jsonl", ".manifest.json")
            encoded_report = encode(report)
            budget.write(staging / manifest_file, encoded_report)
            shard_reports.append({"corpusPath": filename, "corpusSha256": corpus_sha,
                                  "manifestPath": manifest_file, "manifestSha256": sha(encoded_report),
                                  "contentSha256": content.hexdigest(), "bytes": len(raw), "splitCounts": report["splitCounts"], "addedRecords": len(novelty)})
        report = {"schemaVersion": 1, "purpose": "finite broad Spanish plus Latin American causal data shards; no language-totality or model-quality claim",
                  "shards": shard_reports, "sources": source_receipts,
                  "coverage": {"newFamilyToShard": coverage, "retainedNewFamilies": len(new), "assignedNewFamilies": len(coverage),
                               "allRetainedBoundedFamiliesAssignedExactlyOnce": len(new) == len(coverage),
                               "deferredOversizedFamilies": [{"groupId": f["groupId"], "bytes": f["bytes"], "split": f["split"], "recordIds": [e["record"]["id"] for e in f["entries"]]} for f in deferred],
                               "deferredSourceDataPreserved": True},
                  "guards": {"excludedWholeFamilyIds": sorted(excluded), "exclusionReasons": {key: sorted(value) for key, value in reasons.items()},
                             "samefoldHistoricalParagraphOverlapFamilyIds": sorted(samefold_partial_history - excluded),
                             "duplicateIdsRemoved": duplicate_ids, "withinFamilyFullTextDuplicatesRemoved": duplicate_texts,
                             "paragraphMinimumWords": 12, "paragraphDefinition": "blank-line-delimited blocks and individual original lines; >=12 Unicode words",
                             "fullTextNormalization": "original UTF8 exact and whitespace-collapse, no semantic paraphrase model",
                             "initialHistoricalFullTextGuard": "exact original UTF8 only; unchanged initial seed raw whole-record hashes",
                             "wideHistoricalFullTextGuard": ("exact original UTF8 and whitespace-collapse; every fold per digest retained" if extension_receipt else "not applied: wide-v1 is untrained, so its original sources remain eligible"),
                             "wideHistoricalParagraphGuard": ("whitespace-collapse >=12-word original blank-line blocks and lines; every fold retained" if extension_receipt else "not applied: no wide-history seed requested"),
                             "wideHistoryExtension": extension_receipt,
                             "historicalFamilyFoldsPreserved": True,
                             "lexicalRegistries": registry_receipts,
                             "lexicalAliasConflicts": 0,
                             "lexicalRegistryTESTTargetsEmitted": False,
                             "initialHistoricalParagraphGuard": "whitespace-collapse paragraph hashes from original seed",
                             "humanV3FullTextGuard": "exact original UTF8 and whitespace-collapse",
                             "initialGuard": recipe["initialSeed"], "humanV3Guard": recipe["humanV3"], "foldsMoved": 0},
                  "rehearsal": {"source": recipe["rehearsal"], "repeatedAcrossShards": len(bins), "countsAsNewDiversity": False},
                  "testUsedForTraining": False, "testRead": False, "modelsOrTokenizersLoaded": False,
                  "recipeSha256": recipe["recipeSha256"], "helperSha256": recipe["helperSha256"], "parentHelperSha256": PARENT_HELPER_SHA256,
                  "io": {"preflightInputBytes": input_bytes, "preflightReservedOutputBytes": upper_output,
                         "readBytes": budget.read_bytes, "writtenBytesBeforeFinalManifest": budget.written_bytes,
                         "maximumReadPlusWriteBytes": budget.limit},
                  "elapsedSeconds": time.monotonic() - budget.started,
                  "cpuSeconds": time.process_time() - budget.cpu_started,
                  "maxRSSKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "limits": recipe["limits"], "limitations": recipe["limitations"]}
        budget.write(staging / "manifest.json", encode(report))
        os.rename(staging, output)
        print(json.dumps({"status": "prepared", "output": str(output), "shards": shard_reports,
                          "excludedFamilies": len(excluded), "deferredOversizedFamilies": len(deferred),
                          "readBytes": budget.read_bytes, "writtenBytes": budget.written_bytes}, ensure_ascii=False))
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    recipe_raw = args.recipe.read_bytes()
    recipe = json.loads(recipe_raw)
    recipe["recipeSha256"] = sha(recipe_raw)
    recipe["helperSha256"] = sha(Path(__file__).read_bytes())
    io_mib = recipe["limits"]["totalReadWriteMiB"]
    if not 1 <= io_mib <= 128:
        raise ValueError("recipe exceeds 128-MiB finite mixer IO bound")
    parent_raw = (ROOT / "examples/mix_spanish_public_sources.py").read_bytes()
    if sha(parent_raw) != PARENT_HELPER_SHA256:
        raise ValueError("frozen parent mixer changed")
    budget = Budget(io_mib * MIB)
    budget.read_bytes += len(recipe_raw) + Path(__file__).stat().st_size + len(parent_raw)
    mix(recipe, args.output, budget)


if __name__ == "__main__":
    main()
