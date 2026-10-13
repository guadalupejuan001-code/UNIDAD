#!/usr/bin/env python3
"""Bounded source/fold/byte QA of this prospective corpus; no model or TEST."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
MIB = 1 << 20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-recipe", type=Path, required=True)
    parser.add_argument("--curated-output", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    read_bytes = 0
    def read(path):
        nonlocal read_bytes
        raw = path.read_bytes()
        read_bytes += len(raw)
        if read_bytes > 128 * MIB or time.monotonic() - started > 115:
            raise ValueError("finite source QA budget exceeded")
        return raw
    def checked(spec):
        raw = read(ROOT / spec["path"])
        if sha(raw) != spec["sha256"]:
            raise ValueError("source changed: " + spec["path"])
        return raw
    recipe_raw = read(args.wide / "recipe.json")
    recipe = json.loads(recipe_raw)
    curation_recipe_raw = read(args.curation_recipe)
    curation_recipe = json.loads(curation_recipe_raw)
    curation_manifest = json.loads(read(args.curated_output / "manifest.json"))
    assert curation_manifest["recipeSha256"] == sha(curation_recipe_raw)
    assert curation_manifest["helperSha256"] == sha(checked(curation_recipe["helper"]))
    for name, spec in curation_manifest["files"].items():
        assert sha(read(args.curated_output / name)) == spec["sha256"]
    old_recipe = json.loads(read(ROOT / "examples/learning-repair-20261013/wide-latam-v2/recipe.json"))
    normalized = copy.deepcopy(recipe)
    changed_keys = []
    for old, new in zip(old_recipe["sources"], normalized["sources"], strict=True):
        assert old["key"] == new["key"]
        if old != new:
            assert new["key"] in {"wiktionary", "latin-american-definitions"}
            changed_keys.append(new["key"])
            new["corpus"], new["provenance"] = old["corpus"], old["provenance"]
    normalized["limitations"] = [x.replace("subset96entries/211senses", "subset106entries/229senses")
                                 for x in normalized["limitations"]]
    assert normalized == old_recipe
    assert set(changed_keys) == {"wiktionary", "latin-american-definitions"}
    original_regional = list(map(json.loads, checked(curation_recipe["originalCorpus"]).splitlines()))
    original_by_id = {row["id"]: row for row in original_regional}
    excluded = json.loads(read(args.curated_output / "excluded-senses.json"))["excludedSenses"]
    known = [x for x in excluded if x["reason"] != "unexpanded-source-usage-abbreviation"]
    assert len(known) == 11 and len(excluded) == 18
    known_identities = {(original_by_id[x["parentRecordId"]]["source"]["pageId"],
                         original_by_id[x["parentRecordId"]]["source"]["revisionId"]) for x in known}
    expected, expected_source, source_receipts = {}, {}, []
    all_sources = [{**recipe["rehearsal"], "retentionOnly": True}] + recipe["sources"]
    for source in all_sources:
        checked(source["provenance"])
        for raw in checked(source["corpus"]).splitlines(keepends=True):
            row = json.loads(raw)
            assert row["split"] in ("train", "validation")
            normalized_raw = raw if raw.endswith(b"\n") else raw + b"\n"
            if row["id"] in expected:
                assert expected[row["id"]] == sha(normalized_raw)
            expected[row["id"]] = sha(normalized_raw)
            expected_source[row["id"]] = source["key"]
        source_receipts.append({"key": source["key"], "corpus": source["corpus"], "provenance": source["provenance"]})
    manifest_raw = read(args.wide / "manifest.json")
    manifest = json.loads(manifest_raw)
    assert manifest["recipeSha256"] == sha(recipe_raw)
    assert manifest["testRead"] is False and manifest["modelsOrTokenizersLoaded"] is False
    assert manifest["guards"]["foldsMoved"] == 0
    assert manifest["coverage"]["allRetainedBoundedFamiliesAssignedExactlyOnce"]
    counts, sources, groups, full_known_hits, bad_identity_hits = Counter(), Counter(), {}, [], []
    valid_joder_matches, records, byte_matches = 0, 0, 0
    for shard in manifest["shards"]:
        raw_corpus = read(args.wide / shard["corpusPath"])
        assert sha(raw_corpus) == shard["corpusSha256"]
        assert sha(read(args.wide / shard["manifestPath"])) == shard["manifestSha256"]
        for raw in raw_corpus.splitlines(keepends=True):
            row = json.loads(raw)
            assert row["split"] in ("train", "validation")
            assert row["id"] in expected and sha(raw) == expected[row["id"]]
            byte_matches += 1
            records += 1
            counts[row["split"]] += 1
            sources[expected_source[row["id"]]] += 1
            if row["groupId"] in groups:
                assert groups[row["groupId"]] == row["split"]
            groups[row["groupId"]] = row["split"]
            for item in known:
                if item["originalSense"]["definition"] in row["text"]:
                    full_known_hits.append({"recordId": row["id"], "excludedParent": item["parentRecordId"],
                                            "senseNumber": item["senseNumber"]})
            source = row["source"]
            if expected_source[row["id"]] in {"wiktionary", "latin-american-definitions"} and (
                    source.get("pageId"), source.get("revisionId")) in known_identities:
                if expected_source[row["id"]] == "wiktionary":
                    bad_identity_hits.append(row["id"])
                else:
                    for sense in source["regionalSenses"]:
                        if any(row["id"] == x["parentRecordId"] and sense == x["originalSense"] for x in known):
                            bad_identity_hits.append(row["id"])
            if row["id"] == "wiktionary:es:2751:6363175":
                assert "Significado 2: malograr el funcionamiento o propósito de algo." in row["text"]
                valid_joder_matches += 1
    assert not full_known_hits and not bad_identity_hits
    assert valid_joder_matches == 1
    assert sources["latin-american-definitions"] == 96
    assert sources["oasst-rehearsal"] == 2299 * len(manifest["shards"])
    generic_original = checked(curation_recipe["genericOriginalCorpus"])
    generic_retained = read(args.curated_output / "wiktionary-es.train-validation.jsonl")
    generic_removed = json.loads(read(args.curated_output / "generic-excluded-records.json"))["excludedRecords"]
    removed_ids = {x["id"] for x in generic_removed}
    assert generic_retained == b"".join(raw for raw in generic_original.splitlines(keepends=True)
                                        if json.loads(raw)["id"] not in removed_ids)
    result = {"schemaVersion": 1, "status": "passed; prepared future data, NOT ENQUEUED, not trained weights",
        "reviewType": "mechanical project-AI source QA, not independent human semantic validation",
        "recipeSha256": sha(recipe_raw), "curationRecipeSha256": sha(curation_recipe_raw),
        "helperSha256": sha(Path(__file__).read_bytes()), "sourceReceipts": source_receipts,
        "originalRegionalCorpusSha256": curation_recipe["originalCorpus"]["sha256"],
        "originalGenericCorpusSha256": curation_recipe["genericOriginalCorpus"]["sha256"],
        "knownDefectiveSourceMeaningsChecked": 11, "knownExcludedMeaningsRetained": 0,
        "knownExcludedSourceIdentityHits": bad_identity_hits, "knownBadGlossTextHits": full_known_hits,
        "opaqueAdditionalExactSensesRemoved": 7, "regionalSensesRemovedUnion": 18,
        "genericSamePageRevisionWholeRecordsQuarantined": generic_removed,
        "genericAllNonmatchingRowsByteExact": True,
        "genericAdditionalValidSensesConservativelyOmitted": 0,
        "unrelatedValidJoderSynonymRecordRetained": "wiktionary:es:2751:6363175",
        "allMixedRecordsByteExactToTheirPinnedPreparedSource": byte_matches == records,
        "sourceCountsAcrossShards": dict(sources), "recordCount": records,
        "splitCountsAcrossShards": dict(counts), "foldsMoved": 0,
        "familyAssignmentsPassed": True, "originalSeedReferenceAndLimitsUnchanged": True,
        "recipeChangesOnlyTwoCuratedSourcePointersAndCorrectedRegionalDescriptiveCount": True,
        "testBodiesRead": 0, "reservedBodiesRead": 0, "modelsOrTokenizersLoaded": False,
        "torchLoaded": any(x == "torch" or x.startswith("torch.") for x in sys.modules),
        "enqueued": False, "trainingStarted": False, "networkBytes": 0,
        "readBytes": read_bytes, "elapsedSeconds": time.monotonic() - started,
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": {"cpuCores": 1, "memoryMiB": 512, "seconds": 120, "readMiB": 128}}
    assert not result["torchLoaded"]
    with args.receipt.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "knownExcludedMeaningsRetained": 0, "records": records,
        "regionalRecords": sources["latin-american-definitions"], "allSourceBytesExact": True,
        "readBytes": read_bytes, "elapsedSeconds": result["elapsedSeconds"], "peakRssKiB": result["peakRssKiB"]}))


if __name__ == "__main__":
    main()
