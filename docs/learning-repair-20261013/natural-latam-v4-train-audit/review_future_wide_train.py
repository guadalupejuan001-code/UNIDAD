#!/usr/bin/env python3
"""TRAIN-only whole-mixture review of future causal v3; stdlib, no weights.

This review is independent of the source curator. It checks exact source rows
and source-sense identities as well as the eleven concrete defective glosses.
Nested source split metadata is permitted; only the top-level split is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import time

LEX = re.compile(rb'"(?:\\.|[^"\\])*"|[{}\[\]]')
VALUE = re.compile(rb'\s*:\s*"(train|validation|test)"')
SOURCE_SHA = {
    "latin-american-wiktionary.train-validation.jsonl": "9c0057f5dc5d00007d971605615bbb1a01c592a6b585e5937a4af62c0f29199b",
    "wiktionary-es.train-validation.jsonl": "ea0c44aec988e22ba4495cb059b7ea558c63a10d7ac8cf2c83946194646fd5f3",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_header(raw):
    depth, matches = 0, []
    for match in LEX.finditer(raw):
        token = match.group()
        if token in (b"{", b"["):
            depth += 1
        elif token in (b"}", b"]"):
            depth -= 1
        elif depth == 1 and token == b'"split"':
            value = VALUE.match(raw, match.end())
            assert value, "Invalid top-level split"
            matches.append(value.group(1).decode())
    assert depth == 0 and len(matches) == 1, "Ambiguous top-level split"
    return matches[0]


def train_rows(path, skipped):
    for raw in path.open("rb"):
        split = split_header(raw)
        if split != "train":
            skipped[split] += 1
            continue
        row = json.loads(raw)
        assert row["split"] == "train"
        yield row, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "An existing receipt cannot be replaced"
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second review wall limit")))
    signal.alarm(120)
    started = time.monotonic()
    package = args.repo / "examples/learning-repair-20261013/new-sources/latin-american-spanish-curated-v3"
    definitions = package / "definitions"
    wide = args.repo / "examples/learning-repair-20261013/wide-latam-v3"
    exclusions = json.loads((definitions / "excluded-senses.json").read_bytes())
    recipe = json.loads((package / "recipe.json").read_bytes())
    known = [x for x in exclusions["excludedSenses"] if x["reviewBasis"]["path"] in
             (recipe["priorExclusions"]["path"], recipe["finalParentReview"]["path"])]
    assert len(known) == 11
    removed = {(x["parentRecordId"], x["etymology"], str(x["senseNumber"])) for x in exclusions["excludedSenses"]}
    glosses = {x["parentRecordId"] + ":" + str(x["etymology"]) + "." + str(x["senseNumber"]):
               " ".join(x["originalSense"]["definition"].casefold().split()) for x in known}
    generic_bad_ids = {x["parentRecordId"].replace(":regional:", ":") for x in known}
    sources, source_skipped = {}, {}
    for name, expected in SOURCE_SHA.items():
        assert digest(definitions / name) == expected
        skipped = Counter()
        sources[name] = {r["id"]: raw for r, raw in train_rows(definitions / name, skipped)}
        source_skipped[name] = dict(skipped)
    manifest_path = wide / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["testUsedForTraining"] is False and manifest["testRead"] is False
    counts, skipped, hits, hashes = Counter(), Counter(), [], {}
    regional_records = regional_senses = generic_records = 0
    for shard in manifest["shards"]:
        path = wide / shard["corpusPath"]
        hashes[path.name] = digest(path)
        assert hashes[path.name] == shard["corpusSha256"]
        for row, raw in train_rows(path, skipped):
            ident = row["id"]
            counts[row["source"]["dataset"]] += 1
            text = " ".join(row["text"].casefold().split())
            hits.extend({"rowId": ident, "meaning": meaning} for meaning, gloss in glosses.items() if gloss in text)
            assert ident not in generic_bad_ids, "Known defective generic source-page equivalent retained"
            if ident.startswith("wiktionary:es:regional:"):
                assert sources["latin-american-wiktionary.train-validation.jsonl"][ident] == raw
                regional_records += 1
                for sense in row["source"]["regionalSenses"]:
                    assert (ident, sense["etymology"], str(sense["senseNumber"])) not in removed
                    regional_senses += 1
            elif re.fullmatch(r"wiktionary:es:\d+:\d+", ident):
                assert sources["wiktionary-es.train-validation.jsonl"][ident] == raw
                generic_records += 1
        assert digest(path) == hashes[path.name]
    assert not hits, hits
    assert digest(manifest_path) == manifest_hash
    assert all(digest(definitions / name) == expected for name, expected in SOURCE_SHA.items())
    report = {
        "reviewType": "Project AI mechanical TRAIN-only review independent of source curator; not independent human factual certification",
        "widePackage": "examples/learning-repair-20261013/wide-latam-v3",
        "globalManifestSha256": manifest_hash, "shardSha256": hashes, "curatedSourceSha256": SOURCE_SHA,
        "trainRowsExamined": sum(counts.values()), "sourceTrainCounts": dict(counts),
        "knownDefectiveSourceSensesChecked": 11, "knownDefectiveGlossMatches": hits,
        "genericKnownDefectiveSourceEquivalentRowsRetained": 0, "excludedRegionalSourceSensesRetained": 0,
        "retainedRegionalTrainRecordsByteExact": regional_records, "retainedRegionalTrainSenses": regional_senses,
        "retainedGenericTrainRecordsByteExact": generic_records,
        "splitHeaderMethod": "JSON lexical nesting at depth1; nested source split keys ignored; nonTRAIN JSON bodies not decoded",
        "nonTrainHeadersSkippedWithoutBodyDecoding": dict(skipped), "sourceNonTrainHeadersSkipped": source_skipped,
        "validationBodiesDecoded": 0, "testBodiesDecoded": 0, "reservedEvaluationFilesOpened": 0,
        "weightsOrTokenizerLoaded": False, "sourceOrInputsEditedByReview": False,
        "elapsedSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": {"cpuAffinityCount": 1, "cpuSeconds": 120, "wallSeconds": 120, "addressSpaceBytes": 1024 ** 3},
        "helperSha256": digest(Path(__file__)), "status": "passed future-only data QA; not enqueued by reviewer",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
