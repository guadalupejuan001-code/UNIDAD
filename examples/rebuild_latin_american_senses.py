#!/usr/bin/env python3
"""Rebuild clean regional TEXT from verified private revisions or public senses.

The private mode selects only already assigned TRAIN/VALIDATION page IDs. The
portable mode renders already published, licensed structured senses, without
network, model, tokenizer, or TEST bodies. Neither mode changes family folds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time

import prepare_latin_american_wiktionary as regional
import prepare_spanish_wiktionary as common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--acquisition", type=Path)
    mode.add_argument("--parsed-source", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    source = args.acquisition / "manifest.json" if args.acquisition else args.parsed_source
    if common.file_sha(source) != args.expected_source_sha256:
        raise ValueError("frozen source identity differs")
    if args.acquisition:
        original = json.loads(source.read_text())
        path = args.acquisition / "latin-american-wiktionary.train-validation.jsonl"
        if common.file_sha(path) != original["corpusSha256"]:
            raise ValueError("original clean corpus identity differs")
    else:
        path = args.parsed_source
        original = None
    records = [json.loads(line) for line in path.open()]
    if any(record["split"] not in {"train", "validation"} for record in records):
        raise ValueError("rebuild source may contain only TRAIN and VALIDATION")
    args.output.mkdir(parents=True, exist_ok=False)
    changes, verified, read_bytes = [], set(), 0
    if args.acquisition:
        by_page = {record["source"]["pageId"]: record for record in records}
        for receipt in original["acquisition"]["requests"]:
            if "prop=revisions" not in receipt["url"]:
                continue
            response = args.acquisition / "responses" / Path(receipt["responsePath"]).name
            raw_bytes = response.read_bytes()
            if len(raw_bytes) != receipt["bytes"] or common.sha(raw_bytes) != receipt["sha256"]:
                raise ValueError("original revision response identity differs")
            read_bytes += len(raw_bytes)
            for page in json.loads(raw_bytes).get("query", {}).get("pages", []):
                # Unassigned/test page content is never parsed or used for this fix.
                if page.get("pageid") not in by_page:
                    continue
                record = by_page[page["pageid"]]
                revision = page["revisions"][0]
                raw = revision["slots"]["main"]["content"]
                if revision["revid"] != record["source"]["revisionId"] or common.sha(raw.encode()) != record["source"]["wikitextSha256"]:
                    raise ValueError("page revision identity differs")
                senses = regional.regional_senses(common.spanish_section(raw))
                if not senses:
                    raise ValueError("corrected parser rejected a historical family; explicit quarantine required")
                if senses != record["source"]["regionalSenses"]:
                    changes.append({"id": record["id"], "title": record["source"]["title"], "split": record["split"],
                                    "before": record["source"]["regionalSenses"], "after": senses})
                record["source"]["regionalSenses"] = senses
                verified.add(page["pageid"])
        if verified != set(by_page):
            raise ValueError("some public source revisions were not verified")
    for record in records:
        record["text"] = regional.render(record["source"]["title"], record["source"]["regionalSenses"])
        common.validate_text_record(record)
    output = common.write_jsonl(args.output / "latin-american-wiktionary.train-validation.jsonl", records)
    common.dump(args.output / "corrected-records.json", changes)
    country_index = {country: [{"id": record["id"], "title": record["source"]["title"], "split": record["split"],
                    "sense": f"{sense['etymology']}.{sense['senseNumber']}", "qualifier": item["qualifier"]}
        for record in records for sense in record["source"]["regionalSenses"] for item in sense["scope"] if item["label"] == country]
        for country in regional.COUNTRIES}
    common.dump(args.output / "country-sense-index.json", country_index)
    manifest = {"schemaVersion": 1, "dataset": "unidad-latin-american-wiktionary-v2", "objective": "causal_lm",
        "license": common.LICENSE, "licenseUrl": common.LICENSE_URL,
        "attribution": "Wikcionario contributors; individual fixed revision URLs and history links retained",
        "sourceSha256": args.expected_source_sha256, "mode": "fixed-private-primary-revisions" if args.acquisition else "fixed-clean-public-structured-senses",
        "files": {"trainValidation": output}, "corpusSha256": output["sha256"],
        "contentOnlySha256": common.sha("\n".join(sorted(record["text"] for record in records)).encode()),
        "contentOnlyHashMethod": "sha256 of newline-joined lexicographically sorted exact text fields, no trailing newline",
        "trainingFileSplitCounts": dict(Counter(record["split"] for record in records)),
        "trainingRecords": len(records), "trainingFamilies": len({record["groupId"] for record in records}),
        "regionalSenses": sum(len(record["source"]["regionalSenses"]) for record in records),
        "countrySenseCounts": {country: len(entries) for country, entries in country_index.items()},
        "words": sum(len(record["text"].split()) for record in records),
        "correctedRecords": len(changes), "primaryPublicRevisionsVerified": len(verified),
        "sourceReadBytes": read_bytes, "networkBytes": 0,
        "testBodiesUsed": False, "testTokenized": False, "foldsMoved": False, "modelLoaded": False, "tokenizerFitted": False,
        "preparationScriptSha256": common.file_sha(Path(__file__)), "regionalParserSha256": common.file_sha(Path(regional.__file__)),
        "elapsedSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": {"cpuAffinityCount": 1, "addressSpaceBytes": 512 << 20, "wallSeconds": 120,
                   "outputDiskBytes": 2 << 20, "networkBytes": 0},
        "reviewer": "Project AI agents; no independent human review",
        "limitations": ["Country scopes apply to individual source senses, not every use or all countries",
                        "Clean structured public source is an adaptation, not the original complete wikitext",
                        "No evidence of trained model knowledge or conversational competence is claimed"]}
    common.dump(args.output / "manifest.json", manifest)
    disk = sum(file.stat().st_size for file in args.output.rglob("*") if file.is_file())
    if disk > 2 << 20 or time.monotonic() - started > 120:
        raise RuntimeError("regional rebuild finite budget exceeded")
    print(json.dumps({"completed": True, "corpus": output, "correctedRecords": len(changes),
                      "verifiedPublicRevisions": len(verified), "words": manifest["words"],
                      "contentOnlySha256": manifest["contentOnlySha256"], "diskBytes": disk,
                      "elapsedSeconds": manifest["elapsedSeconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
