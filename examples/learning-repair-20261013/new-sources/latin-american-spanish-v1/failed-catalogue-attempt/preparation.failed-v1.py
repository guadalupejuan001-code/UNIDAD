#!/usr/bin/env python3
"""Convert a frozen regional title catalogue into attributed causal TEXT.

The text only says which Wikcionario categories listed an entry. It does not
guess definitions, frequency, exclusive geographic use, or grammar. All folds
are assigned before emitting text; TEST titles remain inventory metadata only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
import time

import prepare_spanish_wiktionary as common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--expected-catalogue-sha256", required=True)
    parser.add_argument("--registry", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.monotonic()
    if common.file_sha(args.catalogue) != args.expected_catalogue_sha256:
        raise ValueError("catalogue identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(args.catalogue.read_text())
    if len(catalog) != 19 or any(not item["completeDirectNamespaceZeroListing"] or item["continuation"] for item in catalog.values()):
        raise ValueError("this full direct catalogue recipe requires all 19 country continuations exhausted")
    parents, historical, source_registries = {}, [], []
    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)
    for path in args.registry:
        registry = json.loads(path.read_text())
        source_registries.append({"path": str(path), "sha256": common.file_sha(path)})
        for family in registry["families"]:
            aliases = sorted({common.family_key(alias) for alias in family["aliases"]})
            if not aliases:
                continue
            for alias in aliases:
                union(aliases[0], alias)
            historical.append({**family, "aliases": aliases})
    inventory = {}
    for country, item in catalog.items():
        for member in item["members"]:
            if member.get("ns") != 0:
                raise ValueError("catalogue includes a non-lexical namespace")
            page = inventory.setdefault(member["pageid"], {"title": member["title"], "countries": []})
            if page["title"] != member["title"]:
                raise ValueError("one page ID has multiple titles in this live snapshot; quarantine rather than guess")
            if country not in page["countries"]:
                page["countries"].append(country)
            find(common.family_key(member["title"]))
    inherited = defaultdict(dict)
    for family in historical:
        inherited[find(family["aliases"][0])][family["groupId"]] = family
    members_by_root = defaultdict(list)
    for page_id, page in inventory.items():
        members_by_root[find(common.family_key(page["title"]))].append((page_id, page))
    records, registry, exclusions, test_ids = [], [], [], []
    split_counts, split_country_counts = Counter(), defaultdict(Counter)
    for root, members in sorted(members_by_root.items()):
        families = inherited[root]
        historical_splits = {family["split"] for family in families.values()}
        if len(historical_splits) > 1:
            exclusions.append({"family": root, "reason": "prior-family-fold-conflict", "sourceGroups": sorted(families)})
            continue
        if families:
            chosen = families[sorted(families)[0]]
            group, split = chosen["groupId"], chosen["split"]
        else:
            group = "wiktionary:es:lexical-family:" + root
            split = common.assign_split(group)
        aliases = sorted({common.family_key(page["title"]) for _, page in members}
                         | {alias for family in families.values() for alias in family["aliases"]})
        registry.append({"groupId": group, "split": split, "aliases": aliases, "historicalGroups": sorted(families)})
        for page_id, page in sorted(members):
            split_counts[split] += 1
            split_country_counts[split].update(page["countries"])
            if split == "test":
                test_ids.append(page_id)
                continue
            title = page["title"]
            text = (f"El catálogo de Wikcionario registra la entrada «{title}» en las categorías regionales de "
                    + ", ".join(page["countries"]) + ". Esta clasificación no especifica sus acepciones ni afirma un uso exclusivo.")
            record = {"id": f"wiktionary:es:regional-catalogue:{page_id}", "groupId": group,
                      "split": split, "text": text,
                      "source": {"dataset": "Wikcionario/es", "kind": "catalogue_metadata_not_definitions",
                        "language": "es", "license": common.LICENSE,
                        "licenseUrl": common.LICENSE_URL, "pageId": page_id, "title": title,
                        "countries": page["countries"], "catalogueSha256": args.expected_catalogue_sha256,
                        "historyUrl": "https://es.wiktionary.org/w/index.php?title=" + common.quote(title, safe="") + "&action=history",
                        "attribution": "Wikcionario contributors; frozen country catalogue and page histories",
                        "syntheticFormatting": "Deterministic project-authored sentence around original category metadata; no model generation"}}
            common.validate_text_record(record)
            records.append(record)
    path = args.output / "latin-american-catalogue.train-validation.jsonl"
    receipt = common.write_jsonl(path, records)
    if receipt["bytes"] > 8 << 20:
        raise RuntimeError("catalogue training corpus exceeds its 8 MiB finite cap")
    content = common.sha("\n".join(sorted(record["text"] for record in records)).encode())
    common.dump(args.output / "lexical-family-registry.json", {"dataset": "Wikcionario/es", "families": registry,
                "sourceRegistries": source_registries})
    common.dump(args.output / "excluded-test-page-ids.json", {"testPageIds": sorted(test_ids),
                "testBodiesEmitted": False, "testTokenized": False})
    common.dump(args.output / "quarantine.json", exclusions)
    manifest = {"schemaVersion": 1, "dataset": "unidad-latin-american-catalogue-v1",
        "objective": "causal_lm", "catalogueSha256": args.expected_catalogue_sha256,
        "catalogueDirectCountryMemberships": sum(len(item["members"]) for item in catalog.values()),
        "catalogueUniquePageIds": len(inventory), "lexicalInventorySplitCounts": dict(split_counts),
        "trainingFileSplitCounts": dict(Counter(record["split"] for record in records)),
        "trainingRecords": len(records), "trainingFamilies": len({record["groupId"] for record in records}),
        "allInventoryFamilies": len(registry), "words": sum(len(record["text"].split()) for record in records),
        "countryMembershipsBySplit": {split: dict(counts) for split, counts in split_country_counts.items()},
        "files": {"trainValidation": receipt}, "corpusSha256": receipt["sha256"], "contentOnlySha256": content,
        "contentOnlyHashMethod": "sha256 of newline-joined lexicographically sorted exact text fields, no trailing newline",
        "priorFamilyRegistries": source_registries, "quarantinedFamilies": len(exclusions),
        "partitionMethod": "Whole normalized accent/case/reflexive and explicitly registered prior lemma/variant components; prior folds preserved; deterministic assign_split only for new families",
        "limits": {"cpuAffinityCount": 1, "wallSeconds": 120, "addressSpaceBytes": 512 << 20,
                   "corpusBytes": 8 << 20, "outputDiskBytes": 16 << 20, "networkBytes": 0},
        "elapsedSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "testBodiesEmitted": False, "testTokenized": False, "modelLoaded": False, "tokenizerFitted": False,
        "preparationScriptSha256": common.file_sha(Path(__file__)),
        "limitations": ["Country category membership is source classification, not a definition or proof of universal/exclusive geographic use",
            "One deterministic sentence format adds catalogue facts, not natural dialogue diversity",
            "Catalogue completeness is limited to direct namespace-zero entries in 19 acquired categories; no subcategory recursion",
            "Unregistered lemma/form relations are not inferred from word endings beyond conservative reflexive normalization",
            "Catalogue metadata were acquired live, not an atomic global historical snapshot; definition samples have separate fixed revisions",
            "Preparation alone does not establish that these facts were trained or learned"]}
    common.dump(args.output / "manifest.json", manifest)
    disk = sum(file.stat().st_size for file in args.output.rglob("*") if file.is_file())
    if disk > 16 << 20 or time.monotonic() - started > 120:
        raise RuntimeError("catalogue finite output/time budget exceeded")
    print(json.dumps({"completed": True, "corpus": receipt, "contentOnlySha256": content,
        "splitCounts": dict(split_counts), "trainingFamilies": manifest["trainingFamilies"],
        "words": manifest["words"], "diskBytes": disk, "elapsedSeconds": manifest["elapsedSeconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
