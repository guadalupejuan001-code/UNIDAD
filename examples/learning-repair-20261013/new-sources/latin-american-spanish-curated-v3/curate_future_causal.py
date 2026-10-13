#!/usr/bin/env python3
"""Prospective exact-sense curation; stdlib only, no model or TEST bodies.

Run this helper from its checked-in package location with a new output path.
Every source/review input is SHA-pinned. Unchanged rows retain original bytes;
changed rows contain only exact retained senses and their deterministic text.
The original corpus and all earlier packages remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
MIB = 1 << 20
OPAQUE = frozenset(("utcs", "utci", "utct", "utcp", "utcsma", "umcp"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def render(title, senses):
    lines = [f"Palabra: {title}."]
    for sense in senses:
        lines += [f"Acepción {sense['etymology']}.{sense['senseNumber']} ({sense['partOfSpeech']}): {sense['definition']}",
                  "Ámbito de esta acepción: " + "; ".join(item["label"] +
                  (f" ({item['qualifier']})" if item["qualifier"] else "")
                  for item in sense["scope"]) + "."]
        if sense["usage"]:
            lines.append("Uso: " + "; ".join(item["label"] +
                         (f" ({item['qualifier']})" if item["qualifier"] else "")
                         for item in sense["usage"]) + ".")
        lines += ["Ejemplo de uso de la fuente: " + example for example in sense["examples"]]
    return "\n".join(lines)


def identity(parent, sense):
    return (parent, sense["etymology"], str(sense["senseNumber"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    if args.output.exists():
        raise ValueError("output already exists; all earlier corpora are immutable")
    recipe_raw = args.recipe.read_bytes()
    recipe = json.loads(recipe_raw)
    if sha(Path(__file__).read_bytes()) != recipe["helper"]["sha256"]:
        raise ValueError("helper bytes differ from the pinned recipe")
    read_bytes = len(recipe_raw) + Path(__file__).stat().st_size
    def checked(spec):
        nonlocal read_bytes
        path = ROOT / spec["path"]
        raw = path.read_bytes()
        read_bytes += len(raw)
        if sha(raw) != spec["sha256"]:
            raise ValueError("pinned input changed: " + spec["path"])
        if read_bytes > 2 * MIB or time.monotonic() - started >= 115:
            raise ValueError("finite preparation input/time budget exceeded")
        return raw
    original = checked(recipe["originalCorpus"])
    original_manifest = json.loads(checked(recipe["originalProvenance"]))
    registry_raw = checked(recipe["familyRegistry"])
    prior = json.loads(checked(recipe["priorExclusions"]))
    final = json.loads(checked(recipe["finalParentReview"]))
    reviewed = json.loads(checked(recipe["reviewedSftManifest"]))
    golden = json.loads(checked(recipe["goldenCases"]))
    generic_original = checked(recipe["genericOriginalCorpus"])
    generic_provenance = json.loads(checked(recipe["genericOriginalProvenance"]))
    if generic_provenance["corpusSha256"] != sha(generic_original):
        raise ValueError("generic source provenance does not match its pinned corpus")
    if original_manifest["corpusSha256"] != sha(original):
        raise ValueError("original provenance differs from original corpus")
    if any(obj.get("sourceCorpusSha256", obj.get("regionalCorpusSha256")) != sha(original)
           for obj in (prior, reviewed, golden)):
        raise ValueError("review or golden cases refers to another source")
    rows, raw_by_id, all_senses = [], {}, {}
    for raw in original.splitlines(keepends=True):
        row = json.loads(raw)
        if row["split"] not in ("train", "validation"):
            raise ValueError("only published TRAIN/VALIDATION source rows are allowed")
        if row["id"] in raw_by_id:
            raise ValueError("duplicate source record identity")
        if row["text"] != render(row["source"]["title"], row["source"]["regionalSenses"]):
            raise ValueError("source text is not an exact rendering of its senses")
        rows.append(row)
        raw_by_id[row["id"]] = raw
        for sense in row["source"]["regionalSenses"]:
            key = identity(row["id"], sense)
            if key in all_senses:
                raise ValueError("duplicate exact source-sense identity")
            all_senses[key] = sense
    removed = {}
    def exclude(parent, sense, reason, basis):
        key = identity(parent, sense)
        if key not in all_senses or all_senses[key] != sense:
            raise ValueError("exclusion does not exactly match pinned source sense")
        removed.setdefault(key, {"parentRecordId": parent, "etymology": sense["etymology"],
                               "senseNumber": sense["senseNumber"], "originalSense": sense,
                               "reason": reason, "reviewBasis": basis})
    if len(prior["excludedSenses"]) != 9:
        raise ValueError("reviewed nine-sense exclusion set changed")
    for item in prior["excludedSenses"]:
        exclude(item["parentRecordId"], item["originalSense"], item["reason"], recipe["priorExclusions"])
    for item in recipe["additionalBotanicalSenses"]:
        parent = item["parentRecordId"]
        if final["removedParents"].get(parent) != item["title"]:
            raise ValueError("additional botanical sense lacks final parent-rejection review")
        key = identity(parent, item)
        sense = all_senses[key]
        if sense["definition"] != item["definition"]:
            raise ValueError("additional botanical gloss differs from reviewed recipe")
        exclude(parent, sense, final["reasons"][item["title"]], recipe["finalParentReview"])
    if len(removed) != 11:
        raise ValueError("expected eleven defective gloss exclusions")
    defective = dict(removed)
    for item in reviewed["exclusions"]:
        if item["reason"] == "unexpanded-source-usage-abbreviation":
            if not any(u["label"] in OPAQUE for u in item["sense"]["usage"]):
                raise ValueError("reviewed opaque usage exclusion has no opaque code")
            exclude(item["id"], item["sense"], item["reason"], recipe["reviewedSftManifest"])
    if len(removed) != 18:
        raise ValueError("reviewed union of eleven bad glosses and seven opaque senses changed")
    parents = {row["id"]: row for row in rows}
    bad_source_identities = {}
    for key, item in defective.items():
        source = parents[key[0]]["source"]
        bad_source_identities.setdefault((source["pageId"], source["revisionId"]), []).append(item)
    generic_kept, generic_rows, generic_removed = [], [], []
    for raw in generic_original.splitlines(keepends=True):
        row = json.loads(raw)
        if row["split"] not in ("train", "validation"):
            raise ValueError("generic source may contain only published TRAIN/VALIDATION")
        source = row["source"]
        matching = bad_source_identities.get((source["pageId"], source["revisionId"]), [])
        if matching:
            # Generic records have no structured sense/etymology mapping.
            # Quarantine whole exact page+revision records conservatively.
            generic_removed.append({"id": row["id"], "groupId": row["groupId"], "split": row["split"],
                "title": source["title"], "sourcePageId": source["pageId"], "sourceRevisionId": source["revisionId"],
                "originalRecordBytesSha256": sha(raw), "defectiveRegionalSenses": matching,
                "granularity": "whole generic record; source schema lacks structured etymology/sense mapping",
                "originalSourceProvenance": source})
        else:
            generic_rows.append(row)
            generic_kept.append(raw)
    if len(generic_removed) != 1 or generic_removed[0]["title"] != "abajeño":
        raise ValueError("generic equivalents of known eleven defects changed")
    generic_corpus = b"".join(generic_kept)
    retained, output_raw, changes, dropped = [], [], [], []
    for original_row in rows:
        parent = original_row["id"]
        senses = original_row["source"]["regionalSenses"]
        keep = [sense for sense in senses if identity(parent, sense) not in removed]
        if not keep:
            dropped.append({"id": parent, "groupId": original_row["groupId"],
                            "split": original_row["split"], "reason": "all-source-senses-excluded"})
            continue
        if keep == senses:
            row, raw = original_row, raw_by_id[parent]
        else:
            row = copy.deepcopy(original_row)
            row["source"]["regionalSenses"] = keep
            row["text"] = render(row["source"]["title"], keep)
            raw = encoded(row)
            original_nontext = {k: v for k, v in original_row.items() if k not in ("text", "source")}
            assert {k: v for k, v in row.items() if k not in ("text", "source")} == original_nontext
            assert {k: v for k, v in row["source"].items() if k != "regionalSenses"} == {
                k: v for k, v in original_row["source"].items() if k != "regionalSenses"}
            changes.append({"id": parent, "split": row["split"], "groupId": row["groupId"],
                            "beforeSenseCount": len(senses), "afterSenseCount": len(keep),
                            "originalRecordBytesSha256": sha(raw_by_id[parent]), "newRecordBytesSha256": sha(raw)})
        assert all(not any(u["label"] in OPAQUE for u in sense["usage"]) for sense in keep)
        retained.append(row)
        output_raw.append(raw)
    qa = []
    for case in golden["cases"]:
        matches = [s for row in retained if row["source"]["title"] == case["title"]
                   for s in row["source"]["regionalSenses"]
                   if s["etymology"] == case["etymology"] and s["senseNumber"] == case["senseNumber"]
                   and any(x["label"] == case["country"] for x in s["scope"])]
        if len(matches) != 1:
            raise ValueError("golden exact country-sense identity lost")
        sense = matches[0]
        if case["contains"] not in sense["definition"] or case.get("mustNotContain", "\0") in sense["definition"]:
            raise ValueError("golden definition failed")
        if case.get("requiredUsage") and case["requiredUsage"] not in [
                u["label"] + (f" ({u['qualifier']})" if u["qualifier"] else "") for u in sense["usage"]]:
            raise ValueError("golden usage warning lost")
        qa.append({**case, "passed": True})
    corpus = b"".join(output_raw)
    index = {}
    for row in retained:
        for sense in row["source"]["regionalSenses"]:
            for scope in sense["scope"]:
                index.setdefault(scope["label"], []).append({"id": row["id"], "title": row["source"]["title"],
                    "split": row["split"], "sense": f"{sense['etymology']}.{sense['senseNumber']}", "qualifier": scope["qualifier"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.output.name + ".", dir=args.output.parent))
    write_bytes = 0
    def write(name, raw):
        nonlocal write_bytes
        write_bytes += len(raw)
        if write_bytes > 2 * MIB:
            raise ValueError("finite output disk budget exceeded")
        with (staging / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    try:
        write("latin-american-wiktionary.train-validation.jsonl", corpus)
        write("wiktionary-es.train-validation.jsonl", generic_corpus)
        write("lexical-family-registry.json", registry_raw)
        write("excluded-senses.json", encoded({"schemaVersion": 1, "reviewType": "Project AI, not independent human validation",
            "originalSource": recipe["originalCorpus"], "defectiveGlosses": 11, "additionalOpaqueUsageSenses": 7,
            "excludedSenses": [removed[k] for k in sorted(removed)], "droppedRecords": dropped,
            "rawOriginalPreserved": True, "noCorrectionsInvented": True}))
        write("changed-records.json", encoded(changes))
        write("generic-excluded-records.json", encoded({"schemaVersion": 1,
            "originalSource": recipe["genericOriginalCorpus"], "knownDefectiveGlossesChecked": 11,
            "sourceIdentityPolicy": "exact original source pageId+revisionId; no unrelated synonym substring bans",
            "excludedRecords": generic_removed, "allRetainedRecordsByteExact": True,
            "originalSourcePreserved": True, "testBodiesRead": 0}))
        write("generic-manifest.json", encoded({"schemaVersion": 1, "dataset": "unidad-wiktionary-spanish-curated-v3",
            "status": "prepared prospective causal data; not queued or trained", "objective": "causal_lm",
            "corpusSha256": sha(generic_corpus), "originalSource": recipe["genericOriginalCorpus"],
            "originalProvenance": recipe["genericOriginalProvenance"], "recipeSha256": sha(recipe_raw),
            "helperSha256": sha(Path(__file__).read_bytes()), "license": "CC-BY-SA-4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/deed.es",
            "attribution": "Wikcionario contributors; all retained original source/license/revision/history fields byte exact",
            "trainingRecords": len(generic_rows), "trainingFamilies": len({r["groupId"] for r in generic_rows}),
            "trainingFileSplitCounts": dict(Counter(r["split"] for r in generic_rows)),
            "removedRecords": len(generic_removed), "removedGranularity": "whole exact source page+revision record",
            "allRetainedRecordsByteExact": True, "foldsMoved": 0, "rawOriginalPreserved": True,
            "testBodiesRead": 0, "reservedBodiesRead": 0, "modelsOrTokenizersLoaded": False,
            "enqueued": False, "trainingStarted": False}))
        write("country-sense-index.json", encoded(index))
        write("golden-country-senses-receipt.json", encoded({"source": recipe["goldenCases"], "cases": qa,
            "purpose": "data QA only; no conversation evaluation or model output", "testBodiesRead": 0}))
        manifest = {"schemaVersion": 1, "dataset": "unidad-latin-american-spanish-curated-v3",
            "status": "prepared prospective causal data; not queued or trained", "objective": "causal_lm",
            "corpusSha256": sha(corpus), "originalSource": recipe["originalCorpus"],
            "originalProvenance": recipe["originalProvenance"], "recipeSha256": sha(recipe_raw),
            "helperSha256": sha(Path(__file__).read_bytes()),
            "files": {name: {"bytes": (staging/name).stat().st_size, "sha256": sha((staging/name).read_bytes())}
                      for name in sorted(p.name for p in staging.iterdir())},
            "license": "CC-BY-SA-4.0", "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/deed.es",
            "attribution": "Wikcionario contributors; exact original fixed revision/history URLs retained per record",
            "sourceSenseField": "source.regionalSenses; original schema preserved",
            "originalRecords": len(rows), "originalSenses": len(all_senses), "trainingRecords": len(retained),
            "trainingFamilies": len({r["groupId"] for r in retained}),
            "regionalSenses": sum(len(r["source"]["regionalSenses"]) for r in retained),
            "trainingFileSplitCounts": dict(Counter(r["split"] for r in retained)),
            "scopeSenseCounts": {country: len(entries) for country, entries in index.items()},
            "removedSenseCount": len(removed), "defectiveGlossesRemoved": 11, "additionalOpaqueUsageSensesRemoved": 7,
            "genericEquivalentWholeRecordsRemoved": len(generic_removed),
            "changedRetainedRecords": len(changes), "droppedRecords": len(dropped),
            "unchangedRecordsByteExact": len(retained) - len(changes), "remainingSensesAndExamplesExact": True,
            "remainingProvenanceExact": True, "foldsMoved": 0, "rawOriginalPreserved": True,
            "allRetainedTextRebuiltFromExactSenses": True, "goldenCasesPassed": len(qa),
            "testBodiesRead": 0, "reservedBodiesRead": 0, "modelsOrTokenizersLoaded": False,
            "torchLoaded": any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "networkBytes": 0, "enqueued": False, "trainingStarted": False,
            "limits": recipe["limits"], "sourceReadBytes": read_bytes, "writtenBytesBeforeManifest": write_bytes,
            "elapsedSeconds": time.monotonic() - started, "cpuSeconds": time.process_time() - cpu_started,
            "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "limitations": ["Exact source exclusions apply prospectively; frozen historical sources and existing weights are not rewritten.",
                "Remaining licensed source senses are a finite sample, not exhaustive Spanish coverage or semantic approval of every remaining gloss.",
                "Country labels apply to individual senses; all original qualifications and usage warnings remain.",
                "Prepared causal data do not prove model knowledge, conversation acceptance or competitive quality."]}
        write("manifest.json", encoded(manifest))
        if time.monotonic() - started > 120:
            raise ValueError("finite preparation wall-time budget exceeded")
        os.rename(staging, args.output)
        print(json.dumps({"status": "prepared", "output": str(args.output), "corpusSha256": sha(corpus),
            "records": len(retained), "senses": manifest["regionalSenses"], "removedSenses": len(removed),
            "unchangedRecordsByteExact": manifest["unchangedRecordsByteExact"], "goldenCasesPassed": len(qa),
            "elapsedSeconds": manifest["elapsedSeconds"], "peakRssKiB": manifest["peakRssKiB"],
            "readBytes": read_bytes, "writtenBytes": write_bytes, "torchLoaded": manifest["torchLoaded"]}))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
