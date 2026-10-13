#!/usr/bin/env python3
"""Acquire a bounded Spanish-American, sense-qualified Wikcionario sample.

Only explicitly regional senses with a conservative clean definition are used.
Country categories select pages, never assign a country to every sense. Original
revision bodies stay private because they may contain excluded external quotes.
No weights, tokenizer, dialogue evaluation, or private conversation is read.
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
import time

import prepare_spanish_wiktionary as common

COUNTRIES = ("México", "Argentina", "Chile", "Colombia", "Perú", "Venezuela",
             "Uruguay", "Bolivia", "Ecuador", "Paraguay", "Guatemala",
             "Costa Rica", "El Salvador", "Honduras", "Nicaragua", "Panamá",
             "Cuba", "Puerto Rico", "República Dominicana")
REGIONS = set(COUNTRIES) | {"América", "América Central", "Centroamérica",
    "Caribe", "Antillas", "Rioplatense", "Río de la Plata", "Cono Sur",
    "Sudamérica", "Suramérica", "Chiapas", "Yucatán", "Zulia", "Chiloé"}
SEEDS = ("guagua", "chamba", "plata", "chévere", "vos", "ustedes", "chido",
         "bacán", "bacano", "pibe", "chamaco", "parce", "pana", "mae",
         "pisto", "pololo", "laburo", "lonche", "ahorita", "al tiro",
         "órale", "cuate", "tuanis", "pata", "chompa", "cotufa")


def regional_senses(section: str) -> list[dict]:
    senses, current, part, etymology = [], None, None, 0
    for raw in section.splitlines():
        etym = re.match(r"^={3,6}\s*Etimología\s*(\d*)\s*={3,6}\s*$", raw, re.I)
        if etym:
            etymology = int(etym.group(1)) if etym.group(1) else etymology + 1
        heading = re.match(r"^={3,6}\s*\{\{([^{}]+)\}\}\s*={3,6}\s*$", raw)
        if heading:
            name, args, _ = common.template_parts(heading.group(1))
            part = name if args and args[0] == "es" and name.startswith((
                "sustantivo", "adjetivo", "verbo", "adverbio", "pronombre",
                "preposición", "conjunción", "interjección", "artículo", "locución")) else None
            current = None
            continue
        if re.match(r"^={3,6}.*={3,6}\s*$", raw):
            part, current = None, None
            continue
        if not part:
            continue
        definition = re.match(r"^;\s*(\d+)(.*?)\s*:\s*(.*)$", raw)
        if definition:
            clean = common.inline(definition.group(3))
            current = {"etymology": etymology, "senseNumber": definition.group(1),
                       "partOfSpeech": part, "definition": clean,
                       "scope": [], "usage": [], "examples": []}
            senses.append(current)
            continue
        if current is None:
            continue
        match = re.fullmatch(r"\s*\{\{(ámbito|ambito|uso)\|(.+)\}\}\s*", raw)
        if match:
            name, args, named = common.template_parts(match.group(1) + "|" + match.group(2))
            if any(not re.fullmatch(r"nota\d*", key) for key in named):
                current["unresolvedScope"] = True
                continue
            values = []
            for index, value in enumerate(args, 1):
                label = common.inline(value)
                note = named.get("nota" if index == 1 else f"nota{index}", "")
                qualifier = common.inline(note) if note else ""
                if label is None or note and qualifier is None:
                    current["unresolvedScope"] = True
                    continue
                values.append({"label": label, "qualifier": qualifier})
            current["scope" if name in {"ámbito", "ambito"} else "usage"].extend(values)
        elif raw.strip().startswith("{{ejemplo|") and raw.strip().endswith("}}"):
            name, args, named = common.template_parts(raw.strip()[2:-2])
            if name == "ejemplo" and len(args) == 1 and not named:
                example = common.inline(args[0])
                if example and 3 <= len(example.split()) <= 70:
                    current["examples"].append(example)
    return [sense for sense in senses if sense["definition"]
            and len(sense["definition"].split()) >= 2
            and not sense.get("unresolvedScope")
            and any(item["label"] in REGIONS for item in sense["scope"])]


def render(title: str, senses: list[dict]) -> str:
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


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[\wáéíóúüñ]+", value.casefold()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--guard-corpus", type=Path, action="append", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-pages", type=int, default=240)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900 or not 1 <= args.max_pages <= 240:
        parser.error("maximum acquisition is 900 seconds and 240 pages")
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads(args.registry.read_text())
    prior_alias = {alias: family for family in prior["families"] for alias in family["aliases"]}
    guard_units, guard_files = defaultdict(set), []
    for path in args.guard_corpus:
        guard_files.append({"path": str(path), "sha256": common.file_sha(path), "bytes": path.stat().st_size})
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["split"] not in {"train", "validation"}:
                    raise ValueError("guard corpus must contain TRAIN/VALIDATION only")
                for paragraph in record["text"].splitlines():
                    # Prefixes are formatting, not independent lexical evidence.
                    paragraph = re.sub(r"^(?:Significado\s+\d+|Ejemplo de uso):\s*", "", paragraph)
                    key = normalized(paragraph)
                    if len(key.split()) >= 12:
                        guard_units[key].add(record["split"])
    source = common.BoundedSource(args.output, args.seconds, 32 << 20)
    site = source.query(meta="siteinfo", siprop="general|rightsinfo")
    rights = site["query"]["rightsinfo"]
    if rights.get("url") != common.LICENSE_URL or site["query"]["general"].get("lang") != "es":
        raise RuntimeError("primary source language/license differs")
    common.dump(args.output / "siteinfo.json", site)
    category_info = source.query(titles="|".join("Categoría:ES:" + country for country in COUNTRIES), prop="categoryinfo")
    common.dump(args.output / "country-category-info.json", category_info)
    selections, by_country, catalog, pages, counts, errors = [], defaultdict(list), {}, [], Counter(), []
    stopped = None
    try:
        catalogue_stopped = False
        for country in COUNTRIES:
            members, continuation, complete = [], {}, False
            while time.monotonic() - source.started < max(1, args.seconds - 150):
                try:
                    value = source.query(list="categorymembers", cmtitle="Categoría:ES:" + country,
                        cmnamespace=0, cmtype="page", cmlimit=500, cmprop="ids|title|sortkeyprefix|timestamp", **continuation)
                except common.ACQUISITION_ERRORS as exc:
                    catalogue_stopped = True
                    stopped = str(exc)[:250]
                    errors.append({"stage": "country-catalogue", "country": country, "error": stopped})
                    break
                members.extend(value["query"]["categorymembers"])
                continuation = value.get("continue", {})
                if not continuation:
                    complete = True
                    break
            catalog[country] = {"category": "Categoría:ES:" + country, "completeDirectNamespaceZeroListing": complete,
                                "members": members, "continuation": continuation}
            available = [page["title"] for page in members if page.get("ns") == 0 and not re.search(r"[:|{}\[\]]", page["title"])]
            selected = [available[(len(available) - 1) * index // 23] for index in range(24)] if available else []
            by_country[country].extend(selected)
            selections.append({"country": country, "category": "Categoría:ES:" + country,
                               "selection": "24 equally spaced positions in the acquired alphabetical catalogue", "titles": selected})
            print(json.dumps({"catalogCountry": country, "directMembers": len(members), "complete": complete,
                              "elapsedSeconds": round(time.monotonic() - source.started, 2)}), flush=True)
            if catalogue_stopped:
                break
        common.dump(args.output / "country-catalog.json", catalog)
        titles = list(SEEDS)
        # Round robin retains all requested countries under a shared page cap.
        for index in range(24):
            for country in COUNTRIES:
                if index < len(by_country[country]):
                    title = by_country[country][index]
                    if title not in titles:
                        titles.append(title)
        titles = titles[:args.max_pages]
        for cursor in range(0, len(titles), 25):
            value = source.query(titles="|".join(titles[cursor:cursor + 25]), prop="revisions",
                                 rvprop="ids|timestamp|content", rvslots="main")
            for page in value.get("query", {}).get("pages", []):
                counts["fetchedPages"] += 1
                if page.get("missing") or not page.get("revisions") or page.get("ns") != 0:
                    counts["missingPages"] += 1
                    continue
                revision = page["revisions"][0]
                raw = revision["slots"]["main"].get("content", "")
                section = common.spanish_section(raw)
                senses = regional_senses(section) if section else []
                if not senses:
                    counts["noCleanExplicitRegionalSenses"] += 1
                    continue
                _, links, _ = common.extract(page["title"], section)
                pages.append({"title": page["title"], "pageId": page["pageid"],
                    "revisionId": revision["revid"], "revisionTimestamp": revision["timestamp"],
                    "wikitextSha256": common.sha(raw.encode()), "sectionSha256": common.sha(section.encode()),
                    "links": sorted(links), "senses": senses, "text": render(page["title"], senses)})
            print(json.dumps({"fetchedPages": counts["fetchedPages"], "usablePages": len(pages),
                "elapsedSeconds": round(time.monotonic() - source.started, 2), "networkBytes": source.network_bytes}), flush=True)
    except common.ACQUISITION_ERRORS as exc:
        stopped = str(exc)[:250]
        errors.append({"stage": "bounded-acquisition", "error": stopped})
        if not (args.output / "country-catalog.json").exists():
            common.dump(args.output / "country-catalog.json", catalog)
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)
    for page in pages:
        key = common.family_key(page["title"])
        for linked in page["links"]:
            union(key, linked)
    components, records, quarantine = defaultdict(list), [], []
    for page in pages:
        components[find(common.family_key(page["title"]))].append(page)
    registry = []
    for root, members in components.items():
        aliases = sorted({alias for page in members for alias in page["links"]})
        historical = {prior_alias[alias]["groupId"]: prior_alias[alias]
                      for alias in aliases if alias in prior_alias}
        if len({family["split"] for family in historical.values()}) > 1:
            quarantine.append({"family": root, "reason": "historical-variant-fold-conflict", "aliases": aliases})
            continue
        if historical:
            chosen = historical[sorted(historical)[0]]
            group, split = chosen["groupId"], chosen["split"]
        else:
            group = "wiktionary:es:lexical-family:" + root
            split = common.assign_split(group)
        conflict = False
        for page in members:
            for sense in page["senses"]:
                for value in [sense["definition"], *sense["examples"]]:
                    key = normalized(value)
                    if len(key.split()) >= 12 and guard_units.get(key, set()) - {split}:
                        conflict = True
        if conflict:
            quarantine.append({"family": root, "reason": "exact-long-lexical-segment-cross-fold", "aliases": aliases})
            continue
        registry.append({"groupId": group, "split": split, "aliases": aliases,
                         "historicalGroups": sorted(historical)})
        for page in members:
            record = {"id": f"wiktionary:es:regional:{page['pageId']}:{page['revisionId']}",
                      "groupId": group, "split": split, "text": page["text"],
                      "source": {"dataset": "Wikcionario/es", "kind": "public-wiki-collaborative",
                        "language": "es", "license": common.LICENSE, "licenseUrl": common.LICENSE_URL,
                        "title": page["title"], "pageId": page["pageId"], "revisionId": page["revisionId"],
                        "revisionTimestamp": page["revisionTimestamp"], "wikitextSha256": page["wikitextSha256"],
                        "spanishSectionSha256": page["sectionSha256"],
                        "revisionUrl": "https://es.wiktionary.org/w/index.php?oldid=" + str(page["revisionId"]),
                        "historyUrl": "https://es.wiktionary.org/w/index.php?title=" + common.quote(page["title"], safe="") + "&action=history",
                        "attribution": "Wikcionario contributors; retained revision and page history",
                        "regionalSenses": page["senses"], "extraction": "Clean explicitly regional Spanish senses; country categories only select pages; no external quotations or dictionary reference text"}}
            common.validate_text_record(record)
            records.append(record)
    # Repeated lexical segments in this new corpus also cannot cross folds.
    new_units, bad_groups = defaultdict(list), set()
    for record in records:
        for sense in record["source"]["regionalSenses"]:
            for value in [sense["definition"], *sense["examples"]]:
                key = normalized(value)
                if len(key.split()) >= 12:
                    new_units[key].append((record["groupId"], record["split"]))
    for values in new_units.values():
        if len({split for _, split in values}) > 1:
            bad_groups.update(group for group, _ in values)
    records = [record for record in records if record["groupId"] not in bad_groups]
    registry = [family for family in registry if family["groupId"] not in bad_groups]
    quarantine += [{"family": group, "reason": "new-exact-long-lexical-segment-cross-fold"} for group in sorted(bad_groups)]
    public = [record for record in records if record["split"] in {"train", "validation"}]
    tests = [record for record in records if record["split"] == "test"]
    if not public:
        common.dump(args.output / "failure-receipt.json", {"status": "no-usable-training-data", "statistics": dict(counts),
                    "errors": errors, "networkBytes": source.network_bytes, "requests": source.requests})
        raise RuntimeError("no usable training rows; original receipts preserved, no empty lot accepted")
    files = {"trainValidation": common.write_jsonl(args.output / "latin-american-wiktionary.train-validation.jsonl", public),
             "testPrivate": common.write_jsonl(args.output / "latin-american-wiktionary.test.jsonl", tests),
             "parsedPublic": common.write_jsonl(args.output / "parsed-regional.train-validation.jsonl", public)}
    common.dump(args.output / "lexical-family-registry.json", {"dataset": "Wikcionario/es", "families": registry,
                "priorRegistrySha256": common.file_sha(args.registry)})
    common.dump(args.output / "quarantine.json", quarantine)
    labels = Counter(item["label"] for record in public for sense in record["source"]["regionalSenses"] for item in sense["scope"])
    country_index = {country: [{"id": record["id"], "title": record["source"]["title"], "split": record["split"],
                    "sense": f"{sense['etymology']}.{sense['senseNumber']}", "qualifier": item["qualifier"]}
        for record in public for sense in record["source"]["regionalSenses"] for item in sense["scope"] if item["label"] == country]
        for country in COUNTRIES}
    common.dump(args.output / "country-sense-index.json", country_index)
    manifest = {"schemaVersion": 1, "dataset": "unidad-latin-american-wiktionary-v1", "objective": "causal_lm",
        "originalApi": common.API, "license": common.LICENSE, "licenseUrl": common.LICENSE_URL,
        "rightsinfo": rights, "files": files, "corpusSha256": files["trainValidation"]["sha256"],
        "splitCounts": dict(Counter(record["split"] for record in records)),
        "trainingFileSplitCounts": dict(Counter(record["split"] for record in public)),
        "trainingRecords": len(public), "trainingFamilies": len({record['groupId'] for record in public}),
        "regionalSenses": sum(len(record["source"]["regionalSenses"]) for record in public),
        "countrySenseCounts": {country: len(country_index[country]) for country in COUNTRIES},
        "scopeLabelCounts": dict(labels), "words": sum(len(record["text"].split()) for record in public),
        "statistics": dict(counts), "quarantinedFamilies": len(quarantine),
        "guardCorpora": guard_files, "priorRegistrySha256": common.file_sha(args.registry),
        "acquisition": {"elapsedSeconds": time.monotonic() - source.started, "networkBytes": source.network_bytes,
            "requests": source.requests, "countryCategorySelections": selections, "seedTitles": list(SEEDS),
            "stoppedReason": stopped, "errors": errors},
        "limits": {"wallSeconds": args.seconds, "maxPages": args.max_pages, "networkBytes": 32 << 20,
            "diskBytes": 32 << 20, "addressSpaceBytes": 512 << 20, "cpuAffinityCount": 1},
        "testUsedForTraining": False, "testTokenized": False, "tokenizerFitted": False, "modelLoaded": False,
        "preparationScriptSha256": common.file_sha(Path(__file__)),
        "extractorDependencySha256": common.file_sha(Path(common.__file__)),
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "catalogue": {"path": str(args.output / "country-catalog.json"), "sha256": common.file_sha(args.output / "country-catalog.json"),
            "directMemberCounts": {country: len(entry['members']) for country, entry in catalog.items()},
            "completedCountries": [country for country, entry in catalog.items() if entry['completeDirectNamespaceZeroListing']],
            "scope": "Direct namespace-zero members of 19 Wikcionario country categories; subcategories, every usage and every Spanish-American term are not covered"},
        "limitations": ["A bounded stratified country-category definition sample; complete direct country title catalogues where the acquisition records exhausted API continuation",
            "Regional scopes are Wikcionario annotations for specific senses and do not assert exclusive or universal usage",
            "Definitions and uncredited examples may contain source errors; no independent human review is claimed",
            "External cited quotations, etymologies, reference dictionary content and unknown templates excluded",
            "No executed training stage or conversational improvement is claimed by preparation alone",
            "Mechanical exact long-segment guards do not detect every semantic paraphrase"]}
    common.dump(args.output / "manifest.json", manifest)
    disk = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if disk > 32 << 20:
        raise RuntimeError("aggregate acquisition disk budget exceeded; source retained private")
    print(json.dumps({"completed": True, "corpusSha256": manifest["corpusSha256"],
        "splitCounts": manifest["splitCounts"], "countrySenseCounts": manifest["countrySenseCounts"],
        "words": manifest["words"], "diskBytes": disk, "networkBytes": source.network_bytes}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
