#!/usr/bin/env python3
"""Prepare a future SFT curriculum; never load a tokenizer or model.

The old corpus stays immutable. Whole procedural TRAIN families are sampled
within their original surface-template strata. Every other old row and every
old validation row retain their exact bytes. Regional answers derive only from
explicit country scopes in a frozen, attributed Wikcionario sense extract.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import resource
import time
from pathlib import Path


PRIOR_SHA = "4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510"
REGIONAL_SHA = "d27c633d6166f42d7fe5d59d865b5a8a3ba67c6d3838035d059178184d1d8f19"
PROCEDURAL = "unidad-composition-v1"
COUNTRIES = frozenset((
    "Argentina", "Bolivia", "Chile", "Colombia", "Costa Rica", "Cuba",
    "Ecuador", "El Salvador", "Guatemala", "Honduras", "México",
    "Nicaragua", "Panamá", "Paraguay", "Perú", "Puerto Rico",
    "República Dominicana", "Uruguay", "Venezuela",
))
SCOPE_ALIASES = {"Bolivia (Tarija)": ("Bolivia", "Tarija")}
UNRESOLVED_USAGE_CODES = frozenset(("utcs", "utci", "utct", "utcp", "utcsma", "umcp"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def country_scopes(sense: dict) -> list[tuple[str, str]]:
    result = []
    for item in sense["scope"]:
        label, qualifier = item["label"], item.get("qualifier", "")
        if label in COUNTRIES:
            result.append((label, qualifier))
        elif label in SCOPE_ALIASES:
            country, local = SCOPE_ALIASES[label]
            result.append((country, "; ".join(x for x in (local, qualifier) if x)))
    return result


def usage(sense: dict) -> str:
    # Keep the actual usage notes, including offensive/historical qualifications.
    parts = []
    for item in sense.get("usage", []):
        text = item["label"]
        if item.get("qualifier"):
            text += " (" + item["qualifier"] + ")"
        parts.append(text)
    return "; ".join(parts)


def gloss(sense: dict) -> str:
    return sense["definition"].strip().rstrip(".")


def answer(title: str, country: str, qualifier: str, sense: dict,
           variant: int = 0, only_gloss: bool = False) -> str:
    definition = gloss(sense)
    if only_gloss:
        text = definition[:1].upper() + definition[1:] + "."
    else:
        location = country + (" (" + qualifier + ")" if qualifier else "")
        templates = (
            f"En {location}, «{title}» puede significar: {definition}.",
            f"Un sentido de «{title}» en {location} es: {definition}.",
            f"Para {location}, la fuente recoge este sentido de «{title}»: {definition}.",
            f"Allí, «{title}» puede referirse a: {definition}.",
        )
        text = templates[variant % len(templates)]
    if only_gloss and qualifier:
        text += f" Ámbito específico: {qualifier}."
    if usage(sense):
        text += " Uso indicado: " + usage(sense) + "."
    return text


def make_row(parent: dict, suffix: str, pairs: list[tuple[str, str]],
             selected: list[dict], behavior: str, error_context: bool = False) -> dict:
    require(bool(pairs), "Empty dialogue")
    source = parent["source"]
    citations = {key: source[key] for key in (
        "dataset", "license", "licenseUrl", "title", "pageId", "revisionId",
        "revisionTimestamp", "revisionUrl", "historyUrl", "wikitextSha256",
        "spanishSectionSha256", "attribution",
    )}
    messages = []
    for question, response in pairs:
        messages.extend(({"role": "user", "content": question},
                         {"role": "assistant", "content": response}))
    return {
        "id": parent["id"] + ":natural-latam-v3:" + suffix,
        "groupId": parent["groupId"], "split": parent["split"],
        "messages": messages,
        "source": {
            "dataset": "unidad-natural-latam-sft-v3", "kind": "authored-synthetic",
            "authoring": "project-programmatic-deterministic-source-grounded",
            "language": "es", "license": "CC-BY-SA-4.0",
            "parentRecordId": parent["id"],
            "parentRecordSha256": digest(encoded(parent)),
            "parentRecordSerialization": "sorted compact UTF-8 JSON followed by LF",
            "regionalCorpusSha256": REGIONAL_SHA,
            "provenance": citations, "selectedSenses": selected,
            "behavior": behavior, "prefixPairs": len(pairs),
            "templateVariationsAreNewFacts": False,
            "modelResponsesUsed": False, "privateConversationUsed": False,
            "erroneousAssistantContext": error_context,
            "supervisionPolicy": "assistant_last_reply",
            "earlierErrorMustNeverBePositiveTarget": error_context,
        },
    }


def generate_regionals(rows: list[dict], seed: str, target_exclusions: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    outputs, exclusions = [], []
    excluded_senses = {(x["parentRecordId"], x["etymology"], x["senseNumber"]): x
                       for x in (target_exclusions or [])}
    country_counts = collections.Counter()
    ordered = sorted(rows, key=lambda r: digest((seed + r["id"]).encode()))
    for ordinal, parent in enumerate(ordered):
        source, title = parent["source"], parent["source"]["title"]
        if "/" in parent["groupId"] or "cita requerida" in parent["groupId"]:
            exclusions.append({"id": parent["id"], "reason": "malformed-source-lexical-family-preserved-not-reassigned"})
            continue
        options = []
        for sense in source["regionalSenses"]:
            require(bool(sense["definition"].strip()), "Empty source definition")
            blocked = excluded_senses.get((parent["id"], sense["etymology"], sense["senseNumber"]))
            if blocked:
                exclusions.append({"id": parent["id"], "reason": blocked["reason"], "sense": sense})
                continue
            if target_exclusions is not None and any(u["label"] in UNRESOLVED_USAGE_CODES for u in sense["usage"]):
                exclusions.append({"id": parent["id"], "reason": "unexpanded-source-usage-abbreviation", "sense": sense})
                continue
            if len(sense["definition"].split()) > 90:
                exclusions.append({"id": parent["id"], "reason": "sense-over-90-words",
                                   "sense": sense})
                continue
            for country, qualifier in country_scopes(sense):
                options.append((country, qualifier, sense))
        if not options:
            exclusions.append({"id": parent["id"],
                               "reason": "no-explicit-supported-country-sense"})
            continue
        chosen = min(options, key=lambda x: (country_counts[x[0]],
                                            digest((seed + parent["id"] + x[0] +
                                                    x[2]["definition"]).encode())))
        country, qualifier, sense = chosen
        country_counts[country] += 1
        explicit = country + (" (" + qualifier + ")" if qualifier else "")
        question_patterns = (
            f"¿Qué puede significar «{title}» en {explicit}? Dime un sentido.",
            f"Escuché «{title}» en {explicit}. Explícame un significado.",
            f"¿Me explicas un uso de «{title}» en {explicit}?",
            f"Necesito entender «{title}» como se usa en {explicit}. Dame una acepción.",
        )
        q = question_patterns[ordinal % len(question_patterns)]
        a = answer(title, country, qualifier, sense, ordinal)
        outputs.append(make_row(parent, "direct", [(q, a)], [sense], "country-meaning"))
        format_q = f"En {explicit}, ¿cuál es un sentido de «{title}»? Solo la definición y las advertencias de uso; sin ejemplos."
        outputs.append(make_row(parent, "format", [(format_q, answer(
            title, country, qualifier, sense, only_gloss=True))], [sense], "requested-format"))
        # This authored error is context only. We do not publish its one-pair prefix.
        repair_pairs = [(q, "Se encuentra junto a la plaza."),
                        (f"No te pregunté dónde está. Quiero saber qué significa «{title}» en {explicit}; dame un sentido.", a)]
        outputs.append(make_row(parent, "repair", repair_pairs, [sense],
                                "recover-from-wrong-task", error_context=True))
        definitions = {gloss(x[2]).casefold() for x in options}
        if len(definitions) > 1:
            ambiguous = [(f"¿Qué significa «{title}»?",
                          "La fuente recoge varios sentidos. ¿En qué país o contexto lo escuchaste?")]
            outputs.append(make_row(parent, "clarify-1", ambiguous, [], "genuine-ambiguity"))
            resolved = ambiguous + [(f"En {explicit}. Dime un sentido de allí.", a)]
            outputs.append(make_row(parent, "clarify-2", resolved, [sense], "resolve-ambiguity"))
            unresolved = ambiguous + [("No sé el país ni recuerdo el contexto.",
                                      "Con esos datos no puedo escoger un sentido con seguridad. Si recuerdas la frase o el país, podremos aclararlo.")]
            outputs.append(make_row(parent, "unknown", unresolved, [], "missing-information"))
        alternatives = [x for x in options if x[0] != country and
                        gloss(x[2]).casefold() != gloss(sense).casefold()]
        if alternatives:
            other = min(alternatives, key=lambda x: (country_counts[x[0]], x[0], gloss(x[2])))
            other_country, other_qualifier, other_sense = other
            country_counts[other_country] += 1
            other_explicit = other_country + (" (" + other_qualifier + ")" if other_qualifier else "")
            b = answer(title, other_country, other_qualifier, other_sense, ordinal + 1)
            corrected = [(q, a), (f"Corrijo el país: era {other_explicit}, no {explicit}. ¿Cuál puede ser un significado allí?", b)]
            outputs.append(make_row(parent, "country-correction", corrected,
                                    [sense, other_sense], "follow-country-correction"))
            speaker_q = (f"Iris escuchó «{title}» en {explicit}; Noé lo escuchó en {other_explicit}. "
                         "Ahora pregunta Noé. ¿Qué puede significar en su caso? Dime un sentido.")
            outputs.append(make_row(parent, "speakers", [(speaker_q, b)],
                                    [other_sense], "distinguish-speakers"))
    return outputs, exclusions


def prepare(prior: Path, regional: Path, output: Path, seed: str,
            max_records: int, design: Path, target_exclusion_receipt: Path | None = None) -> dict:
    started = time.monotonic()
    require(not output.exists(), "Output directory must be new; history is immutable")
    prior_bytes, regional_bytes = prior.read_bytes(), regional.read_bytes()
    require(digest(prior_bytes) == PRIOR_SHA, "Prior corpus hash mismatch")
    require(digest(regional_bytes) == REGIONAL_SHA, "Regional corpus hash mismatch")
    spec = json.loads(design.read_text())
    require(spec["status"] == "specified-before-data-generation", "Missing prospective design")
    originals = [(raw, json.loads(raw)) for raw in prior_bytes.splitlines(keepends=True)]
    regional_rows = [json.loads(raw) for raw in regional_bytes.splitlines()]
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    folds, identifiers = {}, set()
    for raw, row in originals:
        require(row["split"] in ("train", "validation"), "TEST in prior input")
        require(row["id"] not in identifiers, "Duplicate prior ID")
        identifiers.add(row["id"])
        old = folds.setdefault(row["groupId"], row["split"])
        require(old == row["split"], "Prior family crosses folds")
        if row["split"] == "train" and row["source"]["dataset"] == PROCEDURAL:
            strata[row["source"]["templateId"]][row["groupId"]].append(row)
    require(len(strata) == 40 and all(len(g) == 64 for g in strata.values()),
            "Unexpected procedural template/world strata")
    selected, selection = set(), []
    for template, groups in sorted(strata.items()):
        ranked = sorted(groups, key=lambda g: digest((seed + template + g).encode()))
        chosen = ranked[:10]
        selected.update(chosen)
        selection.append({"templateId": template, "availableFamilies": len(groups),
                          "selectedFamilies": chosen,
                          "selectedRows": sum(len(groups[g]) for g in chosen)})
    retained = [(raw, row) for raw, row in originals if not (
        row["split"] == "train" and row["source"]["dataset"] == PROCEDURAL and
        row["groupId"] not in selected)]
    procedural_count = sum(r["split"] == "train" and
                           r["source"]["dataset"] == PROCEDURAL for _, r in retained)
    require(procedural_count <= 1600, "Procedural TRAIN cap exceeded")
    for row in regional_rows:
        require(row["split"] in ("train", "validation"), "TEST in regional input")
        old = folds.setdefault(row["groupId"], row["split"])
        require(old == row["split"], "Regional family crosses folds")
    target_exclusions = (json.loads(target_exclusion_receipt.read_text())["excludedSenses"]
                         if target_exclusion_receipt else None)
    synthetic, exclusions = generate_regionals(regional_rows, seed, target_exclusions)
    require(len(synthetic) <= max_records, "Regional cap exceeded; revise prospective design")
    prompt_folds = {}
    for row in [r for _, r in retained] + synthetic:
        prompt = json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True)
        key = digest(prompt.encode())
        old = prompt_folds.setdefault(key, row["split"])
        require(old == row["split"], "Full prompt crosses folds")
    control = b"".join(raw for raw, _ in retained)
    regional_output = b"".join(encoded(row) for row in synthetic)
    combined = control + regional_output
    require(len(control) + len(regional_output) + len(combined) <= 64 * 1024**2,
            "Output data exceeds 64 MiB budget")
    require(time.monotonic() - started < 600, "Preparation time budget exceeded")
    output.mkdir(parents=True)
    files = {"natural-control.train-validation.jsonl": control,
             "regional-sft.train-validation.jsonl": regional_output,
             "natural-latam-sft.train-validation.jsonl": combined,
             "prospective-design.json": design.read_bytes()}
    if target_exclusion_receipt:
        files["target-exclusions.json"] = target_exclusion_receipt.read_bytes()
    for name, data in files.items():
        (output / name).write_bytes(data)
    arm_manifests = {}
    for arm, records, data, name in (
        ("natural-control", [r for _, r in retained], control,
         "natural-control.train-validation.jsonl"),
        ("natural-latam", [r for _, r in retained] + synthetic, combined,
         "natural-latam-sft.train-validation.jsonl"),
    ):
        content = hashlib.sha256()
        split_counts = collections.Counter()
        for record in records:
            content.update((json.dumps(record["messages"], ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")) + "\n").encode())
            split_counts[record["split"]] += 1
        arm_manifest = {
            "schemaVersion": 1, "arm": arm, "status": "prepared-not-trained",
            "corpusSha256": digest(data), "contentSha256": content.hexdigest(),
            "file": {"path": name, "bytes": len(data), "sha256": digest(data)},
            "splitCounts": {"train": split_counts["train"],
                            "validation": split_counts["validation"], "test": 0},
            "priorCorpusSha256": PRIOR_SHA, "regionalCorpusSha256": REGIONAL_SHA,
            "designSha256": digest(design.read_bytes()),
            "helperSha256": digest(Path(__file__).read_bytes()),
            "policy": "assistant_last_reply; old positive targets/folds unmodified; authored errors context only",
            "regionalDataAdded": arm == "natural-latam",
            "targetExclusionReceiptSha256": digest(target_exclusion_receipt.read_bytes()) if target_exclusion_receipt else None,
            "oldValidationRowsPreservedByteExactly": 1704 if len(originals) == 20292 else sum(
                r["split"] == "validation" for _, r in originals),
            "matchedComparison": "Same own parent, seed, model, generation conditions and 4096 logical updates; treatment first, control second; evaluation uses same external DEV cases. Validation NLLs across these different corpora are not comparable.",
        }
        manifest_name = arm + ".manifest.json"
        (output / manifest_name).write_text(json.dumps(arm_manifest, ensure_ascii=False, indent=2) + "\n")
        arm_manifests[arm] = {"path": manifest_name,
                             "sha256": digest((output / manifest_name).read_bytes()),
                             "corpusSha256": arm_manifest["corpusSha256"],
                             "contentSha256": arm_manifest["contentSha256"],
                             "splitCounts": arm_manifest["splitCounts"]}
    counts = collections.Counter((r["split"], r["source"]["dataset"])
                                 for _, r in retained)
    counts.update((r["split"], r["source"]["dataset"]) for r in synthetic)
    manifest = {
        "schemaVersion": 1, "dataset": "unidad-natural-latam-sft-v3",
        "status": "prepared-not-trained-not-accepted", "seed": seed,
        "priorCorpusSha256": PRIOR_SHA, "regionalCorpusSha256": REGIONAL_SHA,
        "helperSha256": digest(Path(__file__).read_bytes()),
        "designSha256": digest(design.read_bytes()),
        "targetExclusionReceiptSha256": digest(target_exclusion_receipt.read_bytes()) if target_exclusion_receipt else None,
        "files": {name: {"bytes": len(data), "sha256": digest(data)}
                  for name, data in files.items()},
        "arms": arm_manifests,
        "retainedOldRows": len(retained), "originalRows": len(originals),
        "priorValidationRowsPreservedByteExactly": sum(r["split"] == "validation" for _, r in retained),
        "oldNonProceduralRowsPreservedByteExactly": sum(r["source"]["dataset"] != PROCEDURAL for _, r in retained),
        "selectedProceduralTrainFamilies": len(selected),
        "selectedProceduralTrainRows": procedural_count,
        "selectionByTemplate": selection,
        "regionalSyntheticRows": len(synthetic),
        "regionalFamilyCount": len({r["groupId"] for r in synthetic}),
        "regionalCountryScopesUsed": sorted({scope[0] for r in synthetic
            for s in r["source"]["selectedSenses"] for scope in country_scopes(s)}),
        "rowCounts": [{"split": split, "dataset": dataset, "rows": count}
                      for (split, dataset), count in sorted(counts.items())],
        "regionalBehaviorCounts": dict(collections.Counter(r["source"]["behavior"] for r in synthetic)),
        "exclusions": exclusions, "testRowsLoaded": 0,
        "guardLimits": {"fullPromptFoldCheck": True,
                        "familyFoldCheck": True,
                        "semanticParaphraseLeakageCheck": False,
                        "tokenizerOrModelLoaded": False},
        "elapsedSeconds": time.monotonic() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "comparison": "Natural control separates prepared data intervention; no matched control model was trained by this helper.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--regional", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--seed", default="unidad-natural-latam-v2-20261013")
    parser.add_argument("--maximum-regional-records", type=int, default=600)
    parser.add_argument("--target-exclusion-receipt", type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    manifest = prepare(args.prior, args.regional, args.output_dir, args.seed,
                       args.maximum_regional_records, args.design, args.target_exclusion_receipt)
    print(json.dumps({"output": str(args.output_dir),
                      "files": manifest["files"],
                      "regionalRows": manifest["regionalSyntheticRows"],
                      "elapsedSeconds": manifest["elapsedSeconds"],
                      "peakRssMiB": manifest["peakRssMiB"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
