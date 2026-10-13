"""Author a development-only probe from frozen public source, without model imports."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V4 = ROOT / "examples/learning-repair-20261013/natural-latam-sft-v4"
RAW = ROOT / "examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/definitions/latin-american-wiktionary.train-validation.jsonl"
SUITE = "latam-knowledge-development-probe-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def lines(path):
    return [(i, b, json.loads(b)) for i, b in enumerate(path.read_bytes().splitlines(keepends=True), 1)]


def exact(path, line, data):
    return {"path": str(path.relative_to(ROOT)), "line": line, "bytes": len(data),
            "sha256": sha(data), "encoding": "UTF-8", "includesTrailingLf": data.endswith(b"\n"),
            "exactBytesUtf8": data.decode("utf-8")}


def mentions(text, term):
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))


# Hand-authored before generation; country scopes and facts below are verified against raw source.
specs = [
    ("train", "estar al cateo de la laucha", "Chile", "42091:6037645", "Neralvo",
     "¿Cómo dirías en español general «estar al cateo de la laucha» (Chile)?",
     "Soy Neralvo. En Chile oí «estar al cateo de la laucha». ¿Me lo aclaras? Usa una sola línea: «Neralvo: <sentido>».",
     ["Estar a la expectativa de una oportunidad favorable."],
     {"type": "one-line-prefix", "prefix": "Neralvo: "}),
    ("train", "abastecedor", "Costa Rica", "12803:5722010", "Odrilena",
     "En Costa Rica, ¿qué clase de local llaman «abastecedor»?",
     "Odrilena pregunta por «abastecedor» en Costa Rica. ¿Qué es? Devuelve solo JSON con «persona»: «Odrilena» y «significado»: una frase.",
     ["Tienda de alimentación y productos del hogar."],
     {"type": "json-object", "exactKeys": ["persona", "significado"], "fixedValues": {"persona": "Odrilena"}}),
    ("train", "cruzazulear", "México", "970176:6322442", "Talvexo",
     "Si en México un equipo «cruzazulea», ¿qué le pasa?",
     "Talvexo quiere entender «cruzazulear» en México. ¿Qué describe y con qué tono se usa? Contesta con dos viñetas: «Talvexo: <sentido>» y «Uso: <tono>».",
     ["Perder estrepitosamente; especialmente tras una racha positiva, sin convertir esa circunstancia en requisito universal."],
     {"type": "two-bullets", "prefixes": ["Talvexo: ", "Uso: "], "requiredUsage": ["coloquial", "jocoso"]}),
    ("train", "a la zumba marumba", "Honduras", "974807:6107413", "Ismarune",
     "¿Qué manera de actuar describe «a la zumba marumba» en Honduras?",
     "Me llamo Ismarune y oí «a la zumba marumba» en Honduras. ¿Cómo se interpreta? Una línea, con este formato: «Ismarune — <sentido>».",
     ["Actuar de forma alocada, sin planificar."],
     {"type": "one-line-prefix", "prefix": "Ismarune — "}),
    ("train", "buco", "Panamá", "22148:6421374", "Veldunio",
     "¿Qué expresa «buco» al hablar de cantidades en Panamá?",
     "Veldunio consulta por «buco» en Panamá. ¿Qué indica sobre una cantidad? Responde solo «Veldunio | <sentido>», en una línea.",
     ["Gran cantidad de algo; no trasladar el sentido argentino de orificio."],
     {"type": "one-line-prefix", "prefix": "Veldunio | "}),
    ("train", "cofla", "Perú", "1037265:5820672", "Senavira",
     "Si en Perú llaman «cofla» a una persona, ¿cómo la describen?",
     "Senavira pregunta qué describe «cofla» en Perú. ¿Qué le dirías? Usa tres líneas: «Nombre: Senavira», «País: Perú» y «Sentido: <explicación>».",
     ["Persona delgada o flaca; la fuente lo identifica como jerga, sin exigir copiar «Es una jerga»."],
     {"type": "three-lines", "fixedLines": ["Nombre: Senavira", "País: Perú"], "lastPrefix": "Sentido: "}),
    ("validation", "apampichar", "República Dominicana", "18515:5806586", None,
     "En República Dominicana, ¿qué actividad nombra «apampichar»? Si no lo sabes, dilo.", None,
     ["Tocar o bailar merengue al estilo del pambiche, variante más lenta y acompasada; no exigir exclusivamente bailar."], None),
    ("validation", "champa", "Guatemala", "22799:6356442", None,
     "En Guatemala, ¿qué vivienda se llama «champa»? Si no lo sabes, dilo.", None,
     ["Casa de mala calidad propia de gente muy humilde de los suburbios; no trasladar el sentido ni la marca vulgar de Chile."], None),
    ("validation", "a cuenta de oreja de cochino", "Venezuela", "1036260:5613556", None,
     "En Venezuela, ¿qué pretende alguien que pide algo «a cuenta de oreja de cochino»? Si no lo sabes, dilo.", None,
     ["Recibir un beneficio sin ofrecer nada a cambio, creyendo que es donación caritativa, especialmente si es costoso."], None),
]


def main():
    regional_path = V4 / "regional-sft.train-validation.jsonl"
    raw_rows = {d["id"]: (i, b, d) for i, b, d in lines(RAW)}
    regional_rows = lines(regional_path)
    regional = {d["id"]: (i, b, d) for i, b, d in regional_rows}
    assert len(regional) == 463
    assert sha(regional_path.read_bytes()) == "520510f77e26626b66af6c1e7ee0c8492f50c381518a4877e2696841bb70138f"
    assert sha(RAW.read_bytes()) == "d27c633d6166f42d7fe5d59d865b5a8a3ba67c6d3838035d059178184d1d8f19"
    train_path = V4 / "natural-latam-sft.train-validation.jsonl"
    corpus = [d for _, _, d in lines(train_path)]
    assert sha(train_path.read_bytes()) == json.loads((V4 / "natural-latam.manifest.json").read_text())["corpusSha256"]
    train = [r for r in corpus if r["split"] == "train"]
    train_text = "\n".join(m["content"] for r in train for m in r["messages"])
    all_user_text = {m["content"] for r in corpus for m in r["messages"] if m["role"] == "user"}
    train_groups = {r["groupId"] for r in train}
    prev = json.loads((V4 / "preceding-target-exclusions.json").read_text())
    final = json.loads((V4 / "final-review-exclusions.json").read_text())
    banned_parents = {s["parentRecordId"] for s in prev["excludedSenses"]} | set(final["removedParents"])
    banned_terms = {s["title"] for s in prev["excludedSenses"]} | set(final["removedParents"].values())
    supports, scenarios, checks = {}, [], []
    for n, (split, title, country, tail, name, prompt, combo, facts, form) in enumerate(specs, 1):
        parent = "wiktionary:es:regional:" + tail
        support_id = parent + ":natural-latam-v3:direct"
        i, b, d = regional[support_id]
        ri, rb, raw = raw_rows[parent]
        src = d["source"]
        assert d["split"] == raw["split"] == split
        assert title == src["provenance"]["title"] == raw["source"]["title"]
        assert parent not in banned_parents and title not in banned_terms
        assert len(src["selectedSenses"]) == 1
        sense = src["selectedSenses"][0]
        assert country in {s["label"] for s in sense["scope"]}
        assert sense in raw["source"]["regionalSenses"]
        canonical = (json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        assert sha(canonical) == src["parentRecordSha256"]
        assert src["license"] == raw["source"]["license"] == "CC-BY-SA-4.0"
        family_splits = {r["split"] for _, _, r in regional_rows if r["groupId"] == d["groupId"]}
        assert family_splits == {split}
        matching_train_ids = [r["id"] for r in train if any(mentions(m["content"], title) for m in r["messages"])]
        if split == "validation":
            assert d["groupId"] not in train_groups and not matching_train_ids
        if name:
            assert name.casefold() not in train_text.casefold()
        supports[support_id] = {"supportId": support_id, "parentRecordId": parent, "lexeme": title,
            "familyId": d["groupId"], "sourceSplit": split, "country": country,
            "selectedSense": sense, "provenance": src["provenance"], "license": src["license"],
            "trainingSourceRow": exact(regional_path, i, b), "rawLexicalSourceRow": exact(RAW, ri, rb),
            "parentCanonicalSerialization": src["parentRecordSerialization"],
            "parentCanonicalSha256": src["parentRecordSha256"],
            "regionalFamilyRowIds": [r["id"] for _, _, r in regional_rows if r["groupId"] == d["groupId"]]}
        for kind, user in [("train_paraphrase" if split == "train" else "validation_calibration", prompt)] + (
                [("train_composition", combo)] if combo else []):
            assert user not in all_user_text
            assert sense["definition"].casefold().strip(".") not in user.casefold()
            identifier = ("tr" if split == "train" else "val") + f"{n:02d}" + ("p" if kind == "train_paraphrase" else "c" if kind == "train_composition" else "u")
            expected = {"sourceGlossExact": sense["definition"], "mustConvey": facts,
                "sourceUsage": sense["usage"], "country": country,
                "uncertaintyAllowed": split == "validation", "format": form if kind == "train_composition" else None,
                "correctFactRequiredForLearnedKnowledgeCount": split == "train",
                "policy": "Semantic paraphrases accepted; no unsupported specificity, cross-country sense, or exclusivity claim."}
            step = {"id": identifier, "user": user, "critical": False, "kind": kind,
                "lexeme": title, "familyId": d["groupId"], "sourceSplit": split,
                "supportIds": [support_id, parent], "supportRowSha256": sha(b),
                "newInventedName": name if kind == "train_composition" else None, "expected": expected}
            scenarios.append({"id": identifier, "kind": kind, "familyId": d["groupId"], "steps": [step]})
        checks.append({"lexeme": title, "familyId": d["groupId"], "split": split,
            "trainExactLexemeRowMatches": len(matching_train_ids), "trainMatchingRowIds": matching_train_ids,
            "familySplits": sorted(family_splits), "newName": name, "newNameAbsentFromTrain": name is None or name.casefold() not in train_text.casefold(),
            "parentAndLexemeExcluded": False, "singleSelectedSense": True, "countryVerifiedAgainstRawSource": True})
    assert len(scenarios) == 15 and len({s["steps"][0]["user"] for s in scenarios}) == 15
    assert len({s[2] for s in specs if s[0] == "train"}) == 6
    files = [RAW, RAW.parent / "manifest.json", regional_path, train_path,
             V4 / "natural-control.train-validation.jsonl", V4 / "natural-control.manifest.json",
             V4 / "natural-latam.manifest.json", V4 / "natural-latam.manifest.admission-v1.json",
             V4 / "manifest.json", V4 / "preceding-target-exclusions.json", V4 / "final-review-exclusions.json"]
    source_files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in files]
    write("cases.json", {"schemaVersion": 1, "suiteId": SUITE, "purpose": "Diagnostic development only; local learning, paraphrase and composition, not final acceptance.",
        "authorship": "Project AI, manually authored from frozen public source before generation; not human review.",
        "counts": {"scenarios": 15, "turns": 15, "trainParaphrases": 6, "trainCompositions": 6, "validationCalibration": 3, "uniqueTrainFamilies": 6, "uniqueValidationFamilies": 3},
        "historyPolicy": "Each case is one independent user turn; no assistant gold text, source gloss, rubric or support injected into input.",
        "scenarios": scenarios})
    write("source-support.json", {"schemaVersion": 1, "suiteId": SUITE, "sources": source_files, "supports": supports,
        "license": "CC-BY-SA-4.0 adaptations of Wikcionario; each support retains revision, history and attribution.",
        "originalWikitextLimit": "Exact bytes are of frozen public lexical/SFT rows. Original wikitext is identified by retained revision/hash; no new retrieval performed."})
    write("novelty-audit.json", {"schemaVersion": 1, "suiteId": SUITE, "checks": checks,
        "all15PromptsAbsentFromAllV4UserMessages": True, "all6InventedNamesAbsentFromV4Train": True,
        "excludedSenseDecisionsChecked": len(prev["excludedSenses"]) + len(final["removedParents"]),
        "familyInterpretation": "The two TRAIN cases per lexeme share a family and trained fact; six families, not twelve independent learned facts. VAL families and exact lexemes absent from v4 TRAIN.",
        "historicalExposureLimit": "VAL exclusion is verified only against this frozen treatment TRAIN. Base weights may have encountered terms/facts in earlier corpora; never claim global unseen knowledge.",
        "review": "Author plus second project-AI source audit; no independent human validation.",
        "modelsLoaded": 0, "modelOutputsRead": 0, "reservedSuiteFilesRead": 0, "trainerPreparedOrExecuted": False})


if __name__ == "__main__":
    main()
