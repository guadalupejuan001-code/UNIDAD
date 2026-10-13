"""Critical data contracts; no tokenizer, model, or reserved evaluation imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HELPER_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_natural_latam_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_natural_latam_sft", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def sense(country, meaning, usage=None, qualifier=""):
    return {"etymology": 1, "senseNumber": "1", "partOfSpeech": "sustantivo",
            "definition": meaning, "scope": [{"label": country, "qualifier": qualifier}],
            "usage": usage or [], "examples": []}


def parent(title="guagua", split="train", senses=None):
    return {"id": "fixture:" + title, "groupId": "lexical-family:" + title,
            "split": split, "text": "fixture",
            "source": {"title": title, "dataset": "Wikcionario/es",
                       "license": "CC-BY-SA-4.0", "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
                       "pageId": 1, "revisionId": 2, "revisionTimestamp": "2026-10-13T00:00:00Z",
                       "revisionUrl": "https://es.wiktionary.org/w/index.php?oldid=2",
                       "historyUrl": "https://es.wiktionary.org/w/index.php?title=fixture&action=history",
                       "wikitextSha256": "a" * 64, "spanishSectionSha256": "b" * 64,
                       "attribution": "Wikcionario contributors",
                       "regionalSenses": senses or [sense("Chile", "Niño pequeño."),
                                                     sense("Cuba", "Autobús.")]}}


class NaturalLatamDataContracts(unittest.TestCase):
    def test_six_frozen_source_country_senses_and_usage_golden_cases(self):
        repository = HELPER_PATH.parents[1]
        source_path = repository / "examples/learning-repair-20261013/new-sources/latin-american-spanish-v1/definitions/latin-american-wiktionary.train-validation.jsonl"
        golden_path = repository / "examples/learning-repair-20261013/natural-latam-sft-v2/golden-country-senses.json"
        golden = json.loads(golden_path.read_text())
        raw = source_path.read_bytes()
        self.assertEqual(HELPER.digest(raw), golden["sourceCorpusSha256"])
        entries = {row["source"]["title"]: row for row in map(json.loads, raw.splitlines())}
        for case in golden["cases"]:
            with self.subTest(title=case["title"], country=case["country"]):
                matches = [s for s in entries[case["title"]]["source"]["regionalSenses"]
                           if s["etymology"] == case["etymology"] and s["senseNumber"] == case["senseNumber"]
                           and any(q[0] == case["country"] for q in HELPER.country_scopes(s))]
                self.assertEqual(len(matches), 1)
                target = HELPER.answer(case["title"], case["country"], "", matches[0])
                self.assertIn(case["contains"], target)
                if case.get("mustNotContain"):
                    self.assertNotIn(case["mustNotContain"], target)
                if case.get("requiredUsage"):
                    self.assertIn(case["requiredUsage"], target)

    def test_country_sense_correction_does_not_transfer_other_country_meaning(self):
        rows, _ = HELPER.generate_regionals([parent()], "fixture")
        corrections = [r for r in rows if r["source"]["behavior"] == "follow-country-correction"]
        self.assertEqual(len(corrections), 1)
        row = corrections[0]
        question, target = row["messages"][-2]["content"], row["messages"][-1]["content"]
        if "era Cuba" in question:
            self.assertIn("Autobús", target)
            self.assertNotIn("Niño pequeño", target)
        else:
            self.assertIn("era Chile", question)
            self.assertIn("Niño pequeño", target)
            self.assertNotIn("Autobús", target)

    def test_usage_warning_and_local_qualifier_survive_requested_format(self):
        entry = parent("china", senses=[sense("Bolivia", "Mujer indígena.",
            [{"label": "despectivo", "qualifier": "antaño no lo era"}], "Tarija")])
        rows, _ = HELPER.generate_regionals([entry], "fixture")
        row = next(r for r in rows if r["source"]["behavior"] == "requested-format")
        target = row["messages"][-1]["content"]
        self.assertIn("Tarija", target)
        self.assertIn("despectivo (antaño no lo era)", target)

    def test_america_and_page_categories_never_expand_to_unlisted_countries(self):
        entry = parent("fixture", senses=[sense("América", "Un sentido continental.")])
        entry["source"]["pageCategories"] = ["Vocabulario de México"]
        rows, exclusions = HELPER.generate_regionals([entry], "fixture")
        self.assertEqual(rows, [])
        self.assertEqual(exclusions[0]["reason"], "no-explicit-supported-country-sense")

    def test_one_source_meaning_does_not_force_an_ambiguity_question(self):
        rows, _ = HELPER.generate_regionals([parent("birome", senses=[sense("Argentina", "Bolígrafo.")])], "fixture")
        behaviors = {r["source"]["behavior"] for r in rows}
        self.assertNotIn("genuine-ambiguity", behaviors)
        self.assertNotIn("missing-information", behaviors)

    def test_semantic_qa_excludes_bad_sense_without_rewriting_source_or_good_sense(self):
        entry = parent("fixture", senses=[sense("Chile", "fruto de la fixture(3)."),
                                         {**sense("Chile", "Planta comestible."), "senseNumber": "2"}])
        original = copy.deepcopy(entry)
        exclusion = {"parentRecordId": entry["id"], "etymology": 1,
                     "senseNumber": "1", "reason": "Unresolved numbered reference"}
        rows, excluded = HELPER.generate_regionals([entry], "fixture", [exclusion])
        self.assertEqual(entry, original)
        self.assertTrue(rows)
        self.assertTrue(any(e["reason"] == exclusion["reason"] for e in excluded))
        self.assertTrue(all("fixture(3)" not in r["messages"][-1]["content"] for r in rows))
        self.assertTrue(all("Planta comestible" in r["messages"][-1]["content"] for r in rows))

    def test_authored_wrong_answer_is_never_a_positive_last_response(self):
        rows, _ = HELPER.generate_regionals([parent()], "fixture")
        self.assertTrue(any("Se encuentra junto a la plaza." in
                            [m["content"] for m in r["messages"][:-1]] for r in rows))
        for row in rows:
            self.assertNotEqual(row["messages"][-1]["content"], "Se encuentra junto a la plaza.")
            self.assertEqual(row["source"]["supervisionPolicy"], "assistant_last_reply")

    def test_preserves_family_and_source_license_without_mutating_input(self):
        entry = parent(split="validation")
        before = copy.deepcopy(entry)
        rows, _ = HELPER.generate_regionals([entry], "fixture")
        self.assertEqual(entry, before)
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["groupId"], entry["groupId"])
            self.assertEqual(row["split"], "validation")
            self.assertEqual(row["source"]["provenance"]["revisionId"], 2)
            self.assertEqual(row["source"]["license"], "CC-BY-SA-4.0")

    def test_train_selection_is_whole_family_and_validation_is_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, regional, design = root / "prior.jsonl", root / "regional.jsonl", root / "design.json"
            records = []
            for template in range(40):
                for world in range(64):
                    for prefix in (1, 2):
                        family = f"family-{template}-{world}"
                        records.append({"id": f"{family}:{prefix}", "groupId": family,
                            "split": "train", "source": {"dataset": "unidad-composition-v1", "templateId": str(template)},
                            "messages": [{"role": "user", "content": f"{family} pregunta {prefix}"},
                                         {"role": "assistant", "content": "Respuesta."}]})
            raw_validation = b'{ "id":"untouched-validation", "groupId":"old-val", "split":"validation", "source":{"dataset":"manual"}, "messages":[{"role":"user","content":"Original?"},{"role":"assistant","content":"Exact bytes."}] }\n'
            prior.write_bytes(b"".join(HELPER.encoded(r) for r in records) + raw_validation)
            regional.write_bytes(HELPER.encoded(parent()))
            design.write_text(json.dumps({"status": "specified-before-data-generation"}))
            with patch.object(HELPER, "PRIOR_SHA", HELPER.digest(prior.read_bytes())), patch.object(HELPER, "REGIONAL_SHA", HELPER.digest(regional.read_bytes())):
                manifest = HELPER.prepare(prior, regional, root / "output", "fixture", 600, design)
            control_bytes = (root / "output/natural-control.train-validation.jsonl").read_bytes()
            self.assertTrue(control_bytes.endswith(raw_validation))
            selected = [json.loads(line) for line in control_bytes.splitlines()]
            selected_families = {}
            for row in selected[:-1]:
                selected_families.setdefault(row["groupId"], set()).add(row["id"])
            self.assertEqual(len(selected_families), 400)
            self.assertTrue(all(len(ids) == 2 for ids in selected_families.values()))
            self.assertEqual(manifest["priorValidationRowsPreservedByteExactly"], 1)
            self.assertEqual(prior.read_bytes().splitlines(keepends=True)[-1], raw_validation)


if __name__ == "__main__":
    unittest.main()
