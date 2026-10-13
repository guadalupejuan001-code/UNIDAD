"""Meaningful data-boundary and native supervision checks; no model training."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pattern_curriculum_helper", ROOT / "examples/prepare_conversation_pattern_curriculum.py")
curriculum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum)
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("torch", "tokenizers", "safetensors"))


class PatternDataTests(unittest.TestCase):
    def fixtures(self):
        cases = curriculum.build_cases()
        return cases, curriculum.build_records(cases)

    def test_whole_family_folds_and_informative_contrast_pairs(self):
        cases, records = self.fixtures()
        audit = curriculum.audit_splits(cases, records)
        self.assertEqual(audit["pairsCount"], 18)
        self.assertEqual(len(records), 72)
        self.assertEqual(sum(r["split"] == "train" for r in records), 48)
        self.assertEqual(sum(r["split"] == "validation" for r in records), 24)
        families = {}
        for record in records:
            families.setdefault(record["groupId"], set()).add(record["split"])
        self.assertTrue(all(len(folds) == 1 for folds in families.values()))
        self.assertFalse(any(r["split"] == "test" for r in records))
        self.assertTrue(all(p["sameCurrentRequests"] and p["differentFirstCorrectResponse"] for p in audit["pairs"]))

    def test_detects_fixed_answer_or_changed_request_counterfeits(self):
        for corrupt in ("fixed_target", "same_context", "changed_request"):
            cases, records = self.fixtures()
            if corrupt == "fixed_target":
                cases[1]["expectedResponses"][0] = cases[0]["expectedResponses"][0]
            elif corrupt == "same_context":
                cases[1]["initialMessages"] = copy.deepcopy(cases[0]["initialMessages"])
            else:
                cases[1]["userTurns"][0] += " Cambia la petición."
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, "Contrast pair"):
                curriculum.audit_splits(cases, records)

    def test_detects_cross_fold_family_lexical_and_template_leaks(self):
        for corrupt in ("family", "name", "value", "template"):
            cases, records = self.fixtures()
            held = next(c for c in cases if c["split"] == "validation")
            if corrupt == "family":
                held["groupId"] = cases[0]["groupId"]
            elif corrupt == "name":
                # The normalized audit must catch case and accent variants.
                held["names"][0] = cases[0]["names"][0].upper()
            elif corrupt == "value":
                held["values"][0] = cases[0]["values"][0].upper()
            else:
                held["phraseTemplate"] = list(cases[0]["phraseTemplate"])
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, "Cross-fold"):
                curriculum.audit_splits(cases, records)

    def test_wrong_assistant_is_never_a_positive_prefix_and_cases_match_gold(self):
        cases, records = self.fixtures()
        by_id = {r["id"]: r for r in records}
        for case in cases:
            for turn, identity in enumerate(case["recordIds"]):
                record = by_id[identity]
                self.assertEqual(record["messages"][-1]["content"], case["expectedResponses"][turn])
                for index in case["wrongHistoryAssistantMessageIndices"]:
                    self.assertLess(index, len(record["messages"]) - 1)
                    self.assertEqual(record["messages"][index]["role"], "assistant")
                    self.assertNotEqual(record["messages"][index]["content"], record["messages"][-1]["content"])

    def test_existing_output_rejected_before_parent_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)
            (existing / "keep.txt").write_text("Original.")
            if HAS_EXTRAS:
                with self.assertRaisesRegex(ValueError, "immutable"):
                    curriculum.prepare(existing, existing / "missing-parent")
                self.assertEqual((existing / "keep.txt").read_text(), "Original.")


@unittest.skipUnless(HAS_EXTRAS, "optional conversation dependencies unavailable")
class PatternSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.cases = curriculum.build_cases()
        cls.records = curriculum.build_records(cls.cases)
        cls.tokenizer = fit_tokenizer([r for r in cls.records if r["split"] == "train"], vocab_size=384)

    def test_native_encoder_has_only_exact_last_reply_and_eos(self):
        audit = curriculum.audit_supervision(self.records, self.tokenizer, 2048)
        self.assertEqual(audit["historyPositiveTargets"], 0)
        self.assertEqual(audit["wrongAssistantHistoryOccurrencesIgnored"], 12)
        self.assertTrue(all(r["fullContextRetained"] and r["windows"] == 1 for r in audit["recordAudits"]))
        with self.assertRaisesRegex(ValueError, "fit one native window"):
            curriculum.audit_supervision(self.records, self.tokenizer, 32)

    def test_audit_rejects_accidental_history_labels(self):
        from ia_propia.conversation import training
        original = training._record_windows

        def inject_history_label(*args, **kwargs):
            for inputs, labels in original(*args, **kwargs):
                damaged = list(labels)
                damaged[0] = inputs[0]
                yield inputs, damaged

        with patch.object(training, "_record_windows", inject_history_label):
            with self.assertRaisesRegex(ValueError, "History got accidental positive supervision"):
                curriculum.audit_supervision(self.records[:1], self.tokenizer, 2048)


if __name__ == "__main__":
    unittest.main()
