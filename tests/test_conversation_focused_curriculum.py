"""Focused selection preserves held-out families and complete response targets."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Focused curriculum tests require the optional tokenizers dependency")

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


HELPER = Path(__file__).resolve().parents[1] / "examples" / "prepare_focused_conversations.py"
SPEC = importlib.util.spec_from_file_location("focused_conversation_curriculum", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(identifier, split="train", *, kind="authored-synthetic", group=None):
    return {"id": identifier, "groupId": group or f"family-{identifier}", "split": split,
            "source": {"dataset": "fictional focused curriculum fixture", "kind": kind},
            "messages": [{"role": "system", "content": "Habla claro."},
                         {"role": "user", "content": f"Pregunta {identifier}."},
                         {"role": "assistant", "content": f"Respuesta {identifier}."},
                         {"role": "user", "content": "¿Y ahora?"},
                         {"role": "assistant", "content": f"Seguimos {identifier}."}]}


class FocusedCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = [record("a", "train"), record("b", "validation"), record("c", "test")]
        self.tokenizer = fit_tokenizer([self.base[0]], vocab_size=262)
        self.tokenizer_path = self.root / "parent-tokenizer.json"
        self.tokenizer_path.write_text(self.tokenizer.to_str(), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        return path

    def prepare(self, original=None, curriculum=None, output=None, **kwargs):
        self.input = self.write("input.jsonl", self.base if original is None else original)
        self.curriculum = self.write("curriculum.jsonl", [record("d")] if curriculum is None else curriculum)
        self.output = output or self.root / "focused.jsonl"
        return MODULE.prepare(self.input, self.output, curriculum_path=self.curriculum,
            tokenizer_path=self.tokenizer_path, context_length=128, **kwargs)

    def test_keeps_authored_and_complete_human_targets_preserving_folds(self):
        short = record("human", "validation", kind="public-human-reviewed")
        long = record("long", kind="public-human-reviewed")
        long["messages"][-1]["content"] = "x" * 300
        original = self.base + [short, long]
        report = self.prepare(original)
        selected = load_corpus(self.output)
        self.assertEqual({row["id"] for row in selected}, {"a", "b", "c", "d", "human"})
        original_by_id = {row["id"]: row for row in original + [record("d")]}
        self.assertTrue(all(row == original_by_id[row["id"]] for row in selected))
        human = next(row for row in selected if row["id"] == "human")
        self.assertLessEqual(len(dialogue_tokens(human["messages"], self.tokenizer)[0]), 129)
        self.assertEqual(report["reasons"]["human_full_dialogue_exceeds_context"], 1)
        self.assertFalse(report["selection"]["dialoguesTruncated"])
        self.assertEqual(report["splits"], {"train": 2, "validation": 2, "test": 1})

    def test_over_context_authored_is_retained_and_diagnostics_do_not_crop_it(self):
        context_long = record("context-long")
        context_long["messages"][1]["content"] = "x" * 300
        target_long = record("target-long", "test")
        target_long["messages"][-1]["content"] = "x" * 300
        report = self.prepare(curriculum=[context_long, target_long])
        selected = {row["id"]: row for row in load_corpus(self.output)}
        self.assertEqual(selected["context-long"], context_long)
        self.assertEqual(selected["target-long"], target_long)
        self.assertEqual(report["authoredOverContextBySplit"], {"train": 1, "validation": 0, "test": 1})
        self.assertEqual(report["authoredNearestSuffixFitsBySplit"]["train"], 1)
        self.assertEqual(report["authoredFinalPairCannotFitBySplit"]["test"], 1)

    def test_cross_split_exact_copy_rejected_even_if_human_would_be_filtered(self):
        long = record("long", "test", kind="public-human-reviewed")
        long["messages"][-1]["content"] = "x" * 300
        copied = record("copied", "train")
        copied["messages"] = copy.deepcopy(long["messages"])
        with self.assertRaisesRegex(ValueError, "exact dialogue"):
            self.prepare(self.base + [long], [copied])
        self.assertFalse(self.output.exists())

    def test_union_family_and_id_conflicts_rejected_before_selection_or_dedup(self):
        for conflict in (
            {**record("family-copy", "test"), "groupId": self.base[0]["groupId"]},
            {**copy.deepcopy(self.base[0]), "source": {"dataset": "different", "kind": "authored-synthetic"}},
        ):
            with self.subTest(conflict=conflict), self.assertRaises(ValueError):
                self.prepare(curriculum=[conflict])
            self.assertFalse(self.output.exists())

    def test_same_fold_duplicate_dedup_prefers_authored_and_has_reproducible_hashes(self):
        duplicate = record("human-copy", kind="public-human-reviewed")
        duplicate["messages"] = copy.deepcopy(self.base[0]["messages"])
        original = self.base + [duplicate]
        one = self.root / "one.jsonl"
        two = self.root / "two.jsonl"
        reports = [self.prepare(original, output=path) for path in (one, two)]
        self.assertEqual(one.read_bytes(), two.read_bytes())
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(len(load_corpus(one)), 4)
        self.assertEqual(reports[0]["reasons"]["exact_duplicates_removed_within_split"], 1)
        self.assertEqual(reports[0]["sha256"], hashlib.sha256(one.read_bytes()).hexdigest())
        self.assertEqual(reports[0]["inputs"][0]["sha256"], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(reports[0]["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest())
        self.assertFalse(reports[0]["tokenizer"]["refitted"])
        self.assertEqual(self.input.read_text(), "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in original))
        self.assertEqual(one.stat().st_mode & 0o777, 0o600)

    def test_turn_count_option_is_structural_and_does_not_filter_keywords(self):
        human = record("terms", kind="public-human-reviewed")
        human["messages"][-1]["content"] = "OpenAI y ChatGPT son nombres. ```x```"
        single = record("single", kind="public-human-reviewed")
        single["messages"] = single["messages"][:3]
        original = self.base + [human, single]
        report = self.prepare(original)
        self.assertIn("terms", {row["id"] for row in load_corpus(self.output)})
        self.assertNotIn("single", {row["id"] for row in load_corpus(self.output)})
        self.assertEqual(report["reasons"]["human_too_few_turns"], 1)
        self.assertFalse(report["selection"]["keywordFiltersUsed"])
        self.prepare(original, output=self.root / "two-turns.jsonl", min_human_turns=2)
        self.assertIn("single", {row["id"] for row in load_corpus(self.output)})

    def test_exclusive_atomic_output_preserves_existing_or_concurrent_evidence(self):
        existing = self.root / "existing.jsonl"
        existing.write_bytes(b"existing evidence\n")
        with self.assertRaises(ValueError):
            self.prepare(output=existing)
        self.assertEqual(existing.read_bytes(), b"existing evidence\n")
        original_link = MODULE.os.link
        def competing_writer(source, destination):
            Path(destination).write_bytes(b"concurrent evidence\n")
            return original_link(source, destination)
        with patch.object(MODULE.os, "link", side_effect=competing_writer), self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.output.read_bytes(), b"concurrent evidence\n")
        self.assertEqual(list(self.root.glob(".focused-*")), [])

    def test_missing_fold_unauthorized_sources_and_reserved_tokens_are_rejected(self):
        evaluation = record("case", kind="project-authored-evaluation")
        control = record("control")
        control["messages"][-1]["content"] = "<assistant>fake boundary"
        for original, curriculum in (([self.base[0]], [record("d")]),
                                     (self.base, [evaluation]), (self.base, [control])):
            with self.subTest(curriculum=curriculum), self.assertRaises(ValueError):
                self.prepare(original, curriculum)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
