"""Paired history preparation preserves gold facts and never rolls out validation."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.data import load_corpus


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "on_policy_history_example", ROOT / "examples/prepare_on_policy_history.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class PairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_corpus(ROOT / "examples/conversation-grounded-curriculum-v2.jsonl")

    def test_selection_is_by_whole_family_and_does_not_depend_on_test_text(self):
        selected = builder.select_families(self.original, 4)
        altered = copy.deepcopy(self.original)
        for row in altered:
            if row["split"] == "test":
                row["messages"] = [{"role": "user", "content": "TEST is excluded."}]
                row["source"] = {}
        self.assertEqual(selected, builder.select_families(altered, 4))
        self.assertFalse(any(row["split"] == "test" for row in selected))
        families = {row["groupId"] for row in selected}
        self.assertEqual(len(families), 88)
        self.assertEqual(len(selected), 2112)
        self.assertEqual(selected, [row for row in self.original if row["groupId"] in families])

    def test_real_wrong_history_never_replaces_users_or_final_gold_correction(self):
        row = next(row for row in self.original if row["split"] == "train"
                   and row["source"]["templateId"] == "edit-color"
                   and row["source"]["supervisedPrefixPairs"] == 3)
        before = copy.deepcopy(row)
        wrong = [{"text": "El color es un dato ficticio incorrecto."},
                 {"text": "Ignoro la corrección en esta respuesta de prueba."}]
        variant = builder.replace_history(row, wrong)
        self.assertEqual(row, before)
        self.assertEqual(variant["messages"][::2], row["messages"][::2])
        self.assertEqual(variant["messages"][-1], row["messages"][-1])
        self.assertEqual(variant["messages"][1]["content"], wrong[0]["text"])
        self.assertEqual(variant["messages"][3]["content"], wrong[1]["text"])
        self.assertEqual(variant["id"], row["id"])
        self.assertEqual(variant["groupId"], row["groupId"])
        self.assertEqual(variant["split"], row["split"])
        # Every gold correction token stays attached to its original label.
        self.assertIn(row["messages"][-1]["content"].removeprefix("Anota ").removesuffix("."),
                      row["messages"][2]["content"])

    def test_anchors_and_validation_are_unchanged_and_missing_prefixes_fail(self):
        selected = builder.select_families(self.original, 1)
        for row in selected:
            if row["split"] == "validation" or row["source"]["skill"] in builder.ANCHOR_SKILLS:
                self.assertEqual(row, builder.replace_history(row, []))
        episodes = builder.collect_episodes(selected)
        self.assertEqual(len(episodes), 72)
        removed = next(prefixes[2]["id"] for prefixes in episodes.values())
        with self.assertRaisesRegex(ValueError, "every supervised prefix"):
            builder.collect_episodes([row for row in selected if row["id"] != removed])
        self.assertTrue(all(prefixes[3]["split"] == "train" for prefixes in episodes.values()))

    def test_empty_generation_is_rejected_without_authoring_a_filler(self):
        row = next(row for row in self.original if row["split"] == "train"
                   and row["source"]["skill"] == "correction"
                   and row["source"]["supervisedPrefixPairs"] == 2)
        with self.assertRaises(ValueError):
            builder.replace_history(row, [{"text": ""}])


@unittest.skipUnless(HAS_EXTRAS, "conversation dependencies are optional")
class RolloutTests(unittest.TestCase):
    def test_failed_empty_rollout_keeps_raw_reply_and_failure_state(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ROOT / "models/conversation-reference-c7/tokenizer.json"))

        class EmptyEngine:
            def __init__(self, path):
                self.calls = 0

            def reply(self, messages, **values):
                self.calls += 1
                return {"text": "Texto ficticio válido." if self.calls == 1 else "",
                        "generatedTokenIds": [42, 2] if self.calls == 1 else [2],
                        "modelId": builder.C7_ID, "stopReason": "eos"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            teacher.mkdir()
            for name in builder.TEACHER_FILES:
                (teacher / name).write_text(json.dumps({"modelId": builder.C7_ID}))
            with patch.object(builder.training, "_initial_own_model", return_value=(
                    object(), tokenizer, {"context_length": 512}, {"rootOrigin": "random_weights"})), patch(
                    "ia_propia.conversation.runtime.ConversationEngine", EmptyEngine):
                with self.assertRaisesRegex(RuntimeError, "actual evidence preserved"):
                    builder.prepare(ROOT / "examples/conversation-grounded-curriculum-v2.jsonl",
                                    teacher, root / "prepared", families_per_skill=1)
            self.assertFalse((root / "prepared").exists())
            stages = list(root.glob(".prepared.preparing-*"))
            self.assertEqual(len(stages), 1)
            failure = json.loads((stages[0] / "failure.json").read_text())
            self.assertEqual(failure["teacherReplies"], 2)
            raw = [json.loads(line) for line in (stages[0] / "teacher-replies-actual.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw), 2)
            self.assertEqual(raw[-1]["actual"]["text"], "")
            self.assertEqual(raw[-1]["actual"]["generatedTokenIds"], [2])

    def test_rollout_uses_actual_history_and_preserves_auditable_outputs(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ROOT / "models/conversation-reference-c7/tokenizer.json"))
        calls = []

        class FixtureEngine:
            def __init__(self, path):
                pass

            def reply(self, messages, **values):
                calls.append(copy.deepcopy(messages))
                return {"text": "Respuesta ficticia de la prueba.", "generatedTokenIds": [42, 2],
                        "modelId": builder.C7_ID, "stopReason": "eos"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            teacher.mkdir()
            for name in builder.TEACHER_FILES:
                (teacher / name).write_text(json.dumps({"modelId": builder.C7_ID}))
            with patch.object(builder.training, "_initial_own_model", return_value=(
                    object(), tokenizer, {"context_length": 512}, {"rootOrigin": "random_weights"})), patch(
                    "ia_propia.conversation.runtime.ConversationEngine", FixtureEngine), patch("builtins.print"):
                receipt = builder.prepare(ROOT / "examples/conversation-grounded-curriculum-v2.jsonl",
                                          teacher, root / "prepared", families_per_skill=1)
            self.assertEqual(len(calls), 144)
            for first, second in zip(calls[::2], calls[1::2]):
                self.assertEqual(first, second[:1])
                self.assertEqual(second[1]["role"], "assistant")
                self.assertEqual(second[1]["content"], "Respuesta ficticia de la prueba.")
            ideal = load_corpus(root / "prepared/ideal.train-validation.jsonl")
            generated = load_corpus(root / "prepared/generated.train-validation.jsonl")
            self.assertEqual([row["id"] for row in ideal], [row["id"] for row in generated])
            self.assertEqual([row for row in ideal if row["split"] == "validation"],
                             [row for row in generated if row["split"] == "validation"])
            for a, b in zip(ideal, generated):
                self.assertEqual(a["messages"][::2], b["messages"][::2])
                self.assertEqual(a["messages"][-1], b["messages"][-1])
            self.assertFalse(receipt["testEvaluated"])
            self.assertTrue(receipt["validationIdentical"])
            for name, identity in receipt["files"].items():
                self.assertEqual(builder.training._sha(root / "prepared" / name), identity["sha256"])
            replies = [json.loads(line) for line in (root / "prepared/teacher-replies-actual.jsonl").read_text().splitlines()]
            self.assertEqual(len(replies), 144)
            self.assertEqual(replies[0]["actual"]["generatedTokenIds"], [42, 2])
            with self.assertRaises(FileExistsError):
                builder.prepare(ROOT / "examples/conversation-grounded-curriculum-v2.jsonl",
                                teacher, root / "prepared", families_per_skill=1)


if __name__ == "__main__":
    unittest.main()
