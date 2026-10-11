"""Quota preparation keeps fixed folds, complete families and train-only audits."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Grounded preparation requires the optional tokenizers dependency")

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import DEFAULT_SYSTEM_PROMPT, fit_tokenizer

HELPER = Path(__file__).resolve().parents[1] / "examples" / "prepare_grounded_curriculum.py"
SPEC = importlib.util.spec_from_file_location("grounded_preparation", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(identifier, split="train", *, kind="authored-synthetic", group=None):
    return {"id": identifier, "groupId": group or "family-" + identifier, "split": split,
            "source": {"dataset": "fixture-" + kind, "kind": kind, "license": "authored fixture"},
            "messages": [{"role": "user", "content": "Pregunta " + identifier},
                         {"role": "assistant", "content": "Respuesta " + identifier}]}


class GroundedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = [record("old-train"), record("old-val", "validation"), record("old-test", "test")]
        self.tokenizer = fit_tokenizer([self.base[0]], 300)
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer_path.write_text(self.tokenizer.to_str(), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def prepare(self, original=None, curriculum=None, *, name="output.jsonl", **kwargs):
        self.input = self.write("input.jsonl", self.base if original is None else original)
        self.curriculum = self.write("new.jsonl", [record("new")] if curriculum is None else curriculum)
        self.output = self.root / name
        return MODULE.prepare(self.input, self.output, curriculum_path=self.curriculum,
                              tokenizer_path=self.tokenizer_path, **kwargs)

    def test_preserves_authored_records_and_whole_human_families_with_stable_caps(self):
        humans = []
        for split in ("train", "validation", "test"):
            for group in range(4):
                for variant in range(2):
                    humans.append(record(f"{split}-{group}-{variant}", split, kind="public-human-reviewed",
                                         group=f"human-{split}-{group}"))
        old = self.base + humans
        report = self.prepare(old, max_human_train=3, max_human_validation=2, max_human_test=0)
        rows = load_corpus(self.output)
        original_by_id = {row["id"]: row for row in old + [record("new")]}
        self.assertTrue(all(row == original_by_id[row["id"]] for row in rows))
        self.assertTrue({row["id"] for row in self.base + [record("new")]} <= {row["id"] for row in rows})
        for group in {row["groupId"] for row in rows if row["source"]["kind"] == "public-human-reviewed"}:
            self.assertEqual([row for row in rows if row["groupId"] == group],
                             sorted([row for row in humans if row["groupId"] == group], key=lambda row: row["id"]))
        self.assertEqual(report["selection"]["humanRecordsSelectedBySplit"], {"train": 2, "validation": 2, "test": 0})
        first = self.output.read_bytes()
        again = self.prepare(list(reversed(old)), name="again.jsonl", max_human_train=3,
                             max_human_validation=2, max_human_test=0)
        self.assertEqual(first, self.output.read_bytes())
        self.assertEqual(report["sha256"], again["sha256"])
        self.assertEqual(report["trainAudit"]["scope"], "train_only")
        self.assertEqual(report["trainAudit"]["records"], report["splits"]["train"])
        for dataset, values in report["trainAudit"]["byDataset"].items():
            train = [row for row in rows if row["split"] == "train" and row["source"]["dataset"] == dataset]
            expected = sum(len(self.tokenizer.encode(row["messages"][-1]["content"]).ids) + 1 for row in train)
            self.assertEqual(values["targetTokens"], expected)
        manifest = json.loads(Path(str(self.root / "output.jsonl") + ".manifest.json").read_text())
        self.assertEqual(report, manifest)
        self.assertEqual(report["sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual((self.root / "output.jsonl").stat().st_mode & 0o777, 0o600)
        self.assertFalse(report["selection"]["modelResponsesOrEvaluationMetricsUsed"])

    def test_normalized_cross_fold_public_prompts_exclude_entire_families(self):
        one = record("ambiguous-train", kind="public-human-reviewed", group="ambiguous-family-train")
        two = record("ambiguous-val", "validation", kind="public-human-reviewed", group="ambiguous-family-val")
        two["messages"][0] = copy.deepcopy(one["messages"][0])
        two["messages"].insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        sibling = record("sibling", kind="public-human-reviewed", group=one["groupId"])
        report = self.prepare(self.base + [one, two, sibling])
        ids = {row["id"] for row in load_corpus(self.output)}
        self.assertTrue({one["id"], two["id"], sibling["id"]}.isdisjoint(ids))
        self.assertEqual(report["crossSplitPromptAudit"]["ambiguousPublicRecords"], 2)
        self.assertEqual(report["crossSplitPromptAudit"]["excludedPublicFamilies"], 2)
        self.assertEqual(report["reasons"]["human_cross_split_prompt_family"], 3)
        self.assertEqual(load_corpus(self.input), self.base + [one, two, sibling])

    def test_union_conflicts_fail_before_overlength_filter_or_quota(self):
        long = record("long-test", "test", kind="public-human-reviewed")
        long["messages"][-1]["content"] = "x" * 1000
        dialogue_copy = record("copy")
        dialogue_copy["messages"] = copy.deepcopy(long["messages"])
        prompt_copy = record("prompt-copy")
        prompt_copy["messages"][0] = copy.deepcopy(self.base[1]["messages"][0])
        cases = [([long], [dialogue_copy]), ([], [prompt_copy]),
                 ([], [{**copy.deepcopy(self.base[0]), "source": {"dataset": "conflict", "kind": "authored-synthetic"}}]),
                 ([], [record("wrong-fold", "test", group=self.base[0]["groupId"])])]
        for extra, curriculum in cases:
            with self.subTest(curriculum=curriculum), self.assertRaises(ValueError):
                self.prepare(self.base + extra, curriculum, max_human_train=0, max_human_test=0)
            self.assertFalse(self.output.exists())
            self.assertFalse(Path(str(self.output) + ".manifest.json").exists())

    def test_full_fit_only_no_truncation_and_human_family_complete(self):
        long = record("long-author")
        long["messages"][-1]["content"] = "x" * 1000
        human_long = record("human-long", kind="public-human-reviewed", group="long-human")
        human_long["messages"][-1]["content"] = "z" * 1000
        sibling = record("human-short", kind="public-human-reviewed", group="long-human")
        report = self.prepare(self.base + [human_long, sibling], [record("new"), long])
        self.assertEqual({row["id"] for row in load_corpus(self.output)}, {row["id"] for row in self.base + [record("new")]})
        self.assertEqual(report["reasons"]["authored_full_dialogue_exceeds_context"], 1)
        self.assertEqual(report["reasons"]["human_family_exceeds_context"], 2)
        self.assertFalse(report["selection"]["dialoguesTruncated"])
        self.assertTrue(all(stats["max"] <= 257 for stats in report["fullDialogueTokensBySplit"].values()))

    def test_development_collision_only_new_authored_fails_without_scoring(self):
        suite = json.loads((HELPER.parent / "conversation-evaluation.json").read_text())
        collision = record("new-dev-collision")
        collision["messages"] = copy.deepcopy(suite["cases"][0]["messages"]) + [{"role": "assistant", "content": "Respuesta ficticia nueva."}]
        development = self.root / "development.json"
        development.write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "development"):
            self.prepare(curriculum=[collision], exclude_development_cases=development)
        self.assertFalse(self.output.exists())
        # An existing record is preserved; the option does not silently rewrite history.
        report = self.prepare(self.base + [collision], [record("new")], name="existing-dev.jsonl",
                              exclude_development_cases=development, context_length=1024)
        self.assertIn(collision, load_corpus(self.output))
        self.assertFalse(report["developmentExclusion"]["scoresUsed"])

    def test_no_overwrite_pair_atomicity_and_concurrent_receipt_preserved(self):
        self.prepare()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.prepare()
        self.assertEqual(before, self.output.read_bytes())
        original_link = MODULE.os.link
        def concurrent(source, destination):
            if str(destination).endswith(".manifest.json"):
                Path(destination).write_bytes(b"external receipt\n")
            return original_link(source, destination)
        with patch.object(MODULE.os, "link", side_effect=concurrent), self.assertRaisesRegex(ValueError, "concurrently"):
            self.prepare(name="raced.jsonl")
        self.assertFalse(self.output.exists())
        self.assertEqual(Path(str(self.output) + ".manifest.json").read_bytes(), b"external receipt\n")
        self.assertEqual(list(self.root.glob(".grounded-*")), [])
