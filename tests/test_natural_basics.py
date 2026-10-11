"""Audit fictional basic dialogue authoring without models or private corpora."""

from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unicodedata
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Natural-basics authoring tests require optional tokenizers")

from examples import build_natural_basics as GENERATOR
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


class NaturalBasicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {"seed": 413, "episodes_per_skill": 20, "context_length": 4096}
        cls.templates = GENERATOR.templates()
        cls.template_ids = {pattern["templateId"] for bank in cls.templates.values() for pattern in bank}
        # This independent train fixture supplies a complete byte alphabet.
        # No existing tokenizer, holdout, evaluation case or checkpoint is read.
        fixture = [{"split": "train", "messages": [
            {"role": "user", "content": "Hola. ¿Podemos conversar?"},
            {"role": "assistant", "content": "Sí. Cuéntame algo."},
        ]}]
        before = copy.deepcopy(fixture)
        cls.tokenizer = fit_tokenizer(fixture, vocab_size=262)
        if fixture != before:
            raise AssertionError("Tokenizer fitting mutated its independent fixture")
        cls.tokenizer_raw = cls.tokenizer.to_str().encode("utf-8")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cls.tokenizer_path = root / "fixture-tokenizer.json"
        cls.tokenizer_path.write_bytes(cls.tokenizer_raw)
        cls.output = root / "basics.jsonl"
        cls.manifest = root / "basics.manifest.json"
        cls.report = GENERATOR.build(cls.tokenizer_path, cls.output, cls.manifest, **cls.options)
        cls.rows = load_corpus(cls.output)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, output=None, manifest=None):
        return GENERATOR.build(self.tokenizer_path, output or self.root / "new.jsonl",
                               manifest or self.root / "new.manifest.json", **self.options)

    def world(self):
        values = GENERATOR._world(self.options["seed"], "train", 0)
        values.update(name="NombreLatente", alias="ApodoElegido", new_name="NombreCorregido",
                      activity="leer", other_activity="cocinar",
                      title="Proyecto Inicial", other_title="Proyecto Actual")
        return values

    def test_cold_prefix_dedup_retains_one_prototype_per_pattern_in_every_fold(self):
        cold = [row for row in self.rows if row["source"]["prefixPairs"] == 1]
        counts = Counter((row["source"]["templateId"], row["split"]) for row in cold)
        self.assertEqual(len(self.template_ids), 20)
        self.assertEqual(set(counts), {(pattern, split) for pattern in self.template_ids
                                       for split in GENERATOR.SPLITS})
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(Counter(row["split"] for row in cold),
                         Counter({split: 20 for split in GENERATOR.SPLITS}))
        self.assertGreater(self.report["reasons"]["within_fold_duplicate_prefixes_removed"], 0)
        # A generic opening must not erase the personalized continuation of
        # every other episode sharing that opening in its own fold.
        continuations = defaultdict(set)
        for row in self.rows:
            if row["source"]["prefixPairs"] == 3:
                continuations[(row["source"]["templateId"], row["split"])].add(row["groupId"])
        self.assertEqual(set(continuations), set(counts))
        self.assertTrue(all(len(groups) >= 2 for groups in continuations.values()))

    def test_full_prompts_and_episode_groups_do_not_cross_folds(self):
        groups, prompts, targets, patterns = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
        exact = set()
        for row in self.rows:
            groups[row["groupId"]].add(row["split"])
            normalized = tuple((message["role"], " ".join(unicodedata.normalize(
                "NFKC", message["content"]).casefold().split())) for message in row["messages"][:-1])
            prompts[normalized].add(row["split"])
            targets[normalized].add(row["messages"][-1]["content"])
            patterns[row["source"]["templateId"]].add(row["split"])
            key = GENERATOR.canonical(row["messages"])
            self.assertNotIn(key, exact)
            exact.add(key)
            self.assertEqual(row["source"]["coldOpeningFold"], row["split"])
            self.assertEqual(row["source"]["kind"], "authored-synthetic")
            self.assertFalse(row["source"]["privateConversationUsed"])
            self.assertFalse(row["source"]["modelResponsesUsed"])
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))
        self.assertTrue(all(len(folds) == 1 for folds in prompts.values()))
        self.assertTrue(all(len(values) == 1 for values in targets.values()))
        self.assertTrue(all(folds == set(GENERATOR.SPLITS) for folds in patterns.values()))

    def test_cross_fold_prompt_is_rejected_before_context_filtering(self):
        train = copy.deepcopy(GENERATOR.generate(self.options["seed"], 20)[0])
        heldout = copy.deepcopy(train)
        heldout.update(id="injected-validation-record", groupId="injected-validation-family", split="validation")
        with patch.object(GENERATOR, "generate", return_value=[train, heldout]), \
                patch.object(GENERATOR, "dialogue_tokens", return_value=([0] * 4098, None)):
            with self.assertRaisesRegex(ValueError, "full prompt occurs across folds"):
                self.build()
        self.assertFalse((self.root / "new.jsonl").exists())
        self.assertFalse((self.root / "new.manifest.json").exists())

    def test_conflicting_targets_are_rejected_before_dedup_or_context_filtering(self):
        original = copy.deepcopy(GENERATOR.generate(self.options["seed"], 20)[0])
        conflict = copy.deepcopy(original)
        conflict["id"] = "injected-conflicting-record"
        conflict["messages"][-1]["content"] = "Una respuesta contradictoria."
        with patch.object(GENERATOR, "generate", return_value=[original, conflict]), \
                patch.object(GENERATOR, "dialogue_tokens", return_value=([0] * 4098, None)):
            with self.assertRaisesRegex(ValueError, "conflicting authored targets"):
                self.build()
        self.assertFalse((self.root / "new.jsonl").exists())

    def test_historical_prompt_occupancy_is_checked_before_length_filtering(self):
        candidate = GENERATOR.generate(self.options["seed"], 20)[0]
        self.assertEqual(candidate["split"], "train")
        audit_path = self.root / "occupied-prompts.json"
        for old_splits in (("test",), ("train", "test")):
            with self.subTest(occupied=old_splits):
                audit = {"purpose": "data_provenance_collision_audit_no_model_outputs",
                         "collisions": [{"messages": candidate["messages"][:-1],
                                         "old": [{"split": split} for split in old_splits]}]}
                audit_raw = GENERATOR.canonical(audit) + b"\n"
                audit_path.write_bytes(audit_raw)
                with patch.object(GENERATOR, "dialogue_tokens", return_value=([0] * 4098, None)):
                    with self.assertRaisesRegex(ValueError, "conflicts with an occupied split"):
                        GENERATOR.build(self.tokenizer_path, self.root / "new.jsonl",
                                        self.root / "new.manifest.json", **self.options,
                                        collision_audit_path=audit_path)
                self.assertEqual(audit_path.read_bytes(), audit_raw)
                self.assertFalse((self.root / "new.jsonl").exists())
                self.assertFalse((self.root / "new.manifest.json").exists())

    def test_latent_user_names_are_not_invented_and_alias_is_not_a_real_name(self):
        world = self.world()
        for bank in self.templates.values():
            for pattern in bank:
                for split in GENERATOR.SPLITS:
                    with self.subTest(pattern=pattern["templateId"], split=split):
                        messages = GENERATOR.render(pattern, world, split)
                        stated = ""
                        for user, reply in zip(messages[::2], messages[1::2]):
                            stated += " " + user["content"]
                            for field in ("name", "alias", "new_name"):
                                if world[field] in reply["content"]:
                                    self.assertIn(world[field], stated)
        for pattern in self.templates["unknown_user_name"]:
            messages = GENERATOR.render(pattern, world, "train")
            for field in ("name", "alias", "new_name"):
                self.assertNotIn(world[field], messages[1]["content"])
        alias_pattern = next(pattern for pattern in self.templates["self_identity"]
                             if pattern["templateId"] == "identity-alias")
        messages = GENERATOR.render(alias_pattern, world, "train")
        self.assertIn("No has dicho tu nombre real", messages[-1]["content"])
        self.assertIn(world["alias"], messages[-1]["content"])
        self.assertNotIn(world["name"], messages[-1]["content"])

    def test_name_and_goal_changes_follow_the_latest_user_fact(self):
        world = self.world()
        checks = {"unknown-name-introduced-corrected": ("name", "new_name"),
                  "unknown-no-guess": ("name", "new_name"),
                  "greeting-start": ("activity", "other_activity"),
                  "goal-activity-change": ("activity", "other_activity"),
                  "goal-project-change": ("title", "other_title"),
                  "goal-clarify-then-switch": ("title", "other_title")}
        patterns = {pattern["templateId"]: pattern for bank in self.templates.values() for pattern in bank}
        for name, (old, current) in checks.items():
            for split in GENERATOR.SPLITS:
                with self.subTest(pattern=name, split=split):
                    messages = GENERATOR.render(patterns[name], world, split)
                    self.assertIn(world[old], messages[2]["content"])
                    self.assertIn(world[current], messages[-2]["content"])
                    self.assertIn(world[current], messages[-1]["content"])
                    self.assertNotIn(world[old], messages[-1]["content"])

    def test_generation_is_reproducible_full_fit_and_inputs_remain_unchanged(self):
        raw = self.output.read_bytes()
        tokenizer_before = self.tokenizer_path.read_bytes()
        report = self.build()
        self.assertEqual((self.root / "new.jsonl").read_bytes(), raw)
        self.assertEqual(report, self.report)
        self.assertEqual(self.tokenizer_path.read_bytes(), tokenizer_before)
        self.assertEqual(report["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["tokenizer"]["sha256"], hashlib.sha256(tokenizer_before).hexdigest())
        self.assertFalse(report["tokenizer"]["refitted"])
        self.assertFalse(report["authoring"]["dialoguesTruncated"])
        for row in self.rows:
            tokens, _ = dialogue_tokens(row["messages"], self.tokenizer)
            self.assertLessEqual(len(tokens), self.options["context_length"] + 1)
        before = copy.deepcopy(self.templates)
        self.assertEqual(GENERATOR.generate(self.options["seed"], 20),
                         GENERATOR.generate(self.options["seed"], 20))
        self.assertEqual(self.templates, before)

    def test_frozen_outputs_and_concurrent_writer_are_preserved(self):
        before = {path: path.read_bytes() for path in (self.output, self.manifest, self.tokenizer_path)}
        with self.assertRaisesRegex(ValueError, "Existing outputs are immutable"):
            self.build(self.output, self.manifest)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        other_manifest = self.root / "already-frozen.manifest.json"
        other_manifest.write_bytes(b"frozen evidence\n")
        with self.assertRaisesRegex(ValueError, "Existing outputs are immutable"):
            self.build(manifest=other_manifest)
        self.assertEqual(other_manifest.read_bytes(), b"frozen evidence\n")
        self.assertFalse((self.root / "new.jsonl").exists())
        output = self.root / "race.jsonl"

        def race(_source, target):
            Path(target).write_bytes(b"concurrent evidence\n")
            raise FileExistsError("another writer won")

        with patch.object(GENERATOR.os, "link", side_effect=race):
            with self.assertRaisesRegex(ValueError, "Another writer created output"):
                GENERATOR.exclusive_write(output, b"replacement\n")
        self.assertEqual(output.read_bytes(), b"concurrent evidence\n")
        self.assertFalse(list(self.root.glob(".natural-basics-*")))


if __name__ == "__main__":
    unittest.main()
