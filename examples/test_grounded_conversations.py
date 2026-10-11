"""Check fictional curriculum boundaries without loading any existing corpus."""

from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Grounded curriculum tests require optional tokenizers")

try:
    import build_grounded_conversations as GENERATOR
except ModuleNotFoundError as exc:
    if exc.name != "build_grounded_conversations":
        raise
    from examples import build_grounded_conversations as GENERATOR

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class GroundedConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {"seed": 1701, "conversation_episodes": 10, "fact_episodes": 10}
        cls.templates = {**GENERATOR.CONVERSATION_TEMPLATES, **GENERATOR.FACT_TEMPLATES}
        cls.template_ids = {
            template["templateId"] for templates in cls.templates.values() for template in templates
        }
        cls.records = GENERATOR.generate(**cls.options)
        # The only tokenizer fitting input is this independent fictional train
        # fixture. Neither generated validation/test nor an existing tokenizer
        # or corpus is used to fit it.
        cls.training_fixture = [{
            "id": "tokenizer-fixture", "groupId": "tokenizer-fixture-family", "split": "train",
            "messages": [{"role": "user", "content": "Hola. ¿Podemos conversar en español?"},
                         {"role": "assistant", "content": "Sí, podemos conversar."}],
        }]
        original = copy.deepcopy(cls.training_fixture)
        cls.tokenizer = fit_tokenizer(cls.training_fixture, vocab_size=262)
        if cls.training_fixture != original:
            raise AssertionError("Tokenizer fitting mutated its training fixture")
        cls.tokenizer_raw = cls.tokenizer.to_str().encode("utf-8")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer_path = self.root / "fixture-tokenizer.json"
        self.tokenizer_path.write_bytes(self.tokenizer_raw)

    def build(self, output=None, manifest=None):
        return GENERATOR.build(
            self.tokenizer_path,
            output or self.root / "curriculum.jsonl",
            manifest or self.root / "manifest.json",
            context_length=4096,
            **self.options,
        )

    def world(self):
        return {
            "name": "Hugo", "friend": "Nuria", "item": "cuaderno", "other_item": "mapa",
            "place": "Quito", "other_place": "Soria", "day": "lunes", "other_day": "jueves",
            "count": 3, "other_count": 7, "color": "rojo", "other_color": "azul",
            "activity": "leer", "other_activity": "pasear", "feeling": "preocupado",
            "reason": "se atrasó el pedido",
        }

    def test_every_template_and_all_supervised_prefixes_exist_in_every_fold(self):
        coverage = defaultdict(set)
        identifiers = set()
        for record in self.records:
            self.assertNotIn(record["id"], identifiers)
            identifiers.add(record["id"])
            source = record["source"]
            coverage[source["templateId"]].add(record["split"])
            self.assertEqual(len(record["messages"]), 2 * source["supervisedPrefixPairs"])
            self.assertEqual([message["role"] for message in record["messages"]],
                             ["user", "assistant"] * source["supervisedPrefixPairs"])
            self.assertTrue(source["fictional"])
            self.assertFalse(source["privateConversationUsed"])
            self.assertFalse(source["modelResponsesUsed"])
        self.assertEqual(set(coverage), self.template_ids)
        for template_id, folds in coverage.items():
            with self.subTest(template=template_id):
                self.assertEqual(folds, set(GENERATOR.SPLITS))

    def test_paraphrases_counterfactuals_and_prefixes_share_one_world_fold(self):
        groups = defaultdict(list)
        for record in self.records:
            groups[record["groupId"]].append(record)
        self.assertEqual(len(groups), len(self.templates) * 10)
        for group_id, family in groups.items():
            with self.subTest(group=group_id):
                self.assertEqual(len({row["split"] for row in family}), 1)
                self.assertEqual(len({row["source"]["worldId"] for row in family}), 1)
                self.assertEqual(len({row["source"]["nameCohort"] for row in family}), 1)
                skills = {row["source"]["skill"] for row in family}
                self.assertEqual(len(skills), 1)
                skill = next(iter(skills))
                expected = {(template["templateId"], branch, prefix)
                            for template in self.templates[skill]
                            for branch in (0, 1) for prefix in (1, 2, 3)}
                actual = {(row["source"]["templateId"], row["source"]["counterfactualBranch"],
                           row["source"]["supervisedPrefixPairs"]) for row in family}
                self.assertEqual(actual, expected)
                self.assertEqual(len(family), len(expected))
                branches = defaultdict(dict)
                for row in family:
                    source = row["source"]
                    branches[(source["templateId"], source["counterfactualBranch"])][
                        source["supervisedPrefixPairs"]] = row["messages"]
                for prefixes in branches.values():
                    self.assertEqual(prefixes[1], prefixes[3][:2])
                    self.assertEqual(prefixes[2], prefixes[3][:4])

    def test_name_value_associations_are_reserved_but_lexical_pools_are_shared(self):
        lexical = {split: defaultdict(set) for split in GENERATOR.SPLITS}
        associations = {}
        accessible_names = {split: set() for split in GENERATOR.SPLITS}
        for cohort in range(len(GENERATOR.NAMES) // 4):
            names = GENERATOR.NAMES[4 * cohort:4 * cohort + 4]
            for field, values in GENERATOR.POOLS.items():
                pools = {split: set(GENERATOR._pool(cohort, field, split))
                         for split in GENERATOR.SPLITS}
                self.assertEqual(set().union(*pools.values()), set(values))
                for split, pool in pools.items():
                    self.assertGreaterEqual(len(pool), 2)
                    lexical[split][field].update(pool)
                    accessible_names[split].update(names)
                    for name in names:
                        for value in pool:
                            key = (name, field, value)
                            self.assertNotIn(key, associations, (key, split, associations.get(key)))
                            associations[key] = split
                for index, split in enumerate(GENERATOR.SPLITS):
                    for other in GENERATOR.SPLITS[index + 1:]:
                        self.assertFalse(pools[split] & pools[other], (cohort, field, split, other))
        for split in GENERATOR.SPLITS:
            self.assertEqual(accessible_names[split], set(GENERATOR.NAMES))
        for field, values in GENERATOR.POOLS.items():
            self.assertEqual(lexical["train"][field], set(values))
            for split in ("validation", "test"):
                self.assertTrue(lexical[split][field])
                self.assertLessEqual(lexical[split][field], lexical["train"][field])

    def test_sampled_worlds_use_distinct_reserved_values_and_swaps_keep_identity(self):
        rng = random.Random(8128)
        for split in GENERATOR.SPLITS:
            for episode_index in range(32):
                world, cohort = GENERATOR._world(rng, split, episode_index)
                self.assertNotEqual(world["name"], world["friend"])
                cohort_names = GENERATOR.NAMES[4 * cohort:4 * cohort + 4]
                self.assertIn(world["name"], cohort_names)
                self.assertIn(world["friend"], cohort_names)
                before = copy.deepcopy(world)
                swapped = GENERATOR._swapped(world)
                self.assertEqual(world, before)
                self.assertEqual(GENERATOR._swapped(swapped), world)
                self.assertEqual(GENERATOR._world_key("selector", world),
                                 GENERATOR._world_key("selector", swapped))
                for field in GENERATOR.POOLS:
                    self.assertNotEqual(world[field], world["other_" + field])
                    pool = GENERATOR._pool(cohort, field, split)
                    self.assertIn(world[field], pool)
                    self.assertIn(world["other_" + field], pool)
                    self.assertEqual(swapped[field], world["other_" + field])
                    self.assertEqual(swapped["other_" + field], world[field])

    def test_correction_targets_use_the_explicitly_revised_value_in_both_branches(self):
        expected = {
            "edit-date": ("other_day", "No, ahora es el {other_day}."),
            "edit-color": ("other_color", "Anota {other_color}."),
            "edit-destination": ("other_place", "Vas a {other_place}."),
            "edit-count": ("other_count", "La cifra correcta es {other_count}."),
        }
        world = self.world()
        for template in GENERATOR.FACT_TEMPLATES["correction"]:
            field, final_reply = expected[template["templateId"]]
            for branch, values in enumerate((world, GENERATOR._swapped(world))):
                with self.subTest(template=template["templateId"], branch=branch):
                    messages = GENERATOR._render(template, values)
                    self.assertIn(str(values[field]), messages[2]["content"])
                    self.assertEqual(messages[-1]["content"], final_reply.format(**values))

    def test_clarification_does_not_turn_the_speakers_city_into_a_destination(self):
        template = next(template for template in GENERATOR.FACT_TEMPLATES["clarification"]
                        if template["templateId"] == "missing-destination")
        world = self.world()
        for branch, values in enumerate((world, GENERATOR._swapped(world))):
            with self.subTest(branch=branch):
                messages = GENERATOR._render(template, values)
                self.assertIn(values["place"], messages[0]["content"])
                self.assertEqual(messages[1]["content"],
                                 "Todavía no dijiste la ciudad de destino. ¿Cuál es?")
                self.assertEqual(messages[3]["content"], "El destino es {place}.".format(**values))
                self.assertEqual(messages[5]["content"],
                                 "No has dicho el destino del {other_item}.".format(**values))
                for place in (values["place"], values["other_place"]):
                    self.assertNotIn(place, messages[1]["content"])
                    self.assertNotIn(place, messages[5]["content"])

    def test_inventory_counts_notes_even_when_the_other_object_is_an_envelope(self):
        template = next(template for template in GENERATOR.FACT_TEMPLATES["explicit_fact"]
                        if template["templateId"] == "fact-inventory")
        world = {**self.world(), "item": "sobre"}
        messages = GENERATOR._render(template, world)
        self.assertIn("3 notas y un sobre", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Tienes 3 notas.")
        self.assertEqual(messages[5]["content"], "Tienes un sobre.")

    def test_generation_is_reproducible_without_mutating_the_template_bank(self):
        original_templates = copy.deepcopy(self.templates)
        first = GENERATOR.generate(**self.options)
        second = GENERATOR.generate(**self.options)
        self.assertEqual(first, second)
        self.assertEqual(first, self.records)
        self.assertEqual(self.templates, original_templates)
        changed = GENERATOR.generate(**{**self.options, "seed": self.options["seed"] + 1})
        self.assertNotEqual(first, changed)

    def test_build_is_reproducible_has_no_exact_cross_fold_leak_and_keeps_inputs(self):
        outputs = [self.root / "one.jsonl", self.root / "two.jsonl"]
        manifests = [self.root / "one.manifest.json", self.root / "two.manifest.json"]
        before = copy.deepcopy(self.training_fixture)
        reports = [self.build(output, manifest) for output, manifest in zip(outputs, manifests)]
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(manifests[0].read_bytes(), manifests[1].read_bytes())
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
        self.assertEqual(self.training_fixture, before)
        report = reports[0]
        self.assertFalse(report["tokenizer"]["refitted"])
        self.assertEqual(report["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer_raw).hexdigest())
        self.assertEqual(report["sha256"], hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8")), report)
        self.assertFalse(report["authoring"]["dialoguesTruncated"])
        selected = load_corpus(outputs[0])
        candidates = {row["id"]: row for row in self.records}
        folds_by_messages, folds_by_family, coverage = defaultdict(set), defaultdict(set), defaultdict(set)
        for row in selected:
            self.assertEqual(row, candidates[row["id"]])
            folds_by_messages[canonical(row["messages"])].add(row["split"])
            folds_by_family[row["groupId"]].add(row["split"])
            coverage[row["source"]["templateId"]].add(row["split"])
            self.assertLessEqual(len(dialogue_tokens(row["messages"], self.tokenizer)[0]), 4097)
        self.assertTrue(all(len(folds) == 1 for folds in folds_by_messages.values()))
        self.assertTrue(all(len(folds) == 1 for folds in folds_by_family.values()))
        self.assertEqual(set(coverage), self.template_ids)
        self.assertTrue(all(folds == set(GENERATOR.SPLITS) for folds in coverage.values()))
        self.assertEqual(len(selected), report["records"])
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path, *outputs, *manifests})

    def test_existing_dataset_or_manifest_is_never_overwritten(self):
        for existing_name in ("curriculum.jsonl", "manifest.json"):
            with self.subTest(existing=existing_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, manifest = root / "curriculum.jsonl", root / "manifest.json"
                existing = root / existing_name
                existing.write_bytes(b"existing evidence\n")
                with self.assertRaisesRegex(ValueError, "immutable"):
                    self.build(output, manifest)
                self.assertEqual(existing.read_bytes(), b"existing evidence\n")
                self.assertEqual(set(root.iterdir()), {existing})
                self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)

    def test_output_aliases_cannot_overwrite_the_input_tokenizer(self):
        for output, manifest in (
            (self.tokenizer_path, self.root / "manifest.json"),
            (self.root / "curriculum.jsonl", self.tokenizer_path),
            (self.root / "same.json", self.root / "same.json"),
        ):
            with self.subTest(output=output, manifest=manifest):
                with self.assertRaisesRegex(ValueError, "distinct paths"):
                    self.build(output, manifest)
                self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
                self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path})

    def test_competing_publication_preserves_the_other_writers_file(self):
        output = self.root / "race.jsonl"
        original_link = GENERATOR.os.link

        def competing_writer(source, destination):
            Path(destination).write_bytes(b"concurrent evidence\n")
            return original_link(source, destination)

        with patch.object(GENERATOR.os, "link", side_effect=competing_writer):
            with self.assertRaisesRegex(ValueError, "Another writer"):
                GENERATOR._exclusive_write(output, b"new data\n")
        self.assertEqual(output.read_bytes(), b"concurrent evidence\n")
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path, output})


if __name__ == "__main__":
    unittest.main()
