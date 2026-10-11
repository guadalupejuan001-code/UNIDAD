"""Check open fictional facts without reading a corpus, model or evaluation."""

from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Open-fact curriculum tests require optional tokenizers")

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


_HELPER = Path(__file__).resolve().parents[1] / "examples" / "build_open_fact_conversations.py"
_SPEC = importlib.util.spec_from_file_location("open_fact_curriculum", _HELPER)
GENERATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GENERATOR)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class OpenFactConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {"seed": 413, "episodes_per_skill": 32}
        cls.templates = GENERATOR._templates()
        cls.template_ids = {template["templateId"] for bank in cls.templates.values() for template in bank}
        cls.records = GENERATOR.generate(**cls.options)
        # Fit only this independent train fixture, never the generated holdouts
        # or an existing tokenizer. A byte-only vocabulary keeps it portable.
        cls.training_fixture = [{
            "id": "open-fact-tokenizer-fixture", "groupId": "open-fact-tokenizer-family", "split": "train",
            "messages": [{"role": "user", "content": "Hola. ¿Puedes leer un dato ficticio?"},
                         {"role": "assistant", "content": "Sí. Cuéntame ese dato."}],
        }]
        original = copy.deepcopy(cls.training_fixture)
        cls.tokenizer = fit_tokenizer(cls.training_fixture, vocab_size=262)
        if cls.training_fixture != original:
            raise AssertionError("Tokenizer fitting changed its input fixture")
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
            output or self.root / "facts.jsonl",
            manifest or self.root / "manifest.json",
            context_length=4096,
            **self.options,
        )

    def world(self, count=2, other_count=3):
        """Independent fictional values, with derived fields checked by hand."""
        items = ["Ruta Celari", "Taller Morali", "Archivo Tiveli", "Puente Surali", "Mapa Velari"][:count]
        other_items = ["Cuaderno Norali", "Patio Lunari", "Faro Kelari", "Jardín Polari", "Ventana Sarali"][:other_count]
        return {
            "name": "Zamira", "alias": "Talisa", "new_name": "Nureli",
            "place": "Sala Lunari", "other_place": "Sala Torali", "time": "09:15", "other_time": "14:30",
            "title": "Ruta Maveli", "other_title": "Taller Kureli",
            "value": "pluma oeste", "other_value": "niebla sur", "code": "k3-r21", "other_code": "m7-b42",
            "items": items, "other_items": other_items,
            "count": count, "other_count": other_count,
            "items_text": "; ".join(items), "other_items_text": "; ".join(other_items),
            "first": items[0], "last": items[-1], "other_first": other_items[0], "other_last": other_items[-1],
        }

    def test_eight_skills_and_every_template_have_all_three_folds(self):
        expected_skills = {"name_alias", "name_change", "project_title", "attribute",
                           "time_place", "list_count", "missing_data", "declarations"}
        self.assertEqual(set(self.templates), expected_skills)
        self.assertTrue(all(len(bank) == 4 for bank in self.templates.values()))
        self.assertEqual(len(self.template_ids), 32)
        template_folds, skill_folds = defaultdict(set), defaultdict(set)
        ids = set()
        for row in self.records:
            self.assertNotIn(row["id"], ids)
            ids.add(row["id"])
            source = row["source"]
            self.assertEqual(source["kind"], "authored-synthetic")
            self.assertTrue(source["fictional"])
            self.assertFalse(source["privateConversationUsed"])
            self.assertFalse(source["modelResponsesUsed"])
            self.assertEqual(len(row["messages"]), 2 * source["prefixPairs"])
            self.assertEqual([message["role"] for message in row["messages"]],
                             ["user", "assistant"] * source["prefixPairs"])
            self.assertTrue(all(message["content"].strip() for message in row["messages"]))
            template_folds[source["templateId"]].add(row["split"])
            skill_folds[source["skill"]].add(row["split"])
        self.assertEqual(set(template_folds), self.template_ids)
        self.assertEqual(set(skill_folds), expected_skills)
        for category, folds in {**template_folds, **skill_folds}.items():
            with self.subTest(category=category):
                self.assertEqual(folds, {"train", "validation", "test"})

    def test_all_world_variants_and_prefixes_stay_in_one_family_fold(self):
        groups = defaultdict(list)
        for row in self.records:
            groups[row["groupId"]].append(row)
        self.assertEqual(len(groups), 8 * self.options["episodes_per_skill"])
        for group_id, family in groups.items():
            with self.subTest(group=group_id):
                self.assertTrue(group_id.startswith("open-facts-v2-"))
                self.assertEqual(len({row["split"] for row in family}), 1)
                self.assertEqual(len({row["source"]["worldId"] for row in family}), 1)
                skills = {row["source"]["skill"] for row in family}
                self.assertEqual(len(skills), 1)
                skill = next(iter(skills))
                expected = {(template["templateId"], branch, prefix)
                            for template in self.templates[skill]
                            for branch in (0, 1) for prefix in (1, 2, 3)}
                actual = {(row["source"]["templateId"], row["source"]["branch"],
                           row["source"]["prefixPairs"]) for row in family}
                self.assertEqual(actual, expected)
                self.assertEqual(len(family), len(expected))
                branches = defaultdict(dict)
                for row in family:
                    source = row["source"]
                    branches[(source["templateId"], source["branch"])][source["prefixPairs"]] = row["messages"]
                for prefixes in branches.values():
                    self.assertEqual(prefixes[1], prefixes[3][:2])
                    self.assertEqual(prefixes[2], prefixes[3][:4])

    def test_name_corrections_use_the_replacement_in_both_branches(self):
        final_replies = {
            "name-correction": "Debo llamarte {new_name}.",
            "name-new-preference": "Conserva {new_name}.",
            "name-retraction": "No. Lo corregiste a {new_name}.",
            "name-current-choice": "Usas {new_name}.",
        }
        world = self.world()
        for template in self.templates["name_change"]:
            for branch, values in enumerate((world, GENERATOR._swapped(world))):
                with self.subTest(template=template["templateId"], branch=branch):
                    messages = GENERATOR._render(template, values)
                    self.assertIn(values["name"], messages[0]["content"])
                    self.assertIn(values["new_name"], messages[2]["content"])
                    self.assertIn(values["new_name"], messages[3]["content"])
                    self.assertNotIn(values["name"], messages[3]["content"])
                    self.assertEqual(messages[-1]["content"],
                                     final_replies[template["templateId"]].format(**values))
                    self.assertNotIn(values["name"], messages[-1]["content"])

    def test_declarations_acknowledge_recall_and_update_literal_facts(self):
        expected_replies = {
            "declaration-name": ("Hola, {name}.", "Te llamas {name}.", "Entendido, te llamas {new_name}."),
            "declaration-alias": ("De acuerdo, {alias}.", "Quieres que te llame {alias}.", "De acuerdo, {new_name}."),
            "declaration-project": ("Quieres trabajar en {title}.", "Quieres trabajar en {title}.",
                                    "Ahora quieres trabajar en {other_title}."),
            "declaration-preference": ("Prefieres el texto {value}.", "Prefieres {value}.",
                                       "Ahora prefieres {other_value}."),
        }
        self.assertEqual({template["templateId"] for template in self.templates["declarations"]},
                         set(expected_replies))
        world = self.world()
        for branch, values in enumerate((world, GENERATOR._swapped(world))):
            for template in self.templates["declarations"]:
                with self.subTest(template=template["templateId"], branch=branch):
                    messages = GENERATOR._render(template, values)
                    # Exchanges one and three state or correct facts; exchange
                    # two asks for the literal fact introduced by the first.
                    for index in (0, 4):
                        self.assertNotIn("?", messages[index]["content"])
                        self.assertNotIn("¿", messages[index]["content"])
                    self.assertIn("?", messages[2]["content"])
                    expected = [reply.format(**values) for reply in expected_replies[template["templateId"]]]
                    self.assertEqual([message["content"] for message in messages[1::2]], expected)

    def test_lists_have_two_to_five_distinct_entries_and_targets_follow_the_new_list(self):
        for split in ("train", "validation", "test"):
            counts = set()
            for index in range(8):
                values = GENERATOR._world(self.options["seed"], split, index)
                for field, size in (("items", "count"), ("other_items", "other_count")):
                    items = values[field]
                    self.assertGreaterEqual(len(items), 2)
                    self.assertLessEqual(len(items), 5)
                    self.assertEqual(len(items), len(set(items)))
                    self.assertEqual(values[size], len(items))
                    counts.add(len(items))
            self.assertEqual(counts, {2, 3, 4, 5})
        expected_replies = {
            "list-entries": ("Tiene {count} entradas.", "Ahora hay {other_count} entradas.", "La primera es {other_first}."),
            "list-options": ("Son {count} opciones.", "Ahora son {other_count} opciones.", "Primero aparece {other_first}."),
            "list-final-entry": ("Figuran {count} elementos.", "Ahora figuran {other_count} elementos.", "El último es {other_last}."),
            "list-first-and-count": ("La primera es {first}.", "Tiene {count} entradas.", "La nueva tiene {other_count} entradas."),
        }
        for count in range(2, 6):
            world = self.world(count, 2 + (count - 1) % 4)
            for branch, values in enumerate((world, GENERATOR._swapped(world))):
                self.assertEqual(values["count"], len(values["items"]))
                self.assertEqual(values["other_count"], len(values["other_items"]))
                self.assertEqual(values["items_text"], "; ".join(values["items"]))
                self.assertEqual(values["other_items_text"], "; ".join(values["other_items"]))
                for template in self.templates["list_count"]:
                    with self.subTest(template=template["templateId"], count=count, branch=branch):
                        messages = GENERATOR._render(template, values)
                        expected = [reply.format(**values) for reply in expected_replies[template["templateId"]]]
                        self.assertEqual([message["content"] for message in messages[1::2]], expected)

    def test_time_and_place_queries_do_not_update_the_other_field(self):
        expected_replies = {
            "where-before-when": ("La cita es en {place}.", "Es a las {time}.", "Sigue siendo en {place}."),
            "when-before-where": ("Será a las {time}.", "Será en {place}.", "Será a las {time}."),
            "time-correction": ("Está prevista a las {time}.", "Ahora es a las {other_time}.", "No, el lugar sigue siendo {place}."),
            "place-correction": ("Será en {place}.", "Ahora será en {other_place}.", "Quedaron a las {time}."),
        }
        world = self.world()
        for branch, values in enumerate((world, GENERATOR._swapped(world))):
            for template in self.templates["time_place"]:
                with self.subTest(template=template["templateId"], branch=branch):
                    messages = GENERATOR._render(template, values)
                    expected = [reply.format(**values) for reply in expected_replies[template["templateId"]]]
                    self.assertEqual([message["content"] for message in messages[1::2]], expected)

    def test_latent_world_fields_are_not_invented_when_the_user_has_not_supplied_them(self):
        expectations = {
            "missing-time": ("Todavía no has dicho la hora de inicio.", "Empieza a las {time}.",
                             "No has dicho dónde se hará {other_title}."),
            "missing-code": ("No has indicado el código del paquete.", "El código es {code}.",
                             "No has dicho el código de la otra caja."),
            "missing-name": ("No has dicho su nombre.", "Se llama {name}.",
                             "Todavía no has dicho el nombre de esa otra persona."),
            "missing-place": ("Aún no has dicho el lugar de la presentación.", "Será en {place}.",
                              "No has anunciado el lugar de la otra presentación."),
        }
        world = self.world()
        for branch, values in enumerate((world, GENERATOR._swapped(world))):
            for template in self.templates["missing_data"]:
                with self.subTest(template=template["templateId"], branch=branch):
                    messages = GENERATOR._render(template, values)
                    expected = [reply.format(**values) for reply in expectations[template["templateId"]]]
                    self.assertEqual([message["content"] for message in messages[1::2]], expected)

    def test_generation_is_deterministic_and_does_not_mutate_the_bank(self):
        original = copy.deepcopy(self.templates)
        generated = GENERATOR.generate(**self.options)
        self.assertEqual(generated, self.records)
        self.assertEqual(GENERATOR.generate(**self.options), generated)
        self.assertEqual(GENERATOR._templates(), original)
        changed = GENERATOR.generate(**{**self.options, "seed": self.options["seed"] + 1})
        self.assertNotEqual(generated, changed)

    def test_build_is_reproducible_without_cross_fold_copies_or_input_mutation(self):
        outputs = [self.root / "one.jsonl", self.root / "two.jsonl"]
        manifests = [self.root / "one.manifest.json", self.root / "two.manifest.json"]
        original = copy.deepcopy(self.training_fixture)
        reports = [self.build(output, manifest) for output, manifest in zip(outputs, manifests)]
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(manifests[0].read_bytes(), manifests[1].read_bytes())
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
        self.assertEqual(self.training_fixture, original)
        report = reports[0]
        self.assertFalse(report["tokenizer"]["refitted"])
        self.assertEqual(report["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer_raw).hexdigest())
        self.assertEqual(report["sha256"], hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8")), report)
        candidates = {row["id"]: row for row in self.records}
        rows = load_corpus(outputs[0])
        message_folds, prompt_folds, family_folds, coverage = (defaultdict(set) for _ in range(4))
        for row in rows:
            self.assertEqual(row, candidates[row["id"]])
            message_folds[canonical(row["messages"])].add(row["split"])
            prompt_folds[canonical(row["messages"][:-1])].add(row["split"])
            family_folds[row["groupId"]].add(row["split"])
            coverage[row["source"]["templateId"]].add(row["split"])
            self.assertLessEqual(len(dialogue_tokens(row["messages"], self.tokenizer)[0]), 4097)
        self.assertEqual(len(rows), report["records"])
        self.assertTrue(all(len(folds) == 1 for folds in message_folds.values()))
        self.assertTrue(all(len(folds) == 1 for folds in prompt_folds.values()))
        self.assertTrue(all(len(folds) == 1 for folds in family_folds.values()))
        self.assertEqual(set(coverage), self.template_ids)
        self.assertTrue(all(folds == {"train", "validation", "test"} for folds in coverage.values()))
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path, *outputs, *manifests})

    def test_cross_fold_prompt_collision_quarantines_both_complete_families(self):
        rows = copy.deepcopy(self.records)
        train = next(row for row in rows if row["split"] == "train" and row["source"]["prefixPairs"] == 1)
        heldout = next(row for row in rows if row["split"] == "validation" and row["source"]["prefixPairs"] == 1)
        heldout["messages"] = copy.deepcopy(train["messages"])
        excluded = {train["groupId"], heldout["groupId"]}
        with patch.object(GENERATOR, "generate", return_value=rows):
            report = self.build()
        selected = load_corpus(self.root / "facts.jsonl")
        self.assertFalse(excluded & {row["groupId"] for row in selected})
        self.assertEqual(report["excludedWorlds"], 2)
        self.assertGreaterEqual(report["reasons"]["records_in_cross_fold_prompt_families_omitted"], 48)
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)

    def test_conflicting_targets_for_an_identical_prompt_fail_before_publication(self):
        rows = copy.deepcopy(self.records)
        conflicting = copy.deepcopy(rows[0])
        conflicting["id"] += "-conflicting"
        conflicting["messages"][-1]["content"] = "Una respuesta diferente que contradice el dato."
        rows.append(conflicting)
        with patch.object(GENERATOR, "generate", return_value=rows):
            with self.assertRaisesRegex(ValueError, "conflicting targets"):
                self.build()
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path})
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)

    def test_existing_dataset_or_manifest_is_preserved_before_generation(self):
        for existing_name in ("facts.jsonl", "manifest.json"):
            with self.subTest(existing=existing_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, manifest = root / "facts.jsonl", root / "manifest.json"
                existing = root / existing_name
                existing.write_bytes(b"immutable evidence\n")
                with patch.object(GENERATOR, "generate", side_effect=AssertionError("must fail before generation")):
                    with self.assertRaisesRegex(ValueError, "immutable|exist"):
                        self.build(output, manifest)
                self.assertEqual(existing.read_bytes(), b"immutable evidence\n")
                self.assertEqual(set(root.iterdir()), {existing})
                self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)

    def test_output_paths_cannot_alias_each_other_or_the_input_tokenizer(self):
        for output, manifest in (
            (self.tokenizer_path, self.root / "manifest.json"),
            (self.root / "facts.jsonl", self.tokenizer_path),
            (self.root / "same.json", self.root / "same.json"),
        ):
            with self.subTest(output=output, manifest=manifest):
                with self.assertRaises(ValueError):
                    self.build(output, manifest)
                self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
                self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path})


if __name__ == "__main__":
    unittest.main()
