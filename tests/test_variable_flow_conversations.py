"""Independent fictional oracles for the offline dialogue authoring helper."""

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
    raise unittest.SkipTest("Variable-flow curriculum tests require optional tokenizers")

from examples import build_variable_flow_conversations as GENERATOR
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


WORLD = {
    "name": "Samori", "new_name": "Nevasi", "alias": "Tilara",
    "topic": "Puente Luneri", "new_topic": "Jardín Noravi",
    "city": "Puerto Tulela", "new_city": "Villa Coreni",
    "item": "cuaderno", "other_item": "mapa",
    "color": "azul", "other_color": "rojo", "new_color": "ocre",
    "count": 3, "new_count": 8,
}


class VariableFlowOracleTests(unittest.TestCase):
    def author(self, events, expected):
        world = copy.deepcopy(WORLD)
        world_before = copy.deepcopy(world)
        events_before = copy.deepcopy(events)
        messages = GENERATOR.author(events, world, "train")
        self.assertEqual(world, world_before)
        self.assertEqual(events, events_before)
        self.assertEqual(len(messages), 2 * len(events))
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"] * len(events))
        self.assertTrue(all(message["content"].strip() for message in messages))
        self.assertEqual([message["content"] for message in messages[1::2]], expected)
        return messages

    def test_own_identity_survives_unknown_then_introduced_user_name(self):
        events = ["greet", "ask_own_name", "ask_real_name", "introduce_real_name", "ask_own_name", "ask_real_name"]
        # Expected values are written manually; no authoring state or reply
        # selector is called to calculate any expected answer.
        expected = [
            "Hola. Te leo.",
            "Soy Unidad.",
            "Todavía no has dicho tu nombre.",
            "De acuerdo, te llamas Samori.",
            "Soy Unidad.",
            "Te llamas Samori.",
        ]
        self.author(events, expected)

    def test_real_name_correction_preserves_the_independently_chosen_alias(self):
        events = ["introduce_real_name", "set_preferred_alias", "ask_real_name", "correct_real_name",
                  "ask_preferred_alias", "ask_real_name"]
        expected = [
            "De acuerdo, te llamas Samori.",
            "De acuerdo, te llamaré Tilara.",
            "Te llamas Samori.",
            "De acuerdo, te llamas Nevasi.",
            "Quieres que te llame Tilara.",
            "Te llamas Nevasi.",
        ]
        self.author(events, expected)

    def test_city_updates_do_not_reveal_the_undeclared_color_in_the_world(self):
        events = ["ask_color", "set_city", "ask_city", "correct_city", "ask_color", "ask_city"]
        expected = [
            "No has dicho el color del cuaderno.",
            "Tu ciudad es Puerto Tulela.",
            "Tu ciudad es Puerto Tulela.",
            "Tu ciudad es Villa Coreni.",
            "No has dicho el color del cuaderno.",
            "Tu ciudad es Villa Coreni.",
        ]
        self.author(events, expected)

    def test_alias_alone_does_not_supply_a_real_name_or_change_the_models_name(self):
        events = ["greet", "ask_preferred_alias", "set_preferred_alias", "ask_real_name",
                  "ask_own_name", "ask_preferred_alias"]
        expected = [
            "Hola. Te leo.",
            "No has indicado un apodo.",
            "De acuerdo, te llamaré Tilara.",
            "Todavía no has dicho tu nombre.",
            "Soy Unidad.",
            "Quieres que te llame Tilara.",
        ]
        self.author(events, expected)

    def test_topic_update_survives_identity_and_an_unknown_name_question(self):
        events = ["set_topic", "ask_own_name", "ask_topic", "correct_topic", "ask_real_name", "ask_topic"]
        self.author(events, [
            "Hablemos de Puente Luneri.", "Soy Unidad.", "El tema actual es Puente Luneri.",
            "Hablemos de Jardín Noravi.", "Todavía no has dicho tu nombre.", "El tema actual es Jardín Noravi.",
        ])

    def test_correcting_one_objects_color_preserves_the_other_objects_color(self):
        events = ["greet", "set_colors", "correct_color", "ask_other_color", "ask_color", "ask_own_name"]
        self.author(events, [
            "Hola. Te leo.", "El cuaderno es azul y el mapa es rojo.", "El cuaderno es ocre.",
            "El mapa es rojo.", "El cuaderno es ocre.", "Soy Unidad.",
        ])

    def test_count_is_unknown_until_declared_then_uses_the_latest_correction(self):
        events = ["ask_count", "set_count", "ask_count", "correct_count", "ask_own_name", "ask_count"]
        self.author(events, [
            "No has dicho cuántas fichas tienes.", "Tienes 3 fichas.", "Tienes 3 fichas.",
            "Tienes 8 fichas.", "Soy Unidad.", "Tienes 8 fichas.",
        ])

    def test_latest_listening_intent_does_not_revert_after_an_identity_question(self):
        events = ["set_intent_act", "ask_intent", "set_intent_listen", "ask_own_name", "ask_intent", "greet"]
        self.author(events, [
            "¿Qué quieres empezar? Podemos buscar un paso sencillo.", "Quieres que proponga un primer paso.",
            "Te escucho, sin proponer un plan.", "Soy Unidad.", "Quieres que te escuche.", "Hola. Te leo.",
        ])

    def test_all_styles_and_folds_preserve_unknown_name_and_own_identity(self):
        own_replies = ["Soy Unidad.", "Me llamo Unidad.", "Mi nombre es Unidad."]
        for split in ("train", "validation", "test"):
            for style in range(3):
                with self.subTest(split=split, style=style):
                    events = [{"type": "ask_real_name", "style": style},
                              {"type": "ask_own_name", "style": style},
                              {"type": "set_preferred_alias", "style": style},
                              {"type": "ask_real_name", "style": style},
                              {"type": "ask_own_name", "style": style},
                              {"type": "ask_preferred_alias", "style": style}]
                    messages = GENERATOR.author(events, copy.deepcopy(WORLD), split)
                    self.assertEqual([message["content"] for message in messages[1::2]], [
                        "Todavía no has dicho tu nombre.", own_replies[style], "De acuerdo, te llamaré Tilara.",
                        "Todavía no has dicho tu nombre.", own_replies[style], "Quieres que te llame Tilara.",
                    ])


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class VariableFlowCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = {"seed": 1813, "worlds": 30}
        cls.records = GENERATOR.generate(**cls.options)
        # Only this invented training fixture is used to fit a tokenizer.
        # Corpus holdouts, owner text and real tokenizers are never opened.
        fixture = [{"split": "train", "messages": [
            {"role": "user", "content": "Hola. ¿Puedes leer una frase ficticia?"},
            {"role": "assistant", "content": "Sí. Te leo."},
        ]}]
        original = copy.deepcopy(fixture)
        cls.tokenizer = fit_tokenizer(fixture, vocab_size=262)
        if fixture != original:
            raise AssertionError("Tokenizer fitting mutated its training fixture")
        cls.tokenizer_raw = cls.tokenizer.to_str().encode("utf-8")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer_path = self.root / "fixture-tokenizer.json"
        self.tokenizer_path.write_bytes(self.tokenizer_raw)

    def build(self, output=None, manifest=None):
        return GENERATOR.build(self.tokenizer_path, output or self.root / "flows.jsonl",
                               manifest or self.root / "manifest.json", context_length=4096, **self.options)

    def assert_question_positions(self, records):
        positions = {split: defaultdict(set) for split in ("train", "validation", "test")}
        for row in records:
            source = row["source"]
            positions[row["split"]][source["questionEvent"]].add(source["questionPosition"])
        for split, events in positions.items():
            for event in ("ask_own_name", "ask_real_name", "ask_topic", "ask_city"):
                with self.subTest(split=split, event=event):
                    self.assertLessEqual({2, 3, 4, 5, 6}, events[event])

    def test_worlds_keep_all_flow_orders_and_prefixes_in_one_fold(self):
        groups, coverage, identifiers = defaultdict(list), defaultdict(set), set()
        full_lengths = set()
        for row in self.records:
            self.assertNotIn(row["id"], identifiers)
            identifiers.add(row["id"])
            groups[row["groupId"]].append(row)
            source = row["source"]
            self.assertEqual(source["kind"], "authored-synthetic")
            self.assertTrue(source["fictional"])
            self.assertFalse(source["privateConversationUsed"])
            self.assertFalse(source["modelResponsesUsed"])
            self.assertEqual(len(row["messages"]), 2 * source["prefixPairs"])
            self.assertEqual([message["role"] for message in row["messages"]],
                             ["user", "assistant"] * source["prefixPairs"])
            self.assertEqual(source["questionPosition"], source["prefixPairs"])
            self.assertEqual(source["questionEvent"], source["events"][-1])
            self.assertEqual(len(source["events"]), source["prefixPairs"])
            self.assertIn(source["fullFlowPairs"], (4, 5, 6))
            full_lengths.add(source["fullFlowPairs"])
            coverage[source["flowId"]].add(row["split"])
        self.assertEqual(len(groups), 30)
        self.assertEqual(full_lengths, {4, 5, 6})
        self.assertEqual(set(coverage), set(GENERATOR.FLOWS))
        self.assertTrue(all(folds == {"train", "validation", "test"} for folds in coverage.values()))
        for group_id, family in groups.items():
            with self.subTest(group=group_id):
                self.assertTrue(group_id.startswith("variable-flow-v1-"))
                self.assertEqual(len({row["split"] for row in family}), 1)
                self.assertEqual(len({row["source"]["worldId"] for row in family}), 1)
                flows = defaultdict(dict)
                for row in family:
                    source = row["source"]
                    flows[source["flowId"]][source["prefixPairs"]] = row["messages"]
                    self.assertEqual(source["events"], list(GENERATOR.FLOWS[source["flowId"]][:source["prefixPairs"]]))
                self.assertEqual(set(flows), set(GENERATOR.FLOWS))
                for flow_id, prefixes in flows.items():
                    length = len(GENERATOR.FLOWS[flow_id])
                    self.assertEqual(set(prefixes), set(range(1, length + 1)))
                    for prefix, messages in prefixes.items():
                        self.assertEqual(messages, prefixes[length][:2 * prefix])

    def test_each_fold_exercises_important_questions_at_positions_two_through_six(self):
        self.assert_question_positions(self.records)

    def test_generation_is_reproducible_and_does_not_mutate_flow_programs(self):
        original_flows = copy.deepcopy(GENERATOR.FLOWS)
        self.assertEqual(GENERATOR.generate(**self.options), self.records)
        self.assertEqual(GENERATOR.FLOWS, original_flows)
        self.assertNotEqual(GENERATOR.generate(**{**self.options, "seed": 1814}), self.records)

    def test_build_retains_complete_fitting_flows_without_cross_fold_prompts(self):
        outputs, manifests = [self.root / "one.jsonl", self.root / "two.jsonl"], [self.root / "one.json", self.root / "two.json"]
        reports = [self.build(output, manifest) for output, manifest in zip(outputs, manifests)]
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(manifests[0].read_bytes(), manifests[1].read_bytes())
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
        report = reports[0]
        self.assertFalse(report["tokenizer"]["refitted"])
        self.assertEqual(report["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer_raw).hexdigest())
        self.assertEqual(report["sha256"], hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8")), report)
        self.assertFalse(report["authoring"]["dialoguesTruncated"])
        self.assertFalse(report["authoring"]["inferenceUsesAuthoringState"])
        self.assertEqual(report["authoring"]["excludedCompleteFlows"], 0)
        rows = load_corpus(outputs[0])
        candidates = {row["id"]: row for row in self.records}
        prompts, messages, families, full_coverage = (defaultdict(set) for _ in range(4))
        complete = {(row["groupId"], row["source"]["flowId"]) for row in rows
                    if row["source"]["prefixPairs"] == row["source"]["fullFlowPairs"]}
        for row in rows:
            self.assertEqual(row, candidates[row["id"]])
            self.assertLessEqual(len(dialogue_tokens(row["messages"], self.tokenizer)[0]), 4097)
            prompts[canonical(row["messages"][:-1])].add(row["split"])
            messages[canonical(row["messages"])].add(row["split"])
            families[row["groupId"]].add(row["split"])
            source = row["source"]
            if source["prefixPairs"] == source["fullFlowPairs"]:
                full_coverage[source["flowId"]].add(row["split"])
            else:
                # Generic duplicate prefixes can survive under a different
                # prototype world. A complete matching dialogue must still
                # exist in the same fold; no shortened full flow is created.
                if (row["groupId"], source["flowId"]) not in complete:
                    self.assertTrue(any(full["split"] == row["split"] and
                                        full["source"]["flowId"] == source["flowId"] and
                                        full["source"]["prefixPairs"] == full["source"]["fullFlowPairs"] and
                                        full["messages"][:len(row["messages"])] == row["messages"] for full in rows))
        self.assertTrue(all(len(folds) == 1 for folds in prompts.values()))
        self.assertTrue(all(len(folds) == 1 for folds in messages.values()))
        self.assertTrue(all(len(folds) == 1 for folds in families.values()))
        self.assertEqual(set(full_coverage), set(GENERATOR.FLOWS))
        self.assertTrue(all(folds == {"train", "validation", "test"} for folds in full_coverage.values()))
        self.assert_question_positions(rows)
        self.assertEqual(len(rows), report["records"])
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path, *outputs, *manifests})

    def test_existing_outputs_and_input_aliases_are_preserved(self):
        for existing_name in ("flows.jsonl", "manifest.json"):
            with self.subTest(existing=existing_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, manifest = root / "flows.jsonl", root / "manifest.json"
                existing = root / existing_name
                existing.write_bytes(b"immutable evidence\n")
                with patch.object(GENERATOR, "generate", side_effect=AssertionError("must reject before generating")):
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        self.build(output, manifest)
                self.assertEqual(existing.read_bytes(), b"immutable evidence\n")
                self.assertEqual(set(root.iterdir()), {existing})
        for output, manifest in ((self.tokenizer_path, self.root / "manifest.json"),
                                 (self.root / "flows.jsonl", self.tokenizer_path),
                                 (self.root / "same.json", self.root / "same.json")):
            with self.subTest(output=output, manifest=manifest):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    self.build(output, manifest)
                self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
                self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path})

    def test_contradictory_targets_fail_before_output_publication(self):
        records = copy.deepcopy(self.records)
        conflicting = copy.deepcopy(records[0])
        conflicting["id"] += "-contradiction"
        conflicting["messages"][-1]["content"] = "Un dato contradictorio."
        records.append(conflicting)
        with patch.object(GENERATOR, "generate", return_value=records):
            with self.assertRaisesRegex(ValueError, "contradictory targets"):
                self.build()
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path})

    def test_competing_writer_keeps_existing_evidence_and_cleans_temporary_file(self):
        output = self.root / "race.jsonl"
        original_link = GENERATOR.os.link

        def compete(source, destination):
            Path(destination).write_bytes(b"concurrent evidence\n")
            return original_link(source, destination)

        with patch.object(GENERATOR.os, "link", side_effect=compete):
            with self.assertRaisesRegex(ValueError, "Another writer"):
                GENERATOR.exclusive_write(output, b"new evidence\n")
        self.assertEqual(output.read_bytes(), b"concurrent evidence\n")
        self.assertEqual(self.tokenizer_path.read_bytes(), self.tokenizer_raw)
        self.assertEqual(set(self.root.iterdir()), {self.tokenizer_path, output})


if __name__ == "__main__":
    unittest.main()
