"""Independent authored oracles for historical-error curriculum safety.

These fixtures are fictional and local to this test. No published corpus,
evaluation, model, or owner conversation is read. Passing verifies data and
supervision invariants, not a model's ability to recover conversationally.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
if HAS_TOKENIZERS:
    from examples import build_conversation_recovery as recovery
    from ia_propia.conversation import training
    from ia_propia.conversation.data import load_corpus
    from ia_propia.conversation.tokenizer import (
        ASSISTANT_ID, EOS_ID, IGNORE_INDEX, dialogue_tokens, fit_tokenizer,
    )


FOLDS = ("train", "validation", "test")
# Expected replies are authored explicitly, never produced by the generator.
SCENES = {
    "name": {
        "kind": "user-name",
        "events": ["introduce_real_name", "ask_own_name", "correct_real_name", "ask_real_name"],
        "turns": [
            ("Me llamo Samori.", "De acuerdo, te llamas Samori."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("Corrijo mi nombre: es Nevasi.", "De acuerdo, te llamas Nevasi."),
            ("¿Cómo me llamo?", "Te llamas Nevasi."),
        ],
    },
    "city": {
        "kind": "city",
        "events": ["set_city", "ask_own_name", "correct_city", "ask_city"],
        "turns": [
            ("Mi ciudad es Puerto Tulela.", "Tu ciudad es Puerto Tulela."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("Ahora vivo en Villa Coreni.", "Tu ciudad es Villa Coreni."),
            ("¿Cuál es mi ciudad?", "Tu ciudad es Villa Coreni."),
        ],
    },
    "topic": {
        "kind": "topic",
        "events": ["set_topic", "ask_own_name", "correct_topic", "ask_topic"],
        "turns": [
            ("Quiero hablar de Puente Luneri.", "Hablemos de Puente Luneri."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("Cambio de tema: Jardín Noravi.", "Hablemos de Jardín Noravi."),
            ("¿De qué tema hablamos?", "El tema actual es Jardín Noravi."),
        ],
    },
    "intent": {
        "kind": "intent",
        "events": ["set_intent_act", "ask_own_name", "set_intent_listen", "ask_intent"],
        "turns": [
            ("Quiero un primer paso práctico.", "¿Qué quieres empezar? Podemos buscar un paso sencillo."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("Ahora sólo quiero que me escuches, sin consejos.", "Te escucho, sin proponer un plan."),
            ("¿Qué quiero de esta charla ahora?", "Quieres que te escuche."),
        ],
    },
    "own": {
        "kind": "assistant-name",
        "events": ["introduce_real_name", "ask_own_name", "set_topic", "ask_own_name"],
        "turns": [
            ("Me llamo Samori.", "De acuerdo, te llamas Samori."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("Quiero hablar de Puente Luneri.", "Hablemos de Puente Luneri."),
            ("¿Cuál es tu nombre?", "Soy Unidad."),
        ],
    },
    "alias": {
        "kind": "preferred-alias",
        "events": ["introduce_real_name", "set_preferred_alias", "ask_own_name", "ask_preferred_alias"],
        "turns": [
            ("Me llamo Samori.", "De acuerdo, te llamas Samori."),
            ("Llámame Tilara.", "De acuerdo, te llamaré Tilara."),
            ("¿Cómo te llamas?", "Soy Unidad."),
            ("¿Cómo quiero que me llames?", "Quieres que te llame Tilara."),
        ],
    },
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(scene, fold="train", prefix=4):
    spec = SCENES[scene]
    messages = []
    for user, assistant in spec["turns"][:prefix]:
        messages.extend([{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}])
    # Distinct fictional framing prevents test fixture prompts crossing folds.
    setting = {"train": "jardín", "validation": "patio", "test": "taller"}[fold]
    messages[0]["content"] = f"En una historia del {setting}, " + messages[0]["content"]
    return {
        "id": f"recovery-fixture-{scene}-{fold}-p{prefix}",
        "groupId": f"recovery-fixture-{scene}-{fold}",
        "split": fold,
        "source": {
            "dataset": "unidad-variable-flow-v1", "kind": "authored-synthetic",
            "fictional": True, "privateConversationUsed": False,
            "modelResponsesUsed": False, "prefixPairs": prefix,
            "events": spec["events"][:prefix], "fixtureNote": "independent manual oracle",
        },
        "messages": messages,
    }


@unittest.skipUnless(HAS_TOKENIZERS, "Optional conversation extra 'tokenizers' is unavailable")
class ConversationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = fit_tokenizer([{
            "split": "train",
            "messages": [{"role": "user", "content": "Hola. Una frase ficticia."},
                         {"role": "assistant", "content": "Te leo."}],
        }], vocab_size=262)

    def assert_variant(self, parent, variant, scene, seed):
        expected = SCENES[scene]["turns"][len(parent["messages"]) // 2 - 1][1]
        self.assertEqual(variant["messages"][-1]["content"], expected)
        changed = [index for index, (old, new) in
                   enumerate(zip(parent["messages"], variant["messages"])) if old != new]
        index = len(parent["messages"]) - 3
        self.assertEqual(changed, [index])
        self.assertEqual(variant["messages"][index]["role"], "assistant")
        for old, new in zip(parent["messages"], variant["messages"]):
            self.assertEqual(old["role"], new["role"])
            if old["role"] == "user":
                self.assertEqual(old["content"].encode("utf-8"), new["content"].encode("utf-8"))
        self.assertEqual(parent["messages"][-1], variant["messages"][-1])
        self.assertNotEqual(variant["id"], parent["id"])
        self.assertEqual(variant["groupId"], parent["groupId"])
        self.assertEqual(variant["split"], parent["split"])
        for key, value in parent["source"].items():
            if key != "dataset":
                self.assertEqual(variant["source"][key], value)
        self.assertEqual(variant["source"]["dataset"], "unidad-conversation-recovery-v1")
        self.assertIs(variant["source"]["artificialPreviousAssistantError"], True)
        self.assertTrue(variant["source"]["contextNoise"])
        trace = variant["source"]["recovery"]
        expected_trace = {
            "parentDataset": "unidad-variable-flow-v1", "parentRecordId": parent["id"],
            "parentRecordSha256": digest(canonical(parent)),
            "parentMessagesSha256": digest(canonical(parent["messages"])),
            "changedAssistantMessageIndex": index,
            "changedAssistantExchange": len(parent["messages"]) // 2 - 1,
            "errorKind": SCENES[scene]["kind"], "seed": seed,
            "originalResponseSha256": digest(parent["messages"][index]["content"].encode("utf-8")),
            "artificialResponseSha256": digest(variant["messages"][index]["content"].encode("utf-8")),
            "finalResponseSha256": digest(expected.encode("utf-8")),
        }
        for key, value in expected_trace.items():
            self.assertEqual(trace[key], value, key)
        self.assertEqual(trace["supervision"],
                         "last assistant response only; artificial history is never a target")

    def test_manual_oracles_preserve_current_facts_identity_and_alias(self):
        for scene in SCENES:
            for fold in FOLDS:
                with self.subTest(scene=scene, fold=fold):
                    parent = fixture(scene, fold)
                    snapshot = copy.deepcopy(parent)
                    variant = recovery.corrupt_record(parent, seed=13)
                    self.assert_variant(parent, variant, scene, 13)
                    self.assertEqual(parent, snapshot)
                    variant["messages"][0]["content"] = "A changed copy."
                    variant["source"]["events"].append("greet")
                    self.assertEqual(parent, snapshot)

    def test_listening_history_error_opposes_explicit_current_request(self):
        variant = recovery.corrupt_record(fixture("intent"))
        self.assertEqual(variant["messages"][-3]["content"],
                         "Quieres que proponga un plan práctico.")
        self.assertEqual(variant["messages"][-1]["content"], "Quieres que te escuche.")

    def test_reproducible_corruption(self):
        for scene in SCENES:
            parent = fixture(scene)
            self.assertEqual(recovery.corrupt_record(parent, seed=17),
                             recovery.corrupt_record(copy.deepcopy(parent), seed=17))
            self.assertNotEqual(recovery.corrupt_record(parent, seed=17),
                                recovery.corrupt_record(parent, seed=18))

    def test_prefix_one_and_nonrecovery_targets_are_omitted(self):
        self.assertIsNone(recovery.corrupt_record(fixture("name", prefix=1)))
        for event in ("greet", "ask_color", "ask_count"):
            parent = fixture("name")
            parent["source"]["events"][-1] = event
            parent["messages"][-2]["content"] = "Hola."
            parent["messages"][-1]["content"] = "Hola. Te leo."
            with self.subTest(event=event):
                self.assertIsNone(recovery.corrupt_record(parent))

    def test_unclean_or_inconsistent_provenance_is_rejected(self):
        changes = [
            {"dataset": "another-dataset"}, {"kind": "model-generated"},
            {"privateConversationUsed": True}, {"modelResponsesUsed": True},
            {"privateConversationUsed": None}, {"contextNoise": "earlier-error"},
            {"artificialPreviousAssistantError": True},
            {"prefixPairs": 3}, {"prefixPairs": True}, {"events": ["ask_real_name"]},
        ]
        for change in changes:
            with self.subTest(change=change):
                parent = fixture("name")
                parent["source"].update(change)
                snapshot = copy.deepcopy(parent)
                with self.assertRaises(ValueError):
                    recovery.corrupt_record(parent)
                self.assertEqual(parent, snapshot)
        with self.assertRaises(ValueError):
            recovery.corrupt_record(recovery.corrupt_record(fixture("name")))

    def test_real_trainer_masks_all_earlier_assistant_targets(self):
        for scene, spec in SCENES.items():
            with self.subTest(scene=scene):
                variants = [recovery.corrupt_record(fixture(scene, fold)) for fold in FOLDS]
                with patch.object(training, "_torch", side_effect=AssertionError("No ML backend needed")):
                    encoded = training._encoded(variants, self.tokenizer, context_length=1024)
                expected = self.tokenizer.encode(spec["turns"][-1][1], add_special_tokens=False).ids + [EOS_ID]
                for fold in FOLDS:
                    self.assertEqual(len(encoded[fold]), 1)
                    inputs, labels = encoded[fold][0]
                    last = max(i for i, token in enumerate(inputs) if token == ASSISTANT_ID)
                    self.assertEqual(labels[:last], [IGNORE_INDEX] * last)
                    self.assertEqual([label for label in labels if label != IGNORE_INDEX], expected)
                    # The wrong reply remains context, and every label predicting
                    # it is ignored. Shared vocabulary IDs may occur in targets.
                    variant = variants[FOLDS.index(fold)]
                    prefix_tokens, _ = dialogue_tokens(variant["messages"][:-2], self.tokenizer)
                    false_ids = self.tokenizer.encode(variant["messages"][-3]["content"], add_special_tokens=False).ids
                    start = len(prefix_tokens) - len(false_ids) - 1
                    self.assertEqual(inputs[start:start + len(false_ids)], false_ids)
                    self.assertEqual(labels[start - 1:start + len(false_ids)],
                                     [IGNORE_INDEX] * (len(false_ids) + 1))

    def test_real_trainer_windowing_never_teaches_or_repeats_history(self):
        variants = [recovery.corrupt_record(fixture("name", fold)) for fold in FOLDS]
        expected = self.tokenizer.encode("Te llamas Nevasi.", add_special_tokens=False).ids + [EOS_ID]
        for policy in ("contiguous_nonoverlapping", "sliding_context"):
            with self.subTest(policy=policy):
                with patch.object(training, "_torch", side_effect=AssertionError("No ML backend needed")):
                    encoded = training._encoded(variants, self.tokenizer, 32, policy)
                for fold in FOLDS:
                    self.assertTrue(all(len(inputs) <= 32 for inputs, _ in encoded[fold]))
                    supervised = [label for _, labels in encoded[fold]
                                  for label in labels if label != IGNORE_INDEX]
                    self.assertEqual(supervised, expected)

    def make_inputs(self, root):
        records = [fixture(scene, fold) for fold in FOLDS
                   for scene in ("name", "city", "topic", "intent")]
        records.extend(fixture("name", fold, prefix=1) for fold in FOLDS)
        parent, tokenizer = root / "parent.jsonl", root / "tokenizer.json"
        parent.write_bytes(b"".join(canonical(record) + b"\n" for record in records))
        tokenizer.write_text(self.tokenizer.to_str(), encoding="utf-8")
        return parent, tokenizer, records

    def test_build_preserves_folds_full_dialogues_inputs_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, tokenizer, originals = self.make_inputs(root)
            before = (parent.read_bytes(), tokenizer.read_bytes())
            output, manifest = root / "recovery.jsonl", root / "manifest.json"
            report = recovery.build(parent, tokenizer, output, manifest, seed=73, context_length=4096)
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["parent"]["records"], 15)
            self.assertEqual(report["sha256"], digest(output.read_bytes()))
            self.assertEqual(report["parent"]["sha256"], digest(before[0]))
            self.assertEqual(report["tokenizer"]["sha256"], digest(before[1]))
            self.assertIs(report["tokenizer"]["refitted"], False)
            self.assertIs(report["authoring"]["dialoguesTruncated"], False)
            self.assertEqual(report["authoring"]["assistantTargets"], "last_reply_only")
            self.assertEqual(report["reasons"]["prefix1_or_nonrecovery_target_omitted"], 3)
            self.assertEqual(json.loads(manifest.read_bytes()), report)
            by_id = {record["id"]: record for record in originals}
            for variant in load_corpus(output):
                trace = variant["source"]["recovery"]
                clean = by_id[trace["parentRecordId"]]
                scene = clean["id"].split("-")[2]
                self.assert_variant(clean, variant, scene, 73)
                self.assertEqual(trace["parentCorpusSha256"], digest(before[0]))
                tokens, _ = dialogue_tokens(variant["messages"], self.tokenizer)
                self.assertLessEqual(len(tokens), 4097)
            for fold in FOLDS:
                self.assertEqual(report["selection"][fold], {
                    "parentPrefixesAtLeast2": 4, "fullfitRecoveryCandidates": 4,
                    "selected": 1, "fraction": .25,
                })
                self.assertEqual(report["splits"][fold], 1)
            self.assertEqual((parent.read_bytes(), tokenizer.read_bytes()), before)
            self.assertEqual({path.name for path in root.iterdir()},
                             {"parent.jsonl", "tokenizer.json", "recovery.jsonl", "manifest.json"})

    def test_build_reproducible_with_new_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, tokenizer, _ = self.make_inputs(root)
            first = recovery.build(parent, tokenizer, root / "first.jsonl", root / "first.json", context_length=4096)
            second = recovery.build(parent, tokenizer, root / "second.jsonl", root / "second.json", context_length=4096)
            self.assertEqual(first, second)
            self.assertEqual((root / "first.jsonl").read_bytes(), (root / "second.jsonl").read_bytes())
            self.assertEqual((root / "first.json").read_bytes(), (root / "second.json").read_bytes())

    def test_existing_outputs_and_input_aliases_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, tokenizer, _ = self.make_inputs(root)
            before = (parent.read_bytes(), tokenizer.read_bytes())
            output, manifest = root / "out.jsonl", root / "out.json"
            for occupied in (output, manifest):
                occupied.write_bytes(b"preserved existing evidence\n")
                with patch.object(recovery, "corrupt_record", side_effect=AssertionError("Must reject before authoring")):
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        recovery.build(parent, tokenizer, output, manifest, context_length=4096)
                self.assertEqual(occupied.read_bytes(), b"preserved existing evidence\n")
                occupied.unlink()
            for aliases in ((parent, manifest), (tokenizer, manifest), (output, parent), (output, tokenizer), (output, output)):
                with self.subTest(aliases=aliases):
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        recovery.build(parent, tokenizer, *aliases, context_length=4096)
                    self.assertEqual((parent.read_bytes(), tokenizer.read_bytes()), before)
                    self.assertFalse(output.exists())
                    self.assertFalse(manifest.exists())

    def test_too_short_context_refuses_publication_without_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, tokenizer, _ = self.make_inputs(root)
            before = (parent.read_bytes(), tokenizer.read_bytes())
            output, manifest = root / "out.jsonl", root / "out.json"
            with self.assertRaisesRegex(ValueError, "full-fitting"):
                recovery.build(parent, tokenizer, output, manifest, context_length=32)
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual((parent.read_bytes(), tokenizer.read_bytes()), before)

    def test_exclusive_publication_preserves_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "new.jsonl"
            real_link = recovery.os.link

            def racing_link(source, destination):
                Path(destination).write_bytes(b"other writer's evidence\n")
                return real_link(source, destination)

            with patch.object(recovery.os, "link", side_effect=racing_link):
                with self.assertRaisesRegex(ValueError, "preserved"):
                    recovery.exclusive_write(output, b"replacement\n")
            self.assertEqual(output.read_bytes(), b"other writer's evidence\n")
            self.assertEqual(list(root.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
