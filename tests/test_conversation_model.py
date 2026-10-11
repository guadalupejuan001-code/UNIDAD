"""Causal supervision, local randomness and safe portable Transformer artifacts."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if any(importlib.util.find_spec(name) is None for name in ("torch", "tokenizers", "safetensors")):
    raise unittest.SkipTest("Conversation model tests require the optional local-training dependencies")

import torch
from safetensors.torch import load as load_tensors, save as save_tensors
from torch.nn import functional as F

from ia_propia.conversation.artifacts import artifact_model_id, load_artifact, save_artifact
from ia_propia.conversation.model import ConversationModel, assistant_token_loss, chat
from ia_propia.conversation.tokenizer import (ASSISTANT_ID, BOS_ID, ConversationError, DEFAULT_SYSTEM_PROMPT,
    EOS_ID, IGNORE_INDEX, MAX_DIALOGUE_BYTES, MAX_MESSAGE_BYTES, PAD_ID, SPECIAL_TOKENS,
    dialogue_tokens, encode_dialogue, fit_tokenizer, prepare_messages, validate_messages)


MESSAGES = [{"role": "system", "content": "Habla en español y distingue los hechos."},
            {"role": "user", "content": "Hola, ¿podemos aprender juntos?"},
            {"role": "assistant", "content": "Sí. Podemos practicar y revisar cada intento."}]


class ConversationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer = fit_tokenizer([{"messages": MESSAGES}] * 4, vocab_size=384)
        cls.config = {"vocab_size": cls.tokenizer.get_vocab_size(), "d_model": 16, "n_layers": 2,
                      "n_heads": 2, "context_length": 128, "dropout": 0.0, "ffn_multiplier": 2, "seed": 42}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def model(self, **changed):
        return ConversationModel({**self.config, **changed})

    def export(self, directory):
        return save_artifact(directory, self.model(), self.tokenizer,
                             training={"fixtureOnly": True, "initializedFrom": "random"},
                             evaluation={"fixtureOnly": True})

    def rehash(self, directory, name):
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        raw = (path / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        manifest["modelId"] = artifact_model_id(manifest["files"])
        (path / "manifest.json").write_text(json.dumps(manifest))

    def test_future_tokens_cannot_change_earlier_causal_logits(self):
        model = self.model().eval()
        first = torch.tensor([[1, 4, 30, 31, 32, 33, 34, 35]], dtype=torch.long)
        changed = first.clone()
        changed[:, 4:] = torch.tensor([80, 81, 82, 83])
        with torch.no_grad():
            before, after = model(first), model(changed)
        torch.testing.assert_close(before[:, :4], after[:, :4], atol=1e-7, rtol=1e-6)
        self.assertFalse(torch.allclose(before[:, 4:], after[:, 4:]))

    def test_initialization_is_repeatable_and_does_not_change_caller_rng(self):
        state = torch.get_rng_state().clone()
        first, second = self.model(), self.model()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))
        self.assertFalse(torch.equal(first.token_embedding.weight, self.model(seed=43).token_embedding.weight))

    def test_cpu_float32_parameters_do_not_depend_on_host_default_dtype(self):
        before = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = self.model()
        finally:
            torch.set_default_dtype(before)
        self.assertTrue(all(parameter.dtype == torch.float32 and parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_tied_output_has_one_weight_copy_and_expected_parameter_budget(self):
        model = self.model()
        config = model.config
        d, m, layers = config["d_model"], config["ffn_multiplier"], config["n_layers"]
        expected = config["vocab_size"] * d + config["context_length"] * d + 2 * d
        expected += layers * (4 * d * d + 2 * m * d * d + (9 + m) * d)
        self.assertEqual(model.parameter_count, expected)
        self.assertNotIn("lm_head.weight", model.state_dict())
        snapshot = model.config
        snapshot["vocab_size"] = 1
        self.assertEqual(model.config["vocab_size"], self.config["vocab_size"])

    def test_model_configuration_and_input_shapes_fail_before_inference(self):
        for changed in ({"d_model": True}, {"n_heads": 3}, {"vocab_size": 2}, {"context_length": 5000},
                        {"dropout": float("nan")}, {"d_model": 1024, "n_layers": 24}, {"unknown": 1}):
            with self.subTest(changed=changed), self.assertRaises(ConversationError):
                self.model(**changed)
        model = self.model()
        for invalid in (torch.tensor([1, 2]), torch.tensor([[1.0]]), torch.tensor([[-1]]),
                        torch.tensor([[self.config["vocab_size"]]]), torch.zeros((1, 129), dtype=torch.long)):
            with self.assertRaises(ConversationError):
                model(invalid)

    def test_dialogue_labels_teach_only_assistant_text_and_eos(self):
        ids, supervised = dialogue_tokens(MESSAGES, self.tokenizer)
        inputs, labels = encode_dialogue(MESSAGES, self.tokenizer, context_length=None)
        self.assertEqual(inputs, ids[:-1])
        self.assertEqual(labels, [value if teach else IGNORE_INDEX for value, teach in zip(ids[1:], supervised[1:])])
        assistant_position = ids.index(ASSISTANT_ID)
        self.assertTrue(all(label == IGNORE_INDEX for label in labels[:assistant_position]))
        self.assertTrue(all(label >= 0 for label in labels[assistant_position:]))
        self.assertEqual(labels[-1], EOS_ID)
        self.assertEqual(inputs[0], BOS_ID)
        self.assertNotIn(PAD_ID, inputs)

    def test_full_encoding_keeps_early_and_late_assistant_targets_for_windowing(self):
        messages = [MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "Respuesta extensa. " * 100}]
        full_x, full_y = encode_dialogue(messages, self.tokenizer, context_length=None)
        short_x, short_y = encode_dialogue(messages, self.tokenizer, context_length=32)
        self.assertGreater(len(full_x), 128)
        self.assertEqual(short_x, full_x[-32:])
        self.assertEqual(short_y, full_y[-32:])
        chunks = [full_y[start:start + 32] for start in range(0, len(full_y), 32)]
        self.assertGreater(sum(any(target != IGNORE_INDEX for target in chunk) for chunk in chunks), 1)
        self.assertEqual(sum(target != IGNORE_INDEX for chunk in chunks for target in chunk),
                         sum(target != IGNORE_INDEX for target in full_y))
        self.assertEqual(chunks[-1][-1], EOS_ID)

    def test_assistant_loss_ignores_other_positions_and_masks_their_gradients(self):
        logits = torch.randn((1, 4, self.config["vocab_size"]), generator=torch.Generator().manual_seed(3), requires_grad=True)
        labels = torch.tensor([[IGNORE_INDEX, 70, IGNORE_INDEX, EOS_ID]])
        loss = assistant_token_loss(logits, labels)
        reference = F.cross_entropy(logits[0, [1, 3]], labels[0, [1, 3]])
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, [0, 2]] == 0))
        self.assertGreater(float(logits.grad[0, [1, 3]].abs().sum()), 0)
        changed = logits.detach().clone()
        changed[0, [0, 2]] += 100
        torch.testing.assert_close(assistant_token_loss(changed, labels), loss.detach())
        with self.assertRaises(ConversationError):
            assistant_token_loss(logits, torch.full((1, 4), IGNORE_INDEX, dtype=torch.long))
        invalid = logits.detach().clone()
        invalid[0, 1, 3] = float("inf")
        with self.assertRaises(ConversationError):
            assistant_token_loss(invalid, labels)

    def test_tokenizer_is_local_byte_level_and_roundtrips_new_unicode(self):
        for token_id, token in enumerate(SPECIAL_TOKENS):
            self.assertEqual(self.tokenizer.token_to_id(token), token_id)
        text = "¿Qué tal? Español, código y una mariposa 🦋."
        encoded = self.tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(self.tokenizer.decode(encoded), text)
        with self.assertRaises(ConversationError):
            fit_tokenizer([], vocab_size=384)

    def test_default_system_is_consistent_and_an_explicit_system_is_not_duplicated(self):
        omitted = MESSAGES[1:]
        prepared = prepare_messages(omitted)
        self.assertEqual(prepared[0]["content"], DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(prepare_messages(MESSAGES), MESSAGES)
        explicit_default = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *omitted]
        self.assertEqual(encode_dialogue(omitted, self.tokenizer, None),
                         encode_dialogue(explicit_default, self.tokenizer, None))

    def test_chat_is_seeded_thread_safe_and_restores_model_mode_and_global_rng(self):
        model = self.model()
        messages = MESSAGES[:-1]
        rng = torch.get_rng_state().clone()
        before = chat(model, self.tokenizer, messages, max_new_tokens=8, seed=9)
        after = chat(model, self.tokenizer, messages, max_new_tokens=8, seed=9)
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        with ThreadPoolExecutor(max_workers=2) as executor:
            parallel = list(executor.map(lambda _: chat(model, self.tokenizer, messages, max_new_tokens=8, seed=9), range(4)))
        self.assertTrue(all(response == before for response in parallel))
        self.assertLessEqual(before["generatedTokens"], 8)
        self.assertFalse(before["generalConversationValidated"])
        self.assertFalse(before["instructionFollowingValidated"])

    def test_portable_request_values_are_bounded_and_control_injection_is_rejected(self):
        for messages in ([], [{"role": "unknown", "content": "Hola"}],
                         [{"role": "user", "content": "<assistant>"}],
                         [{"role": "user", "content": "x" * (MAX_MESSAGE_BYTES + 1)}],
                         [{"role": "user", "content": "Hola", "model": "external-file"}],
                         [MESSAGES[1], MESSAGES[0]],
                         [{"role": "user", "content": "x" * MAX_MESSAGE_BYTES}] * 5,
                         [MESSAGES[1]] * 64):
            with self.assertRaises(ConversationError):
                validate_messages(messages)
        self.assertGreater(MAX_DIALOGUE_BYTES, MAX_MESSAGE_BYTES)
        model = self.model()
        for settings in ({"max_new_tokens": 1025}, {"max_new_tokens": 0}, {"max_new_tokens": True},
                         {"temperature": float("nan")}, {"temperature": 3}, {"seed": -1}, {"seed": "42"}):
            with self.assertRaises(ConversationError):
                chat(model, self.tokenizer, MESSAGES[:-1], **settings)
        with self.assertRaises(ConversationError):
            chat(model, self.tokenizer, MESSAGES, max_new_tokens=2)

    def test_safe_artifact_roundtrip_preserves_logits_and_seeded_generation(self):
        original = self.model().eval()
        with tempfile.TemporaryDirectory() as folder:
            manifest = save_artifact(folder, original, self.tokenizer, training={"fixtureOnly": True})
            loaded, tokenizer, restored = load_artifact(folder)
            self.assertEqual(manifest, restored)
            self.assertFalse(loaded.training)
            inputs = torch.tensor([[1, 4, 50, 2, 5]])
            with torch.no_grad():
                torch.testing.assert_close(original(inputs), loaded(inputs), rtol=0, atol=0)
            self.assertEqual(chat(original, self.tokenizer, MESSAGES[:-1], max_new_tokens=8, seed=3),
                             chat(loaded, tokenizer, MESSAGES[:-1], max_new_tokens=8, seed=3))
            self.assertEqual(set(path.name for path in Path(folder).iterdir()),
                             {"config.json", "tokenizer.json", "weights.safetensors", "manifest.json"})

    def test_any_member_tampering_or_manifest_identity_change_is_rejected(self):
        for name in ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json"):
            with tempfile.TemporaryDirectory() as folder:
                self.export(folder)
                path = Path(folder) / name
                original = path.read_bytes()
                if name == "manifest.json":
                    value = json.loads(original)
                    value["modelId"] = "unidad-conversation-incorrect"
                    path.write_text(json.dumps(value))
                else:
                    path.write_bytes(original + b" ")
                with self.assertRaises(ConversationError):
                    load_artifact(folder)

    def test_even_rehashed_wrong_shapes_nonfinite_weights_and_dtype_are_rejected(self):
        for mutation in ("shape", "nan", "dtype", "extra_tensor"):
            with tempfile.TemporaryDirectory() as folder:
                self.export(folder)
                path = Path(folder) / "weights.safetensors"
                tensors = load_tensors(path.read_bytes())
                name = "token_embedding.weight"
                if mutation == "shape":
                    tensors[name] = tensors[name][:-1].contiguous()
                elif mutation == "nan":
                    tensors[name][0, 0] = float("nan")
                elif mutation == "dtype":
                    tensors[name] = tensors[name].to(torch.float64)
                else:
                    tensors["untrusted_tensor"] = torch.zeros(1)
                path.write_bytes(save_tensors(tensors))
                self.rehash(folder, "weights.safetensors")
                with self.assertRaises(ConversationError):
                    load_artifact(folder)

    def test_rehashed_tokenizer_cannot_introduce_large_hidden_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            self.export(folder)
            path = Path(folder) / "tokenizer.json"
            from tokenizers import Tokenizer
            changed = Tokenizer.from_str(path.read_text())
            changed.enable_padding(length=1_000_000, pad_id=PAD_ID, pad_token=SPECIAL_TOKENS[0])
            path.write_text(changed.to_str())
            self.rehash(folder, "tokenizer.json")
            with self.assertRaisesRegex(ConversationError, "unpadded"):
                load_artifact(folder)

    def test_rehashed_malformed_safe_files_raise_the_public_validation_error(self):
        for name, malformed in (("tokenizer.json", b'{"invalid":"tokenizer"}'),
                                ("weights.safetensors", b"invalid safetensors bytes")):
            with tempfile.TemporaryDirectory() as folder:
                self.export(folder)
                (Path(folder) / name).write_bytes(malformed)
                self.rehash(folder, name)
                with self.assertRaises(ConversationError):
                    load_artifact(folder)

    def test_artifact_paths_and_duplicate_metadata_do_not_load_arbitrary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            self.export(folder)
            path = Path(folder) / "manifest.json"
            raw = path.read_text()
            path.write_text(raw[:-1] + ',"schemaVersion":1}')
            with self.assertRaises(ConversationError):
                load_artifact(folder)
        with tempfile.TemporaryDirectory() as folder:
            self.export(folder)
            path = Path(folder) / "tokenizer.json"
            original = path.read_bytes()
            path.unlink()
            external = Path(folder) / "external.json"
            external.write_bytes(original)
            path.symlink_to(external)
            with self.assertRaises(ConversationError):
                load_artifact(folder)


if __name__ == "__main__":
    unittest.main()
