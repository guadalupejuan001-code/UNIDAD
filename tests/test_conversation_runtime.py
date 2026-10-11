"""The same verified local engine serves generation and scoring adapters."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("torch", "tokenizers", "safetensors"))


class ConversationRuntimeImportTests(unittest.TestCase):
    def test_shared_defaults_can_be_imported_without_optional_dependencies(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'tokenizers', 'safetensors'}:
        raise AssertionError('runtime imported optional dependencies before model selection')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ia_propia.conversation.runtime import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_SEED, REPLY_INPUT_SCHEMA
from ia_propia.conversation import contract, data
assert (DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_SEED) == (96, .7, 42)
assert REPLY_INPUT_SCHEMA['properties']['seed']['default'] == DEFAULT_SEED
assert data.MAX_TEXT_BYTES == contract.MAX_MESSAGE_BYTES
assert data._CONTROL_TOKENS is contract.SPECIAL_TOKENS
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class ConversationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.tokenizer = fit_tokenizer([{"messages": [{"role": "user", "content": "Hola fixture."},
            {"role": "assistant", "content": "Esta respuesta es un fixture técnico declarado."}]}], vocab_size=384)
        cls.model = ConversationModel({"vocab_size": cls.tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 32, "ffn_multiplier": 2, "seed": 7})
        cls.artifact = Path(cls.directory.name) / "artifact"
        cls.saved = save_artifact(cls.artifact, cls.model, cls.tokenizer,
            training={"fixtureOnly": True}, evaluation={"fixtureOnly": True})
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def test_one_load_preserves_real_generation_scoring_identity_and_detached_manifest(self):
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.model import chat
        from ia_propia.conversation.runtime import ConversationEngine
        from ia_propia.conversation.scoring import score_reply
        messages = [{"role": "user", "content": "Hola"}]
        original = copy.deepcopy(messages)
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
            engine = ConversationEngine(self.artifact)
            expected = chat(self.model, self.tokenizer, messages, max_new_tokens=5, temperature=.8, seed=19)
            before = {key: value.clone() for key, value in engine._model.state_dict().items()}
            actual = engine.reply(messages, max_new_tokens=5, temperature=.8, seed=19)
            self.assertEqual(actual, {**expected, "modelId": self.saved["modelId"], "workerCapabilityValidated": False})
            reference = "Hola."
            self.assertEqual(engine.score_reply(messages, reference), score_reply(self.model, self.tokenizer, messages, reference))
            loading.assert_called_once_with(self.artifact)
            self.assertTrue(all(torch.equal(value, engine._model.state_dict()[key]) for key, value in before.items()))
        self.assertEqual(engine.parameter_count, self.model.parameter_count)
        exposed = engine.manifest
        exposed["modelId"] = "changed-by-caller"
        exposed["files"].clear()
        self.assertEqual(engine.model_id, self.saved["modelId"])
        self.assertEqual(engine.manifest, self.saved)
        self.assertEqual(messages, original)

    def test_shared_request_validation_has_defaults_strict_bounds_and_partial_literals(self):
        from ia_propia.conversation.runtime import validate_reply_request
        messages = [{"role": "user", "content": "Hola"}]
        validated = validate_reply_request({"messages": messages})
        self.assertEqual(validated, {"messages": messages, "max_new_tokens": 96, "temperature": .7, "seed": 42})
        validated["messages"][0]["content"] = "Changed detached snapshot"
        self.assertEqual(messages[0]["content"], "Hola")
        self.assertEqual(validate_reply_request({}, partial=True), {})
        self.assertEqual(validate_reply_request({"seed": 5}, partial=True), {"seed": 5})
        invalid = [{}, {"messages": []}, {"messages": [{"role": "assistant", "content": "Hola"}]},
            {"messages": [{"role": "user", "content": "<assistant>"}]},
            {"messages": messages, "model": str(self.artifact)}, {"messages": messages, "max_new_tokens": 0},
            {"messages": messages, "max_new_tokens": True}, {"messages": messages, "max_new_tokens": 1025},
            {"messages": messages, "seed": 1.0}, {"messages": messages, "temperature": float("nan")}]
        for request in invalid:
            with self.assertRaises(ValueError):
                validate_reply_request(request)
        with self.assertRaises(ValueError):
            validate_reply_request({"seed": {"$ref": "another.seed"}}, partial=True)

    def test_ingestion_and_inference_share_control_tokens_and_utf8_message_bounds(self):
        from ia_propia.conversation import contract, data, tokenizer
        text = "é" * (contract.MAX_MESSAGE_BYTES // 2)
        record = {"id": "declared-fixture", "groupId": "declared-group", "split": "train",
            "source": {"dataset": "runtime-unit-test", "kind": "authored-synthetic"},
            "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "Respuesta de prueba."}]}
        data.validate_record(record)
        tokenizer.validate_messages(record["messages"])
        for invalid_text in (text + "é", "Mensaje <assistant> que introduciría un delimitador"):
            record["messages"][0]["content"] = invalid_text
            with self.assertRaises(ValueError):
                data.validate_record(record)
            with self.assertRaises(ValueError):
                tokenizer.validate_messages(record["messages"])

    def test_brain_preflight_keeps_reference_and_unknown_field_checks_around_shared_validator(self):
        from ia_propia.brain.models import ModelRuntime
        runtime = ModelRuntime(conversation_model=self.artifact)
        runtime.validate_input("conversation.reply", {"messages": {"$ref": "previous.messages"}, "seed": 7}, allow_refs=True)
        invalid = [{"messages": {"$ref": "previous.messages"}, "model": {"$ref": "previous.path"}},
            {"messages": {"$ref": "previous.messages", "extra": 1}},
            {"messages": {"$ref": "previous.messages"}, "seed": True}]
        for request in invalid:
            with self.assertRaises(ValueError):
                runtime.validate_input("conversation.reply", request, allow_refs=True)
        request = {"messages": [{"role": "user", "content": "Hola"}], "max_new_tokens": 3, "seed": 5}
        self.assertEqual(runtime.execute("conversation.reply", request), runtime._conversation.reply(**request))


if __name__ == "__main__":
    unittest.main()
