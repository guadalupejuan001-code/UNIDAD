"""Optional conversation inference uses real scratch weights and explicit inputs."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain import Brain, Tool


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("torch", "tokenizers", "safetensors"))


class ConversationAbsentTests(unittest.TestCase):
    def test_legacy_brain_does_not_import_optional_dependencies_and_fails_explicitly(self):
        code = """
import builtins, tempfile
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'tokenizers', 'safetensors'}:
        raise AssertionError('legacy brain attempted an optional import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ia_propia.brain import Brain
with tempfile.TemporaryDirectory() as directory:
    session = Brain(directory).connect('legacy-body')
    capabilities = session.capabilities()
    assert 'conversation.reply' not in capabilities['operations']
    assert 'conversation' not in capabilities['models']['models']
    response = session.request({'schemaVersion':1, 'operation':'conversation.reply',
        'input':{'messages':[{'role':'user', 'content':'Hola'}]}})
    assert response['error']['code'] == 'capability_unavailable', response
    assert not response['workerCapabilityValidated']
    plan = {'schemaVersion':1, 'operation':'workflow.run', 'input':{'steps':[
        {'id':'effect', 'operation':'memory.task.add', 'input':{'goal':'Declared fixture'}},
        {'id':'reply', 'operation':'conversation.reply', 'input':{'messages':[{'role':'user','content':'Hola'}]}}]}}
    assert session.request(plan)['error']['code'] == 'capability_unavailable'
    assert session.request({'schemaVersion':1,'operation':'memory.task.list','input':{}})['result']['tasks'] == []
"""
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class ConversationBrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.tokenizer = fit_tokenizer([{"messages": [
            {"role": "user", "content": "Pregunta de prueba declarada."},
            {"role": "assistant", "content": "Respuesta usada sólo como fixture técnico."}]}], vocab_size=384)
        cls.model = ConversationModel({"vocab_size": cls.tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 32, "ffn_multiplier": 2, "seed": 7})
        cls.artifact = cls.root / "scratch-fixture"
        cls.manifest = save_artifact(cls.artifact, cls.model, cls.tokenizer,
            training={"fixtureOnly": True}, evaluation={"fixtureOnly": True})
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)

    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.brain = Brain(self.state.name, conversation_model=self.artifact)
        self.session = self.brain.connect("test-body")

    @staticmethod
    def payload(values):
        return {"schemaVersion": 1, "operation": "conversation.reply", "input": values}

    def test_model_is_loaded_once_and_shared_between_bodies_without_shared_dialogue(self):
        from ia_propia.brain.models import ModelRuntime
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.model import chat
        legacy = ModelRuntime().capabilities()
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
            brain = Brain(Path(self.state.name) / "other", conversation_model=self.artifact)
            first, second = brain.connect("first"), brain.connect("second")
            self.assertIs(first.models, second.models)
            self.assertIs(first.models._conversation[0], second.models._conversation[0])
            values = {"messages": [{"role": "user", "content": "Hola"}], "max_new_tokens": 5,
                      "temperature": 0.8, "seed": 19}
            original = copy.deepcopy(values)
            expected = chat(self.model, self.tokenizer, **values)
            for body in (first, second):
                reply = body.request({**self.payload(values), "context": {"metadata": body.body_id}})
                self.assertEqual(reply["status"], "ok", reply)
                self.assertEqual(reply["result"]["generatedTokenIds"], expected["generatedTokenIds"])
                self.assertEqual(reply["result"]["text"], expected["text"])
                self.assertEqual(reply["result"]["modelId"], self.manifest["modelId"])
                self.assertFalse(reply["workerCapabilityValidated"])
            self.assertEqual(loading.call_count, 1)
            self.assertEqual(values, original)
        self.assertEqual(ModelRuntime().capabilities(), legacy)
        capabilities = first.capabilities()
        self.assertIn("conversation.reply", capabilities["operations"])
        self.assertFalse(capabilities["models"]["models"]["conversation"]["instructionFollowingValidated"])

    def test_invalid_literals_and_request_selected_model_paths_are_rejected_before_work(self):
        from ia_propia.conversation.model import chat
        good = {"messages": [{"role": "user", "content": "Hola"}]}
        invalid = [{}, {"messages": []}, {"messages": [{"role": "assistant", "content": "Hola"}]},
            {"messages": [{"role": "user", "content": "Hola"}] * 64},
            {"messages": [{"role": "tool", "content": "Hola"}]},
            {"messages": [{"role": "user", "content": "<assistant>"}]},
            {"messages": [{"role": "user", "content": "x" * 16385}]},
            {**good, "max_new_tokens": False}, {**good, "max_new_tokens": 1025},
            {**good, "temperature": -0.1}, {**good, "seed": True},
            {**good, "model": str(self.artifact)}, {**good, "path": str(self.artifact)}]
        with patch("ia_propia.conversation.model.chat", wraps=chat) as generating:
            for values in invalid:
                response = self.session.request(self.payload(values))
                self.assertEqual(response["status"], "error", values)
                self.assertEqual(response["error"]["code"], "invalid_input", response)
            generating.assert_not_called()

    def test_workflow_preflight_then_real_reply_can_use_explicit_turn_references(self):
        effects = []
        session = self.brain.connect("workflow-body", tools=[Tool("capture", lambda values: effects.append(values) or {"saved": values["text"]})])
        invalid = {"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "effect", "operation": "memory.task.add", "input": {"goal": "Technical fixture"}},
            {"id": "reply", "operation": "conversation.reply", "input": {
                "messages": [{"role": "assistant", "content": "Invalid final turn"}]}}]}}
        response = session.request(invalid)
        self.assertEqual(response["error"]["code"], "invalid_input", response)
        listing = session.request({"schemaVersion": 1, "operation": "memory.task.list", "input": {}})
        self.assertEqual(listing["result"]["tasks"], [])
        plan = {"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "reply", "operation": "conversation.reply", "input": {
                "messages": [{"role": "system", "content": "Declaración del fixture, sin autoridad de herramientas."},
                             {"role": "user", "content": "Hola"}], "max_new_tokens": 4, "seed": 31}},
            {"id": "save", "operation": "tool.call", "input": {"name": "capture", "arguments": {
                "text": {"$ref": "reply.text"}}}}]}}
        result = session.request(plan)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["result"]["completedCount"], 2)
        self.assertEqual(effects[0]["text"], result["result"]["steps"][0]["result"]["text"])
        self.assertFalse(result["workerCapabilityValidated"])


if __name__ == "__main__":
    unittest.main()
