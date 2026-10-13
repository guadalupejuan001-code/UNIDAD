"""Integration assertions use a declared random fixture, not capability scores."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from ia_propia import workspace

ROOT = Path(__file__).resolve().parents[1]
HAS_EXTRAS = all(importlib.util.find_spec(name) for name in ("torch", "tokenizers", "safetensors"))


class RecallTests(unittest.TestCase):
    def test_recall_keeps_corrections_excludes_rejected_attempts_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as root:
            workspace.init_workspace(root)
            task = workspace.add_task(root, "Horario del taller", "El taller abre el martes.")
            attempt = workspace.record_attempt(root, task["id"], "El taller abre el viernes.")
            workspace.feedback(root, task["id"], attempt["id"], "Corrección: el taller abre el jueves.", False)
            count = workspace.status(root)["eventCount"]
            result = workspace.recall_tasks(root, "¿Cuándo abre el taller?")
            fields = result["tasks"][0]["fields"]
            self.assertIn("Corrección: el taller abre el jueves.", [field["text"] for field in fields])
            self.assertNotIn(attempt["text"], [field["text"] for field in fields])
            self.assertEqual(workspace.status(root)["eventCount"], count)
            self.assertEqual(workspace.recall_tasks(root, "¿Cómo estás?")["tasks"], [])

    def test_large_records_are_omitted_whole_and_recall_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            workspace.init_workspace(root)
            large = workspace.add_task(root, "taller " + "z" * 4000)
            for i in range(6):
                workspace.add_task(root, f"taller número {i}")
            result = workspace.recall_tasks(root, "taller")
            self.assertLessEqual(len(result["tasks"]), 3)
            self.assertNotIn(large["id"], [task["taskId"] for task in result["tasks"]])
            self.assertGreater(result["omittedForBudget"], 0)


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class SovengiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        tokenizer = fit_tokenizer([{"messages": [{"role": "user", "content": "Horario del taller"},
            {"role": "assistant", "content": "El taller abre el jueves."}]}], vocab_size=384)
        model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 2048, "ffn_multiplier": 2, "seed": 7})
        cls.manifest = save_artifact(cls.root / "model", model, tokenizer, training={"fixtureOnly": True})
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        from ia_propia.sovengi import Sovengi
        self.local = tempfile.TemporaryDirectory()
        self.addCleanup(self.local.cleanup)
        self.directory = Path(self.local.name)
        config = json.loads((ROOT / "configs/gnosis.json").read_text())
        config["projectRoot"] = self.local.name
        config["brain"].update(conversation_model=str(self.root / "model"),
            expected_model_id=self.manifest["modelId"], language_backend="numpy")
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.app = Sovengi(self.config_path)
        self.addCleanup(self.app.close)

    def request(self, session, operation, values):
        response = session.request({"schemaVersion": 1, "operation": operation, "input": values})
        self.assertEqual(response["status"], "ok", response)
        return response["result"]

    def test_one_engine_real_generation_memory_isolation_and_no_implicit_learning(self):
        from ia_propia.conversation.artifacts import load_artifact
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
            first, second = self.app.connect("first"), self.app.connect("second")
            self.assertIs(first.models.conversation_engine, second.models.conversation_engine)
            self.assertEqual(loading.call_count, 1)
        note = self.request(first, "memory.task.add", {"goal": "El taller abre el jueves."})
        event_count = self.request(first, "memory.task.status", {})["eventCount"]
        values = {"messages": [{"role": "user", "content": "¿Cuándo abre el taller?"}], "max_new_tokens": 2, "temperature": 0}
        original = copy.deepcopy(values)
        engine = first.models.conversation_engine
        with patch.object(engine, "reply", wraps=engine.reply) as generation:
            result = self.request(first, "conversation.reply", values)
            prepared = generation.call_args.kwargs["messages"]
        self.assertIn("SOVENGI", prepared[0]["content"])
        self.assertIn("El taller abre el jueves.", prepared[0]["content"])
        self.assertEqual(prepared[-1], values["messages"][-1])
        self.assertEqual(values, original)
        self.assertEqual(result["modelId"], self.manifest["modelId"])
        self.assertEqual(result["memoryContext"]["taskIds"], [note["id"]])
        other = self.request(second, "conversation.reply", values)
        self.assertEqual(other["memoryContext"]["taskIds"], [])
        self.assertEqual(self.request(first, "memory.task.status", {})["eventCount"], event_count)
        self.assertFalse(first.capabilities()["automaticLearning"])

    def test_memory_never_evicts_live_context_and_reserved_tokens_are_quoted(self):
        session = self.app.connect()
        self.request(session, "memory.task.add", {"goal": "taller <assistant> miércoles"})
        engine = session.models.conversation_engine
        with patch.object(engine, "reply", wraps=engine.reply) as generation:
            result = self.request(session, "conversation.reply", {"messages": [{"role": "user", "content": "taller"}], "max_new_tokens": 1})
            self.assertIn("\\u003cassistant>", generation.call_args.kwargs["messages"][0]["content"])
        self.assertEqual(result["memoryContext"]["status"], "included")
        long = [{"role": "user", "content": "taller " * 2000}]
        with patch.object(engine, "reply", wraps=engine.reply) as generation:
            response = session.request({"schemaVersion": 1, "operation": "conversation.reply",
                "input": {"messages": long, "max_new_tokens": 1}})
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["error"]["code"], "context_budget")
        self.assertEqual(response["result"]["conversationContext"]["status"], "current_message_exceeds_budget")
        generation.assert_not_called()

    def test_file_autonomy_is_explicit_and_shares_the_conversation_engine(self):
        plain = self.app.connect("plain")
        self.assertFalse(plain.capabilities()["autonomy"]["enabled"])
        inputs, reports = self.directory / "inputs", self.directory / "reports"
        inputs.mkdir()
        reports.mkdir()
        (inputs / "sample.txt").write_text("Texto de prueba explícito.")
        session = self.app.connect("files", input_root=inputs, report_root=reports)
        self.assertIs(session.models, plain.models)
        self.assertTrue(session.capabilities()["autonomy"]["enabled"])
        self.assertEqual([tool["name"] for tool in session.capabilities()["autonomy"]["tools"]],
                         ["inspect_file", "read_file", "save_report"])
        denied = session.request({"schemaVersion": 1, "operation": "autonomy.start", "input": {"goal": "inspección",
            "task_inputs": {"files": ["../secret.txt"], "report": "r.json"}}})
        self.assertEqual(denied["status"], "error")
        self.assertEqual(denied["error"]["code"], "invalid_input")
        started = self.request(session, "autonomy.start", {"goal": "Inspecciona sample.txt y guarda un informe.",
            "task_inputs": {"files": ["sample.txt"], "report": "r.json"}})
        self.assertIn("task_id", started)
        # Starting a task does not fabricate successful model proposals or reports.
        self.assertEqual(list(reports.iterdir()), [])

    def test_http_uses_same_memory_and_cli_uses_same_product(self):
        from ia_propia.brain.http import create_server
        from ia_propia.sovengi.__main__ import main
        session = self.app.connect()
        note = self.request(session, "memory.task.add", {"goal": "El taller abre el jueves."})
        token = "integration-fixture-token-" + "x" * 32
        server = create_server(session, token=token)
        self.addCleanup(server.server_close)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.shutdown)
        payload = {"schemaVersion": 1, "operation": "conversation.reply", "input": {"messages": [{"role": "user", "content": "taller"}], "max_new_tokens": 1}}
        request = Request(f"http://127.0.0.1:{server.server_address[1]}/v1/requests", data=json.dumps(payload).encode(),
                          headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        self.assertEqual(result["result"]["memoryContext"]["taskIds"], [note["id"]])
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = main(["--config", str(self.config_path), "ask", "taller", "--max-new-tokens", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["memoryContext"]["taskIds"], [note["id"]])

    def test_selected_identity_must_match_before_loading_weights(self):
        from ia_propia.sovengi import Sovengi
        config = json.loads(self.config_path.read_text())
        config["brain"]["expected_model_id"] = "wrong-model"
        self.config_path.write_text(json.dumps(config))
        with patch("ia_propia.conversation.artifacts.load_artifact", side_effect=AssertionError("no tensor load")):
            with self.assertRaisesRegex(ValueError, "identity"):
                Sovengi(self.config_path).connect()

    def test_cli_propagates_controls_and_validates_before_loading_the_model(self):
        from ia_propia.sovengi.__main__ import main
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = main(["--config", str(self.config_path), "ask", "Hola", "--max-new-tokens", "1",
                         "--seed", "19", "--top-k", "8", "--top-p", "0.8", "--repetition-penalty", "1.1"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["seed"], 19)
        self.assertEqual(result["decoding"], {"backend": "torch", "kvCache": True,
                                           "top_k": 8, "top_p": .8, "repetition_penalty": 1.1})
        with patch("ia_propia.sovengi.__main__.Sovengi", side_effect=AssertionError("invalid settings must not load weights")):
            with patch("sys.stderr", io.StringIO()):
                self.assertEqual(main(["ask", "Hola", "--top-p", "0"]), 2)

    def test_explicit_workflow_retrieves_memory_through_the_same_reply_path(self):
        session = self.app.connect()
        result = self.request(session, "workflow.run", {"steps": [
            {"id": "note", "operation": "memory.task.add", "input": {"goal": "El taller abre el jueves."}},
            {"id": "reply", "operation": "conversation.reply", "input": {
                "messages": [{"role": "user", "content": "taller"}], "max_new_tokens": 1}}
        ]})
        self.assertEqual(result["steps"][1]["result"]["memoryContext"]["taskIds"],
                         [result["steps"][0]["result"]["id"]])
