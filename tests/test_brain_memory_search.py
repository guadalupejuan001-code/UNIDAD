"""Brain/CLI/HTTP memory contracts with model construction disabled."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import unicodedata
from unittest.mock import patch

from ia_propia.brain import Brain
from ia_propia.brain.__main__ import main
from ia_propia.brain.http import create_server
from ia_propia import workspace


class NoModels:
    def __init__(self, **unused):
        pass

    def capabilities(self):
        return {"operations": [], "models": {}}

    def execute(self, *unused):
        raise AssertionError("Memory retrieval must not run inference")


class BrainMemorySearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = patch("ia_propia.brain.core.ModelRuntime", NoModels)
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.brain = Brain(self.root / "brain")

    def request(self, body, operation, values, **extra):
        response = body.request({"schemaVersion": 1, "operation": operation, "input": values, **extra})
        self.assertEqual(response["status"], "ok", response)
        return response["result"]

    def test_search_scope_is_fixed_by_host_and_only_explicit_sharing_works(self):
        editor, panel = self.brain.connect("editor"), self.brain.connect("panel")
        added = self.request(editor, "memory.task.add", {"goal": "Preparar cata\u0301logo azul"})
        self.assertEqual(self.request(editor, "memory.task.search", {"query": "CATÁLOGO"})["tasks"][0]["id"], added["id"])
        self.assertEqual(self.request(panel, "memory.task.search", {"query": "catálogo"},
                                      context={"memory_scope": "editor", "workspace": str(self.root)})["tasks"], [])
        for field in ("path", "bodyId", "memory_scope", "workspace"):
            response = panel.request({"schemaVersion": 1, "operation": "memory.task.search",
                                      "input": {"query": "catálogo", field: "editor"}})
            self.assertEqual(response["error"]["code"], "invalid_input")
        shared_a = self.brain.connect("a", memory_scope="project")
        shared_b = self.brain.connect("b", memory_scope="project")
        shared = self.request(shared_a, "memory.task.add", {"goal": "Ordenar cuaderno"})
        self.assertEqual(self.request(shared_b, "memory.task.search", {"query": "cuaderno"})["tasks"][0]["id"], shared["id"])
        capability = editor.capabilities()["memory"]["retrieval"]
        self.assertFalse(capability["automaticContextInjection"])
        self.assertEqual(capability["maxResults"], 20)
        self.assertEqual(capability["method"], "lexical_all_words_nfc_casefold")

    def test_references_verify_against_show_and_rejection_changes_next_search(self):
        body = self.brain.connect("references")
        task = self.request(body, "memory.task.add", {"goal": "Preparar cartel", "context": "Para feria"})
        attempt = self.request(body, "memory.task.attempt", {"task_id": task["id"], "text": "Diseño violeta"})
        values = {"task_id": task["id"], "attempt_id": attempt["id"], "text": "Aprobado jueves", "accepted": True}
        self.request(body, "memory.task.feedback", values)
        result = self.request(body, "memory.task.search", {"query": "feria violeta jueves"})
        shown = self.request(body, "memory.task.show", {"task_id": task["id"]})
        for match in result["tasks"][0]["matches"]:
            source = match["source"]
            self.assertEqual(source["taskId"], shown["id"])
            if source["field"] == "context":
                text, event = shown["context"], shown["eventId"]
            else:
                original = next(item for item in shown["attempts"] if item["id"] == source["attemptId"])
                record = original if source["field"] == "attempt.text" else original["feedback"][-1]
                text, event = record["text"], record["eventId"]
            self.assertEqual(source["eventId"], event)
            self.assertEqual(source["textSha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(unicodedata.normalize("NFC", text[source["start"]:source["end"]]).casefold(), match["term"])
            self.assertEqual(text[match["excerptStart"]:match["excerptEnd"]], match["excerpt"])
        self.request(body, "memory.task.feedback", {**values, "text": "Mejor sábado", "accepted": False})
        self.assertEqual(self.request(body, "memory.task.search", {"query": "violeta"})["tasks"], [])
        self.assertEqual(self.request(body, "memory.task.search", {"query": "jueves"})["tasks"], [])
        latest = self.request(body, "memory.task.search", {"query": "sábado"})["tasks"][0]["matches"][0]
        self.assertFalse(latest["source"]["accepted"])

    def test_invalid_queries_fail_before_workspace_access_and_workflow_effects(self):
        body = self.brain.connect("validation")
        with patch("ia_propia.brain.core.workspace.search_tasks") as search:
            for values in ({"query": "!"}, {"query": "x", "limit": True}, {"query": "é" * 257},
                           {"query": "x", "limit": 21}, {"query": "x", "path": str(self.root)}):
                response = body.request({"schemaVersion": 1, "operation": "memory.task.search", "input": values})
                self.assertEqual(response["error"]["code"], "invalid_input")
            search.assert_not_called()
        response = body.request({"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "add", "operation": "memory.task.add", "input": {"goal": "Never recorded"}},
            {"id": "find", "operation": "memory.task.search", "input": {"query": "x", "limit": False}},
        ]}})
        self.assertEqual(response["error"]["code"], "invalid_input")
        self.assertEqual(self.request(body, "memory.task.status", {})["taskCount"], 0)

    def test_workflow_query_references_are_resolved_and_revalidated(self):
        body = self.brain.connect("workflow")
        result = self.request(body, "workflow.run", {"steps": [
            {"id": "add", "operation": "memory.task.add", "input": {"goal": "Catálogo nuevo"}},
            {"id": "find", "operation": "memory.task.search", "input": {"query": {"$ref": "add.goal"}}},
            {"id": "show", "operation": "memory.task.show", "input": {"task_id": {"$ref": "find.tasks.0.id"}}},
        ]})
        self.assertEqual(result["completedCount"], 3)
        self.assertEqual(result["steps"][0]["result"]["id"], result["steps"][2]["result"]["id"])
        response = body.request({"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "status", "operation": "memory.task.status", "input": {}},
            {"id": "bad", "operation": "memory.task.search", "input": {"query": {"$ref": "status.taskCount"}}},
        ]}})
        self.assertEqual(response["error"]["code"], "workflow_failed")
        self.assertEqual(response["result"]["failedStep"]["error"]["code"], "invalid_input")

    def test_search_reads_one_snapshot_and_does_not_change_history(self):
        body = self.brain.connect("readonly")
        self.request(body, "memory.task.add", {"goal": "Preparar informe"})
        log = body._memory_directory / "events.jsonl"
        before = log.read_bytes()
        with patch.object(workspace, "_load", wraps=workspace._load) as load, \
                patch.object(workspace, "show", side_effect=AssertionError("No full history copy")):
            self.request(body, "memory.task.search", {"query": "informe"})
        self.assertEqual(load.call_count, 1)
        self.assertEqual(log.read_bytes(), before)

    def test_cli_search_returns_same_contract_and_rejects_cross_body_selection(self):
        state = self.root / "cli"
        def call(operation, values, body="cli-product"):
            payload = json.dumps({"schemaVersion": 1, "operation": operation, "input": values})
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors), patch("sys.stdin", io.StringIO(payload)):
                code = main(["call", "--state-dir", str(state), "--body", body, "--input", "-"])
            self.assertEqual(errors.getvalue(), "")
            return code, json.loads(output.getvalue())
        code, added = call("memory.task.add", {"goal": "Preparar folleto"})
        self.assertEqual(code, 0)
        code, result = call("memory.task.search", {"query": "FOLLETO", "limit": 1})
        self.assertEqual(code, 0)
        self.assertEqual(result["result"]["tasks"][0]["id"], added["result"]["id"])
        self.assertEqual(call("memory.task.search", {"query": "folleto"}, "another")[1]["result"]["tasks"], [])
        self.assertEqual(call("memory.task.search", {"query": "folleto", "limit": False})[0], 2)

    def test_http_search_uses_fixed_session_and_credentials_never_appear_in_logs(self):
        body = self.brain.connect("http-product")
        added = self.request(body, "memory.task.add", {"goal": "Revisar cuaderno"})
        token = "synthetic-search-token-00000000000000"
        logs = io.StringIO()
        server = create_server(body, token, port=0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            with redirect_stderr(logs):
                connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                try:
                    payload = json.dumps({"schemaVersion": 1, "operation": "memory.task.search",
                                          "input": {"query": "cuaderno", "limit": 1},
                                          "context": {"bodyId": "another", "memory_scope": "elsewhere"}})
                    connection.request("POST", "/v1/requests", body=payload,
                                       headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                    response = connection.getresponse()
                    result = json.loads(response.read())
                    self.assertEqual(response.status, 200)
                    self.assertEqual(result["bodyId"], "http-product")
                    self.assertEqual(result["result"]["tasks"][0]["id"], added["id"])
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertNotIn(token, logs.getvalue())
        self.assertNotIn("cuaderno", logs.getvalue())


if __name__ == "__main__":
    unittest.main()
