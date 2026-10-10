"""Verify real models and product effects across two interfaces of one brain."""

from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest

from ia_propia.brain import Brain, Tool
from ia_propia.brain.http import create_server


ROOT = Path(__file__).resolve().parents[1]


class BrainIntegrationTests(unittest.TestCase):
    def test_same_models_across_python_and_http_with_isolated_product_tools_and_memory(self):
        expected = json.loads((ROOT / "docs/language-demo-generation.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "product-document.txt"

            def save(arguments):
                destination.write_text(arguments["text"], encoding="utf-8")
                return {"saved": True, "bytes": destination.stat().st_size}

            brain = Brain(state_dir=Path(temporary) / "state")
            editor = brain.connect("editor", tools=[Tool("draft.save", save)])
            dashboard = brain.connect("dashboard")
            generation = {"prompt": "Una tarea ", "length": 160, "seed": 42, "temperature": 0.8}
            workflow = editor.request({"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
                {"id": "draft", "operation": "language.generate", "input": generation},
                {"id": "store", "operation": "tool.call", "input": {
                    "name": "draft.save", "arguments": {"text": {"$ref": "draft.continuation"}}}}]}})
            self.assertEqual(workflow["status"], "ok", workflow)
            self.assertEqual(workflow["result"]["completedCount"], 2)
            self.assertEqual(workflow["result"]["steps"][0]["result"], expected)
            self.assertEqual(destination.read_bytes(), expected["continuation"].encode("utf-8"))
            added = editor.request({"schemaVersion": 1, "operation": "memory.task.add", "input": {
                "goal": "Example private editor assignment", "context": "Fictional test data"}})
            self.assertEqual(added["status"], "ok", added)
            token = "unidad-test-only-credential-0123456789"
            with create_server(dashboard, token) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    def post(operation, inputdict):
                        client = HTTPConnection(*server.server_address, timeout=10)
                        try:
                            client.request("POST", "/v1/requests", body=json.dumps({"schemaVersion": 1,
                                "operation": operation, "input": inputdict}).encode("utf-8"), headers={
                                    "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                            response = client.getresponse()
                            return response.status, json.loads(response.read())
                        finally:
                            client.close()

                    code, via_http = post("language.generate", generation)
                    self.assertEqual(code, 200, via_http)
                    self.assertEqual(via_http["bodyId"], "dashboard")
                    self.assertEqual(via_http["result"], expected)
                    code, listing = post("memory.task.list", {})
                    self.assertEqual(code, 200, listing)
                    self.assertEqual(listing["result"]["tasks"], [])
                    code, denied = post("tool.call", {"name": "draft.save", "arguments": {"text": "overwrite"}})
                    self.assertEqual(code, 403, denied)
                    self.assertEqual(denied["status"], "error")
                    self.assertEqual(destination.read_bytes(), expected["continuation"].encode("utf-8"))
                finally:
                    server.shutdown()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
