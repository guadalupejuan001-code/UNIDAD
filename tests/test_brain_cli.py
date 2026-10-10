"""Public brain commands use the same local operations without guessing intent."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain.__main__ import main


class BrainCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / "private-brain"

    def call(self, arguments, *, stdin=None):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            if stdin is None:
                code = main(arguments, prog="unidad brain")
            else:
                with patch("sys.stdin", io.StringIO(stdin)):
                    code = main(arguments, prog="unidad brain")
        return code, output.getvalue(), errors.getvalue()

    def request(self, operation, input_data, *, body="cli-product", request_id="synthetic-cli-request"):
        payload = {"schemaVersion": 1, "requestId": request_id, "operation": operation, "input": input_data}
        code, output, errors = self.call(["call", "--state-dir", str(self.state), "--body", body,
                                          "--input", "-"], stdin=json.dumps(payload))
        self.assertEqual(errors, "")
        return code, json.loads(output)

    def test_help_has_the_portable_public_commands_and_localhost_default(self):
        for command in ([], ["capabilities"], ["call"], ["serve"]):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                main([*command, "--help"], prog="unidad brain")
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn("usage: unidad brain", output.getvalue())
        from ia_propia.brain.__main__ import parser
        configured = parser().parse_args(["serve"])
        self.assertEqual(configured.host, "127.0.0.1")
        self.assertEqual(configured.body, "product-cli")
        self.assertEqual(configured.token_env, "UNIDAD_API_TOKEN")

    def test_actual_capabilities_identify_configured_body_and_model_limits(self):
        code, output, error = self.call(["capabilities", "--state-dir", str(self.state), "--body", "capabilities-product"])
        self.assertEqual(code, 0, error)
        capabilities = json.loads(output)
        self.assertEqual(capabilities["bodyId"], "capabilities-product")
        self.assertFalse(capabilities["workerCapabilityValidated"])
        self.assertIn("language.generate", capabilities["operations"])
        self.assertIn("memory.task.add", capabilities["operations"])

    def test_request_file_returns_envelope_without_overwriting_input(self):
        path = self.root / "explicit-request.json"
        original = json.dumps({"schemaVersion": 1, "requestId": "file-request", "operation": "memory.task.add",
                               "input": {"goal": "Synthetic CLI task", "context": "Declared local fixture"}}).encode()
        path.write_bytes(original)
        code, output, error = self.call(["call", "--state-dir", str(self.state), "--body", "file-product", "--input", str(path)])
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        self.assertEqual(result["requestId"], "file-request")
        self.assertEqual(result["bodyId"], "file-product")
        self.assertEqual(result["result"]["goal"], "Synthetic CLI task")
        self.assertEqual(path.read_bytes(), original)

    def test_stdin_memory_and_model_requests_use_one_explicit_body(self):
        code, added = self.request("memory.task.add", {"goal": "Synthetic free-form task"})
        self.assertEqual(code, 0, added)
        code, listed = self.request("memory.task.list", {})
        self.assertEqual(code, 0)
        self.assertEqual(listed["result"]["tasks"][0]["id"], added["result"]["id"])
        _, other = self.request("memory.task.list", {}, body="other-cli-product")
        self.assertEqual(other["result"]["tasks"], [])
        code, generated = self.request("language.generate", {"prompt": "Unidad", "length": 8, "seed": 7})
        self.assertEqual(code, 0, generated)
        self.assertEqual(generated["bodyId"], "cli-product")
        self.assertFalse(generated["workerCapabilityValidated"])

    def test_invalid_and_oversized_input_returns_json_error_without_echoing_payload(self):
        from ia_propia.brain.protocol import MAX_REQUEST_BYTES
        inputs = ['{"input":{},"input":{"secret":"PRIVATE_INPUT_MARKER"}}', '{"x":NaN}',
                  '{"x":Infinity}', '{"x":1e999}', '{"x":"' + "a" * MAX_REQUEST_BYTES + '"}']
        for raw in inputs:
            code, output, error = self.call(["call", "--state-dir", str(self.state), "--input", "-"], stdin=raw)
            self.assertEqual(code, 2)
            self.assertEqual(error, "")
            envelope = json.loads(output)
            self.assertEqual(envelope["status"], "error")
            self.assertEqual(envelope["error"]["code"], "invalid_json")
            self.assertNotIn("PRIVATE_INPUT_MARKER", output)

    def test_unsupported_operations_and_unbound_tools_fail_without_guessing_intent(self):
        for operation, arguments in (("write_a_novel", {"goal": "Synthetic text"}),
                                     ("tool.call", {"name": "unbound-tool", "arguments": {}}),
                                     ("language.generate", {"model": "private/arbitrary.json"})):
            code, result = self.request(operation, arguments)
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "error")

    def test_missing_or_short_http_token_is_rejected_before_creating_private_state(self):
        for value in (None, "SHORT_TOKEN_VALUE", "\n" + "x" * 40, "ñ" * 40):
            with patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["UNIDAD_API_TOKEN"] = value
                with patch("ia_propia.brain.__main__.Brain") as brain, patch("ia_propia.brain.__main__.create_server") as server:
                    code, output, errors = self.call(["serve", "--state-dir", str(self.state)])
                    self.assertEqual(code, 2)
                    self.assertEqual(output, "")
                    self.assertIn("API token", errors)
                    if value:
                        self.assertNotIn(value, errors)
                    brain.assert_not_called()
                    server.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_serve_keyboard_interrupt_closes_server_without_printing_credentials(self):
        class Session:
            body_id = "fixture-server-body"

        class Server:
            server_address = ("127.0.0.1", 45678)
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                self.closed = True

            def serve_forever(self):
                raise KeyboardInterrupt

        token = "synthetic-cli-test-token-00000000000000"
        configured = Server()
        with patch.dict(os.environ, {"UNIDAD_API_TOKEN": token}), patch("ia_propia.brain.__main__.Brain") as brain:
            brain.return_value.connect.return_value = Session()
            with patch("ia_propia.brain.__main__.create_server", return_value=configured):
                code, output, errors = self.call(["serve", "--state-dir", str(self.state), "--body", "fixture-server-body"])
        self.assertEqual(code, 0)
        self.assertTrue(configured.closed)
        self.assertEqual(output, "")
        self.assertNotIn(token, errors)
        self.assertEqual(json.loads(errors)["bodyId"], Session.body_id)


if __name__ == "__main__":
    unittest.main()
