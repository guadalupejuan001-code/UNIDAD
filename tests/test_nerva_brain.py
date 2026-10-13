"""The trusted host selects NERVA; requests cannot replace runtimes or weights."""
from contextlib import redirect_stderr, redirect_stdout
from importlib import resources
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain import Brain
from ia_propia.brain.__main__ import main as brain_main
from ia_propia.brain.models import INPUT_SCHEMAS, ModelRuntime, ModelRuntimeError
from ia_propia.language.__main__ import main as language_main


class NervaHostSelectionTests(unittest.TestCase):
    def test_schema_is_effective_detached_and_does_not_change_numpy_defaults(self):
        nerva = ModelRuntime(language_backend="nerva")
        schema = nerva.capabilities()["inputSchemas"]["language.generate"]["properties"]["temperature"]
        self.assertEqual(schema["const"], 0)
        self.assertEqual(schema["default"], 0)
        self.assertEqual(schema["maximum"], 0)
        self.assertEqual(INPUT_SCHEMAS["language.generate"]["properties"]["temperature"]["default"], 0.8)
        numpy = ModelRuntime()
        self.assertNotIn("const", numpy.capabilities()["inputSchemas"]["language.generate"]["properties"]["temperature"])
        self.assertNotIn("executionBackend", numpy.capabilities()["models"]["language"])
        schema["const"] = 1
        self.assertEqual(nerva.capabilities()["inputSchemas"]["language.generate"]["properties"]["temperature"]["const"], 0)

    def test_invalid_host_backend_is_rejected(self):
        for backend in ("other", "conversation", None, False, ["nerva"]):
            with self.subTest(backend=backend):
                with self.assertRaises(ModelRuntimeError):
                    ModelRuntime(language_backend=backend)

    def test_conversation_selection_preserves_independent_language_backend_schema(self):
        with patch("ia_propia.conversation.runtime.ConversationEngine") as engine:
            engine.return_value.model_id = "own-conversation-test-artifact"
            engine.return_value.parameter_count = 123
            runtime = ModelRuntime(conversation_model="/trusted/model", language_backend="nerva")
        capabilities = runtime.capabilities()
        temperature = capabilities["inputSchemas"]["language.generate"]["properties"]["temperature"]
        self.assertEqual(temperature["const"], 0)
        self.assertEqual(temperature["default"], 0)
        self.assertIn("conversation.reply", capabilities["inputSchemas"])
        self.assertEqual(capabilities["models"]["conversation"]["modelId"], "own-conversation-test-artifact")

    def test_requests_cannot_choose_backend_runtime_or_paths(self):
        runtime = ModelRuntime(language_backend="nerva")
        for request in ({"backend": "numpy"}, {"language_backend": "numpy"},
                        {"node": "/arbitrary/program"}, {"model": "/arbitrary/weights"},
                        {"temperature": 0.8}):
            with self.subTest(request=request):
                with self.assertRaises(ModelRuntimeError):
                    runtime.validate_input("language.generate", request)

    def test_actual_missing_node_failure_uses_existing_structured_brain_error(self):
        from ia_propia.language.nerva_backend import generate_nerva
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(directory, language_backend="nerva")
            brain.models._language_generate = lambda artifact, **request: generate_nerva(
                artifact, **request, node="/nonexistent/unidad-node")
            result = brain.connect("nerva-test").request({"schemaVersion": 1, "requestId": "missing-node",
                "operation": "language.generate", "input": {"prompt": "Hola", "length": 1}})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "operation_failed")
        self.assertNotIn("result", result)


@unittest.skipUnless(shutil.which("node"), "The optional NERVA backend requires Node.js 18+")
class NervaPublicIntegrationTests(unittest.TestCase):
    def test_actual_brain_operation_runs_the_complete_rnn_backend(self):
        expected = ModelRuntime().execute("language.generate", {"prompt": "Hola", "length": 16, "temperature": 0})
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(directory, language_backend="nerva")
            result = brain.connect("nerva-test").request({"schemaVersion": 1, "requestId": "own-rnn",
                "operation": "language.generate", "input": {"prompt": "Hola", "length": 16}})
        self.assertEqual(result["status"], "ok", result)
        actual = result["result"]
        self.assertEqual(actual.pop("executionBackend"), "nerva-0.2")
        self.assertEqual(actual, expected)

    def test_brain_cli_host_selector_and_default_greedy_request(self):
        payload = {"schemaVersion": 1, "requestId": "cli-nerva", "operation": "language.generate",
                   "input": {"prompt": "Hola", "length": 8}}
        with tempfile.TemporaryDirectory() as directory:
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors), patch("sys.stdin", io.StringIO(json.dumps(payload))):
                code = brain_main(["call", "--language-backend", "nerva", "--state-dir", directory, "--input", "-"])
        self.assertEqual(code, 0, errors.getvalue())
        self.assertEqual(errors.getvalue(), "")
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["executionBackend"], "nerva-0.2")
        self.assertEqual(result["result"]["temperature"], 0)

    def test_language_cli_requires_explicit_greedy_and_executes_when_selected(self):
        model = resources.files("ia_propia.assets").joinpath("language-demo.json")
        with resources.as_file(model) as path:
            for extra, expected_code in (([], 2), (["--temperature", "0"], 0)):
                with self.subTest(extra=extra):
                    output, errors = io.StringIO(), io.StringIO()
                    with redirect_stdout(output), redirect_stderr(errors):
                        code = language_main(["generate", "--model", str(path), "--backend", "nerva",
                                              "--prompt", "Hola", "--length", "8", *extra])
                    self.assertEqual(code, expected_code, errors.getvalue())
                    if code == 0:
                        self.assertEqual(json.loads(output.getvalue())["executionBackend"], "nerva-0.2")
                    else:
                        self.assertIn("temperature=0", errors.getvalue())
                        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
