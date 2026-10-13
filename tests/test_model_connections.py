"""Model-service wiring with actual auxiliary weights and declared engine fixtures.

The facade records requests without loading conversational weights. These are
protocol, evidence and compatibility checks, not generated-language acceptance.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ia_propia.brain import Brain, Tool
from ia_propia.brain.models import ModelRuntime
from ia_propia.gnosis.project import GnosisProject
from ia_propia.language.scoring import score_text
from tests.test_gnosis import configuration


ROOT = Path(__file__).resolve().parents[1]
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
AS_OF = 1728950400000
SIGNAL = "wikimedia:en.wikipedia.org:Moo_Deng"


class DeclaredFacade:
    """Declared transport fixture deliberately has no model/tokenizer internals."""

    model_id = "declared-facade-no-conversation-weights"

    def __init__(self):
        self.requests, self.usages = [], []

    def prompt_usage(self, messages):
        self.usages.append(deepcopy(messages))
        return {"tokens": 2 + sum(2 + len(message["content"].encode("utf-8")) for message in messages),
                "capacity": 16384, "discarded": 0}

    def reply(self, **request):
        self.requests.append(deepcopy(request))
        return {"text": '{"action":"stop","reason":"Declared facade fixture"}',
                "modelId": "declared-selected-member-no-conversation-weights"}


class AuxiliaryConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_protocol_scores_with_the_loaded_rnn_and_rejects_request_selected_paths(self):
        brain = Brain(self.directory / "state")
        session = brain.connect("auxiliary-scoring")
        before = deepcopy(brain.models._language)
        expected = score_text(brain.models._language, "hola unidad", max_chars=32)
        payload = {"schemaVersion": 1, "operation": "language.score",
                   "input": {"text": "hola unidad", "max_chars": 32}}
        with patch("pathlib.Path.read_bytes", side_effect=AssertionError("Scoring must reuse loaded weights")):
            result = session.request(payload)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["result"], expected)
        self.assertEqual(brain.models._language, before)
        for operation, values in (("language.score", {"text": "hola", "model": "/private/other.json"}),
                ("attention.forecast", {"observations": "/private/observations.json"}),
                ("language.score", {"text": "hola", "max_chars": True})):
            response = session.request({"schemaVersion": 1, "operation": operation, "input": values})
            self.assertEqual(response["status"], "error", response)
            self.assertEqual(response["error"]["code"], "invalid_input")

    def test_gnosis_preserves_explicit_host_auxiliary_paths_without_enabling_learning(self):
        config = configuration(self.directory)
        language_path, attention_path = ROOT / "models/language-demo.json", ROOT / "models/attention-v1.json"
        config["brain"].update(language_model=str(language_path), attention_model=str(attention_path))
        original = deepcopy(config)
        project = GnosisProject(config, config_dir=self.directory)
        default = ModelRuntime().capabilities()["models"]
        self.assertEqual(project.config["brain"]["language_model"], str(language_path))
        self.assertEqual(project.config["brain"]["attention_model"], str(attention_path))
        self.assertEqual(project.brain.models.capabilities()["models"], default)
        self.assertEqual(config, original)
        self.assertFalse(project.brain.capabilities()["automaticLearning"])

    def test_learning_runs_cannot_replace_host_selected_auxiliary_artifacts(self):
        for key in ("language_model", "attention_model"):
            config = configuration(self.directory)
            run = config["learning"]["stages"]["foundation"]["run_dir"]
            config["brain"][key] = str(Path(run) / "artifact" / "manifest.json")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overwrite"):
                GnosisProject(config, config_dir=self.directory)


@unittest.skipUnless(HAS_TOKENIZERS, "conversation request validation requires optional tokenizers")
class ForecastAndFacadeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.runtime = ModelRuntime()
        self.facade = DeclaredFacade()
        self.runtime._conversation = self.facade
        self.rows = json.loads((ROOT / "examples/attention-observations.json").read_text())

    def forecast(self):
        return {"observations": deepcopy(self.rows), "signal_id": SIGNAL,
                "source": "wikimedia:en.wikipedia.org", "metric": "views",
                "as_of": AS_OF, "historical": True}

    def test_forecast_evidence_matches_actual_mlp_output_and_keeps_original_request(self):
        values = {"messages": [{"role": "system", "content": "Sistema de fixture; conserva los datos."},
                               {"role": "user", "content": "Explica el pronóstico explícito."}],
                  "forecast": self.forecast(), "max_new_tokens": 1, "temperature": 0}
        original = deepcopy(values)
        expected = self.runtime.execute("attention.forecast", values["forecast"])
        result = self.runtime.execute("conversation.forecast", values)
        self.assertEqual(values, original)
        self.assertEqual(result["metricEvidence"]["forecasts"], expected["forecasts"])
        self.assertTrue(result["metricEvidence"]["predictionNotObservedFact"])
        prompt = self.facade.requests[-1]["messages"]
        self.assertTrue(prompt[0]["content"].startswith(values["messages"][0]["content"]))
        self.assertEqual(prompt[-1], values["messages"][-1])
        self.assertIn(json.dumps(expected["forecasts"][0]["predictedValue"]), prompt[0]["content"])

    def test_wrong_metric_or_private_observations_block_generation_before_any_reply(self):
        mutations = []
        source = self.forecast()
        source["source"] = "an-untrained-source"
        mutations.append(source)
        metric = self.forecast()
        metric["metric"] = "shares"
        mutations.append(metric)
        private = self.forecast()
        private["observations"][0]["privacy"] = "PRIVATE"
        mutations.append(private)
        for forecast in mutations:
            with self.subTest(kind=forecast.get("metric")), self.assertRaises(ValueError):
                self.runtime.execute("conversation.forecast", {"messages": [{"role": "user", "content": "Explica."}],
                    "forecast": forecast, "max_new_tokens": 1})
        self.assertEqual(self.facade.requests, [])

    def test_workflow_resolves_forecast_observations_before_shared_generation(self):
        brain = Brain(self.directory / "state")
        brain.models = self.runtime
        session = brain.connect("forecast-workflow", tools=[Tool("capture", lambda _: {"rows": deepcopy(self.rows)})])
        forecast = self.forecast()
        forecast["observations"] = {"$ref": "capture.rows"}
        response = session.request({"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "capture", "operation": "tool.call", "input": {"name": "capture", "arguments": {}}},
            {"id": "explain", "operation": "conversation.forecast", "input": {
                "messages": [{"role": "user", "content": "Explica estas observaciones explícitas."}],
                "forecast": forecast, "max_new_tokens": 1, "temperature": 0}}
        ]}})
        self.assertEqual(response["status"], "ok", response)
        self.assertEqual(response["result"]["completedCount"], 2)
        self.assertTrue(response["result"]["steps"][1]["result"]["metricEvidence"]["forecasts"])
        self.assertEqual(len(self.facade.requests), 1)

    def test_autonomy_preflight_uses_facade_contract_and_tracks_selected_model_identity(self):
        from ia_propia.brain.autonomy_model import ModelActionProposer

        proposer = ModelActionProposer(SimpleNamespace(conversation_engine=self.facade), max_new_tokens=1)
        result = proposer({"task": {"goal": "Lee el ensayo ficticio autorizado."}, "tools": []})
        self.assertEqual(result, '{"action":"stop","reason":"Declared facade fixture"}')
        self.assertEqual(len(self.facade.usages), 1)
        self.assertEqual(proposer.last_metadata["modelId"], "declared-selected-member-no-conversation-weights")
        self.assertFalse(hasattr(self.facade, "_model"))
        self.assertFalse(hasattr(self.facade, "_tokenizer"))
        self.assertEqual(proposer.last_metadata["discardedPromptTokens"], 0)

    def test_sovengi_forecast_failure_keeps_completed_workflow_effect_receipts(self):
        from ia_propia.sovengi.application import SovengiSession

        effects = []
        def mark(_):
            effects.append("completed")
            return {"effect": "completed"}
        brain = Brain(self.directory / "state")
        brain.models = self.runtime
        session = SovengiSession(brain.connect("failed-forecast-workflow", tools=[Tool("mark", mark)]))
        forecast = self.forecast()
        forecast["observations"] = forecast["observations"][:1]
        response = session.request({"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "effect", "operation": "tool.call", "input": {"name": "mark", "arguments": {}}},
            {"id": "explain", "operation": "conversation.forecast", "input": {
                "messages": [{"role": "user", "content": "Explica estas observaciones."}],
                "forecast": forecast, "max_new_tokens": 1, "temperature": 0}}
        ]}})
        self.assertEqual(effects, ["completed"])
        self.assertEqual(response["status"], "error", response)
        self.assertEqual(response["error"]["code"], "workflow_failed", response)
        self.assertEqual(response["result"]["completedCount"], 1)
        self.assertFalse(response["result"]["effectsRolledBack"])
        self.assertEqual(response["result"]["steps"][0]["result"], {"effect": "completed"})
        self.assertEqual(self.facade.requests, [])


if __name__ == "__main__":
    unittest.main()
