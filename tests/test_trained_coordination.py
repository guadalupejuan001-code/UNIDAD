"""Host-selected coordination weights with real, tiny random Transformers.

The selector learns fictional feature preferences; the Transformers remain
untrained fixtures. Passing these checks does not prove conversational quality.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain import Brain
from ia_propia.brain.models import ModelRuntime
from ia_propia.conversation.selection import FEATURE_NAMES, save_selector, train_selector
from ia_propia.gnosis.project import GnosisProject


ROOT = Path(__file__).resolve().parents[1]
HAS_EXTRAS = all(importlib.util.find_spec(name) for name in ("torch", "tokenizers", "safetensors"))


def train_fixture_selector(path):
    high, low = [0.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES)
    high[0] = 1.0
    artifact = train_selector(
        [{"cohortId": "train/fictional-family-A", "features": [high, low], "utilities": [1, 0]}],
        [{"cohortId": "validation/fictional-family-B", "features": [low, high], "utilities": [0, 1]}],
        epochs=40, max_seconds=5, seed=7,
        provenance={"kind": "declared fictional feature-signal fixture", "personalData": False})
    save_selector(path, artifact)
    return artifact


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class TrainedCoordinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer

        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.models = []
        tokenizer = fit_tokenizer([{"messages": [
            {"role": "user", "content": "El taller ficticio abre el jueves. Revisemos el ensayo."},
            {"role": "assistant", "content": "Este texto corresponde a un fixture declarado."}]}], vocab_size=384)
        for index, seed in enumerate((7, 11)):
            model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
                "n_layers": 1, "n_heads": 2, "context_length": 512,
                "ffn_multiplier": 2, "seed": seed})
            directory = cls.root / f"transformer-{index}"
            manifest = save_artifact(directory, model, tokenizer,
                training={"fixtureOnly": True, "trainedFrom": "random_weights"})
            cls.models.append((directory, manifest))
        cls.selector_path = cls.root / "selector.json"
        cls.selector_artifact = train_fixture_selector(cls.selector_path)
        entries = [{"name": directory.name, "path": directory.name, "modelId": manifest["modelId"],
            "weightsSha256": manifest["files"]["weights.safetensors"]["sha256"],
            "tokenizerSha256": manifest["files"]["tokenizer.json"]["sha256"],
            "config": manifest["config"], "parameterCount": manifest["training"]["parameterCount"],
            "artifactKind": "complete", "rootOrigin": "random_weights", "productionAccepted": False}
            for directory, manifest in cls.models]
        cls.catalog = cls.root / "catalog.json"
        cls.catalog.write_text(json.dumps({"schemaVersion": 1, "entries": entries}))
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)

    def setUp(self):
        self.local = tempfile.TemporaryDirectory()
        self.addCleanup(self.local.cleanup)

    def runtime_args(self):
        return {"conversation_model": self.models[0][0], "conversation_models": [
            {"name": directory.name, "model_dir": directory, "expected_model_id": manifest["modelId"]}
            for directory, manifest in self.models], "ensemble_primary": "transformer-0",
            "ensemble_budget": {"max_candidates": 2, "min_candidates": 2, "max_resident": 2,
                "total_new_tokens": 16, "max_score_calls": 4, "critic_count": 2, "max_seconds": 60},
            "conversation_selector": self.selector_path}

    def config(self):
        config = json.loads((ROOT / "configs/gnosis.json").read_text())
        config["projectRoot"] = str(self.root)
        config["brain"].update(state_dir=str(Path(self.local.name) / "brain"), language_backend="numpy",
            conversation_model="transformer-0", expected_model_id=self.models[0][1]["modelId"],
            conversation_selector="selector.json", conversation_ensemble={"catalog": "catalog.json",
                "primary": "transformer-0", "max_candidates": 2, "min_candidates": 2,
                "max_resident": 2, "total_new_tokens": 16, "max_score_calls": 4})
        return config

    def test_all_bodies_share_the_loaded_trained_selector_and_its_identity(self):
        project = GnosisProject(self.config(), config_dir=self.root)
        runtime = project.brain.models
        self.addCleanup(runtime.close)
        first, second = project.connect("body-a"), project.connect("body-b")
        engine = first.models.conversation_engine
        self.assertIs(engine, second.models.conversation_engine)
        self.assertIs(engine._learned_selector, second.models.conversation_engine._learned_selector)
        self.assertEqual(engine._learned_selector.model_id, self.selector_artifact["modelId"])
        self.assertEqual(engine._learned_selector.parameter_count, 18)
        self.assertGreater(self.selector_artifact["training"]["selectedEpoch"], 0)
        caps = first.capabilities()["models"]["models"]["conversation"]["ensemble"]
        self.assertTrue(caps["learnedSelector"])
        self.assertEqual(caps["selectorModelId"], self.selector_artifact["modelId"])

    def test_trained_selection_returns_one_exact_generated_candidate_and_minimum_real_models(self):
        from ia_propia.conversation.runtime import ConversationEngine

        brain = Brain(Path(self.local.name) / "state", **self.runtime_args())
        self.addCleanup(brain.models.close)
        session = brain.connect("trained-candidate-selection")
        observed = []
        original_reply = ConversationEngine.reply
        def capture(engine, *args, **kwargs):
            result = original_reply(engine, *args, **kwargs)
            observed.append(deepcopy(result))
            return result
        messages = [{"role": "user", "content": "¿Cuándo abre el taller ficticio?"}]
        with patch.object(ConversationEngine, "reply", capture):
            response = session.request({"schemaVersion": 1, "operation": "conversation.reply",
                "input": {"messages": messages, "max_new_tokens": 8, "temperature": 0}})
        self.assertEqual(response["status"], "ok", response)
        result = response["result"]
        self.assertEqual(len({candidate["modelId"] for candidate in observed}), 2)
        selected = next(candidate for candidate in observed if candidate["modelId"] == result["modelId"])
        self.assertEqual(result["text"], selected["text"])
        self.assertEqual(result["generatedTokenIds"], selected["generatedTokenIds"])
        self.assertTrue(result["ensemble"]["minimumCandidatesSatisfied"])
        self.assertTrue(result["ensemble"]["learnedSelectorApplied"])
        self.assertEqual(result["ensemble"]["learnedSelection"]["modelId"], self.selector_artifact["modelId"])

    def test_request_cannot_change_selector_path_or_parameters(self):
        brain = Brain(Path(self.local.name) / "state", **self.runtime_args())
        self.addCleanup(brain.models.close)
        session = brain.connect("host-only-selector")
        selector = brain.models.conversation_engine._learned_selector
        for field, value in (("conversation_selector", "/private/other.json"), ("selector", {"weights": [0] * 18})):
            response = session.request({"schemaVersion": 1, "operation": "conversation.reply", "input": {
                "messages": [{"role": "user", "content": "Hola."}], field: value}})
            self.assertEqual(response["status"], "error")
            self.assertEqual(response["error"]["code"], "invalid_input")
        self.assertIs(brain.models.conversation_engine._learned_selector, selector)
        self.assertEqual(selector.model_id, self.selector_artifact["modelId"])

    def test_corrupt_selector_fails_before_loading_transformer_tensors(self):
        broken = deepcopy(self.selector_artifact)
        broken["weights"][0] += 0.01
        path = Path(self.local.name) / "corrupt.json"
        path.write_text(json.dumps(broken))
        args = self.runtime_args()
        args["conversation_selector"] = path
        with patch("ia_propia.conversation.artifacts.load_artifact", side_effect=AssertionError("No tensors admitted")):
            with self.assertRaisesRegex(ValueError, "hash|identity"):
                ModelRuntime(**args)

    def test_training_run_cannot_replace_the_coordination_artifact(self):
        config = self.config()
        config["brain"]["conversation_selector"] = "private/gnosis/foundation/artifact/selector.json"
        with self.assertRaisesRegex(ValueError, "overwrite"):
            GnosisProject(config, config_dir=self.root)

    def test_single_model_configuration_rejects_a_trained_coordination_selector(self):
        with self.assertRaisesRegex(ValueError, "ensemble"):
            ModelRuntime(conversation_model=self.models[0][0], conversation_selector=self.selector_path)
        config = self.config()
        config["brain"].pop("conversation_ensemble")
        with self.assertRaisesRegex(ValueError, "ensemble"):
            GnosisProject(config, config_dir=self.root)


if __name__ == "__main__":
    unittest.main()
