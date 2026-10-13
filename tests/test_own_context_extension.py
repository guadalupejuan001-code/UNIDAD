"""Tiny synthetic native fixtures for byte-preserving own-context extension."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("own_context_extension",
    Path(__file__).resolve().parents[1] / "examples/extend_own_conversation_context.py")
extension = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extension)
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
    for name in ("torch", "tokenizers", "safetensors"))


@unittest.skipUnless(HAS_EXTRAS, "optional conversation dependencies are not installed")
class OwnContextExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [{"messages": [{"role": "user", "content": "Este diálogo es una prueba sintética."},
            {"role": "assistant", "content": "Comprobemos los pesos y el contexto."}]}]
        self.tokenizer = fit_tokenizer(self.rows, vocab_size=300)
        self.config = {"vocab_size": self.tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 16,
            "ffn_multiplier": 2, "dropout": 0, "seed": 97}
        self.model = ConversationModel(self.config).eval()
        self.config = self.model.config
        self.parent = self.root / "parent"
        self.corpus_sha = hashlib.sha256(b"synthetic fixture, not reserved examples").hexdigest()
        self.parent_manifest = save_artifact(self.parent, self.model, self.tokenizer,
            training={"trainedFrom": "random_weights", "corpusSha256": self.corpus_sha,
                      "optimizerSteps": 123, "fixtureOnly": True})
        self.before = {path.name: path.read_bytes() for path in self.parent.iterdir()}

    def tearDown(self):
        self.temporary.cleanup()

    def extend(self, name="child", **changes):
        return extension.extend_context_artifact(self.parent, self.root / name,
            expected_model_config=self.config, context_length=32, **changes)

    def test_preserves_all_original_tensor_bytes_tokenizer_and_prefix_logits(self):
        from ia_propia.conversation.artifacts import load_artifact
        result = self.extend()
        child, tokenizer, manifest = load_artifact(result["output"])
        self.assertEqual(self.before, {path.name: path.read_bytes() for path in self.parent.iterdir()})
        self.assertEqual(child.config, {**self.config, "context_length": 32})
        for name, old in self.model.state_dict().items():
            value = child.state_dict()[name]
            inherited = value[:16] if name == extension.POSITION_KEY else value
            self.assertTrue(self.torch.equal(old.view(self.torch.uint8), inherited.view(self.torch.uint8)), name)
        self.assertEqual(manifest["files"]["tokenizer.json"], self.parent_manifest["files"]["tokenizer.json"])
        self.assertEqual(tokenizer.to_str(), self.tokenizer.to_str())
        with self.torch.inference_mode():
            for length in (1, 7, 16):
                tokens = self.torch.arange(length, dtype=self.torch.long)[None] % self.config["vocab_size"]
                self.assertTrue(self.torch.equal(self.model(tokens), child(tokens)))
            self.assertTrue(self.torch.isfinite(child(self.torch.ones((1, 32), dtype=self.torch.long))).all())
        self.assertEqual(child.parameter_count - self.model.parameter_count, 16 * 16)
        self.assertNotEqual(manifest["modelId"], self.parent_manifest["modelId"])

    def test_initializes_only_extra_rows_deterministically_without_rng_or_storage_alias(self):
        state = self.torch.get_rng_state().clone()
        with patch("ia_propia.conversation.model.ConversationModel._initialize",
                   side_effect=AssertionError("a backbone must not be randomly reinitialized")):
            first = extension._extend_model(self.model, {**self.config, "context_length": 32}, 97)
        self.assertTrue(self.torch.equal(state, self.torch.get_rng_state()))
        second = extension._extend_model(self.model, {**self.config, "context_length": 32}, 97)
        third = extension._extend_model(self.model, {**self.config, "context_length": 32}, 98)
        new = first.position_embedding.weight[16:]
        self.assertTrue(self.torch.equal(new, second.position_embedding.weight[16:]))
        self.assertFalse(self.torch.equal(new, third.position_embedding.weight[16:]))
        self.assertEqual(new.dtype, self.torch.float32)
        self.assertTrue(self.torch.isfinite(new).all())
        self.assertNotEqual(first.position_embedding.weight.data_ptr(), self.model.position_embedding.weight.data_ptr())
        self.assertFalse(self.torch.equal(new, self.model.position_embedding.weight))
        with self.torch.no_grad():
            new.zero_()
        self.assertTrue(self.torch.equal(self.model.position_embedding.weight,
                                       second.position_embedding.weight[:16]))

    def test_native_512_to_1024_boundary_preserves_a_recorded_own_parent(self):
        from ia_propia.conversation.artifacts import load_artifact, save_artifact
        from ia_propia.conversation.model import ConversationModel
        model = ConversationModel({**self.config, "context_length": 512}).eval()
        previous_lineage = {"parentModelId": self.parent_manifest["modelId"],
            "parentWeightsSha256": self.parent_manifest["files"]["weights.safetensors"]["sha256"],
            "parentCorpusSha256": self.corpus_sha, "rootOrigin": "random_weights"}
        save_artifact(self.parent, model, self.tokenizer,
            training={"trainedFrom": "own_checkpoint", "corpusSha256": self.corpus_sha,
                      "lineage": previous_lineage, "fixtureOnly": True})
        source_bytes = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        result = extension.extend_context_artifact(self.parent, self.root / "context1024",
            expected_model_config=model.config)
        child, _, manifest = load_artifact(result["output"])
        self.assertEqual(child.config["context_length"], 1024)
        self.assertEqual(child.parameter_count - model.parameter_count, 512 * 16)
        self.assertEqual(manifest["training"]["contextExtension"]["parentLineage"], previous_lineage)
        self.assertEqual(manifest["training"]["contextExtension"]["parentTrainedFrom"], "own_checkpoint")
        self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in self.parent.iterdir()})
        with self.torch.inference_mode():
            old_tokens = self.torch.ones((1, 512), dtype=self.torch.long)
            self.assertTrue(self.torch.equal(model(old_tokens), child(old_tokens)))
            self.assertTrue(self.torch.isfinite(child(self.torch.ones((1, 1024), dtype=self.torch.long))).all())

    def test_explicit_2048_context_preserves_native_own_backbone_and_512_prefix(self):
        from ia_propia.conversation.artifacts import load_artifact, save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation import training
        model = ConversationModel({**self.config, "context_length": 512}).eval()
        lineage = {"parentModelId": self.parent_manifest["modelId"],
            "parentWeightsSha256": self.parent_manifest["files"]["weights.safetensors"]["sha256"],
            "parentCorpusSha256": self.corpus_sha, "rootOrigin": "random_weights"}
        save_artifact(self.parent, model, self.tokenizer,
            training={"trainedFrom": "own_checkpoint", "corpusSha256": self.corpus_sha,
                      "lineage": lineage, "fixtureOnly": True})
        source_bytes = {path.name: path.read_bytes() for path in self.parent.iterdir()}
        result = extension.extend_context_artifact(self.parent, self.root / "context2048",
            expected_model_config=model.config, context_length=2048)
        child, _, manifest = load_artifact(result["output"])
        self.assertEqual(child.config, {**model.config, "context_length": 2048})
        self.assertEqual(child.parameter_count - model.parameter_count, 1536 * 16)
        for name, old in model.state_dict().items():
            value = child.state_dict()[name]
            inherited = value[:512] if name == extension.POSITION_KEY else value
            self.assertTrue(self.torch.equal(old.view(self.torch.uint8), inherited.view(self.torch.uint8)), name)
        self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in self.parent.iterdir()})
        self.assertEqual(manifest["training"]["contextExtension"]["status"], "untrained")
        self.assertEqual(manifest["training"]["contextExtension"]["parentLineage"], lineage)
        self.assertEqual(manifest["training"]["contextExtension"]["initializedPositionRange"], [512, 2048])
        self.assertEqual(manifest["files"]["tokenizer.json"], self.parent_manifest["files"]["tokenizer.json"])
        admitted, _, config, next_lineage = training._initial_own_model(result["output"], None)
        self.assertEqual(config, child.config)
        self.assertEqual(admitted.config, child.config)
        self.assertEqual(next_lineage["parentModelId"], manifest["modelId"])
        with self.torch.inference_mode():
            old_tokens = self.torch.ones((1, 512), dtype=self.torch.long)
            self.assertTrue(self.torch.equal(model(old_tokens), child(old_tokens)))
            self.assertTrue(self.torch.isfinite(child(self.torch.ones((1, 2048), dtype=self.torch.long))).all())

    def test_untrained_metadata_is_accepted_by_native_own_stage_loader(self):
        from ia_propia.conversation import training
        result = self.extend()
        manifest = result["manifest"]
        metadata = manifest["training"]
        lineage = {"parentModelId": self.parent_manifest["modelId"],
                   "parentWeightsSha256": self.parent_manifest["files"]["weights.safetensors"]["sha256"],
                   "parentCorpusSha256": self.corpus_sha, "rootOrigin": "random_weights"}
        self.assertEqual(metadata["trainedFrom"], "own_checkpoint")
        self.assertEqual(metadata["lineage"], lineage)
        self.assertFalse(metadata["trainingPerformed"])
        self.assertFalse(metadata["promoted"])
        self.assertFalse(metadata["trainingCorpusRead"])
        self.assertEqual(metadata["optimizerSteps"], 0)
        self.assertNotIn("fixtureOnly", metadata)
        self.assertEqual(metadata["contextExtension"]["status"], "untrained")
        self.assertEqual(metadata["contextExtension"]["parentTokenizerSha256"],
                         self.parent_manifest["files"]["tokenizer.json"]["sha256"])
        # This helper reads only the portable artifact; no corpus or TEST files.
        model, _, config, next_lineage = training._initial_own_model(result["output"], None)
        self.assertEqual(model.config, config)
        self.assertEqual(config["context_length"], 32)
        self.assertEqual(next_lineage["parentModelId"], manifest["modelId"])
        self.assertEqual(next_lineage["rootOrigin"], "random_weights")
        self.assertFalse(manifest["evaluation"]["productionAccepted"])

    def test_rejects_config_and_resource_mismatches_before_loading_weights(self):
        from ia_propia.conversation.configuration import ConversationError
        count = self.model.parameter_count
        for changes in ({"context_length": 16}, {"context_length": 2049},
                        {"context_length": True}, {"max_parameters": count},
                        {"max_parameters": 20_000_001}, {"seed": -1}):
            options = {"context_length": 32, **changes}
            with self.subTest(changes=changes), patch("ia_propia.conversation.artifacts.load_artifact",
                    side_effect=AssertionError("inadmissible model payload must not be loaded")):
                with self.assertRaises(ConversationError):
                    extension.extend_context_artifact(self.parent, self.root / "bad",
                        expected_model_config=self.config, **options)
        with self.assertRaisesRegex(ConversationError, "admitted model configuration"):
            extension.extend_context_artifact(self.parent, self.root / "bad-pin",
                expected_model_config={**self.config, "seed": 96}, context_length=32)
        self.assertFalse((self.root / "bad-pin").exists())

    def test_destination_is_always_new_and_never_a_parent_member(self):
        from ia_propia.conversation.configuration import ConversationError
        for destination in (self.parent, self.parent / "child", self.root):
            with self.subTest(destination=destination), self.assertRaises(ConversationError):
                extension.extend_context_artifact(self.parent, destination,
                    expected_model_config=self.config, context_length=32)
        default = extension.extend_context_artifact(self.parent, expected_model_config=self.config,
                                                    context_length=32)
        output = Path(default["output"])
        self.assertEqual(output.parent, self.parent.parent)
        self.assertIn("context32-untrained", output.name)
        with self.assertRaises(ConversationError):
            extension.extend_context_artifact(self.parent, output,
                expected_model_config=self.config, context_length=32)
        self.assertEqual(self.before, {path.name: path.read_bytes() for path in self.parent.iterdir()})

    def test_rejects_external_or_incomplete_lineage_and_nonfinite_weights(self):
        from ia_propia.conversation.artifacts import save_artifact
        for metadata in ({"trainedFrom": "external_pretrained", "corpusSha256": self.corpus_sha},
                         {"trainedFrom": "random_weights"},
                         {"trainedFrom": "own_checkpoint", "corpusSha256": self.corpus_sha,
                          "lineage": {"rootOrigin": "external_pretrained"}}):
            save_artifact(self.parent, self.model, self.tokenizer, training=metadata)
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.extend()
        with self.torch.no_grad():
            self.model.position_embedding.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite float32"):
            extension._extend_model(self.model, {**self.config, "context_length": 32}, 97)

    def test_executable_uses_pinned_configuration_and_reports_untrained_export(self):
        config_path = self.root / "admitted-config.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        command = [sys.executable, str(Path(extension.__file__)), str(self.parent),
                   "--expected-config", str(config_path), "--context-length", "32",
                   "--output", str(self.root / "cli-child")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["manifest"]["training"]["contextExtension"]["status"], "untrained")
        self.assertEqual(self.before, {path.name: path.read_bytes() for path in self.parent.iterdir()})


if __name__ == "__main__":
    unittest.main()
