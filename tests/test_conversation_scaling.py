"""Opt-in architecture scaling and bounded, fixed-identity artifact IO."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.configuration import (ConversationError, DEFAULT_CONFIG,
    MAX_MODEL_PARAMETERS, MAX_PARAMETER_BUDGET, count_config_parameters, validate_config)


PROFILES = [
    ({"vocab_size": 8192, "d_model": 512, "n_layers": 14, "n_heads": 8,
      "context_length": 1024, "parameter_budget": 50_000_000}, 48_852_992),
    ({"vocab_size": 8192, "d_model": 640, "n_layers": 19, "n_heads": 10,
      "context_length": 1024, "parameter_budget": 100_000_000}, 99_446_400),
    ({"vocab_size": 16_384, "d_model": 1536, "n_layers": 34, "n_heads": 24,
      "context_length": 2048, "parameter_budget": 1_000_000_000}, 991_586_304),
]


class ArchitecturePlanningTests(unittest.TestCase):
    def test_large_profiles_count_exactly_without_constructing_weights(self):
        for config, expected in PROFILES:
            with self.subTest(parameters=expected):
                self.assertEqual(count_config_parameters(config), expected)
                original = dict(config)
                checked = validate_config(config)
                self.assertEqual(config, original)
                self.assertEqual(checked["parameter_budget"], config["parameter_budget"])

    def test_configuration_import_requires_only_standard_library(self):
        code = """
import builtins, sys
original = builtins.__import__
def bounded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'tokenizers', 'safetensors', 'numpy'}:
        raise AssertionError('planning imported optional numerical dependencies')
    return original(name, *args, **kwargs)
builtins.__import__ = bounded
from ia_propia.conversation.configuration import count_config_parameters
assert count_config_parameters({'vocab_size':16384,'d_model':1536,'n_layers':34,
    'n_heads':24,'context_length':2048,'parameter_budget':1_000_000_000}) == 991_586_304
assert not any(name in sys.modules for name in ['torch','tokenizers','safetensors','numpy'])
"""
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_defaults_and_limits_remain_implicit(self):
        self.assertEqual(validate_config({"vocab_size": 384}), {**DEFAULT_CONFIG, "vocab_size": 384})
        self.assertNotIn("parameter_budget", validate_config({"vocab_size": 384}))
        for config, _ in PROFILES:
            implicit = {key: value for key, value in config.items() if key != "parameter_budget"}
            with self.subTest(config=implicit), self.assertRaises(ConversationError):
                validate_config(implicit)
        narrow = {"vocab_size": 384, "d_model": 16, "n_heads": 2, "n_layers": 25}
        with self.assertRaises(ConversationError):
            validate_config(narrow)
        self.assertEqual(validate_config({**narrow, "parameter_budget": MAX_MODEL_PARAMETERS})["n_layers"], 25)

    def test_budget_is_strict_and_cannot_exceed_absolute_ceiling(self):
        for bad in (True, False, 20_000_000.0, "100000000", 19_999_999, 0,
                    MAX_PARAMETER_BUDGET + 1, float("nan")):
            with self.subTest(budget=bad), self.assertRaises(ConversationError):
                validate_config({"vocab_size": 384, "parameter_budget": bad})
        config, _ = PROFILES[1]
        with self.assertRaisesRegex(ConversationError, "parameter budget"):
            validate_config({**config, "parameter_budget": 50_000_000})
        with self.assertRaisesRegex(ConversationError, "parameter budget"):
            validate_config({"vocab_size": 16384, "d_model": 4096, "n_layers": 64,
                             "n_heads": 64, "parameter_budget": MAX_PARAMETER_BUDGET})

    def test_expanded_architecture_bounds_and_copy_head_still_validate(self):
        basic = {"vocab_size": 384, "d_model": 128, "n_layers": 1,
                 "parameter_budget": MAX_MODEL_PARAMETERS}
        self.assertEqual(validate_config({**basic, "n_heads": 32})["n_heads"], 32)
        for changes in ({"d_model": 4097}, {"n_layers": 65}, {"n_heads": 65},
                        {"n_heads": 3}, {"context_length": 4097}, {"vocab_size": 16385},
                        {"copy_head": {"version": 1, "rank": 16, "seed": 0}}):
            with self.subTest(changes=changes), self.assertRaises(ConversationError):
                validate_config({**basic, **changes})
        head = {"version": 1, "rank": 32, "seed": 7}
        with_head = {**basic, "copy_head": head}
        self.assertEqual(count_config_parameters(with_head) - count_config_parameters(basic), 8321)
        validated = validate_config(with_head)
        validated["copy_head"]["seed"] = 9
        self.assertEqual(head["seed"], 7)


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


@unittest.skipUnless(HAS_EXTRAS, "Optional conversation dependencies are not installed")
class ScaledArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        messages = [{"role": "user", "content": "¿Qué podemos aprender hoy?"},
                    {"role": "assistant", "content": "Podemos practicar español y comprobar el resultado."}]
        cls.tokenizer = fit_tokenizer([{"messages": messages}], vocab_size=384)
        cls.config = {"vocab_size": cls.tokenizer.get_vocab_size(), "d_model": 16,
                      "n_layers": 2, "n_heads": 2, "context_length": 64,
                      "ffn_multiplier": 2, "dropout": 0, "seed": 73}

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def model(self, **changes):
        from ia_propia.conversation.model import ConversationModel
        return ConversationModel({**self.config, **changes})

    def test_accounting_matches_real_modules_and_meta_billion_profile(self):
        from ia_propia.conversation.model import ConversationModel
        for multiplier in (1, 2, 4, 8):
            for head in (False, True):
                changes = {"ffn_multiplier": multiplier}
                if head:
                    changes["copy_head"] = {"version": 1, "rank": 32, "seed": 47}
                model = self.model(**changes)
                self.assertEqual(count_config_parameters(model.config), model.parameter_count)
        config, expected = PROFILES[2]
        skeleton = ConversationModel._loading_skeleton(config)
        self.assertEqual(skeleton.parameter_count, expected)
        self.assertTrue(all(parameter.is_meta for parameter in skeleton.parameters()))

    def test_stream_export_preserves_legacy_bytes_and_does_not_clone_cpu_weights(self):
        from safetensors.torch import save
        from ia_propia.conversation.artifacts import save_artifact
        model = self.model()
        legacy = save({name: value.detach().cpu().contiguous().clone()
                       for name, value in model.state_dict().items()})
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(self.torch.Tensor, "clone", side_effect=AssertionError("unexpected full weight clone")):
                manifest = save_artifact(folder, model, self.tokenizer)
            exported = Path(folder, "weights.safetensors").read_bytes()
            self.assertEqual(exported, legacy)
            self.assertEqual(manifest["files"]["weights.safetensors"]["sha256"], hashlib.sha256(legacy).hexdigest())
            self.assertNotIn("parameter_budget", manifest["config"])
            self.assertEqual(set(path.name for path in Path(folder).iterdir()),
                             {"config.json", "tokenizer.json", "weights.safetensors", "manifest.json"})

    def test_loading_uses_meta_and_bounded_reads_then_independent_cpu_snapshot(self):
        from ia_propia.conversation import artifacts
        from ia_propia.conversation.model import ConversationModel
        original = self.model(parameter_budget=MAX_MODEL_PARAMETERS).eval()
        inputs = self.torch.tensor([[1, 4, 40, 41, 2, 5]], dtype=self.torch.long)
        with tempfile.TemporaryDirectory() as folder:
            saved = artifacts.save_artifact(folder, original, self.tokenizer)
            bounded_read = artifacts._bounded_read
            def metadata_only(path, limit):
                if path.name == "weights.safetensors":
                    raise AssertionError("weight payload read into one bytes object")
                return bounded_read(path, limit)
            rng = self.torch.get_rng_state().clone()
            with patch.object(artifacts, "_bounded_read", side_effect=metadata_only), \
                    patch.object(ConversationModel, "_initialize", side_effect=AssertionError("random full model initialized")):
                loaded, _, restored = artifacts.load_artifact(folder)
            self.assertEqual(saved, restored)
            self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
            self.assertTrue(all(parameter.device.type == "cpu" and not parameter.is_meta
                                for parameter in loaded.parameters()))
            with self.torch.inference_mode():
                expected = original(inputs).clone()
                self.torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)
            # Edits of the backing file after loading cannot change fixed weights.
            path = Path(folder, "weights.safetensors")
            with path.open("r+b") as stream:
                stream.seek(-4, 2)
                stream.write(b"\x00\x00\xc0\x7f")
            with self.torch.inference_mode():
                self.torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)
            with self.assertRaises(ConversationError):
                artifacts.load_artifact(folder)

    def test_explicit_artifact_bounds_follow_exact_model_not_requested_budget(self):
        from ia_propia.conversation.artifacts import (MAX_SAFETENSORS_HEADER_BYTES, MAX_WEIGHT_BYTES,
                                                     _weight_limit)
        for config, count in PROFILES:
            self.assertEqual(_weight_limit(config), 4 * count + 8 + MAX_SAFETENSORS_HEADER_BYTES)
        self.assertEqual(_weight_limit(self.config), MAX_WEIGHT_BYTES)
        tiny = {**self.config, "parameter_budget": MAX_PARAMETER_BUDGET}
        self.assertLess(_weight_limit(tiny), 2 * MAX_SAFETENSORS_HEADER_BYTES)

    def test_admitted_architecture_is_checked_before_weight_reads_or_allocation(self):
        from ia_propia.conversation import artifacts
        from ia_propia.conversation.model import ConversationModel
        with tempfile.TemporaryDirectory() as folder:
            model = self.model()
            saved = artifacts.save_artifact(folder, model, self.tokenizer)
            admitted = {**model.config, "d_model": 24}
            with patch.object(artifacts, "_file_descriptor", side_effect=AssertionError("unadmitted weight IO")), \
                    patch.object(artifacts, "load_file", side_effect=AssertionError("unadmitted weight mapping")), \
                    patch.object(ConversationModel, "_loading_skeleton", side_effect=AssertionError("unadmitted model allocation")), \
                    self.assertRaisesRegex(ConversationError, "admitted model configuration"):
                artifacts.load_artifact(folder, expected_model_config=admitted)
            loaded, _, manifest = artifacts.load_artifact(folder, expected_model_config=model.config)
            self.assertEqual(manifest, saved)
            self.assertEqual(loaded.config, model.config)
            for name, tensor in model.state_dict().items():
                self.assertTrue(self.torch.equal(tensor, loaded.state_dict()[name]))

    def test_oversized_header_is_rejected_before_payload_mapping(self):
        from ia_propia.conversation import artifacts
        with tempfile.TemporaryDirectory() as folder:
            artifacts.save_artifact(folder, self.model(), self.tokenizer)
            path = Path(folder, "weights.safetensors")
            raw = bytearray(path.read_bytes())
            raw[:8] = (artifacts.MAX_SAFETENSORS_HEADER_BYTES + 1).to_bytes(8, "little")
            path.write_bytes(raw)
            manifest_path = Path(folder, "manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["weights.safetensors"]["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest["modelId"] = artifacts.artifact_model_id(manifest["files"])
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(artifacts, "load_file", side_effect=AssertionError("unsafe payload mapping")), \
                    self.assertRaisesRegex(ConversationError, "header"):
                artifacts.load_artifact(folder)

    def test_export_cannot_replace_a_symlink_or_publish_nonfinite_parameters(self):
        from ia_propia.conversation.artifacts import save_artifact
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "external"
            external.write_bytes(b"preserve")
            target = root / "model"
            target.mkdir()
            (target / "weights.safetensors").symlink_to(external)
            with self.assertRaisesRegex(ConversationError, "symlink"):
                save_artifact(target, self.model(), self.tokenizer)
            self.assertEqual(external.read_bytes(), b"preserve")
            with self.torch.no_grad():
                model = self.model()
                model.token_embedding.weight[0, 0] = float("inf")
            with self.assertRaisesRegex(ConversationError, "finite"):
                save_artifact(root / "nonfinite", model, self.tokenizer)
            self.assertFalse((root / "nonfinite").exists())


if __name__ == "__main__":
    unittest.main()
