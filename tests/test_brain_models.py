import copy
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain.models import ModelRuntime, ModelRuntimeError
from ia_propia.language.model import generate, load_artifact as load_language
from ia_propia.trend.data import DataError, load_observations, observations_from_payload
from ia_propia.trend.model import load_artifact as load_attention, predict


ROOT = Path(__file__).resolve().parents[1]
SIGNAL = "wikimedia:en.wikipedia.org:Moo_Deng"
SOURCE = "wikimedia:en.wikipedia.org"
AS_OF = 1728950400000


class BrainModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ModelRuntime()
        cls.rows = json.loads((ROOT / "examples/attention-observations.json").read_text())

    def test_packaged_files_match_saved_weights_and_manifest_exactly(self):
        root = resources.files("ia_propia.assets")
        manifest = json.loads(root.joinpath("manifest.json").read_bytes())
        for name, descriptor in manifest["models"].items():
            raw = root.joinpath(descriptor["file"]).read_bytes()
            self.assertEqual(raw, (ROOT / "models" / descriptor["file"]).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor["fileSha256"])
            self.assertEqual(self.runtime.capabilities()["models"][name]["modelId"], descriptor["modelId"])
            self.assertEqual(self.runtime.capabilities()["models"][name]["parameterCount"], descriptor["parameterCount"])

    def test_defaults_load_without_working_directory_or_original_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")
            process = subprocess.run([sys.executable, "-c", "from ia_propia.brain.models import ModelRuntime; import json; print(json.dumps(ModelRuntime().capabilities()))"],
                                     cwd=directory, env=environment, text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout), self.runtime.capabilities())

    def test_capabilities_are_honest_and_detached_from_runtime_configuration(self):
        capabilities = self.runtime.capabilities()
        self.assertEqual(capabilities["models"]["language"]["parameterCount"], 2903)
        self.assertEqual(capabilities["models"]["attention"]["parameterCount"], 49)
        self.assertFalse(capabilities["networkRequired"])
        self.assertFalse(capabilities["workerCapabilityValidated"])
        self.assertFalse(capabilities["productionAccepted"])
        capabilities["models"]["language"]["modelId"] = "altered"
        capabilities["inputSchemas"]["language.generate"]["properties"]["length"]["maximum"] = 100000
        self.assertNotEqual(self.runtime.capabilities()["models"]["language"]["modelId"], "altered")
        self.assertEqual(self.runtime.capabilities()["inputSchemas"]["language.generate"]["properties"]["length"]["maximum"], 4096)

    def test_language_reuses_loaded_model_and_preserves_deterministic_outputs(self):
        expected = generate(load_language(ROOT / "models/language-demo.json"), "La red ", length=40, seed=8, temperature=0.8)
        request = {"prompt": "La red ", "length": 40, "seed": 8, "temperature": 0.8}
        with patch("pathlib.Path.read_bytes", side_effect=AssertionError("A request must not reload model files")):
            self.assertEqual(self.runtime.execute("language.generate", request), expected)
            self.assertEqual(self.runtime.execute("language.generate", request), expected)

    def test_raw_forecast_matches_existing_model_for_exact_same_values(self):
        model = load_attention(ROOT / "models/attention-v1.json")
        observations, _ = load_observations(ROOT / "examples/attention-observations.json", source=SOURCE, metric="views")
        expected = predict(model, [row for row in observations if row.signal_id == SIGNAL], as_of=AS_OF, historical=True)
        request = {"observations": self.rows, "signal_id": SIGNAL, "as_of": AS_OF, "historical": True}
        result = self.runtime.execute("attention.forecast", request)
        self.assertEqual(result, {"schemaVersion": 1, "forecasts": expected})
        self.assertEqual(result, self.runtime.execute("attention.forecast", request))
        result["forecasts"][0]["evidence"][0]["provenance"]["source"] = "changed output"
        self.assertEqual(self.runtime.execute("attention.forecast", request)["forecasts"], expected)

    def test_raw_payload_helper_preserves_file_loader_results(self):
        now = 2_000_000_000_000
        expected, file_metadata = load_observations(ROOT / "examples/attention-observations.json", source=SOURCE, metric="views", now_ms=now)
        actual, raw_metadata = observations_from_payload(self.rows, source=SOURCE, metric="views", now_ms=now)
        self.assertEqual(actual, expected)
        self.assertEqual(raw_metadata, file_metadata)
        state = {"schemaVersion": 1, "observations": {row["id"]: row for row in self.rows},
                 "signals": {row["signalId"]: {"id": row["signalId"], "privacy": "PUBLIC"} for row in self.rows}}
        self.assertEqual(observations_from_payload(state, source=SOURCE, metric="views", now_ms=now)[0], expected)

    def test_strict_requests_cannot_select_files_or_coerce_types(self):
        invalid = [("unknown", {}), ("language.generate", []), ("language.generate", {"model": "/tmp/file"}),
                   ("language.generate", {"length": True}), ("language.generate", {"length": "5"}),
                   ("language.generate", {"temperature": float("nan")}), ("language.generate", {"prompt": {"$ref": "value"}}),
                   ("attention.forecast", {}), ("attention.forecast", {"observations": "/tmp/file"}),
                   ("attention.forecast", {"observations": self.rows, "historical": "true"}),
                   ("attention.forecast", {"observations": self.rows, "as_of": True}),
                   ("attention.forecast", {"observations": self.rows, "source": "another-source"}),
                   ("attention.forecast", {"observations": self.rows, "metric": "shares"}),
                   ("attention.forecast", {"observations": self.rows, "path": "/tmp/file"})]
        for operation, request in invalid:
            with self.assertRaises(ValueError, msg=str((operation, request))):
                self.runtime.execute(operation, request)

    def test_scope_privacy_and_conflicting_ids_validate_before_signal_filter(self):
        request = {"observations": copy.deepcopy(self.rows), "signal_id": SIGNAL, "as_of": AS_OF, "historical": True}
        other = next(row for row in request["observations"] if row["signalId"] != SIGNAL)
        other["privacy"] = "PRIVATE"
        with self.assertRaises(DataError):
            self.runtime.execute("attention.forecast", request)
        request["observations"] = copy.deepcopy(self.rows)
        private = copy.deepcopy(request["observations"][-1])
        private.update(id="private-extra", signalId=SIGNAL, privacy="PRIVATE", metrics={"shares": 1})
        private["provenance"]["source"] = "another-source"
        request["observations"].append(private)
        with self.assertRaises(DataError):
            self.runtime.execute("attention.forecast", request)
        request["observations"] = copy.deepcopy(self.rows)
        changed = copy.deepcopy(request["observations"][0]); changed["metrics"]["views"] += 1
        request["observations"].append(changed)
        with self.assertRaises(DataError):
            self.runtime.execute("attention.forecast", request)

    def test_explicit_trusted_custom_paths_preserve_the_same_model_identities(self):
        selected = ModelRuntime(language_model=ROOT / "models/language-demo.json", attention_model=ROOT / "models/attention-v1.json")
        self.assertEqual(selected.capabilities(), self.runtime.capabilities())

    def test_workflow_validation_accepts_only_whole_field_references(self):
        self.runtime.validate_input("language.generate", {"prompt": {"$ref": "steps.note.text"}}, allow_refs=True)
        self.runtime.validate_input("attention.forecast", {"observations": {"$ref": "steps.capture.observations"}}, allow_refs=True)
        for request in ({"prompt": {"$ref": ""}}, {"prompt": {"$ref": "x", "extra": 1}}, {"length": False}):
            with self.assertRaises(ValueError):
                self.runtime.validate_input("language.generate", request, allow_refs=True)


if __name__ == "__main__":
    unittest.main()
