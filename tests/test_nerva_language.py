"""Real subprocess parity and contract tests for the optional NERVA backend."""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ia_propia.language.model import (LanguageError, digest, encode, generate,
                                     seal_artifact, validate_artifact)
from ia_propia.language.nerva_backend import generate_nerva


def demo_artifact():
    return validate_artifact(json.loads(resources.files("ia_propia.assets")
                                        .joinpath("language-demo.json").read_text(encoding="utf-8")))


class NervaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = demo_artifact()

    def test_refuses_sampling_without_starting_any_process(self):
        with patch("ia_propia.language.nerva_backend.subprocess.run") as run:
            with self.assertRaisesRegex(LanguageError, "temperature=0"):
                generate_nerva(self.artifact, "Hola", temperature=0.8)
        run.assert_not_called()

    def test_preserves_existing_validation_for_invalid_inputs(self):
        cases = ({"length": -1}, {"length": True}, {"length": 4097},
                 {"seed": True}, {"seed": -1}, {"temperature": True},
                 {"temperature": float("nan")}, {"temperature": 3},
                 {"prompt": 3}, {"prompt": "x" * 4097})
        for arguments in cases:
            with self.subTest(arguments={key: type(value).__name__ for key, value in arguments.items()}):
                with patch("ia_propia.language.nerva_backend.subprocess.run") as run:
                    with self.assertRaises(LanguageError):
                        generate_nerva(self.artifact, **arguments)
                run.assert_not_called()

    def test_validates_provenance_before_execution(self):
        changed = deepcopy(self.artifact)
        changed["weights"]["Wxh"][0][0] += 1
        with patch("ia_propia.language.nerva_backend.subprocess.run") as run:
            with self.assertRaises(LanguageError):
                generate_nerva(changed, "Hola")
        run.assert_not_called()

    def test_missing_runtime_is_an_explicit_error_not_a_numpy_fallback(self):
        with self.assertRaisesRegex(LanguageError, r"Node\.js 18\+"):
            generate_nerva(self.artifact, "Hola", node="/nonexistent/unidad-node")

    def test_runtime_timeout_is_reported_without_fallback(self):
        with patch("ia_propia.language.nerva_backend.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("node", 1)):
            with self.assertRaisesRegex(LanguageError, "timeout"):
                generate_nerva(self.artifact, "Hola", timeout=1)

    def test_runtime_budget_failure_is_reported_without_fallback(self):
        result = subprocess.CompletedProcess([], 1, "", '{"error":"Tensor operation budget exceeded"}')
        with patch("ia_propia.language.nerva_backend.subprocess.run", return_value=result):
            with self.assertRaisesRegex(LanguageError, "operation budget"):
                generate_nerva(self.artifact, "Hola")

    def test_invalid_runtime_output_is_rejected(self):
        outputs = ("not json", "{}", '{"version":"0.1","steps":1,"records":[]}',
                   '{"version":"0.2.0","steps":1,"records":[]}',
                   json.dumps({"version": "0.2.0", "steps": 1, "records": [
                       {"kind": "print", "version": "0.2", "values": ["continuation", ""]}]}))
        for output in outputs:
            with self.subTest(output=output):
                result = subprocess.CompletedProcess([], 0, output, "")
                with patch("ia_propia.language.nerva_backend.subprocess.run", return_value=result):
                    with self.assertRaises(LanguageError):
                        generate_nerva(self.artifact, "Hola", length=1)


@unittest.skipUnless(shutil.which("node"), "The optional NERVA backend requires Node.js 18+")
class NervaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = demo_artifact()

    def test_complete_rnn_matches_original_own_weights_across_prompts(self):
        for prompt in ("Una tarea ", "Hola", "Aprender ", "IA", "El futuro ", "", "🧠未知"):
            with self.subTest(prompt=prompt):
                expected = generate(self.artifact, prompt, length=64, temperature=0, seed=42)
                actual = generate_nerva(self.artifact, prompt, length=64, seed=42)
                self.assertEqual(actual.pop("executionBackend"), "nerva-0.2")
                self.assertEqual(actual, expected)

    def test_zero_characters_preserves_empty_prompt_fallback_and_large_seed(self):
        for prompt in ("", "Hola"):
            with self.subTest(prompt=prompt):
                expected = generate(self.artifact, prompt, length=0, temperature=0, seed=2**63 - 1)
                actual = generate_nerva(self.artifact, prompt, length=0, seed=2**63 - 1)
                actual.pop("executionBackend")
                self.assertEqual(actual, expected)

    def test_backend_does_not_mutate_the_artifact(self):
        before = deepcopy(self.artifact)
        generate_nerva(self.artifact, "Hola", length=8)
        self.assertEqual(self.artifact, before)

    def test_excluded_unk_logit_still_preserves_numeric_overflow_guard(self):
        artifact = deepcopy(self.artifact)
        weights = artifact["weights"]
        state = np.zeros(artifact["config"]["hiddenSize"])
        for token in encode("Hola", artifact["tokenizer"]):
            state = np.tanh(np.asarray(weights["Wxh"])[:, token]
                            + np.asarray(weights["Whh"]) @ state + weights["bh"])
        weights["Why"][0] = [1.7e308 if value >= 0 else -1.7e308 for value in state]
        weights["by"][0] = 1.7e308
        artifact["training"]["selectedWeightsSha256"] = digest(weights)
        seal_artifact(artifact)
        validate_artifact(artifact)
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(LanguageError, "logits overflow"):
                generate(artifact, "Hola", length=1, temperature=0)
        with self.assertRaises(LanguageError):
            generate_nerva(artifact, "Hola", length=1)


if __name__ == "__main__":
    unittest.main()
