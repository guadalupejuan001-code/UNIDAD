"""Character scoring preserves own weights, original text and training objective."""

from copy import deepcopy
from importlib import resources
import json
import math
import unittest

import numpy as np

from ia_propia.language.model import LanguageError
from ia_propia.language.scoring import score_text


class LanguageScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(resources.files("ia_propia.assets").joinpath("language-demo.json").read_bytes())

    def test_score_matches_independent_next_character_forward_pass(self):
        text = "hola unidad"
        vocabulary = {char: index for index, char in enumerate(self.artifact["tokenizer"]["tokens"])}
        tokens = [vocabulary.get(char, 0) for char in text]
        weights = {name: np.asarray(values, dtype=np.float64)
                   for name, values in self.artifact["weights"].items()}
        state = np.zeros(self.artifact["config"]["hiddenSize"])
        expected = []
        for current, target in zip(tokens, tokens[1:]):
            state = np.tanh(weights["Wxh"][:, current] + weights["Whh"] @ state + weights["bh"])
            logits = weights["Why"] @ state + weights["by"]
            shifted = logits - np.max(logits)
            expected.append(float(math.log(float(np.exp(shifted).sum())) - shifted[target]))
        result = score_text(self.artifact, text)
        self.assertAlmostEqual(result["meanNll"], math.fsum(expected) / len(expected), places=12)
        self.assertEqual(result["targetChars"], len(text) - 1)
        self.assertFalse(result["includesEos"])

    def test_prefix_bound_reports_truncation_without_scoring_or_echoing_suffix(self):
        original = "hola unidad, revisemos el ensayo ficticio. " * 20
        first = score_text(self.artifact, original, max_chars=16)
        second = score_text(self.artifact, original[:16], max_chars=16)
        self.assertEqual(first["meanNll"], second["meanNll"])
        self.assertEqual(first["targetChars"], 15)
        self.assertEqual(first["charLength"], len(original))
        self.assertEqual(first["scoredCharacters"], 16)
        self.assertTrue(first["truncated"])
        self.assertFalse(second["truncated"])
        self.assertNotIn("text", first)
        self.assertNotIn("continuation", first)

    def test_unknown_coverage_includes_initial_input_and_exact_original_characters(self):
        result = score_text(self.artifact, "🙂aЖ")
        self.assertEqual(result["unknownRatio"], 2 / 3)
        self.assertEqual(result["targetChars"], 2)
        self.assertTrue(math.isfinite(result["meanNll"]))
        one = score_text(self.artifact, "🙂")
        self.assertIsNone(one["meanNll"])
        self.assertEqual(one["targetChars"], 0)
        self.assertEqual(one["unknownRatio"], 1)
        self.assertFalse(one["scoreable"])
        self.assertEqual(one["reason"], "no_next_character_target")

    def test_scoring_is_deterministic_preserves_artifact_and_makes_no_quality_claim(self):
        initial = deepcopy(self.artifact)
        first = score_text(self.artifact, "hola unidad")
        second = score_text(self.artifact, "hola unidad")
        self.assertEqual(first, second)
        self.assertEqual(self.artifact, initial)
        self.assertEqual(first["modelId"], self.artifact["modelId"])
        self.assertEqual(first["artifactSha256"], self.artifact["artifactSha256"])
        self.assertFalse(first["parametersUpdated"])
        self.assertFalse(first["automaticLearning"])
        self.assertFalse(first["provenance"]["factualAccuracyValidated"])
        self.assertFalse(first["provenance"]["generalConversationValidated"])

    def test_rejects_unbounded_invalid_inputs_and_corrupted_weights(self):
        for text in (None, "", "x" * 16_385, "🙂" * 4097, "\ud800"):
            with self.subTest(type=type(text).__name__):
                with self.assertRaises(LanguageError):
                    score_text(self.artifact, text)
        for limit in (True, 0, 513, 2.5, None):
            with self.subTest(limit=limit):
                with self.assertRaises(LanguageError):
                    score_text(self.artifact, "hola", max_chars=limit)
        corrupt = deepcopy(self.artifact)
        corrupt["weights"]["by"][0] += 0.01
        with self.assertRaises(LanguageError):
            score_text(corrupt, "hola")


if __name__ == "__main__":
    unittest.main()
