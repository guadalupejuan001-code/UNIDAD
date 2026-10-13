import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.selection import (
    FEATURE_NAMES, LearnedSelector, candidate_features, load_selector,
    save_selector, train_selector,
)


MESSAGES = [{"role": "user", "content": "¿Dónde está la carpeta violeta?"}]


def candidate(name, text="La carpeta está en la caja violeta.", *, support=0.9,
              critic=0.2, likelihood=3.0, unknown=0.0):
    auxiliary = {} if likelihood is None else {
        "meanNll": likelihood, "unknownRatio": unknown, "scoreable": True,
    }
    return {"name": name, "text": text, "criticRankCost": critic,
            "lexicalSourceSupport": support, "auxiliary": auxiliary,
            "generatedTokens": 12}


def cohort(identity, high=0.9, low=0.1, *, reverse=False):
    good, bad = [0.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES)
    good[FEATURE_NAMES.index("lexical_source_support")] = high
    bad[FEATURE_NAMES.index("lexical_source_support")] = low
    return {"cohortId": identity, "features": [good, bad],
            "utilities": [0, 1] if reverse else [1, 0]}


class ConversationSelectionTests(unittest.TestCase):
    def train(self, **kwargs):
        return train_selector([cohort("train/family-A")],
                              [cohort("validation/family-B", 0.8, 0.2)],
                              epochs=100, **kwargs)

    def test_features_ignore_candidate_identity_and_use_current_question(self):
        first = candidate("historic-reference")
        renamed = copy.deepcopy(first)
        renamed["name"] = "brand-new-version"
        renamed["modelId"] = "not-a-feature"
        original = candidate_features(MESSAGES, [first])[0]
        self.assertEqual(original, candidate_features(MESSAGES, [renamed])[0])
        messages = [{"role": "user", "content": "¿Qué hora es?"}, *MESSAGES]
        self.assertEqual(original, candidate_features(messages, [first])[0])
        self.assertEqual(len(original), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in original))

    def test_rnn_ranks_are_relative_and_keep_ties_and_missing_explicit(self):
        candidates = [candidate("a", likelihood=2), candidate("b", likelihood=5),
                      candidate("c", likelihood=5), candidate("missing", likelihood=None)]
        features = candidate_features(MESSAGES, candidates)
        position = FEATURE_NAMES.index("rnn_preference")
        self.assertEqual([row[position] for row in features], [1, 0.25, 0.25, 0])
        available = FEATURE_NAMES.index("rnn_available")
        self.assertEqual([row[available] for row in features], [1, 1, 1, 0])
        changed = copy.deepcopy(candidates)
        for item in changed[:3]:
            item["auxiliary"]["meanNll"] = 11 + 7 * item["auxiliary"]["meanNll"]
        self.assertEqual(features, candidate_features(MESSAGES, changed))
        changed[0]["auxiliary"]["scoreable"] = False
        self.assertEqual(candidate_features(MESSAGES, changed)[0][available], 0)

    def test_long_diverse_text_and_repetition_remain_bounded(self):
        diverse = "".join(chr(0x4E00 + index) for index in range(600))
        features = candidate_features(MESSAGES, [candidate("unicode", diverse),
            candidate("repeat", "A" * 600), candidate("word-repeat", "caja " * 150)])
        self.assertTrue(all(0 <= value <= 1 for row in features for value in row))
        run = FEATURE_NAMES.index("maximum_character_run")
        repeat = FEATURE_NAMES.index("repeated_word_fraction")
        self.assertEqual(features[1][run], 1)
        self.assertGreater(features[2][repeat], 0.9)

    def test_training_learns_a_preference_and_selects_exact_member(self):
        artifact = self.train(provenance={"corpusSha256": "a" * 64,
                                          "kind": "fictional-test-fixture"})
        training = artifact["training"]
        self.assertLess(training["selectedTrainLoss"], training["initialTrainLoss"])
        self.assertEqual(training["rootOrigin"], "random_weights")
        self.assertFalse(training["testUsedForTraining"])
        self.assertEqual(artifact["parameterCount"], len(FEATURE_NAMES))
        selector = LearnedSelector(artifact)
        candidates = [candidate("bad", "AAAAAA", support=0.1), candidate("good", support=0.9)]
        before = copy.deepcopy(candidates)
        result = selector.select(MESSAGES, candidates)
        self.assertEqual(result["selectedName"], "good")
        self.assertEqual(result["modelId"], selector.model_id)
        self.assertTrue(result["learned"])
        self.assertEqual(candidates, before)
        self.assertNotIn("text", result)
        self.assertGreater(result["candidateScores"][1]["score"], result["candidateScores"][0]["score"])

    def test_validation_selects_initial_weights_when_training_would_hurt(self):
        artifact = train_selector([cohort("train/A")],
            [cohort("validation/B", reverse=True)], epochs=20, seed=42)
        rng = random.Random(42)
        self.assertEqual(artifact["weights"], [rng.gauss(0, 0.02) for _ in FEATURE_NAMES])
        self.assertEqual(artifact["training"]["selectedEpoch"], 0)
        self.assertEqual(artifact["training"]["completedEpochs"], 20)

    def test_same_seed_and_data_reproduce_weights_and_losses(self):
        first, second = self.train(seed=19), self.train(seed=19)
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["training"]["selectedValidationLoss"],
                         second["training"]["selectedValidationLoss"])
        self.assertEqual(first["training"]["trainingCohortsSha256"],
                         second["training"]["trainingCohortsSha256"])

    def test_training_rejects_absent_preferences_and_overlapping_cohorts(self):
        tied = cohort("train/A")
        tied["utilities"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "No preference signal"):
            train_selector([tied], [cohort("validation/B")])
        indistinguishable = cohort("train/A", high=0.5, low=0.5)
        with self.assertRaisesRegex(ValueError, "No distinguishable preference signal"):
            train_selector([indistinguishable], [cohort("validation/B")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            train_selector([cohort("same/A")], [cohort("same/A")])

    def test_deadline_returns_valid_initial_snapshot_with_real_stop_reason(self):
        with patch("ia_propia.conversation.selection.perf_counter", side_effect=[0, 0.002, 0.003]):
            artifact = train_selector([cohort("train/A")], [cohort("validation/B")], max_seconds=0.001)
        self.assertEqual(artifact["training"]["completedEpochs"], 0)
        self.assertEqual(artifact["training"]["stopReason"], "time_budget")
        self.assertEqual(LearnedSelector(artifact).parameter_count, len(FEATURE_NAMES))

    def test_saved_artifact_preserves_identity_and_rejects_corruption(self):
        artifact = self.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "selector.json"
            self.assertEqual(save_selector(path, artifact), artifact["modelId"])
            self.assertEqual(load_selector(path), artifact)
            corrupted = copy.deepcopy(artifact)
            corrupted["weights"][0] += 0.1
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or identity"):
                load_selector(path)
            path.write_bytes(b"x" * 131073)
            with self.assertRaisesRegex(ValueError, "byte limit"):
                load_selector(path)

    def test_missing_diagnostics_single_member_and_exact_tie_are_supported(self):
        artifact = self.train()
        selector = LearnedSelector(artifact)
        first = candidate("first", critic=None, likelihood=None)
        second = copy.deepcopy(first)
        second["name"] = "second"
        self.assertEqual(selector.select(MESSAGES, [first, second])["selectedName"], "first")
        self.assertEqual(selector.select(MESSAGES, [second])["selectedName"], "second")

    def test_loader_rejects_duplicate_fields_and_nonstandard_json_constants(self):
        artifact = self.train()
        raw = json.dumps(artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            # The last duplicate has the valid value and would otherwise retain
            # the original identity after an ordinary json.loads collapsed it.
            path.write_text('{"schemaVersion":99,' + raw[1:], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate key"):
                load_selector(path)
            path.write_text(raw.replace('"experimental": true', '"experimental": NaN'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                load_selector(path)
            path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_selector(path)

    def test_rejects_nonfinite_signals_invalid_contract_and_forged_provenance(self):
        for key, value in (("criticRankCost", float("nan")), ("lexicalSourceSupport", 2),
                           ("generatedTokens", True), ("text", "")):
            item = candidate("bad")
            item[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                candidate_features(MESSAGES, [item])
        with self.assertRaises(ValueError):
            candidate_features(MESSAGES, [candidate("same"), candidate("same")])
        artifact = self.train()
        artifact["training"]["rootOrigin"] = "external_pretrained"
        with self.assertRaisesRegex(ValueError, "provenance"):
            LearnedSelector(artifact)
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            self.train(provenance={"invalid": float("inf")})


if __name__ == "__main__":
    unittest.main()
