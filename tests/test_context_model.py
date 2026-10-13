"""Independent model integrity, source attribution, bounded input and resumability."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.context.model import ContextRanker, FEATURE_NAMES, load_ranker
from ia_propia.context.training import fictional_corpus, train_ranker, validate_corpus


class ContextModelTests(unittest.TestCase):
    def test_packaged_reference_is_the_delivered_own_model(self):
        packaged = load_ranker()
        root = Path(__file__).resolve().parents[1]
        delivered = load_ranker(root / "models/context-reference-v1/ranker.json")
        self.assertEqual(packaged.model_id, delivered.model_id)
        self.assertEqual(packaged.weights, delivered.weights)
        self.assertEqual(packaged.parameter_count, len(FEATURE_NAMES) + 1)
        self.assertTrue(any(weight != 0 for weight in packaged.weights))
        self.assertFalse(packaged.artifact()["generatesAnswers"])

    def test_ranker_recalls_a_correction_using_the_previous_thread(self):
        ranker = load_ranker()
        candidates = [
            {"id": "old", "text": "El taller Nebula abre el jueves.", "turn_index": 0},
            {"id": "corrected", "text": "Corrijo la apertura del taller Nebula: abre el lunes.", "turn_index": 1},
            {"id": "noise", "text": "Hablemos de una canción que escuché ayer.", "turn_index": 2},
        ]
        scored = ranker.score_candidates("¿Y eso?", candidates, thread="Revisemos el día de apertura del taller Nebula.")
        self.assertEqual(max(scored, key=lambda candidate: candidate["score"])["id"], "corrected")
        self.assertTrue(all(0 <= candidate["score"] <= 1 for candidate in scored))

    def test_ranking_keeps_exact_source_text_and_provenance_without_mutating_input(self):
        candidate = {"id": "source", "text": "No publiques los datos personales del taller Nebula.",
                     "role": "user", "turn_index": 4, "kind": "constraint",
                     "source": {"messageId": "real-source", "private": True}}
        before = deepcopy(candidate)
        result = load_ranker().score_candidates("¿Y la condición?", [candidate], thread="Datos personales del taller Nebula.")
        self.assertEqual(candidate, before)
        self.assertEqual(result[0]["text"], candidate["text"])
        self.assertEqual(result[0]["source"], candidate["source"])
        result[0]["source"]["messageId"] = "different"
        self.assertEqual(candidate, before)

    def test_artifact_corruption_and_nonfinite_parameters_are_rejected(self):
        artifact = load_ranker().artifact()
        artifact["weights"][0] += 0.5
        with self.assertRaisesRegex(ValueError, "hash"):
            ContextRanker.from_artifact(artifact)
        with self.assertRaises(ValueError):
            ContextRanker([float("nan")] * len(FEATURE_NAMES), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.json"
            path.write_text('{"schemaVersion":1,"schemaVersion":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_ranker(path)

    def test_epoch_boundary_resume_conserves_exact_weights(self):
        corpus = fictional_corpus()
        uninterrupted, _ = train_ranker(corpus, epochs=120, seed=123)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            train_ranker(corpus, epochs=50, seed=123, checkpoint=checkpoint)
            resumed, _ = train_ranker(corpus, epochs=120, seed=123, checkpoint=checkpoint, resume=True)
            self.assertEqual(resumed.artifact(), uninterrupted.artifact())
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            state["weights"][0] += 1
            checkpoint.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                train_ranker(corpus, epochs=121, seed=123, checkpoint=checkpoint, resume=True)

    def test_training_rejects_family_leakage_and_changed_resume_corpus(self):
        corpus = fictional_corpus()
        altered = deepcopy(corpus)
        altered[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "partition leakage"):
            validate_corpus(altered)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            train_ranker(corpus, epochs=1, checkpoint=checkpoint)
            changed = deepcopy(corpus)
            changed[0]["query"] += " distinto"
            with self.assertRaisesRegex(ValueError, "configuration or corpus"):
                train_ranker(changed, epochs=2, checkpoint=checkpoint, resume=True)

    def test_input_bounds_and_source_roles_are_checked_before_scoring(self):
        ranker = load_ranker()
        with self.assertRaises(ValueError):
            ranker.score_candidates("query", [{"text": ""}])
        with self.assertRaises(ValueError):
            ranker.score_candidates("query", [{"text": "source", "role": "tool-admin"}])
        with self.assertRaises(ValueError):
            ranker.score_candidates("x" * 16385, [{"text": "source"}])
        with self.assertRaises(ValueError):
            ranker.score_candidates("query", [{"text": "source", "turn_index": True}])


if __name__ == "__main__":
    unittest.main()
