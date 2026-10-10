import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from ia_propia.language.model import (LanguageError, chunks, digest, encode, generate,
                                     initialize, load_artifact, loss_and_gradients,
                                     read_documents, seal_artifact, split_documents,
                                     tokenizer, train, validate_artifact, write_json)


class LanguageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "corpus.txt"
        self.documents = [f"ab cab cab cab cab. {index} ab cab cab cab cab. " * 2 for index in range(10)]
        self.path.write_text("\n---\n".join(self.documents), encoding="utf-8")

    def trained(self):
        return train(self.path, hidden_size=8, sequence_length=24, epochs=8, patience=3, seed=7)

    def test_full_bptt_matches_finite_differences_for_every_parameter(self):
        parameters = initialize(4, 3, 3)
        inputs = np.asarray([1, 2, 1, 3])
        target = np.asarray([2, 1, 3, 2])
        _, gradients = loss_and_gradients(parameters, inputs, target)
        epsilon = 1e-6
        for name, values in parameters.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + epsilon
                positive, _ = loss_and_gradients(parameters, inputs, target, gradients=False)
                values[index] = original - epsilon
                negative, _ = loss_and_gradients(parameters, inputs, target, gradients=False)
                values[index] = original
                self.assertAlmostEqual(gradients[name][index], (positive - negative) / (2 * epsilon), delta=2e-7)

    def test_split_and_tokenizer_cannot_see_heldout_characters(self):
        documents = ["ab first", "ab second", "ab third", "ab fourth", "ab fifth", "ab sixth", "Ω val", "Ω next", "Ж test", "Ж last"]
        splits, indices = split_documents(documents)
        vocabulary = tokenizer(splits["train"])
        self.assertNotIn("Ω", vocabulary["tokens"])
        self.assertNotIn("Ж", vocabulary["tokens"])
        self.assertEqual(encode("ΩЖ", vocabulary).tolist(), [0, 0])
        self.assertEqual(indices, {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]})
        all_indices = sum(indices.values(), [])
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_chunks_preserve_every_transition_without_crossing_documents(self):
        vocabulary = tokenizer(["abcdef", "zzab"])
        sequences = chunks(["abcdef", "zzab"], vocabulary, 2)
        observed = [(int(a), int(b)) for inputs, targets in sequences for a, b in zip(inputs, targets)]
        expected = [(int(a), int(b)) for text in ("abcdef", "zzab") for a, b in zip(encode(text, vocabulary)[:-1], encode(text, vocabulary)[1:])]
        self.assertEqual(observed, expected)
        self.assertTrue(all(len(inputs) <= 2 for inputs, _ in sequences))

    def test_training_changes_random_weights_and_learns_sequence(self):
        artifact, report = self.trained()
        self.assertGreater(artifact["training"]["bestEpoch"], 0)
        self.assertNotEqual(artifact["training"]["initialWeightsSha256"], artifact["training"]["selectedWeightsSha256"])
        self.assertLess(artifact["evaluation"]["splits"]["train"]["rnnNllNats"], artifact["training"]["initialTrainNllNats"])
        self.assertLess(artifact["evaluation"]["splits"]["test"]["rnnNllNats"], artifact["evaluation"]["splits"]["test"]["unigramNllNats"])
        self.assertFalse(artifact["evaluation"]["productionAccepted"])
        self.assertEqual(len(report["learningHistory"]), artifact["training"]["epochsCompleted"])

    def test_test_data_changes_neither_weights_tokenizer_baseline_nor_checkpoint(self):
        first, _ = self.trained()
        modified = self.documents[:8] + ["Ω unknown final document", "Ж another final document"]
        self.path.write_text("\n---\n".join(modified), encoding="utf-8")
        second, _ = self.trained()
        for name in ("config", "tokenizer", "weights", "modelId"):
            self.assertEqual(first[name], second[name])
        self.assertEqual(first["training"]["bestEpoch"], second["training"]["bestEpoch"])
        self.assertEqual(first["training"]["unigram"], second["training"]["unigram"])
        self.assertNotEqual(first["evaluation"]["splits"]["test"], second["evaluation"]["splits"]["test"])

    def test_roundtrip_generation_is_deterministic_and_labeled(self):
        artifact, _ = self.trained()
        path = Path(self.tmp.name) / "model.json"
        write_json(path, artifact)
        loaded = load_artifact(path)
        first = generate(artifact, "ab ", length=30, seed=123)
        second = generate(loaded, "ab ", length=30, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first["continuation"]), 30)
        self.assertTrue(first["experimental"])
        self.assertFalse(first["instructionFollowingValidated"])
        self.assertFalse(first["workerCapabilityValidated"])
        empty = generate(artifact, length=0)
        self.assertEqual(empty["continuation"], "")
        self.assertEqual(len(empty["effectivePrompt"]), 1)

    def test_invalid_configs_and_cross_split_duplicate_documents_are_rejected(self):
        for kwargs in ({"hidden_size": 0}, {"hidden_size": True}, {"sequence_length": 257},
                       {"epochs": 0}, {"learning_rate": float("nan")}, {"clip_norm": 0},
                       {"seed": -1}, {"train_fraction": 0.8, "validation_fraction": 0.3}):
            with self.assertRaises(LanguageError):
                train(self.path, **kwargs)
        with self.assertRaises(LanguageError):
            split_documents(["same", "middle", "same"])
        self.path.write_text("single document", encoding="utf-8")
        with self.assertRaises(LanguageError):
            read_documents(self.path)

    def test_artifact_shape_type_finite_hash_and_metadata_guards(self):
        artifact, _ = self.trained()
        malformed = []
        value = copy.deepcopy(artifact); value["weights"]["Whh"] = [[1]]; malformed.append(value)
        value = copy.deepcopy(artifact); value["weights"]["by"][0] = True; malformed.append(value)
        value = copy.deepcopy(artifact); value["weights"]["by"][0] = "1.0"; malformed.append(value)
        value = copy.deepcopy(artifact); value["weights"]["by"][0] = float("nan"); malformed.append(value)
        value = copy.deepcopy(artifact); value["config"]["hiddenSize"] = 10000; malformed.append(value)
        value = copy.deepcopy(artifact); value["schemaVersion"] = True; malformed.append(value)
        value = copy.deepcopy(artifact); value["tokenizer"]["tokens"][1] = "two characters"; malformed.append(value)
        value = copy.deepcopy(artifact); value["weights"]["by"][0] += 0.1; malformed.append(value)
        value = copy.deepcopy(artifact); value["training"]["unigram"]["probabilities"] = [1]; seal_artifact(value); malformed.append(value)
        value = copy.deepcopy(artifact); value["evaluation"]["newValue"] = float("inf"); malformed.append(value)
        value = copy.deepcopy(artifact); value["training"] = {}; seal_artifact(value); malformed.append(value)
        value = copy.deepcopy(artifact); value["evaluation"]["productionAccepted"] = True; seal_artifact(value); malformed.append(value)
        value = copy.deepcopy(artifact); value["training"]["splitDocumentIndices"]["test"] = [0, 1]; seal_artifact(value); malformed.append(value)
        for value in malformed:
            with self.assertRaises(LanguageError):
                validate_artifact(value)
        for kwargs in ({"length": -1}, {"length": 4097}, {"seed": True}, {"temperature": float("inf")}, {"temperature": -0.1}):
            with self.assertRaises(LanguageError):
                generate(artifact, **kwargs)

    def test_json_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        artifact, _ = self.trained()
        encoded = json.dumps(artifact)
        path = Path(self.tmp.name) / "artifact.json"
        path.write_text('{"schemaVersion": 1, ' + encoded[1:], encoding="utf-8")
        with self.assertRaisesRegex(LanguageError, "Duplicate"):
            load_artifact(path)
        path.write_text(encoded.replace('"rnnNllNats": ', '"rnnNllNats": NaN, "other": ', 1), encoding="utf-8")
        with self.assertRaisesRegex(LanguageError, "Nonfinite"):
            load_artifact(path)


if __name__ == "__main__":
    unittest.main()
