"""Pin a declared corpus before imports, allocations and run-directory creation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training


class ExpectedCorpusIdentityTests(unittest.TestCase):
    def test_malformed_expected_identity_fails_without_read_import_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(training, "_torch") as importing, patch.object(training, "_sha") as reading:
                for value in (True, 64, "", "f" * 63, "f" * 65, "g" * 64, " " + "f" * 64):
                    with self.subTest(value=value), self.assertRaisesRegex(ValueError, "expected_corpus_sha256"):
                        training.train(root / "missing.jsonl", root / "run", expected_corpus_sha256=value)
                importing.assert_not_called()
                reading.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_corpus_mutated_after_declaration_rejected_before_allocation_and_run_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            original = b'{"id":"fixture"}\n'
            corpus.write_bytes(original)
            declared = hashlib.sha256(original).hexdigest()
            # Whitespace preserves parsed content but changes declared bytes.
            corpus.write_bytes(original + b"\n")
            with patch.object(training, "_torch") as importing, patch.object(training, "_run_lock") as locking:
                with self.assertRaisesRegex(ValueError, "expected declared identity"):
                    training.train(corpus, root / "new-parent/run", expected_corpus_sha256=declared)
                importing.assert_not_called()
                locking.assert_not_called()
            self.assertEqual(list(root.iterdir()), [corpus])


class ExpectedArchitectureIdentityTests(unittest.TestCase):
    CONFIG = {"vocab_size": 300, "d_model": 16, "n_layers": 1, "n_heads": 2,
              "context_length": 32, "dropout": 0.0, "ffn_multiplier": 1, "seed": 17}

    def test_malformed_expected_architecture_precedes_corpus_read_and_run_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(training, "_torch") as importing, patch.object(training, "_sha") as reading:
                for value in (True, [], {"vocab_size": 4}, {**self.CONFIG, "d_model": True}):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        training.train(root / "absent.jsonl", root / "run", expected_model_config=value)
                importing.assert_not_called()
                reading.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_initial_model_mismatch_precedes_artifact_or_tensor_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(self.CONFIG))
            # A mismatched bounded header must be sufficient to refuse the
            # artifact; no optional dependency or weights file is required.
            with patch.object(training, "_torch") as importing:
                with self.assertRaisesRegex(ValueError, "expected declared architecture"):
                    training._initial_own_model(root, None,
                        expected_model_config={**self.CONFIG, "d_model": 8})
                importing.assert_not_called()
            self.assertEqual([p.name for p in root.iterdir()], ["config.json"])

    def test_verified_checkpoint_mismatch_precedes_tensor_constructor_and_weight_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints/fixture"
            checkpoint.mkdir(parents=True)
            payloads = {"state.json": json.dumps({"model_config": self.CONFIG}).encode(),
                        "weights.safetensors": b"not tensors; loader must not read these",
                        "resume.pt": b"not a pickle; loader must not deserialize these",
                        "tokenizer.json": b"{}"}
            for name, raw in payloads.items():
                (checkpoint / name).write_bytes(raw)
            (checkpoint / "manifest.json").write_text(json.dumps({"schemaVersion": 1,
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}))
            with patch.object(training, "_torch") as importing:
                with self.assertRaisesRegex(ValueError, "expected declared architecture"):
                    training._load_checkpoint(root, "fixture",
                        expected_model_config={**self.CONFIG, "d_model": 8})
                importing.assert_not_called()

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                           for name in ("torch", "tokenizers", "safetensors")),
                         "optional conversation dependencies unavailable")
    def test_matching_expected_architecture_preserves_initial_artifact_and_checkpoint(self):
        import torch
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        from ia_propia.conversation.artifacts import save_artifact
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = [{"id": f"row-{i}", "groupId": f"group-{i}", "split": split,
                         "source": {"dataset": "authored-identity-tests", "kind": "authored-synthetic"},
                         "messages": [{"role": "user", "content": f"Hola Lía {i}."},
                                      {"role": "assistant", "content": "Hola Lía."}]}
                        for i, split in enumerate(("train", "validation", "test"))]
                corpus = root / "corpus.jsonl"
                corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
                tokenizer = fit_tokenizer([rows[0]], 300)
                model = ConversationModel({**self.CONFIG, "context_length": 256,
                                           "vocab_size": tokenizer.get_vocab_size()})
                parent = root / "parent"
                save_artifact(parent, model, tokenizer,
                              training={"trainedFrom": "random_weights", "corpusSha256": training._sha(corpus)})
                inherited, _, actual, _ = training._initial_own_model(parent, None,
                    expected_model_config=model.config)
                self.assertEqual(actual, model.config)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, inherited.state_dict()[name]), name)
                result = training.train(corpus, root / "run", initial_model=parent,
                    expected_corpus_sha256=training._sha(corpus), expected_model_config=model.config,
                    training_config={"max_steps": 1, "max_epochs": 1, "torch_threads": 1,
                                     "validation_every": 1, "checkpoint_every": 1})
                loaded, _, state, _ = training._load_checkpoint(root / "run", result["lastCheckpoint"],
                    expected_model_config=model.config)
                self.assertEqual(loaded.config, model.config)
                self.assertEqual(state["corpus_sha256"], training._sha(corpus))
                self.assertNotIn("expected_model_config", state)
                self.assertNotIn("expected_corpus_sha256", state)
        finally:
            torch.set_num_threads(previous_threads)

    def test_matching_expected_sha_accepts_both_hex_cases_and_absence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_bytes(b"identity-only fixture; parsing is intentionally not reached")
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            for expected in (None, digest, digest.upper()):
                with self.subTest(expected=expected):
                    with patch.object(training, "_torch", side_effect=RuntimeError("passed corpus identity gate")) as importing:
                        with self.assertRaisesRegex(RuntimeError, "passed corpus identity gate"):
                            training.train(corpus, root / "run", expected_corpus_sha256=expected)
                        importing.assert_called_once()
            self.assertEqual(list(root.iterdir()), [corpus])


if __name__ == "__main__":
    unittest.main()
