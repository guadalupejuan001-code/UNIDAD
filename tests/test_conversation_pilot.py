"""The pilot's cooperative boundary must preserve the native optimizer and RNG."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training


SPEC = importlib.util.spec_from_file_location(
    "conversation_pilot_example", Path(__file__).resolve().parents[1] / "examples/run_conversation_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class StopOwnershipTests(unittest.TestCase):
    def test_unrelated_stop_request_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"pid": 999999999, "processBirth": "missing", "owner": "another-worker"}
            training._atomic_json(root / "stop-request.json", request)
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                pilot._clear_own_stop(root, {"runnerId": "fixture"})
            self.assertEqual(training._json_read(root / "stop-request.json"), request)

    def test_live_previous_owner_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"pid": 123456789, "processBirth": "alive", "owner": pilot.OWNER,
                       "runnerId": "fixture"}
            training._atomic_json(root / "stop-request.json", request)
            with patch.object(training, "_birth", return_value="alive"):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    pilot._clear_own_stop(root, {"runnerId": "fixture"})
            self.assertTrue((root / "stop-request.json").exists())


@unittest.skipUnless(HAS_EXTRAS, "conversation dependencies are optional")
class PilotNativeTests(unittest.TestCase):
    def setUp(self):
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rows = [{"id": f"r{index}", "groupId": f"g{index}", "split": split,
                 "source": {"dataset": "pilot-fixture", "kind": "authored-synthetic"},
                 "messages": [{"role": "user", "content": f"Soy la persona ficticia {index}."},
                              {"role": "assistant", "content": f"Hola, persona ficticia {index}."}]}
                for index, split in enumerate(("train", "train", "validation"))]
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        self.digest = hashlib.sha256(self.corpus.read_bytes()).hexdigest()
        tokenizer = fit_tokenizer(rows[:2], vocab_size=300)
        model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 128, "dropout": 0.1,
            "ffn_multiplier": 2, "seed": 7})
        self.parent = self.root / "parent"
        save_artifact(self.parent, model, tokenizer,
                      training={"trainedFrom": "random_weights", "corpusSha256": self.digest})
        self.config = dict(pilot.CONFIG, batch_size=1, max_steps=4, max_seconds=60,
                           validation_every=2, checkpoint_every=2, torch_threads=1,
                           warmup_steps=1, patience=20)

    def tearDown(self):
        self.temporary.cleanup()

    def run_pilot(self, name, boundary, resume=False):
        with patch.object(pilot, "CONFIG", self.config):
            return pilot.run_pilot(self.corpus, self.digest, self.parent, self.root / name,
                                   stop_after_step=boundary, budget_seconds=30, resume=resume)

    def test_native_step_stop_and_resume_are_exact(self):
        import torch
        original_log = training._log
        continuous = self.run_pilot("continuous", 4)
        first = self.run_pilot("resumed", 2)
        self.assertEqual(first["nativeStatus"]["status"], "stopped")
        self.assertEqual(first["nativeStatus"]["step"], 2)
        self.assertTrue(first["cooperativeStepStopWritten"])
        resumed = self.run_pilot("resumed", 4, resume=True)
        self.assertIs(training._log, original_log)
        self.assertEqual(resumed["nativeStatus"]["step"], 4)
        self.assertFalse(resumed["testEvaluated"])
        self.assertFalse(resumed["conversationAccepted"])
        self.assertTrue(resumed["inputsAndSourceUnchanged"])
        a, _, state_a, payload_a = training._load_checkpoint(
            self.root / "continuous", continuous["nativeStatus"]["lastCheckpoint"])
        b, _, state_b, payload_b = training._load_checkpoint(
            self.root / "resumed", resumed["nativeStatus"]["lastCheckpoint"])
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[name]), name)
        for index, moments in payload_a["optimizer"]["state"].items():
            for name, value in moments.items():
                self.assertTrue(torch.equal(value, payload_b["optimizer"]["state"][index][name]))
        self.assertEqual(payload_a["scheduler"], payload_b["scheduler"])
        self.assertTrue(torch.equal(payload_a["torch_rng"], payload_b["torch_rng"]))
        self.assertEqual(state_a["python_rng"], state_b["python_rng"])
        self.assertEqual(state_a["last_validation_step"], state_b["last_validation_step"])
        self.assertEqual(state_a["best_validation_nll"], state_b["best_validation_nll"])
        self.assertEqual(state_a["training_config"], state_b["training_config"])
        self.assertEqual(state_a["lineage"], state_b["lineage"])

    def test_wrong_identity_and_test_records_are_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "declared identity"):
            pilot.run_pilot(self.corpus, "0" * 64, self.parent, self.root / "wrong")
        rows = [json.loads(line) for line in self.corpus.read_text().splitlines()]
        rows[-1]["split"] = "test"
        self.corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.digest = hashlib.sha256(self.corpus.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "TEST"):
            self.run_pilot("test-rejected", 2)
        self.assertFalse((self.root / "test-rejected/status.json").exists())

    def test_native_failure_restores_logging(self):
        original_log = training._log
        with patch.object(pilot, "CONFIG", self.config), patch.object(
                training, "train", side_effect=RuntimeError("fixture trainer failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture trainer failure"):
                pilot.run_pilot(self.corpus, self.digest, self.parent, self.root / "failed",
                                stop_after_step=2)
        self.assertIs(training._log, original_log)

    def test_custom_stage_policy_is_saved_and_resume_cannot_change_it(self):
        overrides = {"learning_rate": 0.00006, "loss_reduction": "sequence_mean"}
        directory = self.root / "custom-policy"
        with patch.object(pilot, "CONFIG", self.config):
            first = pilot.run_pilot(self.corpus, self.digest, self.parent, directory,
                                    stop_after_step=2, training_config=overrides)
            design = training._json_read(directory / "pilot-design.json")
            self.assertEqual(design["trainingConfig"]["loss_reduction"], "sequence_mean")
            self.assertEqual(design["trainingConfig"]["learning_rate"], 0.00006)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                pilot.run_pilot(self.corpus, self.digest, self.parent, directory,
                                stop_after_step=4, resume=True,
                                training_config={"learning_rate": 0.0002})
            resumed = pilot.run_pilot(self.corpus, self.digest, self.parent, directory,
                                      stop_after_step=4, resume=True)
        self.assertEqual(first["nativeStatus"]["step"], 2)
        self.assertEqual(resumed["nativeStatus"]["step"], 4)
        _, _, state, _ = training._load_checkpoint(directory, resumed["nativeStatus"]["lastCheckpoint"])
        self.assertEqual(state["training_config"], design["trainingConfig"])


if __name__ == "__main__":
    unittest.main()
