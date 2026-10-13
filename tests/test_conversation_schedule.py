"""Optional cosine LR must preserve checkpoint checks and exact continuation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class LearningRateScheduleTests(unittest.TestCase):
    def test_legacy_configuration_bytes_and_warmup_are_unchanged(self):
        historical = {
            "batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.01,
            "grad_clip": 1.0, "max_epochs": 100, "max_steps": 100000,
            "max_seconds": 86400, "validation_every": 100, "checkpoint_every": 100,
            "patience": 12, "min_delta": 0.0001, "seed": 42,
            "warmup_steps": 100, "torch_threads": 4,
        }
        config = training._configs(None, None)[1]
        self.assertEqual(training._json_bytes(config), training._json_bytes(historical))
        self.assertNotIn("learning_rate_schedule", training.TRAIN_DEFAULTS)
        for warmup in (0, 1, 2, 100):
            legacy = {**config, "warmup_steps": warmup}
            for step in (0, 1, 2, 99, 100, 1000):
                self.assertEqual(training._lr_multiplier(legacy, step),
                                 min(1.0, (step + 1) / max(1, warmup)))
        # Old runs may finish before their warmup; do not add a new constraint.
        training._configs(None, {"warmup_steps": 100, "max_steps": 1})

    def test_cosine_warmup_midpoint_horizon_and_plateau(self):
        config = {"warmup_steps": 2, "learning_rate_schedule": {
            "type": "cosine", "decay_steps": 6, "min_lr_ratio": 0.2}}
        for step, multiplier in ((0, 0.5), (1, 1.0), (2, 1.0),
                                 (4, 0.6), (6, 0.2), (7, 0.2), (100, 0.2)):
            self.assertAlmostEqual(training._lr_multiplier(config, step), multiplier)
        config["warmup_steps"] = 0
        config["learning_rate_schedule"]["min_lr_ratio"] = 0
        self.assertEqual(training._lr_multiplier(config, 0), 1)
        self.assertEqual(training._lr_multiplier(config, 6), 0)
        config["learning_rate_schedule"]["min_lr_ratio"] = 1
        self.assertEqual(training._lr_multiplier(config, 3), 1)

    def test_schedule_configuration_is_strict_and_detached(self):
        valid = {"type": "cosine", "decay_steps": 6, "min_lr_ratio": 0.2}
        invalid = [None, True, [], {}, {**valid, "type": "constant"},
                   {**valid, "extra": 1}, {"type": "cosine", "decay_steps": 6}]
        invalid += [{**valid, "decay_steps": value} for value in (True, 0, -1, 2, 9, 6.0)]
        invalid += [{**valid, "min_lr_ratio": value}
                    for value in (True, -0.1, 1.1, float("nan"), float("inf"), "0.2")]
        for schedule in invalid:
            with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, "learning_rate_schedule"):
                training._configs(None, {"warmup_steps": 2, "max_steps": 8,
                                         "learning_rate_schedule": schedule})
        config = training._configs(None, {"warmup_steps": 2, "max_steps": 8,
                                          "learning_rate_schedule": valid})[1]
        valid["min_lr_ratio"] = 0.8
        self.assertEqual(config["learning_rate_schedule"]["min_lr_ratio"], 0.2)
        with self.assertRaisesRegex(ValueError, "Unknown configuration fields"):
            training._configs(None, {"learning_rate_scheduler": valid})


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class NativeCosineResumeTests(unittest.TestCase):
    def test_native_resume_preserves_weights_adam_rng_and_rejects_corruption(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            rows = []
            for index, split in enumerate(("train", "train", "validation", "test")):
                rows.append({"id": f"schedule-{index}", "groupId": f"schedule-{index}",
                             "split": split, "source": {
                                 "dataset": "authored-unit-tests", "kind": "authored-synthetic"},
                             "messages": [{"role": "user", "content": f"¿Qué guardo en la caja {index}?"},
                                          {"role": "assistant", "content": f"Guarda la cinta azul {index} en la caja."}]})
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            architecture = {"vocab_size": 300, "d_model": 8, "n_layers": 1, "n_heads": 2,
                            "context_length": 16, "dropout": 0.2, "ffn_multiplier": 1, "seed": 7}
            config = {"batch_size": 2, "max_steps": 8, "max_epochs": 50,
                      "validation_every": 2, "checkpoint_every": 2, "patience": 100,
                      "min_delta": 0, "seed": 7, "warmup_steps": 2, "torch_threads": 1,
                      "learning_rate_schedule": {"type": "cosine", "decay_steps": 6,
                                                 "min_lr_ratio": 0.25}}
            continuous = training.train(corpus, root / "continuous", model_config=architecture,
                                        training_config=config)
            model_a, _, state_a, payload_a = training._load_checkpoint(
                root / "continuous", continuous["lastCheckpoint"])
            checkpoint = training._checkpoint
            for cut in (4, 6):
                with self.subTest(cut=cut):
                    directory = root / f"resumed-{cut}"

                    def pause(*args, **kwargs):
                        name = checkpoint(*args, **kwargs)
                        if args[5]["step"] == cut:
                            training._atomic_json(directory / "stop-request.json", {
                                "pid": os.getpid(), "processBirth": training._birth(os.getpid())})
                        return name

                    with patch.object(training, "_checkpoint", side_effect=pause):
                        stopped = training.train(corpus, directory, model_config=architecture,
                                                 training_config=config)
                    self.assertEqual((stopped["status"], stopped["step"]), ("stopped", cut))
                    (directory / "stop-request.json").unlink()
                    changed = copy.deepcopy(config)
                    changed["learning_rate_schedule"]["min_lr_ratio"] = 0.5
                    with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                        training.train(corpus, directory, resume=True, training_config=changed)
                    resumed = training.train(corpus, directory, resume=True)
                    model_b, _, state_b, payload_b = training._load_checkpoint(
                        directory, resumed["lastCheckpoint"])
                    self.assertEqual(resumed["step"], 8)
                    for key, value in model_a.state_dict().items():
                        self.assertTrue(torch.equal(value, model_b.state_dict()[key]), key)
                    self.assertEqual(payload_a["optimizer"]["param_groups"],
                                     payload_b["optimizer"]["param_groups"])
                    for index, moments in payload_a["optimizer"]["state"].items():
                        for key, value in moments.items():
                            self.assertTrue(torch.equal(value, payload_b["optimizer"]["state"][index][key]))
                    self.assertEqual(payload_a["scheduler"], payload_b["scheduler"])
                    self.assertEqual(payload_a["scheduler"]["lr_lambdas"], [None])
                    self.assertTrue(torch.equal(payload_a["torch_rng"], payload_b["torch_rng"]))
                    for key in ("python_rng", "order", "cursor", "epoch", "training_config",
                                "best_validation_nll", "bad_validations", "last_validation_step"):
                        self.assertEqual(state_a[key], state_b[key], key)
            saved_config = state_a["training_config"]
            optimizer = torch.optim.AdamW(model_a.parameters(), lr=saved_config["learning_rate"],
                                          weight_decay=saved_config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: training._lr_multiplier(saved_config, step))
            training._validate_resume_payload(payload_a, optimizer, scheduler, model_a, state_a)
            self.assertEqual(payload_a["optimizer"]["param_groups"][0]["lr"],
                             saved_config["learning_rate"] * 0.25)
            for value in (True, float("nan"), saved_config["learning_rate"]):
                broken = copy.deepcopy(payload_a)
                broken["optimizer"]["param_groups"][0]["lr"] = value
                with self.assertRaisesRegex(ValueError, "learning rate mismatch"):
                    training._validate_resume_payload(broken, optimizer, scheduler, model_a, state_a)
            for key, value in (("_last_lr", [0.123]), ("last_epoch", 7), ("extra", 1)):
                broken = copy.deepcopy(payload_a)
                broken["scheduler"][key] = value
                with self.assertRaisesRegex(ValueError, "scheduler|Scheduler"):
                    training._validate_resume_payload(broken, optimizer, scheduler, model_a, state_a)


if __name__ == "__main__":
    unittest.main()
