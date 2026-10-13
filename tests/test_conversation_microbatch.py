"""Logical-batch loss, memory release and resumability on authored fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training

HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class MicrobatchConfigurationTests(unittest.TestCase):
    def test_optin_and_invalid_combinations_before_torch_or_run_write(self):
        self.assertNotIn("micro_batch_size", training.TRAIN_DEFAULTS)
        self.assertNotIn("micro_batch_size", training._configs(None, None)[1])
        self.assertNotIn("trainingMicrobatch", training._loss_metadata({}))
        self.assertNotIn("parameter_budget", training._configs(None, None)[0])
        invalid = [{"batch_size": 4, "micro_batch_size": value}
                   for value in (None, True, 0, -1, 5, 1.5, "2")]
        invalid += [{"micro_batch_size": 2, "copy_supervision": {}},
                    {"micro_batch_size": 2, "adaptive_sampling": {}}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(training, "_torch") as importing:
                for config in invalid:
                    with self.subTest(config=config), self.assertRaises(ValueError):
                        training.train(root / "absent.jsonl", root / "run", training_config=config)
                importing.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
        for value in (None, True, 19_999_999, 1_200_000_001):
            with self.subTest(parameter_budget=value), self.assertRaisesRegex(ValueError, "parameter_budget"):
                training._configs({"parameter_budget": value}, None)
        self.assertEqual(training._configs({"parameter_budget": 100_000_000}, None)[0]["parameter_budget"], 100_000_000)


@unittest.skipUnless(HAS_EXTRAS, "optional conversation dependencies unavailable")
class NativeMicrobatchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        self.previous_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"batch_size": 2, "max_steps": 4, "max_epochs": 50, "validation_every": 2,
                       "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 29,
                       "warmup_steps": 2, "torch_threads": 1, "projection_mode": "supervised",
                       "micro_batch_size": 1}
        self.model_config = {"vocab_size": 300, "d_model": 16, "n_layers": 2, "n_heads": 2,
                             "context_length": 256, "dropout": 0.2, "ffn_multiplier": 2, "seed": 17}
        self.rows = [{"id": f"fixture-{index}", "groupId": f"family-{index}", "split": split,
                      "source": {"dataset": "authored-microbatch-tests", "kind": "authored-synthetic"},
                      "messages": [{"role": "user", "content": f"Saluda a Lía {index}."},
                                   {"role": "assistant", "content": ("Hola Lía." if index % 2 else "Hola Lía. ¿Qué te gustaría conversar?")}]}
                     for index, split in enumerate(("train", "train", "train", "validation", "test"))]
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()
        self.torch.set_num_threads(self.previous_threads)
        self.torch.set_rng_state(self.previous_rng)
        random.setstate(self.previous_python_rng)

    def run_train(self, name, config=None, model_config=None):
        return training.train(self.corpus, self.root / name, model_config=model_config or self.model_config,
                              training_config=config or self.config)

    def assert_tree_equal(self, a, b):
        if isinstance(a, self.torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(self.torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(type(a), type(b))
            self.assertEqual(a, b)

    def test_global_loss_gradients_and_adam_match_with_unequal_target_counts(self):
        from ia_propia.conversation.model import ConversationModel, assistant_token_loss
        torch = self.torch
        inputs = torch.tensor([[7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20],
                               [21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34],
                               [35, 36, 37, 38, 39, 40, 41]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 42],
                               [-100, -100, -100, 43, 44, 45, 46], [-100] * 7,
                               [-100, -100, -100, -100, 47, 48, 49],
                               [-100, -100, -100, -100, -100, 50, 51]])
        for projection in ("full", "supervised"):
            for reduction in ("token_mean", "sequence_mean"):
                for micro_size in (1, 2, 3):
                    with self.subTest(projection=projection, reduction=reduction, micro_size=micro_size):
                        a = ConversationModel({**self.model_config, "dropout": 0.0})
                        b = ConversationModel(a.config)
                        oa = torch.optim.AdamW(a.parameters(), lr=0.0001)
                        ob = torch.optim.AdamW(b.parameters(), lr=0.0001)
                        logits = a.selected_logits(inputs, labels != -100) if projection == "supervised" else a(inputs)
                        expected = assistant_token_loss(logits, labels, reduction, projection_mode=projection)
                        expected.backward()
                        actual = training._microbatch_backward(b, inputs, labels,
                            {"micro_batch_size": micro_size, "projection_mode": projection, "loss_reduction": reduction})
                        torch.testing.assert_close(actual, expected.detach(), rtol=2e-6, atol=2e-6)
                        for left, right in zip(a.parameters(), b.parameters()):
                            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
                        oa.step(); ob.step()
                        for left, right in zip(a.parameters(), b.parameters()):
                            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
                        for index, moments in oa.state_dict()["state"].items():
                            for key, value in moments.items():
                                torch.testing.assert_close(value, ob.state_dict()["state"][index][key], rtol=2e-5, atol=2e-6)

    def test_two_plus_two_resume_exact_with_partial_last_batch_fp32_and_bf16(self):
        for precision in (None, "cpu_bfloat16"):
            for reduction in ("token_mean", "sequence_mean"):
                with self.subTest(precision=precision, reduction=reduction):
                    config = {**self.config, "loss_reduction": reduction}
                    if precision:
                        config["train_precision"] = precision
                    name = f"{precision}-{reduction}"
                    full = self.run_train(name + "-full", config)
                    native_checkpoint = training._checkpoint
                    stopped = []

                    def checkpoint(*args, **kwargs):
                        pointer = native_checkpoint(*args, **kwargs)
                        if args[5]["step"] == 2 and not stopped:
                            stopped.append(True)
                            os.kill(os.getpid(), signal.SIGTERM)
                        return pointer

                    with patch.object(training, "_checkpoint", side_effect=checkpoint):
                        paused = self.run_train(name + "-paused", config)
                    self.assertEqual(paused["step"], 2)
                    resumed = training.train(self.corpus, self.root / (name + "-paused"), resume=True)
                    a, _, sa, pa = training._load_checkpoint(self.root / (name + "-full"), full["lastCheckpoint"])
                    b, _, sb, pb = training._load_checkpoint(self.root / (name + "-paused"), resumed["lastCheckpoint"])
                    self.assert_tree_equal(a.state_dict(), b.state_dict())
                    self.assert_tree_equal(pa, pb)
                    for key in ("order", "cursor", "epoch", "python_rng", "best_validation_nll"):
                        self.assert_tree_equal(sa[key], sb[key])
                    self.assertEqual((sa["step"], sa["epoch"], sa["cursor"]), (4, 2, 0))
                    exported = training.export_best(self.root / (name + "-paused"), self.root / (name + "-export"))
                    metadata = exported["training"]["trainingMicrobatch"]
                    self.assertEqual(metadata["logicalBatchSize"], 2)
                    self.assertEqual(metadata["maxMicroBatchSize"], 1)
                    self.assertEqual(metadata["optimizerStepsUnit"], "logical_batch")
                    self.assertFalse(exported["training"]["testEvaluated"])

    def test_single_logical_step_scheduler_and_clipping_after_three_microbatches(self):
        config = {**self.config, "batch_size": 3, "max_steps": 1}
        native_clip = self.torch.nn.utils.clip_grad_norm_
        calls = []
        native_backward = training._microbatch_backward

        def backward(*args, **kwargs):
            calls.append(args[1].shape[0])
            return native_backward(*args, **kwargs)

        with patch.object(training, "_microbatch_backward", side_effect=backward), \
                patch.object(self.torch.nn.utils, "clip_grad_norm_", wraps=native_clip) as clip:
            result = self.run_train("one-step", config)
        self.assertEqual(calls, [3])
        self.assertEqual(clip.call_count, 1)
        _, _, state, payload = training._load_checkpoint(self.root / "one-step", result["lastCheckpoint"])
        self.assertEqual(state["step"], 1)
        self.assertEqual(payload["scheduler"]["last_epoch"], 1)
        self.assertTrue(all(float(value["step"]) == 1 for value in payload["optimizer"]["state"].values()))

    def test_resume_rejects_changed_microbatch_and_resealed_invalid_config(self):
        result = self.run_train("invalid-resume")
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, self.root / "invalid-resume", resume=True,
                           training_config={**self.config, "micro_batch_size": 2})
        _, tokenizer, state, _ = training._load_checkpoint(self.root / "invalid-resume", result["lastCheckpoint"])
        state["training_config"]["micro_batch_size"] = 0
        examples = training._encoded(self.rows, tokenizer, 256)
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            training._validate_resume_state(state, examples)

    def test_failed_second_microbatch_keeps_completed_checkpoint_and_resumes_exactly(self):
        from ia_propia.conversation.model import ConversationModel
        continuous = self.run_train("uninterrupted")
        native_projection = ConversationModel.selected_logits
        calls = []

        def projection(model, *args, **kwargs):
            if model.training:
                calls.append(True)
                if len(calls) == 2:
                    raise RuntimeError("fabricated second-microbatch failure")
            return native_projection(model, *args, **kwargs)

        with patch.object(ConversationModel, "selected_logits", projection):
            with self.assertRaisesRegex(RuntimeError, "second-microbatch"):
                self.run_train("interrupted")
        status = training._json_read(self.root / "interrupted/status.json")
        self.assertEqual((status["status"], status["step"]), ("failed", 0))
        _, _, saved, payload = training._load_checkpoint(self.root / "interrupted", status["lastCheckpoint"])
        self.assertEqual(saved["step"], 0)
        self.assertEqual(payload["optimizer"]["state"], {})
        resumed = training.train(self.corpus, self.root / "interrupted", resume=True)
        a, _, _, pa = training._load_checkpoint(self.root / "uninterrupted", continuous["lastCheckpoint"])
        b, _, _, pb = training._load_checkpoint(self.root / "interrupted", resumed["lastCheckpoint"])
        self.assert_tree_equal(a.state_dict(), b.state_dict())
        self.assert_tree_equal(pa, pb)

    def test_absence_preserves_metadata_and_size_equal_to_batch_matches_legacy(self):
        config = {key: value for key, value in self.config.items() if key != "micro_batch_size"}
        model_config = {**self.model_config, "dropout": 0.0}
        legacy = self.run_train("legacy", config, model_config)
        equal = self.run_train("equal", {**config, "micro_batch_size": 2}, model_config)
        a, _, _, pa = training._load_checkpoint(self.root / "legacy", legacy["lastCheckpoint"])
        b, _, _, pb = training._load_checkpoint(self.root / "equal", equal["lastCheckpoint"])
        for key, value in a.state_dict().items():
            self.torch.testing.assert_close(value, b.state_dict()[key], rtol=2e-5, atol=2e-6)
        self.assertEqual(pa["scheduler"], pb["scheduler"])
        self.assertNotIn("trainingMicrobatch", legacy)
        exported = training.export_best(self.root / "legacy", self.root / "legacy-export")
        self.assertNotIn("trainingMicrobatch", exported["training"])
        self.assertNotIn("micro_batch_size", exported["training"]["config"])

    def test_backward_releases_saved_activation_graph_before_next_microbatch(self):
        # Live saved-for-backward bytes directly measure activation lifetime,
        # independently of allocator caching, RSS or tensor-output size.
        from ia_propia.conversation.model import ConversationModel, assistant_token_loss
        torch = self.torch
        inputs = torch.arange(7, 7 + 4 * 32).reshape(4, 32)
        labels = inputs.clone()
        labels[:, :16] = -100

        class Saved:
            def __init__(self, tensor, counter):
                self.tensor, self.counter = tensor, counter
                self.bytes = tensor.numel() * tensor.element_size()
                counter["live"] += self.bytes
                counter["peak"] = max(counter["peak"], counter["live"])

            def __del__(self):
                self.counter["live"] -= self.bytes

        def measure(micro):
            model = ConversationModel({**self.model_config, "dropout": 0.0})
            parameter_storages = {value.untyped_storage().data_ptr() for value in model.parameters()}
            counter = {"live": 0, "peak": 0}

            def pack(tensor):
                if tensor.untyped_storage().data_ptr() in parameter_storages:
                    return tensor
                return Saved(tensor, counter)

            def unpack(value):
                return value.tensor if isinstance(value, Saved) else value

            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if micro:
                    training._microbatch_backward(model, inputs, labels,
                        {"micro_batch_size": 1, "projection_mode": "supervised"})
                else:
                    assistant_token_loss(model.selected_logits(inputs, labels != -100), labels,
                                         projection_mode="supervised").backward()
            self.assertEqual(counter["live"], 0)
            return counter["peak"]

        full, micro = measure(False), measure(True)
        self.assertGreater(full, 0)
        self.assertLess(micro, full / 2)


if __name__ == "__main__":
    unittest.main()
