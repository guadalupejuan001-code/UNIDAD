"""Optional CPU BF16 preserves F32 storage and exact same-runtime resume.

All records are invented unit fixtures. These checks establish technical
contracts, not conversational quality or a cross-mode numerical equivalence.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training

HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class PrecisionConfigurationTests(unittest.TestCase):
    def test_precision_is_explicit_and_absent_from_legacy_defaults_and_metadata(self):
        defaults = training._configs(None, None)[1]
        self.assertNotIn("train_precision", defaults)
        self.assertNotIn("train_precision", training.TRAIN_DEFAULTS)
        self.assertEqual(training._loss_metadata({}), {})
        legacy = training._loss_metadata({"projection_mode": "supervised",
                                          "loss_reduction": "sequence_mean"})
        self.assertEqual(legacy, {"projectionMode": "supervised", "lossReduction": "sequence_mean",
                                  "sequenceUnit": "encoded_window"})
        config = training._configs(None, {"train_precision": "cpu_bfloat16"})[1]
        self.assertEqual(config["train_precision"], "cpu_bfloat16")
        self.assertEqual(training._loss_metadata(config), {"trainPrecision": "cpu_bfloat16"})

    def test_precision_rejects_coercions_aliases_and_other_devices(self):
        for value in (None, True, False, 0, 1, 0.0, "float32", "cpu_float16", "cuda_bfloat16", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cpu_bfloat16"):
                training._configs(None, {"train_precision": value})


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class ConversationPrecisionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        self.previous_torch_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "authored.jsonl"
        records = []
        for index, split in enumerate(["train"] * 8 + ["validation"] * 2 + ["test"]):
            records.append({
                "id": f"precision-record-{index}", "groupId": f"precision-group-{index}", "split": split,
                "source": {"dataset": "precision-left" if index % 2 else "precision-right",
                           "kind": "authored-synthetic"},
                "messages": [{"role": "system", "content": "Repite únicamente el dato recibido."},
                             {"role": "user", "content": f"El dato ficticio es Caliza-{index}. Repítelo."},
                             {"role": "assistant", "content": f"Caliza-{index}."}],
            })
        self.records = records
        self.corpus.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                      for record in records), encoding="utf-8")
        self.model_config = {"vocab_size": 300, "d_model": 16, "n_layers": 1, "n_heads": 2,
                             "context_length": 64, "dropout": 0.2, "ffn_multiplier": 1, "seed": 17}
        self.config = {"batch_size": 2, "max_steps": 6, "max_epochs": 50, "validation_every": 2,
                       "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 17,
                       "warmup_steps": 2, "torch_threads": 1}

    def tearDown(self):
        self.temporary.cleanup()
        self.torch.set_num_threads(self.previous_threads)
        self.torch.set_rng_state(self.previous_torch_rng)
        random.setstate(self.previous_python_rng)

    def model(self, **changed):
        from ia_propia.conversation.model import ConversationModel
        return ConversationModel({**self.model_config, **changed})

    def assert_tree_equal(self, first, second, path="payload"):
        torch = self.torch
        if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
            self.assertIsInstance(first, torch.Tensor, path)
            self.assertIsInstance(second, torch.Tensor, path)
            self.assertEqual(first.dtype, second.dtype, path)
            self.assertEqual(first.shape, second.shape, path)
            self.assertTrue(torch.equal(first, second), path)
        elif isinstance(first, dict):
            self.assertIs(type(second), type(first), path)
            self.assertEqual(set(first), set(second), path)
            for key in first:
                self.assert_tree_equal(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (list, tuple)):
            self.assertIs(type(second), type(first), path)
            self.assertEqual(len(first), len(second), path)
            for index, (left, right) in enumerate(zip(first, second)):
                self.assert_tree_equal(left, right, f"{path}[{index}]")
        else:
            self.assertIs(type(second), type(first), path)
            self.assertEqual(first, second, path)

    def run_train(self, name, config, *, stop_after=None):
        directory = self.root / name
        original_log = training._log
        stopped = []

        def log(run_dir, event):
            result = original_log(run_dir, event)
            if event.get("event") == "train" and event["step"] == stop_after and not stopped:
                # Exercise the public trainer's cooperative stop without a process signal.
                training._atomic_json(Path(run_dir) / "stop-request.json",
                                      {"pid": os.getpid(), "processBirth": training._birth(os.getpid())})
                stopped.append(True)
            return result

        with patch.object(training, "_log", side_effect=log):
            status = training.train(self.corpus, directory, model_config=self.model_config,
                                    training_config=config)
        if stop_after is not None:
            self.assertEqual(status["status"], "stopped")
            self.assertEqual(status["step"], stop_after)
        return directory, status

    def load_last(self, directory, status):
        return training._load_checkpoint(directory, status["lastCheckpoint"])

    def test_bf16_forward_real_dtype_and_f32_loss_gradients_and_adam(self):
        from ia_propia.conversation.model import assistant_token_loss
        torch = self.torch
        inputs = torch.tensor([[1, 4, 7, 8, 9], [1, 4, 10, 11, 12]])
        labels = torch.tensor([[-100, -100, 13, 14, 2], [-100, 15, 16, 17, 2]])
        selected = labels != -100
        for projection in ("full", "supervised"):
            with self.subTest(projection=projection):
                model = self.model(dropout=0.0)
                name = "forward" if projection == "full" else "selected_logits"
                original = getattr(model, name)
                observed = []

                def project(*args):
                    value = original(*args)
                    observed.append((value.dtype, torch.is_autocast_enabled("cpu")))
                    return value

                with patch.object(model, name, side_effect=project):
                    logits = training._bf16_train_forward(model, inputs,
                                                         selected if projection == "supervised" else None)
                self.assertEqual(observed, [(torch.bfloat16, True)])
                self.assertEqual(logits.dtype, torch.float32)
                self.assertFalse(torch.is_autocast_enabled("cpu"))
                loss = assistant_token_loss(logits, labels, loss_reduction="sequence_mean",
                                            projection_mode=projection)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all()
                                    for value in gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)
                optimizer.step()
                training._precision_storage_guard(model, optimizer)
                self.assertTrue(all(value.dtype == torch.float32 for value in model.state_dict().values()))
                self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all()
                                    for moments in optimizer.state.values() for value in moments.values()))

    def test_checkpoint_storage_rejects_nonfinite_adam_and_non_f32_master_weights(self):
        torch = self.torch
        model = self.model(dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.tensor([[1, 4, 7, 8]])
        model(inputs).sum().backward()
        optimizer.step()
        training._precision_storage_guard(model, optimizer)
        moment = next(iter(optimizer.state.values()))["exp_avg"]
        original = moment.clone()
        moment.reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite F32"):
            training._precision_storage_guard(model, optimizer)
        moment.copy_(original)
        model.double()
        with self.assertRaisesRegex(ValueError, "finite F32"):
            training._precision_storage_guard(model, optimizer)

    def test_evaluation_overrides_outer_autocast_and_restores_model_mode(self):
        torch = self.torch
        from torch.nn import functional
        model = self.model()
        examples = [([1, 4, 7, 8], [-100, -100, 9, 2]),
                    ([1, 4, 10, 11, 12], [-100, -100, 13, 14, 2])]
        original_ce = functional.cross_entropy
        for projection in ("full", "supervised"):
            for reduction in ("token_mean", "sequence_mean"):
                with self.subTest(projection=projection, reduction=reduction):
                    reference = training._evaluate(model, examples, 2, reduction, projection)
                    observed = []

                    def ce(logits, *args, **kwargs):
                        observed.append((logits.dtype, torch.is_autocast_enabled("cpu")))
                        return original_ce(logits, *args, **kwargs)

                    with patch.object(functional, "cross_entropy", side_effect=ce):
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            actual = training._evaluate_configured(model, examples,
                                {"batch_size": 2, "train_precision": "cpu_bfloat16",
                                 "projection_mode": projection, "loss_reduction": reduction})
                            self.assertTrue(torch.is_autocast_enabled("cpu"))
                    self.assertTrue(observed)
                    self.assertTrue(all(value == (torch.float32, False) for value in observed))
                    self.assertEqual(actual, reference)
                    self.assertTrue(model.training)

    def test_dropout_resume_exact_for_both_projections_losses_and_adaptive_sampling(self):
        torch = self.torch
        cases = [(projection, reduction, False) for projection in ("full", "supervised")
                 for reduction in ("token_mean", "sequence_mean")]
        cases.append(("full", "sequence_mean", True))
        for projection, reduction, adaptive in cases:
            with self.subTest(projection=projection, reduction=reduction, adaptive=adaptive):
                name = f"{projection}-{reduction}-{'adaptive' if adaptive else 'uniform'}"
                config = {**self.config, "train_precision": "cpu_bfloat16",
                          "projection_mode": projection, "loss_reduction": reduction}
                if adaptive:
                    config.update(adaptive_sampling={"mode": "train_loss_ema", "refresh_every_steps": 2},
                                  sampling_weights=[{"dataset": "precision-left", "factor": 2}])
                continuous_dir, continuous = self.run_train(name + "-continuous", config)
                resumed_dir, paused = self.run_train(name + "-resumed", config, stop_after=3)
                _, _, paused_state, _ = self.load_last(resumed_dir, paused)
                self.assertEqual(paused_state["last_validation_step"], 2)
                if adaptive:
                    self.assertGreater(paused_state["cursor"], 0)  # Stop inside the second adaptive chunk.
                (resumed_dir / "stop-request.json").unlink()
                resumed = training.train(self.corpus, resumed_dir, resume=True)
                self.assertEqual(resumed["step"], 6)
                a, tokenizer_a, state_a, payload_a = self.load_last(continuous_dir, continuous)
                b, tokenizer_b, state_b, payload_b = self.load_last(resumed_dir, resumed)
                self.assert_tree_equal(a.state_dict(), b.state_dict(), name + ".weights")
                self.assert_tree_equal(payload_a, payload_b, name + ".resume_payload")
                self.assertEqual(tokenizer_a.to_str(), tokenizer_b.to_str())
                fields = ["step", "epoch", "cursor", "order", "python_rng", "training_config", "model_config",
                          "best_validation_nll", "bad_validations", "last_validation_step", "train_precision_runtime"]
                if adaptive:
                    fields += ["adaptive_sampling", "sampling", "epoch_cursor"]
                    self.assertEqual(sum(state_a["adaptive_sampling"]["counts"]), 12)
                for field in fields:
                    self.assert_tree_equal(state_a[field], state_b[field], name + "." + field)
                initial = self.model(**a.config)
                self.assertTrue(any(not torch.equal(value, initial.state_dict()[key])
                                    for key, value in a.state_dict().items()))
                self.assertTrue(payload_a["optimizer"]["state"])
                self.assertTrue(all(float(moments["step"]) == 6
                                    for moments in payload_a["optimizer"]["state"].values()))

    def test_precision_pin_is_strict_and_changed_mode_cannot_resume_or_publish(self):
        config = {**self.config, "train_precision": "cpu_bfloat16", "projection_mode": "supervised"}
        directory, paused = self.run_train("pin", config, stop_after=3)
        (directory / "stop-request.json").unlink()
        before = {name: (directory / name).read_bytes() for name in ("status.json", "events.jsonl")}
        original_load = training._load_checkpoint

        def reject(call, phrase):
            with patch.object(self.torch.optim.AdamW, "step") as step:
                with self.assertRaisesRegex(ValueError, phrase):
                    call()
                step.assert_not_called()
            self.assertEqual(before, {name: (directory / name).read_bytes() for name in before})

        reject(lambda: training.train(self.corpus, directory, resume=True,
                                       training_config={key: value for key, value in config.items()
                                                        if key != "train_precision"}), "configuration mismatch")
        corruptions = [lambda value: value.update(torchThreads=True),
                       lambda value: value.update(mkldnnEnabled=int(value["mkldnnEnabled"])),
                       lambda value: value.update(torchVersion=value["torchVersion"] + "-different"),
                       lambda value: value.pop("cpuFlagsSha256"),
                       lambda value: value.update(unrecordedField=1)]
        for index, corrupt in enumerate(corruptions):
            with self.subTest(corruption=index):
                def load(*args):
                    model, tokenizer, state, payload = original_load(*args)
                    corrupt(state["train_precision_runtime"])
                    return model, tokenizer, state, payload

                with patch.object(training, "_load_checkpoint", side_effect=load):
                    reject(lambda: training.train(self.corpus, directory, resume=True), "runtime mismatch")
        _, tokenizer, state, _ = self.load_last(directory, paused)
        examples = training._encoded(self.records, tokenizer, state["model_config"]["context_length"],
                                     state["window_policy"])
        missing = copy.deepcopy(state)
        missing.pop("train_precision_runtime")
        with self.assertRaisesRegex(ValueError, "runtime policy mismatch"):
            training._validate_resume_state(missing, examples)
        extra = copy.deepcopy(state)
        extra["training_config"].pop("train_precision")
        with self.assertRaisesRegex(ValueError, "runtime policy mismatch"):
            training._validate_resume_state(extra, examples)

    def test_default_and_bf16_exports_keep_native_f32_storage_and_inference(self):
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.model import chat
        for mode in (None, "cpu_bfloat16"):
            with self.subTest(mode=mode):
                config = {**self.config, "max_steps": 2}
                if mode is not None:
                    config["train_precision"] = mode
                name = "export-" + (mode or "default")
                directory, status = self.run_train(name, config)
                _, _, state, _ = self.load_last(directory, status)
                self.assertEqual("train_precision_runtime" in state, mode is not None)
                self.assertEqual("trainPrecision" in status, mode is not None)
                manifest = training.export_best(directory, self.root / (name + "-artifact"))
                self.assertEqual("trainPrecision" in manifest["training"], mode is not None)
                self.assertEqual("train_precision" in manifest["training"]["config"], mode is not None)
                self.assertFalse(manifest["training"]["testEvaluated"])
                model, tokenizer, loaded = load_artifact(self.root / (name + "-artifact"))
                self.assertEqual(loaded["modelId"], manifest["modelId"])
                self.assertTrue(all(value.dtype == self.torch.float32 and self.torch.isfinite(value).all()
                                    for value in model.state_dict().values()))
                observations = []
                native_selected_logits = model.selected_logits

                def observe_selected_logits(*args, **kwargs):
                    logits = native_selected_logits(*args, **kwargs)
                    observations.append((logits.dtype, self.torch.is_autocast_enabled("cpu")))
                    return logits

                # Selective decoding bypasses module forward hooks.
                with patch.object(model, "selected_logits", side_effect=observe_selected_logits):
                    result = chat(model, tokenizer, [{"role": "user", "content": "Hola."}],
                                  max_new_tokens=2, temperature=0, seed=42)
                self.assertTrue(result["experimental"])
                self.assertTrue(observations)
                self.assertTrue(all(value == (self.torch.float32, False) for value in observations))


if __name__ == "__main__":
    unittest.main()
