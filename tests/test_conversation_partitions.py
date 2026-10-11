"""Explicit TRAIN/VAL-only runs preserve legacy policy and native resume.

These invented fixtures verify training mechanics, not conversation competence.
No reserved corpus or development-evaluation cases are used.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ia_propia.conversation import jobs, training

POLICY = {"partition_policy": "train_validation_only"}
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


def fixture(index, split, *, text=False):
    record = {"id": f"partition-record-{index}", "groupId": f"partition-group-{index}",
              "split": split, "source": {"dataset": "partition-fixture", "kind": "authored-synthetic"}}
    if text:
        record["text"] = f"El registro ficticio {index} describe una caja y un papel."
    else:
        record["messages"] = [{"role": "user", "content": f"Repite el dato ficticio Veral-{index}."},
                              {"role": "assistant", "content": f"Veral-{index}."}]
    return record


def write_corpus(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


class PartitionConfigurationTests(unittest.TestCase):
    def test_exact_opt_in_absent_from_defaults_and_legacy_metadata(self):
        self.assertNotIn("partition_policy", training.TRAIN_DEFAULTS)
        self.assertNotIn("partition_policy", training._configs(None, None)[1])
        self.assertEqual(training._encoding_kwargs({}), {})
        self.assertEqual(training._loss_metadata({}), {})
        self.assertEqual(training._loss_metadata({"projection_mode": "supervised"}),
                         {"projectionMode": "supervised"})
        config = training._configs(None, POLICY)[1]
        self.assertEqual(config["partition_policy"], "train_validation_only")
        self.assertEqual(training._loss_metadata(POLICY),
                         {"partitionPolicy": "train_validation_only", "testPartitionPresent": False})

    def test_invalid_policy_fails_before_torch_worker_and_any_run_write(self):
        class StringAlias(str):
            pass
        invalid = (None, True, False, 0, 1, 0.0, [], {}, "", "all_splits",
                   "TRAIN_VALIDATION_ONLY", StringAlias("train_validation_only"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(training, "_torch") as importing, patch.object(jobs.subprocess, "Popen") as spawning:
                for value in invalid:
                    with self.subTest(value=value):
                        config = {"partition_policy": value}
                        with self.assertRaisesRegex(ValueError, "partition_policy"):
                            training.train(root / "missing.jsonl", root / "run", training_config=config)
                        with self.assertRaisesRegex(ValueError, "partition_policy"):
                            jobs.start(root / "missing.jsonl", root / "worker", training_config=config)
                importing.assert_not_called()
                spawning.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_canonical_loader_rejects_test_rows_for_both_objectives(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            for text in (False, True):
                with self.subTest(text=text):
                    records = [fixture(0, "train", text=text), fixture(1, "validation", text=text)]
                    config = {**POLICY, **({"objective": "causal_lm"} if text else {})}
                    write_corpus(path, records)
                    self.assertEqual(training._load_training_corpus(path, config), records)
                    write_corpus(path, records + [fixture(2, "test", text=text)])
                    with self.assertRaisesRegex(ValueError, "must not contain TEST"):
                        training._load_training_corpus(path, config)

    def test_cli_request_and_worker_preserve_the_existing_config_contract(self):
        from ia_propia.conversation.__main__ import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, config_path, directory = root / "corpus.jsonl", root / "config.json", root / "run"
            write_corpus(corpus, [fixture(0, "train"), fixture(1, "validation")])
            config_path.write_text(json.dumps(POLICY))
            with patch.object(jobs, "start", return_value={"status": "planned"}) as start:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["start", "--corpus", str(corpus), "--run-dir", str(directory),
                                           "--training-config", str(config_path)]), 0)
                self.assertEqual(start.call_args.kwargs["training_config"], POLICY)
            fake = types.SimpleNamespace(pid=999999901, wait=lambda: 0)
            with patch.object(jobs, "status", side_effect=[{"status": "unplanned", "processAlive": False},
                                                           {"status": "planned", "processAlive": False}]), \
                 patch.object(jobs, "_birth", return_value="fixture-birth"), \
                 patch.object(jobs.subprocess, "Popen", return_value=fake):
                jobs.start(corpus, directory, training_config=POLICY)
            request_path, = directory.glob("request-*.json")
            request = training._json_read(request_path)
            self.assertEqual(request["training_config"], POLICY)
            with patch.object(sys, "argv", ["training", "worker", "--request", str(request_path)]), \
                 patch.object(training, "train") as train:
                training._main()
            self.assertEqual(train.call_args.kwargs, request)


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class PartitionTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        self.previous_torch_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [fixture(index, split) for index, split in enumerate(["train"] * 8 + ["validation"] * 2)]
        self.corpus = self.root / "corpus.jsonl"
        write_corpus(self.corpus, self.rows)
        self.architecture = {"vocab_size": 300, "d_model": 16, "n_layers": 1, "n_heads": 2,
                             "context_length": 64, "dropout": 0.2, "ffn_multiplier": 1, "seed": 37}
        self.config = {**POLICY, "batch_size": 2, "max_steps": 6, "max_epochs": 20, "validation_every": 2,
                       "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 37,
                       "warmup_steps": 2, "torch_threads": 1, "projection_mode": "supervised",
                       "window_policy": "sliding_context"}

    def tearDown(self):
        self.temporary.cleanup()
        self.torch.set_num_threads(self.previous_threads)
        self.torch.set_rng_state(self.previous_torch_rng)
        random.setstate(self.previous_python_rng)

    def train(self, name, config=None, *, stop_after=None):
        config = self.config if config is None else config
        directory = self.root / name
        original = training._log
        def log(run_dir, event):
            result = original(run_dir, event)
            if event.get("event") == "train" and event["step"] == stop_after:
                training._atomic_json(Path(run_dir) / "stop-request.json",
                                      {"pid": os.getpid(), "processBirth": training._birth(os.getpid())})
            return result
        with patch.object(training, "_log", side_effect=log):
            status = training.train(self.corpus, directory, model_config=self.architecture, training_config=config)
        if stop_after is not None:
            self.assertEqual((status["status"], status["step"]), ("stopped", stop_after))
        return directory, status

    def assert_tree_equal(self, first, second, path="payload"):
        if isinstance(first, self.torch.Tensor):
            self.assertIsInstance(second, self.torch.Tensor, path)
            self.assertEqual(first.dtype, second.dtype, path)
            self.assertEqual(first.shape, second.shape, path)
            self.assertTrue(self.torch.equal(first, second), path)
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

    def test_no_test_encoding_preserves_train_val_targets_and_default_rejects(self):
        from ia_propia.conversation.tokenizer import fit_tokenizer, fit_text_tokenizer
        for text in (False, True):
            rows = [fixture(0, "train", text=text), fixture(1, "validation", text=text)]
            fitting = fit_text_tokenizer if text else fit_tokenizer
            tokenizer = fitting(rows[:1], vocab_size=300)
            objective = {"objective": "causal_lm"} if text else {}
            for window in ("contiguous_nonoverlapping", "sliding_context"):
                with self.subTest(text=text, window=window):
                    with self.assertRaisesRegex(ValueError, "Each split"):
                        training._encoded(rows, tokenizer, 16, window, **objective)
                    actual = training._encoded(rows, tokenizer, 16, window, **objective, **POLICY)
                    expected = training._encoded(rows + [fixture(2, "test", text=text)], tokenizer, 16, window, **objective)
                    self.assertEqual(actual["train"], expected["train"])
                    self.assertEqual(actual["validation"], expected["validation"])
                    self.assertEqual(actual["test"], [])
                    for split in ("train", "validation"):
                        self.assertTrue(actual[split])
                        self.assertTrue(all(any(label != -100 for label in labels) for _, labels in actual[split]))
                    for retained in ("train", "validation"):
                        with self.assertRaisesRegex(ValueError, "Each split"):
                            training._encoded([row for row in rows if row["split"] == retained], tokenizer,
                                              16, window, **objective, **POLICY)

    def test_test_rows_rejected_before_bpe_encoding_and_optimizer_updates(self):
        from ia_propia.conversation import tokenizer as tokens
        for text in (False, True):
            with self.subTest(text=text):
                corpus = self.root / ("text.jsonl" if text else "dialogue.jsonl")
                rows = [fixture(0, "train", text=text), fixture(1, "validation", text=text), fixture(2, "test", text=text)]
                write_corpus(corpus, rows)
                config = {**self.config, **({"objective": "causal_lm"} if text else {})}
                with patch.object(tokens, "fit_tokenizer") as fit, patch.object(tokens, "fit_text_tokenizer") as fit_text, \
                     patch.object(training, "_encoded_data") as encoding, patch.object(self.torch.optim.AdamW, "step") as step:
                    with self.assertRaisesRegex(ValueError, "must not contain TEST"):
                        training.train(corpus, self.root / f"rejected-{text}", model_config=self.architecture, training_config=config)
                    fit.assert_not_called()
                    fit_text.assert_not_called()
                    encoding.assert_not_called()
                    step.assert_not_called()
                self.assertFalse((self.root / f"rejected-{text}" / "status.json").exists())
                with patch.object(tokens, "dialogue_tokens") as dialogues, patch.object(tokens, "text_tokens") as documents:
                    with self.assertRaisesRegex(ValueError, "must not contain TEST"):
                        training._encoded_data(rows, None, 16, "sliding_context", **training._encoding_kwargs(config))
                    dialogues.assert_not_called()
                    documents.assert_not_called()

    def test_real_train_is_train_only_bpe_val_only_selection_and_native_resume_exact(self):
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation import tokenizer as tokens
        torch = self.torch
        cases = [("token_mean", None), ("sequence_mean", None), ("token_mean", "cpu_bfloat16")]
        for reduction, precision in cases:
            with self.subTest(reduction=reduction, precision=precision):
                config = {**self.config, "loss_reduction": reduction}
                if precision is not None:
                    config["train_precision"] = precision
                name = reduction + (precision or "fp32")
                encoded, observed = {}, []
                original_encode, original_eval, original_fit = training._encoded, training._evaluate_configured, tokens.fit_tokenizer
                def encode(*args, **kwargs):
                    result = original_encode(*args, **kwargs)
                    encoded.update(result)
                    return result
                def evaluate(model, examples, supplied):
                    self.assertIs(examples, encoded["validation"])
                    observed.append(examples)
                    return original_eval(model, examples, supplied)
                def fit(records, **kwargs):
                    self.assertTrue(records)
                    self.assertTrue(all(record["split"] == "train" for record in records))
                    return original_fit(records, **kwargs)
                with patch.object(training, "_encoded", side_effect=encode), \
                     patch.object(training, "_evaluate_configured", side_effect=evaluate), patch.object(tokens, "fit_tokenizer", side_effect=fit):
                    continuous_dir, continuous = self.train(name + "-continuous", config)
                self.assertTrue(observed)
                self.assertEqual(encoded["test"], [])
                self.assertEqual(continuous["step"], 6)
                paused_dir, paused = self.train(name + "-resumed", config, stop_after=3)
                paused_state = training._load_checkpoint(paused_dir, paused["lastCheckpoint"])[2]
                self.assertEqual(paused_state["last_validation_step"], 2)
                (paused_dir / "stop-request.json").unlink()
                resumed = training.train(self.corpus, paused_dir, resume=True)
                a, ta, sa, pa = training._load_checkpoint(continuous_dir, continuous["lastCheckpoint"])
                b, tb, sb, pb = training._load_checkpoint(paused_dir, resumed["lastCheckpoint"])
                self.assert_tree_equal(a.state_dict(), b.state_dict(), name + ".weights")
                self.assert_tree_equal(pa, pb, name + ".optimizer_rng")
                self.assertEqual(ta.to_str(), tb.to_str())
                for field in ("training_config", "order", "cursor", "epoch", "python_rng", "step",
                              "best_validation_nll", "bad_validations", "last_validation_step"):
                    self.assert_tree_equal(sa[field], sb[field], name + "." + field)
                initial = ConversationModel(a.config)
                self.assertTrue(any(not torch.equal(value, initial.state_dict()[key]) for key, value in a.state_dict().items()))
                self.assertTrue(pa["optimizer"]["state"])
                self.assertTrue(all(float(moment["step"]) == 6 for moment in pa["optimizer"]["state"].values()))
                self.assertTrue(all(torch.isfinite(value).all() for value in a.state_dict().values()))
                events = [json.loads(line) for line in (continuous_dir / "events.jsonl").read_text().splitlines()]
                self.assertTrue(all(self.torch.isfinite(self.torch.tensor(event["nll"]))
                                    for event in events if event["event"] == "train"))

    def test_changed_policy_resume_rejected_before_update_or_publication(self):
        directory, paused = self.train("policy", stop_after=3)
        (directory / "stop-request.json").unlink()
        before = {name: (directory / name).read_bytes() for name in ("status.json", "events.jsonl")}
        legacy = {key: value for key, value in self.config.items() if key != "partition_policy"}
        with patch.object(self.torch.optim.AdamW, "step") as step:
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                training.train(self.corpus, directory, resume=True, training_config=legacy)
            step.assert_not_called()
        self.assertEqual(before, {name: (directory / name).read_bytes() for name in before})
        _, tokenizer, state, _ = training._load_checkpoint(directory, paused["lastCheckpoint"])
        encoded = training._encoded(self.rows, tokenizer, 64, state["window_policy"], **POLICY)
        polluted = copy.deepcopy(encoded)
        polluted["test"] = polluted["train"][:1]
        with self.assertRaisesRegex(ValueError, "partition mismatch"):
            training._validate_resume_state(state, polluted)
        for invalid in (True, None, "three_splits"):
            corrupted = copy.deepcopy(state)
            corrupted["training_config"]["partition_policy"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "partition_policy"):
                training._validate_resume_state(corrupted, encoded)

    def test_export_false_and_finalize_rejection_do_not_read_corpus_or_seal_run(self):
        from ia_propia.conversation.artifacts import load_artifact
        directory, paused = self.train("export", stop_after=3)
        before = {name: (directory / name).read_bytes() for name in ("status.json", "events.jsonl")}
        output = self.root / "artifact"
        manifest = training.export_best(directory, output)
        self.assertFalse(manifest["training"]["testEvaluated"])
        self.assertFalse(manifest["training"]["testPartitionPresent"])
        self.assertEqual(manifest["training"]["partitionPolicy"], "train_validation_only")
        self.assertEqual(manifest["training"]["selectionSplit"], "validation")
        self.assertFalse(manifest["evaluation"]["productionAccepted"])
        loaded, _, portable = load_artifact(output)
        selected, _, _, _ = training._load_checkpoint(directory, paused["bestCheckpoint"])
        self.assertEqual(portable["modelId"], manifest["modelId"])
        self.assert_tree_equal(loaded.state_dict(), selected.state_dict())
        self.assertTrue(all(value.dtype == self.torch.float32 and self.torch.isfinite(value).all()
                            for value in loaded.state_dict().values()))
        with patch.object(training, "_load_training_corpus") as reading, patch.object(training, "_evaluate_configured") as evaluate, \
             patch.object(training, "_baseline") as baseline:
            with self.assertRaisesRegex(ValueError, "no TEST partition"):
                training.finalize(directory, corpus_path=self.root / "forbidden-missing-corpus.jsonl",
                                  export_dir=self.root / "forbidden-export")
            reading.assert_not_called()
            evaluate.assert_not_called()
            baseline.assert_not_called()
        self.assertFalse((directory / "evaluation.json").exists())
        self.assertFalse((self.root / "forbidden-export").exists())
        self.assertEqual(before, {name: (directory / name).read_bytes() for name in before})
        (directory / "stop-request.json").unlink()
        self.assertEqual(training.train(self.corpus, directory, resume=True)["step"], 6)


if __name__ == "__main__":
    unittest.main()
