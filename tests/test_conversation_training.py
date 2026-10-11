import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import jobs, training

HAS_EXTRAS = all(importlib.util.find_spec(package) is not None for package in ("torch", "tokenizers", "safetensors"))


class JobStateTests(unittest.TestCase):
    def test_unplanned_and_stale_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(jobs.status(temporary)["status"], "unplanned")
            training._atomic_json(Path(temporary) / "status.json", {"status": "running", "pid": 999999999,
                                                                   "processBirth": "123", "heartbeatAt": 0})
            result = jobs.status(temporary)
            self.assertEqual(result["status"], "stale")
            self.assertEqual(result["recordedStatus"], "running")
            self.assertFalse(result["processAlive"])
            self.assertFalse(jobs.stop(temporary)["stopRequested"])

    def test_pid_birth_mismatch_never_signals(self):
        with tempfile.TemporaryDirectory() as temporary:
            training._atomic_json(Path(temporary) / "status.json", {"status": "running", "pid": os.getpid(),
                                                                   "processBirth": "wrong", "heartbeatAt": 0})
            with patch("ia_propia.conversation.jobs.signal.pidfd_send_signal") as send:
                self.assertFalse(jobs.stop(temporary)["stopRequested"])
                send.assert_not_called()

    def test_configuration_and_single_writer(self):
        for field, value in (("batch_size", True), ("learning_rate", float("nan")), ("max_seconds", 86401),
                             ("loss_reduction", "example_sum"), ("loss_reduction", None)):
            with self.assertRaises(ValueError):
                training._configs(None, {field: value})
        with self.assertRaises(ValueError):
            training._configs({"vocab_size": 261}, None)
        self.assertNotIn("loss_reduction", training._configs(None, None)[1])
        self.assertEqual(training._configs(None, {"loss_reduction": "sequence_mean"})[1]["loss_reduction"],
                         "sequence_mean")
        with tempfile.TemporaryDirectory() as temporary:
            with training._run_lock(temporary):
                with self.assertRaises(RuntimeError):
                    with training._run_lock(temporary):
                        pass


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class ConversationLossTests(unittest.TestCase):
    def test_sequence_mean_equal_response_weight_and_masked_gradients(self):
        import torch
        from torch.nn import functional as functional
        from ia_propia.conversation.model import assistant_token_loss
        logits = torch.zeros((3, 4, 2), dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([[0, -100, -100, -100], [0, 0, 0, 0], [-100, -100, -100, -100]])
        sequence_loss = assistant_token_loss(logits, labels, "sequence_mean")
        self.assertTrue(torch.equal(sequence_loss, functional.cross_entropy(logits[labels != -100], labels[labels != -100])))
        sequence_loss.backward()
        gradients = logits.grad.detach().clone()
        # The short and long response each contribute half the objective.
        self.assertAlmostEqual(float(gradients[0].abs().sum()), float(gradients[1].abs().sum()), places=7)
        self.assertEqual(float(gradients[0, 1:].abs().sum()), 0)
        self.assertEqual(float(gradients[2].abs().sum()), 0)
        logits.grad = None
        assistant_token_loss(logits, labels).backward()
        self.assertAlmostEqual(float(logits.grad[1].abs().sum()), 4 * float(logits.grad[0].abs().sum()), places=6)
        with self.assertRaisesRegex(ValueError, "reduction"):
            assistant_token_loss(logits, labels, "unknown")
        with self.assertRaisesRegex(ValueError, "targets"):
            assistant_token_loss(logits, torch.full_like(labels, -100), "sequence_mean")
        invalid = logits.detach().clone()
        invalid[1, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            assistant_token_loss(invalid, labels, "sequence_mean")

    def test_evaluation_and_baseline_same_aggregation_independent_of_batches(self):
        import math
        import torch
        from torch.nn import functional as functional

        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor([[0.0, 2.0], [2.0, 0.0]])[inputs]

        model = FixedModel()
        examples = [([0], [0]), ([1, 1, 1, 1], [0, 0, 0, 0])]
        small = float(functional.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0])))
        large = float(functional.cross_entropy(torch.tensor([[0.0, 2.0]]), torch.tensor([0])))
        expected_macro, expected_tokens = (large + small) / 2, (large + 4 * small) / 5
        for batch in (1, 2):
            score = training._evaluate(model, examples, batch, "sequence_mean")
            self.assertAlmostEqual(score["nll"], expected_macro, places=6)
            self.assertAlmostEqual(score["sequenceMeanNll"], expected_macro, places=6)
            self.assertAlmostEqual(score["tokenWeightedNll"], expected_tokens, places=6)
            self.assertAlmostEqual(score["perplexity"], math.exp(expected_macro), places=6)
            self.assertEqual(score["targetTokens"], 5)
            self.assertEqual(score["lossReduction"], "sequence_mean")
            self.assertEqual(score["sequenceUnit"], "encoded_window")
            self.assertTrue(model.training)
        legacy = training._evaluate(model, examples, 2)
        self.assertAlmostEqual(legacy["nll"], expected_tokens, places=6)
        self.assertEqual(set(legacy), {"nll", "targetTokens", "perplexity", "examples"})
        baseline_examples = [([0], [0]), ([1, 1, 1, 1], [1, 1, 1, 1])]
        baseline = training._baseline_score(baseline_examples, [large, small], "sequence_mean")
        self.assertAlmostEqual(baseline["nll"], expected_macro, places=6)
        self.assertAlmostEqual(baseline["tokenWeightedNll"], expected_tokens, places=6)
        self.assertEqual(set(training._baseline_score(baseline_examples, [large, small])), {"nll", "targetTokens"})


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class ConversationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        rows = []
        for index, split in enumerate(("train", "train", "validation", "test")):
            rows.append({"id": f"record-{index}", "groupId": f"group-{index}", "split": split,
                         "source": {"dataset": "authored-unit-tests", "kind": "authored-synthetic"},
                         "messages": [{"role": "user", "content": f"Hola {index}. ¿Cómo empiezo?"},
                                      {"role": "assistant", "content": f"Puedes probar una tarea pequeña {index}. Después revisa lo que ocurrió."}]})
        self.corpus.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        self.model_config = {"vocab_size": 300, "d_model": 8, "n_layers": 1, "n_heads": 2,
                             "context_length": 16, "dropout": 0.2, "ffn_multiplier": 1, "seed": 7}
        self.config = {"batch_size": 2, "max_steps": 6, "max_epochs": 50, "validation_every": 2,
                       "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 7,
                       "warmup_steps": 2, "torch_threads": 1}

    def tearDown(self):
        self.temporary.cleanup()

    def run_train(self, name, **kwargs):
        return training.train(self.corpus, self.root / name, model_config=self.model_config,
                              training_config=self.config, **kwargs)

    def test_actual_learning_checkpoints_holdout_and_finalization(self):
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.model import ConversationModel
        observed = []
        original_encoded, original_evaluate = training._encoded, training._evaluate
        captured = {}

        def encoded(*args):
            result = original_encoded(*args)
            captured.update(result)
            return result

        def evaluate(model, examples, batch_size):
            self.assertIsNot(examples, captured["test"])
            observed.append(examples)
            return original_evaluate(model, examples, batch_size)

        with patch.object(training, "_encoded", side_effect=encoded), patch.object(training, "_evaluate", side_effect=evaluate):
            result = self.run_train("run")
        self.assertEqual(result["step"], 6)
        self.assertEqual(result["status"], "completed")
        self.assertGreater(len(observed), 1)
        self.assertTrue(all(examples is captured["validation"] for examples in observed))
        run_dir = self.root / "run"
        model, tokenizer, state, _ = training._load_checkpoint(run_dir, result["lastCheckpoint"])
        initial = ConversationModel(state["model_config"])
        self.assertTrue(any(not torch.equal(value, initial.state_dict()[key]) for key, value in model.state_dict().items()))
        self.assertLessEqual(len(list((run_dir / "checkpoints").iterdir())), 2)
        evaluation = training.finalize(run_dir, export_dir=self.root / "export")
        self.assertFalse(evaluation["testUsedForSelection"])
        self.assertGreater(evaluation["test"]["targetTokens"], 0)
        self.assertTrue(evaluation["test"]["nll"] > 0)
        self.assertTrue(evaluation["unigramBaseline"]["test"]["nll"] > 0)
        exported, _, _ = load_artifact(self.root / "export")
        selected, _, _, _ = training._load_checkpoint(run_dir, result["bestCheckpoint"])
        for key, value in selected.state_dict().items():
            self.assertTrue(torch.equal(value, exported.state_dict()[key]))
        with self.assertRaisesRegex(ValueError, "Finalized"):
            training.train(self.corpus, run_dir, resume=True)

    def test_pause_resume_exact_weights_optimizer_rng_and_cadence(self):
        import torch
        uninterrupted = self.run_train("continuous")
        original_checkpoint = training._checkpoint
        requested = []

        def checkpoint(*args, **kwargs):
            name = original_checkpoint(*args, **kwargs)
            state = args[5]
            if state["step"] == 2 and not requested:
                requested.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return name

        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            paused = self.run_train("resumed")
        self.assertEqual(paused["status"], "stopped")
        self.assertEqual(paused["step"], 2)
        resumed = training.train(self.corpus, self.root / "resumed", resume=True)
        self.assertEqual(resumed["step"], uninterrupted["step"])
        a, _, sa, pa = training._load_checkpoint(self.root / "continuous", uninterrupted["lastCheckpoint"])
        b, _, sb, pb = training._load_checkpoint(self.root / "resumed", resumed["lastCheckpoint"])
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        for index, moments in pa["optimizer"]["state"].items():
            for key, value in moments.items():
                self.assertTrue(torch.equal(value, pb["optimizer"]["state"][index][key]))
        self.assertEqual(pa["scheduler"], pb["scheduler"])
        self.assertTrue(torch.equal(pa["torch_rng"], pb["torch_rng"]))
        self.assertEqual(sa["python_rng"], sb["python_rng"])
        self.assertEqual(sa["best_validation_nll"], sb["best_validation_nll"])
        self.assertEqual(sa["bad_validations"], sb["bad_validations"])
        self.assertEqual(sa["last_validation_step"], sb["last_validation_step"])

    def test_resume_rejects_changed_data_config_and_corrupt_checkpoint(self):
        result = self.run_train("run", budget_seconds=0.001)
        run_dir = self.root / "run"
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, run_dir, resume=True, training_config={"batch_size": 3})
        original = self.corpus.read_bytes()
        self.corpus.write_bytes(original.replace(b"Hola 0", b"Hola cambiado 0"))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            training.train(self.corpus, run_dir, resume=True)
        self.corpus.write_bytes(original)
        checkpoint = run_dir / "checkpoints" / result["lastCheckpoint"] / "weights.safetensors"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"damage")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            training.train(self.corpus, run_dir, resume=True)

    def test_sequence_mean_resume_exact_selection_and_export_metadata(self):
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        config = {**self.config, "window_policy": "sliding_context", "loss_reduction": "sequence_mean"}
        continuous = training.train(self.corpus, self.root / "macro-continuous",
                                    model_config=self.model_config, training_config=config)
        original = training._checkpoint
        stopped = []

        def checkpoint(*args, **kwargs):
            name = original(*args, **kwargs)
            if args[5]["step"] == 2 and not stopped:
                stopped.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return name

        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            paused = training.train(self.corpus, self.root / "macro-paused",
                                    model_config=self.model_config, training_config=config)
        self.assertEqual(paused["step"], 2)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, self.root / "macro-paused", resume=True,
                           training_config={**config, "loss_reduction": "token_mean"})
        resumed = training.train(self.corpus, self.root / "macro-paused", resume=True)
        a, _, sa, pa = training._load_checkpoint(self.root / "macro-continuous", continuous["lastCheckpoint"])
        b, _, sb, pb = training._load_checkpoint(self.root / "macro-paused", resumed["lastCheckpoint"])
        self.assertEqual(sa["training_config"], sb["training_config"])
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        for index, moments in pa["optimizer"]["state"].items():
            for key, value in moments.items():
                self.assertTrue(torch.equal(value, pb["optimizer"]["state"][index][key]))
        self.assertEqual(pa["scheduler"], pb["scheduler"])
        self.assertTrue(torch.equal(pa["torch_rng"], pb["torch_rng"]))
        self.assertEqual(sa["python_rng"], sb["python_rng"])
        events = [json.loads(line) for line in (self.root / "macro-paused" / "events.jsonl").read_text().splitlines()]
        scores = [event["score"] for event in events if event["event"] == "validation"]
        self.assertTrue(all(score["nll"] == score["sequenceMeanNll"] for score in scores))
        self.assertEqual(sb["best_validation_nll"], min(score["nll"] for score in scores))
        self.assertEqual(resumed["lossReduction"], "sequence_mean")
        exported = training.export_best(self.root / "macro-paused", self.root / "macro-export")
        self.assertEqual(exported["training"]["lossReduction"], "sequence_mean")
        self.assertEqual(exported["training"]["sequenceUnit"], "encoded_window")
        self.assertEqual(exported["training"]["validationNll"], sb["best_validation_nll"])
        evaluation = training.finalize(self.root / "macro-paused", export_dir=self.root / "macro-final")
        _, _, manifest = load_artifact(self.root / "macro-final")
        self.assertEqual(manifest["training"]["lossReduction"], "sequence_mean")
        self.assertEqual(evaluation["lossReduction"], "sequence_mean")
        for split in ("train", "validation", "test"):
            self.assertEqual(evaluation[split]["nll"], evaluation[split]["sequenceMeanNll"])
            self.assertEqual(evaluation["unigramBaseline"][split]["nll"],
                             evaluation["unigramBaseline"][split]["sequenceMeanNll"])

    def test_export_best_is_inference_only_and_does_not_seal(self):
        result = self.run_train("snapshot", budget_seconds=0.001)
        run = self.root / "snapshot"
        with patch.object(training, "_evaluate", side_effect=AssertionError("export must not evaluate")):
            manifest = training.export_best(run, self.root / "snapshot-export")
        self.assertFalse(manifest["training"]["testEvaluated"])
        self.assertEqual(manifest["training"]["selectedCheckpoint"], result["bestCheckpoint"])
        self.assertFalse((run / "evaluation.json").exists())
        resumed = training.train(self.corpus, run, resume=True)
        self.assertEqual(resumed["step"], 6)

    def test_holdout_group_leakage_rejected(self):
        rows = [json.loads(line) for line in self.corpus.read_text().splitlines()]
        rows[-1]["groupId"] = rows[0]["groupId"]
        self.corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_train("bad")

    def test_last_reply_only_and_complete_chunk_targets(self):
        from ia_propia.conversation.tokenizer import fit_tokenizer, dialogue_tokens, ASSISTANT_ID
        record = {"split": "train", "messages": [{"role": "user", "content": "Primer encargo."},
                  {"role": "assistant", "content": "Esta respuesta anterior sirve como contexto."},
                  {"role": "user", "content": "Sigue con una frase larga."},
                  {"role": "assistant", "content": "Nueva respuesta final " * 30}]}
        tokenizer = fit_tokenizer([record], vocab_size=300)
        records = [dict(record, split=split) for split in ("train", "validation", "test")]
        chunks = training._encoded(records, tokenizer, 16)
        ids, masks = dialogue_tokens(record["messages"], tokenizer)
        last = max(index for index, token in enumerate(ids) if token == ASSISTANT_ID)
        expected_targets = [token for token, teach in zip(ids[last + 1:], masks[last + 1:]) if teach]
        actual_targets = [label for _, labels in chunks["train"] for label in labels if label != -100]
        self.assertEqual(actual_targets, expected_targets)
        self.assertEqual(actual_targets[-1], 2)
        self.assertGreater(len(chunks["train"]), 1)
        self.assertTrue(all(len(inputs) <= 16 for inputs, _ in chunks["train"]))

    def test_sliding_targets_once_overlap_masked_and_earlier_assistant_ignored(self):
        from ia_propia.conversation.tokenizer import fit_tokenizer, dialogue_tokens, ASSISTANT_ID
        record = {"split": "train", "messages": [{"role": "user", "content": "Primer encargo."},
                  {"role": "assistant", "content": "Respuesta anterior sólo como contexto."},
                  {"role": "user", "content": "Explica una idea sencilla."},
                  {"role": "assistant", "content": "Nueva explicación detallada " * 40}]}
        tokenizer = fit_tokenizer([record], vocab_size=300)
        records = [dict(record, split=split) for split in ("train", "validation", "test")]
        ids, masks = dialogue_tokens(record["messages"], tokenizer)
        final = max(index for index, token in enumerate(ids) if token == ASSISTANT_ID)
        expected = [token for token, teach in zip(ids[final + 1:], masks[final + 1:]) if teach]
        sliding = training._encoded(records, tokenizer, 16, "sliding_context")["train"]
        contiguous = training._encoded(records, tokenizer, 16)["train"]
        for windows in (sliding, contiguous):
            self.assertEqual([label for _, labels in windows for label in labels if label != -100], expected)
        self.assertEqual(expected[-1], 2)
        self.assertTrue(all(len(inputs) == 16 for inputs, _ in sliding))
        self.assertTrue(all(any(label == -100 for label in labels) for _, labels in sliding[1:]))
        positions = []
        for start, end, previous_end in training._window_ranges(len(ids) - 1, 16, "sliding_context"):
            positions.extend(index + 1 for index in range(max(start, previous_end, final), end) if masks[index + 1])
        self.assertEqual(positions, list(range(final + 1, len(ids))))
        self.assertEqual(len(positions), len(set(positions)))
        self.assertEqual(list(training._window_ranges(17, 16, "sliding_context")), [(0, 16, 0), (1, 17, 16)])
        self.assertEqual(list(training._window_ranges(17, 16, "contiguous_nonoverlapping")), [(0, 16, 0), (16, 17, 16)])

    def test_sliding_retains_brief_multiturn_question_at_chunk_boundary(self):
        from ia_propia.conversation.tokenizer import fit_tokenizer, dialogue_tokens, USER_ID, ASSISTANT_ID
        record = {"messages": [{"role": "user", "content": "Primer mensaje."},
                  {"role": "assistant", "content": "Ya vimos una opción."},
                  {"role": "user", "content": "¿Cuál es el siguiente paso?"},
                  {"role": "assistant", "content": "Revisa esa opción con calma y confirma qué ocurrió."}]}
        tokenizer = fit_tokenizer([record], vocab_size=300)
        ids, masks = dialogue_tokens(record["messages"], tokenizer)
        assistant = max(index for index, token in enumerate(ids) if token == ASSISTANT_ID)
        user = max(index for index, token in enumerate(ids[:assistant]) if token == USER_ID)
        # Put the boundary exactly at the assistant marker: legacy loses every user token.
        context = assistant
        self.assertGreater(context, assistant - user)
        records = [dict(record, split=split) for split in ("train", "validation", "test")]
        legacy = training._encoded(records, tokenizer, context)["train"]
        sliding = training._encoded(records, tokenizer, context, "sliding_context")["train"]
        self.assertEqual(legacy[0][0][0], ASSISTANT_ID)
        question = ids[user:assistant]
        self.assertTrue(any(inputs[index:index + len(question)] == question
                            for index in range(len(sliding[0][0]) - len(question) + 1)
                            for inputs in [sliding[0][0]]))
        labels = [label for _, labels in sliding for label in labels if label != -100]
        self.assertEqual(labels, ids[assistant + 1:])

    def test_window_audit_tokenizes_train_only_and_never_forwards_model(self):
        import importlib.util
        from ia_propia.conversation import model as model_module
        artifact, _ = self._parent_artifact()
        script = Path(__file__).resolve().parents[1] / "examples" / "audit_conversation_windows.py"
        specification = importlib.util.spec_from_file_location("audit_conversation_windows", script)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        actual_tokens = module.dialogue_tokens
        seen = []

        def training_only(messages, tokenizer):
            self.assertIn(messages[0]["content"], ("Hola 0. ¿Cómo empiezo?", "Hola 1. ¿Cómo empiezo?"))
            seen.append(messages)
            return actual_tokens(messages, tokenizer)

        with patch.object(module, "dialogue_tokens", side_effect=training_only), patch.object(model_module.ConversationModel, "forward", side_effect=AssertionError("audit must not score model")):
            report = module.audit(self.corpus, artifact)
        self.assertEqual(len(seen), 2)
        self.assertFalse(report["heldoutEvaluated"])
        self.assertEqual(report["modelForwardCalls"], 0)
        self.assertEqual(report["corpusSha256"], training._sha(self.corpus))
        legacy = report["policies"]["contiguous_nonoverlapping"]
        sliding = report["policies"]["sliding_context"]
        self.assertEqual(legacy["targetTokens"], sliding["targetTokens"])
        self.assertTrue(sliding["eachFinalReplyTargetExactlyOnce"])

    def test_sliding_pause_resume_exact_and_legacy_schema_unchanged(self):
        import torch
        from unittest.mock import patch
        config = {**self.config, "window_policy": "sliding_context"}
        uninterrupted = training.train(self.corpus, self.root / "sliding-continuous",
                                       model_config=self.model_config, training_config=config)
        original = training._checkpoint
        paused_once = []

        def checkpoint(*args, **kwargs):
            name = original(*args, **kwargs)
            if args[5]["step"] == 2 and not paused_once:
                paused_once.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return name

        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            paused = training.train(self.corpus, self.root / "sliding-paused", model_config=self.model_config,
                                    training_config=config)
        self.assertEqual(paused["step"], 2)
        resumed = training.train(self.corpus, self.root / "sliding-paused", resume=True)
        a, _, sa, pa = training._load_checkpoint(self.root / "sliding-continuous", uninterrupted["lastCheckpoint"])
        b, _, sb, pb = training._load_checkpoint(self.root / "sliding-paused", resumed["lastCheckpoint"])
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        for index, moments in pa["optimizer"]["state"].items():
            for key, value in moments.items():
                self.assertTrue(torch.equal(value, pb["optimizer"]["state"][index][key]))
        self.assertTrue(torch.equal(pa["torch_rng"], pb["torch_rng"]))
        self.assertEqual(pa["scheduler"], pb["scheduler"])
        self.assertEqual(sa["python_rng"], sb["python_rng"])
        self.assertEqual(sa["window_policy"], "sliding_context")
        self.assertEqual(sa["best_validation_nll"], sb["best_validation_nll"])
        exported = training.export_best(self.root / "sliding-paused", self.root / "sliding-export")
        self.assertEqual(exported["training"]["windowPolicy"], "sliding_context")
        self.assertEqual(exported["training"]["config"]["window_policy"], "sliding_context")
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, self.root / "sliding-paused", resume=True, training_config=self.config)
        legacy = self.run_train("legacy-schema", budget_seconds=0.001)
        _, _, legacy_state, _ = training._load_checkpoint(self.root / "legacy-schema", legacy["lastCheckpoint"])
        self.assertNotIn("window_policy", legacy_state["training_config"])
        self.assertEqual(legacy_state["window_policy"], "contiguous_nonoverlapping")
        training.train(self.corpus, self.root / "legacy-schema", resume=True)

    def _parent_artifact(self):
        self.run_train("parent")
        return self.root / "parent-artifact", training.export_best(self.root / "parent", self.root / "parent-artifact")

    def test_new_stage_preserves_parent_weights_tokenizer_and_resets_optimizer(self):
        import shutil
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        artifact, manifest = self._parent_artifact()
        parent, tokenizer, _ = load_artifact(artifact)
        with patch("ia_propia.conversation.tokenizer.fit_tokenizer", side_effect=AssertionError("warm stage must not refit")):
            result = training.train(self.corpus, self.root / "warm", initial_model=artifact,
                                    training_config=self.config, budget_seconds=0.001,
                                    model_config={"d_model": parent.config["d_model"]})
        model, child_tokenizer, state, payload = training._load_checkpoint(self.root / "warm", result["lastCheckpoint"])
        self.assertEqual(state["step"], 0)
        self.assertEqual(model.config, parent.config)
        self.assertEqual(tokenizer.to_str(), child_tokenizer.to_str())
        self.assertEqual(payload["optimizer"]["state"], {})
        self.assertEqual(payload["scheduler"]["last_epoch"], 0)
        for key, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        lineage = {"parentModelId": manifest["modelId"],
                   "parentWeightsSha256": manifest["files"]["weights.safetensors"]["sha256"],
                   "parentCorpusSha256": manifest["training"]["corpusSha256"], "rootOrigin": "random_weights"}
        self.assertEqual(state["lineage"], lineage)
        self.assertEqual(result["lineage"], lineage)
        self.assertEqual(result["trainedFrom"], "own_checkpoint")
        shutil.rmtree(artifact)  # Resume is self-contained; the parent path is unnecessary.
        learned = training.train(self.corpus, self.root / "warm", resume=True,
                                 model_config={"d_model": parent.config["d_model"]})
        trained, _, trained_state, _ = training._load_checkpoint(self.root / "warm", learned["lastCheckpoint"])
        self.assertTrue(any(not torch.equal(value, trained.state_dict()[key]) for key, value in parent.state_dict().items()))
        self.assertEqual(trained_state["lineage"], lineage)
        exported = training.export_best(self.root / "warm", self.root / "warm-export")
        self.assertEqual(exported["training"]["trainedFrom"], "own_checkpoint")
        self.assertEqual(exported["training"]["lineage"], lineage)
        self.assertTrue(exported["training"]["optimizerReset"])
        self.assertTrue(exported["training"]["schedulerReset"])
        finalized = training.finalize(self.root / "warm", export_dir=self.root / "warm-final")
        _, _, final_manifest = load_artifact(self.root / "warm-final")
        self.assertEqual(finalized["trainedFrom"], "own_checkpoint")
        self.assertEqual(final_manifest["training"]["lineage"], lineage)

    def test_warm_stage_pause_resume_exact(self):
        import torch
        artifact, _ = self._parent_artifact()
        continuous = training.train(self.corpus, self.root / "warm-continuous", initial_model=artifact,
                                    training_config=self.config)
        original = training._checkpoint
        stopped = []

        def checkpoint(*args, **kwargs):
            name = original(*args, **kwargs)
            if args[5]["step"] == 2 and not stopped:
                stopped.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return name

        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            paused = training.train(self.corpus, self.root / "warm-paused", initial_model=artifact,
                                    training_config=self.config)
        self.assertEqual(paused["step"], 2)
        resumed = training.train(self.corpus, self.root / "warm-paused", resume=True)
        a, _, sa, pa = training._load_checkpoint(self.root / "warm-continuous", continuous["lastCheckpoint"])
        b, _, sb, pb = training._load_checkpoint(self.root / "warm-paused", resumed["lastCheckpoint"])
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        for index, moments in pa["optimizer"]["state"].items():
            for key, value in moments.items():
                self.assertTrue(torch.equal(value, pb["optimizer"]["state"][index][key]))
        self.assertTrue(torch.equal(pa["torch_rng"], pb["torch_rng"]))
        self.assertEqual(pa["scheduler"], pb["scheduler"])
        self.assertEqual(sa["python_rng"], sb["python_rng"])
        self.assertEqual(sa["best_validation_nll"], sb["best_validation_nll"])
        self.assertEqual(sa["lineage"], sb["lineage"])

    def test_warm_start_rejects_configuration_external_origin_and_resume_reinitialization(self):
        from ia_propia.conversation.artifacts import load_artifact, save_artifact
        artifact, _ = self._parent_artifact()
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, self.root / "incompatible", initial_model=artifact,
                           model_config={"d_model": 16}, training_config=self.config)
        for operation in (training.train, jobs.start):
            with self.assertRaisesRegex(ValueError, "only allowed for a new stage"):
                operation(self.corpus, self.root / "parent", resume=True, initial_model=artifact)
        model, tokenizer, manifest = load_artifact(artifact)
        save_artifact(self.root / "external", model, tokenizer,
                      training={**manifest["training"], "trainedFrom": "external_pretrained"})
        with self.assertRaisesRegex(ValueError, "own random-weight"):
            training.train(self.corpus, self.root / "external-run", initial_model=self.root / "external", training_config=self.config)
        warm = training.train(self.corpus, self.root / "lineage", initial_model=artifact,
                              training_config=self.config, budget_seconds=0.001)
        directory = self.root / "lineage" / "checkpoints" / warm["lastCheckpoint"]
        state = training._json_read(directory / "state.json")
        state["lineage"]["rootOrigin"] = "external_pretrained"
        training._atomic_json(directory / "state.json", state)
        integrity = training._json_read(directory / "manifest.json")
        integrity["files"]["state.json"] = training._sha(directory / "state.json")
        training._atomic_json(directory / "manifest.json", integrity)
        with self.assertRaisesRegex(ValueError, "lineage"):
            training.train(self.corpus, self.root / "lineage", resume=True)

    def test_periodic_checkpoint_survives_validation_plateau_without_changing_best(self):
        from ia_propia.conversation import model as model_module
        import torch
        actual_loss = model_module.assistant_token_loss
        updates, validations = [], []

        def loss(*args):
            updates.append(1)
            if len(updates) == 3:
                raise ValueError("injected failure after periodic checkpoint")
            return actual_loss(*args)

        def plateau(*args):
            validations.append(1)
            return {"nll": 1.0 if len(validations) == 1 else 2.0,
                    "targetTokens": 1, "perplexity": 2.0, "examples": 1}

        with patch.object(training, "_evaluate", side_effect=plateau), patch.object(model_module, "assistant_token_loss", side_effect=loss):
            with self.assertRaisesRegex(ValueError, "after periodic checkpoint"):
                self.run_train("plateau")
        run = self.root / "plateau"
        status = jobs.status(run)
        self.assertEqual(status["status"], "failed")
        self.assertNotEqual(status["lastCheckpoint"], status["bestCheckpoint"])
        latest, _, latest_state, payload = training._load_checkpoint(run, status["lastCheckpoint"])
        best, _, best_state, _ = training._load_checkpoint(run, status["bestCheckpoint"])
        self.assertEqual(latest_state["step"], 2)
        self.assertEqual(latest_state["last_validation_step"], 2)
        self.assertEqual(latest_state["bad_validations"], 1)
        self.assertEqual(best_state["step"], 0)
        self.assertEqual(latest_state["best_validation_nll"], 1.0)
        self.assertEqual(latest_state["best_checkpoint"], status["bestCheckpoint"])
        self.assertTrue(all(float(moment["step"]) == 2 for moment in payload["optimizer"]["state"].values()))
        self.assertTrue(any(not torch.equal(value, best.state_dict()[key]) for key, value in latest.state_dict().items()))

    def test_failed_step_preserves_prior_complete_checkpoint(self):
        from ia_propia.conversation import model as model_module
        actual_loss = model_module.assistant_token_loss
        calls = []

        def bad_loss(*args):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("injected failed step")
            return actual_loss(*args)

        with patch.object(model_module, "assistant_token_loss", side_effect=bad_loss):
            with self.assertRaisesRegex(ValueError, "injected failed step"):
                self.run_train("failure")
        status = jobs.status(self.root / "failure")
        self.assertEqual(status["status"], "failed")
        _, _, state, payload = training._load_checkpoint(self.root / "failure", status["lastCheckpoint"])
        self.assertEqual(state["step"], 0)
        self.assertEqual(payload["optimizer"]["state"], {})
        result = training.train(self.corpus, self.root / "failure", resume=True)
        self.assertEqual(result["step"], 6)

    def test_detached_planned_stop_and_invalid_worker_status(self):
        import time
        run = self.root / "detached"
        started = jobs.start(self.corpus, run, model_config=self.model_config, training_config=self.config)
        self.assertEqual(started["status"], "planned")
        self.assertTrue(started["processAlive"])
        self.assertTrue(jobs.stop(run)["stopRequested"])
        deadline = time.monotonic() + 10
        value = jobs.status(run)
        while value["status"] in ("planned", "running") and time.monotonic() < deadline:
            time.sleep(0.05)
            value = jobs.status(run)
        self.assertEqual(value["status"], "stopped", (value, (run / "worker.log").read_text()))
        self.assertIsNotNone(value["lastCheckpoint"])
        self.assertEqual((run.stat().st_mode & 0o777), 0o700)
        malformed = self.root / "bad.jsonl"
        malformed.write_text("invalid corpus", encoding="utf-8")
        badrun = self.root / "invalid-job"
        jobs.start(malformed, badrun)
        deadline = time.monotonic() + 10
        value = jobs.status(badrun)
        while value["status"] in ("planned", "running") and time.monotonic() < deadline:
            time.sleep(0.05)
            value = jobs.status(badrun)
        self.assertEqual(value["status"], "failed", value)

    def test_resealed_invalid_checkpoint_state_and_dtype_rejected(self):
        from safetensors.torch import load_file, save_file
        result = self.run_train("state", budget_seconds=0.001)
        run = self.root / "state"
        directory = run / "checkpoints" / result["lastCheckpoint"]
        state_path = directory / "state.json"
        state = training._json_read(state_path)
        state["cursor"] = -1
        training._atomic_json(state_path, state)
        manifest = training._json_read(directory / "manifest.json")
        manifest["files"]["state.json"] = training._sha(state_path)
        training._atomic_json(directory / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "counters"):
            training.train(self.corpus, run, resume=True)
        state["cursor"] = 0
        training._atomic_json(state_path, state)
        weights_path = directory / "weights.safetensors"
        weights = load_file(str(weights_path))
        weights[next(iter(weights))] = weights[next(iter(weights))].double()
        save_file(weights, str(weights_path))
        manifest["files"]["weights.safetensors"] = training._sha(weights_path)
        manifest["files"]["state.json"] = training._sha(state_path)
        training._atomic_json(directory / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "dtypes"):
            training.train(self.corpus, run, resume=True)


if __name__ == "__main__":
    unittest.main()
