"""Native frozen-backbone training and exact resume on disposable fixtures."""

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
HEAD_NAMES = {"copy_query.weight", "copy_key.weight", "copy_gate.weight", "copy_gate.bias"}


class TrainableConfigurationTests(unittest.TestCase):
    def test_optional_policy_preserves_legacy_defaults_and_validates_literals(self):
        self.assertNotIn("trainable_parameters", training.TRAIN_DEFAULTS)
        self.assertNotIn("trainable_parameters", training._configs(None, None)[1])
        self.assertEqual(training._parameter_metadata({}), {})
        for invalid in (None, True, 1, [], {}, "head", "none"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "trainable_parameters"):
                training._configs(None, {"trainable_parameters": invalid})
        for policy in ("all", "copy_head"):
            self.assertEqual(training._configs(None, {"trainable_parameters": policy})[1]["trainable_parameters"], policy)


@unittest.skipUnless(HAS_EXTRAS, "optional conversation dependencies unavailable")
class NativeTrainablePolicyTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        self.previous_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        self.rows = [{"id": f"fixture-{index}", "groupId": f"family-{index}", "split": split,
                      "source": {"dataset": "authored-trainable-policy-tests", "kind": "authored-synthetic"},
                      "messages": [{"role": "user", "content": f"La caja es azul {index}. ¿Qué color tiene?"},
                                   {"role": "assistant", "content": f"Es azul {index}."}]}
                     for index, split in enumerate(("train", "train", "train", "validation", "test"))]
        self.corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")
        self.model_config = {"vocab_size": 300, "d_model": 8, "n_layers": 1, "n_heads": 2,
                             "context_length": 512, "dropout": 0.2, "ffn_multiplier": 1, "seed": 17,
                             "copy_head": {"version": 1, "rank": 32, "seed": 31}}
        self.config = {"batch_size": 2, "max_steps": 4, "max_epochs": 50, "validation_every": 2,
                       "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 29,
                       "warmup_steps": 2, "torch_threads": 1, "projection_mode": "supervised",
                       "copy_supervision": {"alignment_weight": 0.2, "gate_weight": 0.2}}

    def tearDown(self):
        self.temporary.cleanup()
        self.torch.set_num_threads(self.previous_threads)
        self.torch.set_rng_state(self.previous_rng)
        random.setstate(self.previous_python_rng)

    def run_train(self, name, config=None, **kwargs):
        return training.train(self.corpus, self.root / name, model_config=self.model_config,
                              training_config=config or self.config, **kwargs)

    def load(self, name, result):
        return training._load_checkpoint(self.root / name, result["lastCheckpoint"])

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

    def paused_at_two(self, name, config):
        native_checkpoint = training._checkpoint
        stopped = []

        def checkpoint(*args, **kwargs):
            pointer = native_checkpoint(*args, **kwargs)
            if args[5]["step"] == 2 and not stopped:
                stopped.append(True)
                os.kill(os.getpid(), signal.SIGTERM)
            return pointer

        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            result = self.run_train(name, config)
        self.assertEqual(result["step"], 2)
        self.assertEqual(result["reason"], "signal")
        return result

    def test_missing_copy_head_rejected_and_all_mode_restores_trainability(self):
        from ia_propia.conversation.model import ConversationModel
        plain = ConversationModel({key: value for key, value in self.model_config.items() if key != "copy_head"})
        with self.assertRaisesRegex(ValueError, "existing model copy head"):
            training._apply_trainable_parameters(plain, {"trainable_parameters": "copy_head"})
        headed = ConversationModel(self.model_config)
        proof = training._apply_trainable_parameters(headed, {"trainable_parameters": "copy_head"})
        self.assertEqual(set(proof["trainableTensorNames"]), HEAD_NAMES)
        self.assertEqual({name for name, value in headed.named_parameters() if value.requires_grad}, HEAD_NAMES)
        self.assertEqual(proof["trainableParameterCount"], 2 * 8 * 32 + 8 + 1)
        self.assertEqual(proof["frozenParameterCount"] + proof["trainableParameterCount"], headed.parameter_count)
        self.assertEqual(len(proof["frozenBackboneSha256"]), 64)
        self.assertIsNone(training._apply_trainable_parameters(headed, {}))
        self.assertTrue(all(value.requires_grad for value in headed.parameters()))

    def test_native_two_plus_two_exactly_matches_four_for_both_policies(self):
        for policy in ("all", "copy_head"):
            with self.subTest(policy=policy):
                config = {**self.config, "trainable_parameters": policy}
                full = self.run_train(policy + "-continuous", config)
                self.paused_at_two(policy + "-paused", config)
                resumed = training.train(self.corpus, self.root / (policy + "-paused"), resume=True)
                self.assertEqual(resumed["step"], 4)
                a, _, sa, pa = self.load(policy + "-continuous", full)
                b, _, sb, pb = self.load(policy + "-paused", resumed)
                self.assert_tree_equal(a.state_dict(), b.state_dict())
                self.assert_tree_equal(pa, pb)
                for key in ("order", "cursor", "epoch", "python_rng", "best_validation_nll", "parameter_policy"):
                    self.assert_tree_equal(sa[key], sb[key])
                manifest = training.export_best(self.root / (policy + "-paused"), self.root / (policy + "-export"))
                self.assertEqual(manifest["training"]["trainableParameters"], sb["parameter_policy"])
                self.assertEqual(manifest["training"]["config"]["trainable_parameters"], policy)
                self.assertFalse(manifest["training"]["testEvaluated"])
                self.assertTrue(manifest["evaluation"]["experimental"])

    def test_frozen_backbone_remains_byte_exact_with_adamw_weight_decay(self):
        from ia_propia.conversation.model import ConversationModel
        result = self.run_train("frozen", {**self.config, "trainable_parameters": "copy_head", "weight_decay": 1.0})
        model, _, state, payload = self.load("frozen", result)
        initial = ConversationModel(state["model_config"])
        changed = set()
        for name, value in model.state_dict().items():
            self.assertEqual(value.dtype, self.torch.float32)
            if not self.torch.equal(value.view(self.torch.uint8), initial.state_dict()[name].view(self.torch.uint8)):
                changed.add(name)
        self.assertEqual(changed, HEAD_NAMES)
        named = list(model.named_parameters())
        expected_moments = {index for index, (name, _) in enumerate(named) if name in HEAD_NAMES}
        self.assertEqual(set(payload["optimizer"]["state"]), expected_moments)
        self.assertEqual(state["parameter_policy"]["frozenBackboneSha256"],
                         training._frozen_parameters_sha256(initial, set(initial.state_dict()) - HEAD_NAMES))

    def test_legacy_absence_matches_explicit_all_without_new_metadata(self):
        legacy = self.run_train("legacy")
        explicit = self.run_train("all-explicit", {**self.config, "trainable_parameters": "all"})
        a, _, sa, pa = self.load("legacy", legacy)
        b, _, _, pb = self.load("all-explicit", explicit)
        self.assert_tree_equal(a.state_dict(), b.state_dict())
        self.assert_tree_equal(pa, pb)
        self.assertNotIn("parameter_policy", sa)
        self.assertNotIn("trainableParameters", legacy)
        exported = training.export_best(self.root / "legacy", self.root / "legacy-export")
        self.assertNotIn("trainableParameters", exported["training"])
        self.assertNotIn("trainable_parameters", exported["training"]["config"])

    def test_bf16_head_only_resume_keeps_frozen_fp32_master_parameters(self):
        config = {**self.config, "trainable_parameters": "copy_head", "train_precision": "cpu_bfloat16"}
        continuous = self.run_train("bf16-continuous", config)
        self.paused_at_two("bf16-paused", config)
        resumed = training.train(self.corpus, self.root / "bf16-paused", resume=True)
        a, _, sa, pa = self.load("bf16-continuous", continuous)
        b, _, sb, pb = self.load("bf16-paused", resumed)
        self.assert_tree_equal(a.state_dict(), b.state_dict())
        self.assert_tree_equal(pa, pb)
        self.assertEqual(sa["parameter_policy"], sb["parameter_policy"])
        self.assertTrue(all(value.dtype == self.torch.float32 for value in b.state_dict().values()))
        training._validate_parameter_policy(b, sb)

    def test_resume_rejects_policy_change_and_optimizer_state_for_frozen_tensors(self):
        config = {**self.config, "trainable_parameters": "copy_head"}
        paused = self.paused_at_two("reject", config)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            training.train(self.corpus, self.root / "reject", resume=True,
                           model_config=self.model_config,
                           training_config={**config, "trainable_parameters": "all"})
        model, _, state, payload = self.load("reject", paused)
        optimizer = self.torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 0.0003),
                                          weight_decay=config.get("weight_decay", 0.01))
        scheduler = self.torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 2))
        native = copy.deepcopy(payload)
        index = next(iter(native["optimizer"]["state"]))
        frozen_index = next(i for i, value in enumerate(model.parameters()) if not value.requires_grad)
        native["optimizer"]["state"][frozen_index] = copy.deepcopy(native["optimizer"]["state"][index])
        with self.assertRaisesRegex(ValueError, "parameter state mismatch"):
            training._validate_resume_payload(native, optimizer, scheduler, model, state)
        missing = copy.deepcopy(payload)
        del missing["optimizer"]["state"][index]
        with self.assertRaisesRegex(ValueError, "parameter state mismatch"):
            training._validate_resume_payload(missing, optimizer, scheduler, model, state)
        broken = copy.deepcopy(payload)
        broken["optimizer"]["state"][index]["exp_avg"] = self.torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "moment tensor"):
            training._validate_resume_payload(broken, optimizer, scheduler, model, state)

    def test_frozen_proof_detects_tensor_or_metadata_mutation(self):
        result = self.run_train("proof", {**self.config, "trainable_parameters": "copy_head"})
        model, _, state, _ = self.load("proof", result)
        changed = copy.deepcopy(state)
        changed["parameter_policy"]["trainableTensorNames"] = []
        with self.assertRaisesRegex(ValueError, "parameter proof"):
            training._validate_parameter_policy(model, changed)
        with self.torch.no_grad():
            model.token_embedding.weight[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "parameter proof"):
            training._validate_parameter_policy(model, state)

    def test_own_lineage_is_preserved_in_head_only_export(self):
        from ia_propia.conversation.artifacts import save_artifact, load_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        tokenizer = fit_tokenizer([row for row in self.rows if row["split"] == "train"], 300)
        parent = self.root / "parent"
        model = ConversationModel({**self.model_config, "vocab_size": tokenizer.get_vocab_size()})
        original = save_artifact(parent, model, tokenizer,
                                 training={"trainedFrom": "random_weights", "corpusSha256": training._sha(self.corpus)})
        run = self.root / "own-run"
        result = training.train(self.corpus, run, initial_model=parent,
                                model_config=model.config,
                                training_config={**self.config, "trainable_parameters": "copy_head"})
        exported = training.export_best(run, self.root / "own-export")
        self.assertEqual(exported["training"]["trainedFrom"], "own_checkpoint")
        self.assertEqual(exported["training"]["lineage"]["parentModelId"], original["modelId"])
        self.assertEqual(exported["training"]["lineage"]["rootOrigin"], "random_weights")
        self.assertEqual(exported["training"]["trainableParameters"]["frozenBackboneSha256"],
                         result["trainableParameters"]["frozenBackboneSha256"])
        loaded, _, _ = load_artifact(self.root / "own-export")
        self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
