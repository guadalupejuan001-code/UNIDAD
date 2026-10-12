"""Portable native-copy contracts using only disposable authored fixtures.

No private artifact, captured legacy source, external model, corpus, owner
message or held-out evaluation case is required. Actual own-B1 byte-exact
verification remains separate private evidence. Tiny scoped test pins exercise
bootstrap validation without changing production identity constants.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training

HEAD = {"version": 1, "rank": 32, "seed": 73}
COEFFICIENTS = {"alignment_weight": 0.2, "gate_weight": 0.2}
NEW_KEYS = {"copy_query.weight", "copy_key.weight", "copy_gate.weight", "copy_gate.bias"}
HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


class CopyConfigurationTests(unittest.TestCase):
    def test_absent_defaults_and_invalid_optins_fail_before_torch_or_run_write(self):
        self.assertNotIn("copy_head", training.MODEL_DEFAULTS)
        self.assertNotIn("copy_supervision", training.TRAIN_DEFAULTS)
        self.assertEqual(training._loss_metadata({}), {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = [
                (None, {"copy_supervision": COEFFICIENTS, "projection_mode": "supervised"}),
                ({"copy_head": None}, None),
                ({"copy_head": {**HEAD, "version": True}}, None),
                ({"copy_head": {**HEAD, "rank": 16}}, None),
                ({"copy_head": {**HEAD, "seed": -1}}, None),
                ({"copy_head": HEAD}, {"copy_supervision": {**COEFFICIENTS, "gate_weight": True}, "projection_mode": "supervised"}),
                ({"copy_head": HEAD}, {"copy_supervision": {**COEFFICIENTS, "gate_weight": float("nan")}, "projection_mode": "supervised"}),
                ({"copy_head": HEAD}, {"copy_supervision": COEFFICIENTS, "projection_mode": "full"}),
                ({"copy_head": HEAD}, {"copy_supervision": COEFFICIENTS, "projection_mode": "supervised", "objective": "causal_lm"}),
            ]
            with patch.object(training, "_torch") as importing:
                for model, config in invalid:
                    with self.subTest(model=model, config=config), self.assertRaises(ValueError):
                        training.train(root / "missing.jsonl", root / "run", model_config=model, training_config=config)
                importing.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
        model, config = training._configs({"copy_head": HEAD}, {
            "copy_supervision": {"alignment_weight": 0, "gate_weight": 0}, "projection_mode": "supervised"})
        self.assertEqual(model["copy_head"], HEAD)
        self.assertEqual(config["copy_supervision"], {"alignment_weight": 0, "gate_weight": 0})


@unittest.skipUnless(HAS_EXTRAS, "optional conversation dependencies unavailable")
class CopyNumericalTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        self.previous_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base_config = {"vocab_size": 262, "d_model": 16, "n_layers": 1, "n_heads": 2,
                            "context_length": 64, "dropout": 0.0, "ffn_multiplier": 1, "seed": 19}

    def tearDown(self):
        self.temporary.cleanup()
        self.torch.set_num_threads(self.previous_threads)
        self.torch.set_rng_state(self.previous_rng)
        random.setstate(self.previous_python_rng)

    def model(self, **overrides):
        from ia_propia.conversation.model import ConversationModel
        return ConversationModel({**self.base_config, "copy_head": HEAD, **overrides})

    def fixture(self):
        from ia_propia.conversation.contract import BOS_ID, USER_ID, ASSISTANT_ID, EOS_ID, IGNORE_INDEX
        ids = self.torch.tensor([[BOS_ID, USER_ID, 7, 8, 7, EOS_ID, ASSISTANT_ID, 7, 8]])
        labels = self.torch.tensor([[IGNORE_INDEX] * 6 + [7, 8, EOS_ID]])
        return ids, labels

    def assert_tree_equal(self, a, b, path="root"):
        if isinstance(a, self.torch.Tensor):
            self.assertIsInstance(b, self.torch.Tensor, path)
            self.assertEqual(a.dtype, b.dtype, path)
            self.assertEqual(a.shape, b.shape, path)
            self.assertTrue(self.torch.equal(a, b), path)
        elif isinstance(a, dict):
            self.assertIs(type(b), type(a), path)
            self.assertEqual(set(a), set(b), path)
            for key in a:
                self.assert_tree_equal(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            self.assertIs(type(b), type(a), path)
            self.assertEqual(len(a), len(b), path)
            for index, (left, right) in enumerate(zip(a, b)):
                self.assert_tree_equal(left, right, f"{path}.{index}")
        else:
            self.assertIs(type(b), type(a), path)
            self.assertEqual(a, b, path)

    def test_legacy_absence_preserves_config_rng_tensor_schema_and_projection(self):
        from ia_propia.conversation.model import ConversationModel, DEFAULT_CONFIG
        from torch.nn import functional
        torch = self.torch
        torch.manual_seed(831)
        before = torch.get_rng_state().clone()
        plain = ConversationModel(self.base_config).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertNotIn("copy_head", plain.config)
        self.assertNotIn("copy_head", DEFAULT_CONFIG)
        self.assertEqual(plain.config, {**DEFAULT_CONFIG, **self.base_config})
        expected = {"token_embedding.weight", "position_embedding.weight", "final_norm.weight", "final_norm.bias"}
        layer_fields = ("attention_norm.weight", "attention_norm.bias", "attention.qkv.weight", "attention.qkv.bias",
                        "attention.projection.weight", "attention.projection.bias", "feedforward_norm.weight",
                        "feedforward_norm.bias", "feedforward.0.weight", "feedforward.0.bias", "feedforward.2.weight",
                        "feedforward.2.bias")
        expected.update(f"blocks.{index}.{field}" for index in range(self.base_config["n_layers"]) for field in layer_fields)
        self.assertEqual(set(plain.state_dict()), expected)
        ids, labels = self.fixture()
        hidden = plain._hidden(ids)
        self.assertTrue(torch.equal(plain(ids), functional.linear(hidden, plain.token_embedding.weight)))
        self.assertTrue(torch.equal(plain.selected_logits(ids, labels != -100),
                                   functional.linear(hidden[labels != -100], plain.token_embedding.weight)))
        headed = self.model()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(set(headed.state_dict()) - set(plain.state_dict()), NEW_KEYS)
        for key, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value.view(torch.uint8), headed.state_dict()[key].view(torch.uint8)), key)
        dimensions = self.base_config["d_model"]
        self.assertEqual(headed.parameter_count - plain.parameter_count, 2 * dimensions * 32 + dimensions + 1)

    def test_head_config_is_detached_and_strict(self):
        from ia_propia.conversation.model import ConversationModel, validate_config
        request = {**self.base_config, "copy_head": dict(HEAD)}
        model = ConversationModel(request)
        request["copy_head"]["seed"] = 99
        exposed = model.config
        exposed["copy_head"]["rank"] = 1
        self.assertEqual(model.config["copy_head"], HEAD)
        for bad in (None, True, {}, {**HEAD, "seed": True}, {**HEAD, "version": 2}, {**HEAD, "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config({**self.base_config, "copy_head": bad})

    def test_user_only_mask_and_future_invariance(self):
        from ia_propia.conversation.contract import BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, EOS_ID
        torch = self.torch
        model = self.model().eval()
        ids = torch.tensor([[BOS_ID, SYSTEM_ID, 20, EOS_ID, USER_ID, 10, 11, EOS_ID, ASSISTANT_ID, 10, USER_ID, 12]])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[0, 8] = True
        first, details = model._copy_project(ids, mask, details=True)
        allowed_positions = details["sourcePositions"][details["allowed"][0]].tolist()
        self.assertEqual(allowed_positions, [5, 6])
        future = ids.clone()
        future[0, 11] = 13
        self.assertTrue(torch.equal(first, model.selected_logits(future, mask)))
        self.assertEqual(int(details["allowed"].sum()), 2)
        self.assertTrue(torch.all((details["sourcePositions"][None, :] <= details["times"][:, None]) | ~details["allowed"]))

    def test_zero_sources_return_raw_lm_and_full_selected_agree(self):
        from ia_propia.conversation.contract import BOS_ID, SYSTEM_ID, ASSISTANT_ID, EOS_ID
        from torch.nn import functional
        torch = self.torch
        model = self.model().eval()
        for ids in (torch.tensor([[BOS_ID, SYSTEM_ID, 7, EOS_ID, ASSISTANT_ID, 8]]),
                    torch.tensor([[7, 8, EOS_ID, ASSISTANT_ID, 10]])):
            expected = functional.linear(model._hidden(ids), model.token_embedding.weight)
            self.assertTrue(torch.equal(model(ids), expected))
            _, details = model._copy_project(ids, details=True)
            self.assertFalse(bool(details["allowed"].any()))
            self.assertEqual(float(details["effectiveGate"].sum()), 0)
        ids, labels = self.fixture()
        mask = labels != -100
        torch.testing.assert_close(model(ids)[mask], model.selected_logits(ids, mask), rtol=1e-5, atol=1e-6)

    def test_single_hidden_auxiliary_reductions_and_eos_negative(self):
        torch = self.torch
        model = self.model()
        ids, labels = self.fixture()
        with patch.object(model, "_hidden", wraps=model._hidden) as hidden:
            _, alignment, gate, stats = model.copy_training_outputs(ids, labels)
            self.assertEqual(hidden.call_count, 1)
        self.assertEqual((stats["eligibleQueries"], stats["matchingPositions"], stats["gateNegativeQueries"]), (2, 3, 1))
        _, details = model._copy_project(ids, labels != -100, details=True)
        gold = labels[labels != -100]
        # Independent scalar formula for this declared three-target fixture:
        # token7 occurs at two USER positions, token8 at one, EOS never copies.
        source_ids = details["sourceIds"][0].tolist()
        masses = []
        for query, token in enumerate([7, 8]):
            masses.append(sum(float(details["attention"][query, index].detach())
                              for index, value in enumerate(source_ids)
                              if value == token and bool(details["allowed"][query, index])))
        self.assertAlmostEqual(float(alignment.detach()), sum(-math.log(value) for value in masses) / 2, places=6)
        expected_gate = sum(float((self.torch.nn.functional.softplus(logit) - target * logit).detach())
                            for logit, target in zip(details["gateLogits"], [1, 1, 0])) / 3
        self.assertAlmostEqual(float(gate.detach()), expected_gate, places=6)
        model.zero_grad(set_to_none=True)
        (alignment + gate).backward()
        self.assertGreater(float(model.copy_query.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.copy_key.weight.grad.abs().sum()), 0)
        self.assertLess(float(model.copy_gate.bias.grad), 0)
        from ia_propia.conversation.contract import BOS_ID, USER_ID, ASSISTANT_ID, EOS_ID
        eos_ids = torch.tensor([[BOS_ID, USER_ID, 7, EOS_ID, ASSISTANT_ID]])
        eos_labels = torch.tensor([[-100, -100, -100, -100, EOS_ID]])
        model.zero_grad(set_to_none=True)
        _, zero, negative, stats = model.copy_training_outputs(eos_ids, eos_labels)
        self.assertEqual(float(zero.detach()), 0)
        self.assertEqual((stats["gatePositiveQueries"], stats["gateNegativeQueries"]), (0, 1))
        negative.backward()
        self.assertGreater(float(model.copy_gate.bias.grad), 0)

    def test_partial_or_falsified_labels_reject_before_hidden(self):
        from ia_propia.conversation.contract import ASSISTANT_ID, EOS_ID, IGNORE_INDEX, USER_ID
        torch = self.torch
        model = self.model()
        ids, labels = self.fixture()
        mutations = []
        for index, replacement in [(8, IGNORE_INDEX), (6, IGNORE_INDEX), (7, IGNORE_INDEX), (6, 9), (6, EOS_ID), (4, 7)]:
            bad = labels.clone()
            bad[0, index] = replacement
            mutations.append((ids, bad))
        no_marker = ids.clone()
        no_marker[0, 6] = 15
        mutations.append((no_marker, labels))
        mutations.append((torch.cat((ids, torch.tensor([[USER_ID]])), 1),
                          torch.cat((labels, torch.tensor([[IGNORE_INDEX]])), 1)))
        mutations.append((ids[:, :0], labels[:, :0]))
        for bad_ids, bad_labels in mutations:
            with self.subTest(shape=bad_ids.shape, labels=bad_labels.tolist()):
                with patch.object(model, "_hidden", wraps=model._hidden) as hidden:
                    with self.assertRaises(ValueError):
                        model.copy_training_outputs(bad_ids, bad_labels)
                    hidden.assert_not_called()
        padded_ids = torch.cat((ids, torch.zeros((1, 2), dtype=torch.long)), 1)
        padded_labels = torch.cat((labels, torch.full((1, 2), IGNORE_INDEX)), 1)
        model.copy_training_outputs(padded_ids, padded_labels)
        # An older complete ASSISTANT reply stays masked even if it copied a
        # valid USER token; supervising it as well must fail explicitly.
        history = torch.tensor([[1, 4, 7, 2, 5, 7, 2, 4, 8, 2, 5, 8]])
        old_labels = torch.tensor([[-100] * 4 + [7] + [-100] * 5 + [8, 2]])
        with self.assertRaises(ValueError):
            model.copy_training_outputs(history, old_labels)

    def test_fp32_and_bf16_forward_gradients_storage_and_eval(self):
        from ia_propia.conversation.model import assistant_token_loss
        torch = self.torch
        for precision in (None, "cpu_bfloat16"):
            with self.subTest(precision=precision):
                model = self.model()
                ids, labels = self.fixture()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
                before = {key: value.clone() for key, value in model.state_dict().items() if key in NEW_KEYS}
                if precision:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        scores, alignment, gate, _ = model.copy_training_outputs(ids, labels)
                    self.assertEqual(scores.dtype, torch.bfloat16)
                    scores = scores.float()
                else:
                    scores, alignment, gate, _ = model.copy_training_outputs(ids, labels)
                    self.assertEqual(scores.dtype, torch.float32)
                self.assertEqual((alignment.dtype, gate.dtype), (torch.float32, torch.float32))
                ce = assistant_token_loss(scores, labels, "sequence_mean", projection_mode="supervised")
                loss = ce + 0.2 * alignment + 0.2 * gate
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                training._precision_storage_guard(model, optimizer)
                self.assertTrue(all(not torch.equal(before[key], model.state_dict()[key]) for key in NEW_KEYS))
                self.assertTrue(all(value.dtype == torch.float32 for value in model.state_dict().values()))
                examples = [(ids[0].tolist(), labels[0].tolist())]
                config = {"batch_size": 1, "projection_mode": "supervised", "loss_reduction": "sequence_mean"}
                if precision:
                    config["train_precision"] = precision
                original = model.selected_logits
                dtypes = []
                def observe(*args):
                    result = original(*args)
                    dtypes.append(result.dtype)
                    return result
                with patch.object(model, "selected_logits", side_effect=observe):
                    if precision:
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            evaluation = training._evaluate_configured(model, examples, config)
                    else:
                        evaluation = training._evaluate_configured(model, examples, config)
                self.assertEqual(dtypes, [torch.float32])
                self.assertTrue(math.isfinite(evaluation["nll"]))
                self.assertTrue(model.training)

    def test_tiny_pinned_own_bootstrap_preserves_weights_and_loads_native_engine(self):
        from ia_propia.conversation.artifacts import load_artifact, save_artifact
        from ia_propia.conversation.runtime import ConversationEngine
        from ia_propia.conversation.tokenizer import fit_tokenizer
        from ia_propia.conversation.model import ConversationModel
        torch = self.torch
        rows = [{"id": "copy-bootstrap-unit", "groupId": "copy-bootstrap-unit-world", "split": "train",
                 "source": {"dataset": "copy-bootstrap-unit", "kind": "authored-synthetic"},
                 "messages": [{"role": "system", "content": "Habla brevemente."},
                              {"role": "user", "content": "Dato ficticio de una prueba."},
                              {"role": "assistant", "content": "Dato ficticio."}]}]
        raw = (json.dumps(rows[0], ensure_ascii=False, sort_keys=True) + "\n").encode()
        tokenizer = fit_tokenizer(rows, vocab_size=self.base_config["vocab_size"])
        base = ConversationModel({**self.base_config, "vocab_size": tokenizer.get_vocab_size()}).eval()
        # Perturb every inherited tensor: regenerating only by constructor seed
        # would no longer pass the exact-copy assertion. No training is claimed.
        with torch.no_grad():
            for index, parameter in enumerate(base.parameters()):
                parameter.flatten()[0].add_((index + 1) / 1024)
        parent = self.root / "tiny-own-parent"
        parent_manifest = save_artifact(parent, base, tokenizer,
            training={"trainedFrom": "random_weights", "corpusSha256": hashlib.sha256(raw).hexdigest(),
                      "fixtureOnly": True, "manualParameterPerturbation": True, "newTrainingSteps": 0})
        # Production pins remain untouched outside this scoped unit fixture.
        # This exercises the pin/clone logic, NOT verification of private B1.
        with patch.multiple(training, _COPY_B1_ID=parent_manifest["modelId"],
                            _COPY_B1_WEIGHTS=parent_manifest["files"]["weights.safetensors"]["sha256"],
                            _COPY_B1_TOKENIZER=parent_manifest["files"]["tokenizer.json"]["sha256"]):
            rng = torch.get_rng_state().clone()
            expanded, inherited_tokenizer, config, lineage = training._initial_own_model(parent, {"copy_head": HEAD})
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(expanded.state_dict()) - set(base.state_dict()), NEW_KEYS)
            for key, value in base.state_dict().items():
                self.assertTrue(torch.equal(value.contiguous().view(torch.uint8),
                                            expanded.state_dict()[key].contiguous().view(torch.uint8)), key)
            self.assertEqual(tokenizer.to_str(), inherited_tokenizer.to_str())
            proof = expanded._copy_bootstrap_proof
            self.assertTrue(proof["inheritedTensorsByteExact"])
            self.assertEqual(set(proof["baseTensorKeys"]), set(base.state_dict()))
            invalid = copy.deepcopy(proof)
            invalid["baseTensorKeys"] = ["fabricated.weight"]
            with self.assertRaises(ValueError):
                training._validate_copy_bootstrap(invalid, config, lineage, expected_base_keys=set(base.state_dict()))
            with self.assertRaises(ValueError):
                training._initial_own_model(parent, {"copy_head": HEAD, "context_length": 32})
        output = self.root / "bootstrapped-native"
        exported = save_artifact(output, expanded, inherited_tokenizer,
            training={"trainedFrom": "own_checkpoint", "lineage": lineage,
                      "corpusSha256": parent_manifest["training"]["corpusSha256"], "copyBootstrap": proof,
                      "fixtureOnly": True, "bootstrapOnly": True, "newTrainingSteps": 0})
        engine = ConversationEngine(output)
        self.assertEqual(engine.model_id, exported["modelId"])
        self.assertEqual(engine.parameter_count, expanded.parameter_count)
        self.assertEqual(engine.manifest["config"]["copy_head"], HEAD)
        generated = engine.reply([{"role": "user", "content": "Prueba ficticia."}],
                                 max_new_tokens=1, temperature=0, seed=42)
        self.assertEqual(generated["modelId"], exported["modelId"])
        self.assertFalse(generated["generalConversationValidated"])

    def test_native_dropout_resume_adam_rng_and_export_for_each_precision(self):
        from ia_propia.conversation.artifacts import load_artifact
        torch = self.torch
        corpus = self.root / "disposable-fixture.jsonl"
        rows = []
        for index, split in enumerate(["train"] * 8 + ["validation"] * 2):
            rows.append({"id": f"copy-unit-{index}", "groupId": f"copy-unit-world-{index}", "split": split,
                         "source": {"dataset": "copy-unit-fixture", "kind": "authored-synthetic"},
                         "messages": [{"role": "system", "content": "Repite el dato."},
                                      {"role": "user", "content": f"Dato ficticio: Ficha-{index}."},
                                      {"role": "assistant", "content": f"Ficha-{index}."}]})
        corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        architecture = {**self.base_config, "vocab_size": 300, "context_length": 128,
                        "dropout": 0.2, "copy_head": HEAD}
        settings = {"batch_size": 2, "max_steps": 6, "max_epochs": 30, "validation_every": 2,
                    "checkpoint_every": 2, "patience": 100, "min_delta": 0, "seed": 19,
                    "warmup_steps": 2, "torch_threads": 1, "learning_rate": 0.002,
                    "projection_mode": "supervised", "loss_reduction": "sequence_mean",
                    "partition_policy": "train_validation_only", "copy_supervision": COEFFICIENTS}
        original_log = training._log
        for precision in (None, "cpu_bfloat16"):
            with self.subTest(precision=precision):
                config = {**settings, **({"train_precision": precision} if precision else {})}
                whole = self.root / f"whole-{precision}"
                resumed_dir = self.root / f"resumed-{precision}"
                complete = training.train(corpus, whole, model_config=architecture, training_config=config)
                def log(run_dir, event):
                    result = original_log(run_dir, event)
                    if event.get("event") == "train" and event["step"] == 3:
                        training._atomic_json(Path(run_dir) / "stop-request.json",
                            {"pid": os.getpid(), "processBirth": training._birth(os.getpid())})
                    return result
                with patch.object(training, "_log", side_effect=log):
                    stopped = training.train(corpus, resumed_dir, model_config=architecture, training_config=config)
                self.assertEqual((stopped["status"], stopped["step"]), ("stopped", 3))
                (resumed_dir / "stop-request.json").unlink()
                before = {name: (resumed_dir / name).read_bytes() for name in ("status.json", "events.jsonl")}
                for changed_model, changed_config in [
                    ({"copy_head": {**HEAD, "seed": 74}}, None),
                    (architecture, {**config, "copy_supervision": {**COEFFICIENTS, "gate_weight": 0.3}}),
                ]:
                    with patch.object(torch.optim.AdamW, "step") as updating, self.assertRaisesRegex(ValueError, "configuration mismatch"):
                        training.train(corpus, resumed_dir, resume=True,
                                       model_config=changed_model, training_config=changed_config)
                    updating.assert_not_called()
                    self.assertEqual(before, {name: (resumed_dir / name).read_bytes() for name in before})
                resumed = training.train(corpus, resumed_dir, resume=True)
                self.assertEqual((complete["step"], resumed["step"]), (6, 6))
                a, ta, sa, pa = training._load_checkpoint(whole, complete["lastCheckpoint"])
                b, tb, sb, pb = training._load_checkpoint(resumed_dir, resumed["lastCheckpoint"])
                self.assert_tree_equal(a.state_dict(), b.state_dict())
                self.assert_tree_equal(pa, pb)
                self.assertEqual(ta.to_str(), tb.to_str())
                fields = ["step", "epoch", "cursor", "order", "python_rng", "training_config", "model_config",
                          "best_validation_nll", "bad_validations", "last_validation_step"]
                if precision:
                    fields.append("train_precision_runtime")
                for field in fields:
                    self.assert_tree_equal(sa[field], sb[field], field)
                self.assertTrue(all(float(moment["step"]) == 6 for moment in pa["optimizer"]["state"].values()))
                initial = self.model(**a.config)
                self.assertTrue(any(not torch.equal(initial.state_dict()[key], a.state_dict()[key]) for key in NEW_KEYS))
                output = self.root / f"selected-{precision}"
                exported = training.export_best(whole, output)
                loaded, _, manifest = load_artifact(output)
                self.assertEqual(manifest["modelId"], exported["modelId"])
                self.assertEqual(loaded.config["copy_head"], HEAD)
                self.assertFalse(manifest["training"]["testEvaluated"])
                self.assertEqual(manifest["training"]["copySupervision"], COEFFICIENTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
