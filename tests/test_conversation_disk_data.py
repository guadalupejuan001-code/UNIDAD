"""Disk loading must preserve supervision, heldout boundaries and exact resume."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation import training
from ia_propia.conversation.data import CorpusError, load_corpus
from ia_propia.conversation.language_data import load_text_corpus


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "tokenizers", "safetensors"))


def rows_for(objective):
    rows = []
    for index, split in enumerate(("train", "train", "validation", "test")):
        row = {"id": f"record-{index}", "groupId": f"family-{index}", "split": split,
               "source": {"dataset": "disk-loading-fixture", "kind": "authored-synthetic",
                          "note": "Metadata is not a training target."}}
        text = (f"El documento ficticio {index} conserva café, pingüinos y palabras distintas.\n"
                + "Una explicación clara conecta los hechos disponibles. " * 4)
        if objective == "causal_lm":
            row["text"] = text
        else:
            row["messages"] = [
                {"role": "user", "content": f"La taza de Inés tiene el número {index}."},
                {"role": "assistant", "content": "La taza es de Inés."},
                {"role": "user", "content": "Aclara el dato actual con calma."},
                {"role": "assistant", "content": text},
            ]
        rows.append(row)
    return rows


class DiskFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def corpus(self, objective, rows=None, name="corpus.jsonl", **config):
        from ia_propia.conversation.disk_data import StreamingCorpus
        path = self.write(name, rows if rows is not None else rows_for(objective))
        return StreamingCorpus(path, {"objective": objective, **config})


class StreamingCorpusTests(DiskFixture):
    def test_reiteration_counts_and_sha_preserve_canonical_bytes(self):
        for objective, loader in (("causal_lm", load_text_corpus),
                                  ("assistant_last_reply", load_corpus)):
            with self.subTest(objective=objective):
                rows = rows_for(objective)
                corpus = self.corpus(objective, rows)
                self.assertEqual(corpus.counts, {"train": 2, "validation": 1, "test": 1})
                self.assertEqual(corpus.sha256, hashlib.sha256(corpus.path.read_bytes()).hexdigest())
                expected = loader(corpus.path)
                self.assertEqual(list(corpus), expected)
                self.assertEqual(list(corpus), expected)

    def test_streaming_does_not_read_entire_corpus_into_bytes(self):
        from ia_propia.conversation.disk_data import StreamingCorpus
        path = self.write("corpus.jsonl", rows_for("causal_lm"))
        with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file bytes allocation")):
            corpus = StreamingCorpus(path, {"objective": "causal_lm"})
            self.assertEqual(len(list(corpus)), 4)

    def test_oversized_source_is_rejected_before_hashing(self):
        from ia_propia.conversation import data, disk_data
        path = self.write("too-large.jsonl", rows_for("causal_lm"))
        self.assertGreater(path.stat().st_size, 512)
        with patch.object(data, "MAX_FILE_BYTES", 512), patch.object(
                disk_data, "_sha256", side_effect=AssertionError("oversized corpus hashed")):
            with self.assertRaises(CorpusError):
                disk_data.StreamingCorpus(path, {"objective": "causal_lm"})

    def test_global_ids_and_families_are_checked_before_encoding(self):
        from ia_propia.conversation.disk_data import StreamingCorpus
        for objective in ("causal_lm", "assistant_last_reply"):
            original = rows_for(objective)
            duplicate_id = copy.deepcopy(original)
            duplicate_id[-1]["id"] = duplicate_id[0]["id"]
            leaking_family = copy.deepcopy(original)
            leaking_family[-1]["groupId"] = leaking_family[0]["groupId"]
            for rows in (duplicate_id, leaking_family):
                with self.subTest(objective=objective, rows=rows), self.assertRaises(CorpusError):
                    StreamingCorpus(self.write("invalid.jsonl", rows), {"objective": objective})

    def test_text_duplicates_in_same_or_different_split_are_rejected(self):
        for destination in (1, 3):
            rows = rows_for("causal_lm")
            rows[destination]["text"] = rows[0]["text"]
            with self.subTest(destination=destination), self.assertRaises(CorpusError):
                self.corpus("causal_lm", rows)

    def test_json_utf8_and_record_schema_keep_existing_guards(self):
        from ia_propia.conversation.disk_data import StreamingCorpus
        for raw in (b'\n', b'{"id":"a","id":"b"}\n', b'{"value":NaN}\n',
                    b'{"value":"\xff"}\n', b'{"value":"\\ud800"}\n'):
            path = self.root / "invalid.jsonl"
            path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(CorpusError):
                StreamingCorpus(path, {"objective": "causal_lm"})
        for objective in ("causal_lm", "assistant_last_reply"):
            rows = rows_for(objective)
            rows[0]["unexpected"] = "not canonical"
            with self.subTest(objective=objective), self.assertRaises(CorpusError):
                self.corpus(objective, rows)

    def test_explicit_train_validation_policy_rejects_test_before_preparation(self):
        with self.assertRaisesRegex(ValueError, "TEST"):
            self.corpus("causal_lm", partition_policy="train_validation_only")
        corpus = self.corpus("causal_lm", rows_for("causal_lm")[:-1],
                             partition_policy="train_validation_only")
        self.assertEqual(corpus.counts, {"train": 2, "validation": 1, "test": 0})


class DiskConfigurationTests(unittest.TestCase):
    def test_default_configuration_keeps_legacy_fields_and_rejects_unsupported_combinations(self):
        self.assertNotIn("data_loading", training._configs(None, None)[1])
        self.assertEqual(training._configs(None, {"data_loading": "mmap"})[1]["data_loading"], "mmap")
        for value in (None, True, "memory", "streaming", 1, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training._configs(None, {"data_loading": value})
        with self.assertRaises(ValueError):
            training._configs(None, {"data_loading": "mmap", "sampling_weights": [
                {"dataset": "disk-loading-fixture", "factor": 2}]})
        with self.assertRaises(ValueError):
            training._configs(None, {"data_loading": "mmap", "adaptive_sampling": {
                "mode": "train_loss_ema", "ema_decay": .9, "exponent": .5, "coverage": .25,
                "min_priority": .5, "max_priority": 2, "refresh_every_steps": 2}})


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class DiskWindowsTests(DiskFixture):
    def fit(self, objective, rows, vocabulary=300):
        from ia_propia.conversation.tokenizer import fit_text_tokenizer, fit_tokenizer
        fit = fit_text_tokenizer if objective == "causal_lm" else fit_tokenizer
        return fit((row for row in rows if row["split"] == "train"), vocabulary)

    def build(self, corpus, tokenizer, cache="cache", context=32, policy="sliding_context", **kwargs):
        from ia_propia.conversation.disk_data import build_or_load_windows
        return build_or_load_windows(self.root / cache, corpus, tokenizer, context, policy,
                                     corpus_sha256=corpus.sha256, **kwargs)

    def close(self, windows):
        from ia_propia.conversation.disk_data import close_windows
        close_windows(windows)

    def test_exact_windows_and_once_only_targets_for_both_objectives_and_policies(self):
        for objective in ("causal_lm", "assistant_last_reply"):
            rows = rows_for(objective)
            corpus = self.corpus(objective, rows)
            tokenizer = self.fit(objective, rows)
            for context in (8, 32, 512):
                for policy in ("contiguous_nonoverlapping", "sliding_context"):
                    with self.subTest(objective=objective, context=context, policy=policy):
                        expected = training._encoded(rows, tokenizer, context, policy, objective=objective)
                        actual = self.build(corpus, tokenizer, f"cache-{objective}-{context}-{policy}",
                                            context, policy, objective=objective, include_test=True)
                        try:
                            self.assertEqual(set(actual), set(expected))
                            for split in expected:
                                self.assertEqual(list(actual[split]), expected[split])
                                self.assertEqual(actual[split][-1], expected[split][-1])
                                self.assertEqual(actual[split][1:-1:2], expected[split][1:-1:2])
                                with self.assertRaises(IndexError):
                                    actual[split][len(actual[split])]
                                targets = [label for _, labels in actual[split] for label in labels if label != -100]
                                self.assertEqual(targets.count(2), sum(row["split"] == split for row in rows))
                                self.assertNotIn(0, targets)
                                self.assertNotIn(1, targets)
                        finally:
                            self.close(actual)

    def test_training_cache_never_tokenizes_reserved_test_text_or_dialogues(self):
        from ia_propia.conversation import tokenizer as encoding
        for objective in ("causal_lm", "assistant_last_reply"):
            rows = rows_for(objective)
            corpus = self.corpus(objective, rows)
            tokenizer = self.fit(objective, rows)
            observed = []
            function = "text_tokens" if objective == "causal_lm" else "dialogue_tokens"
            original = getattr(encoding, function)

            def encode(value, tokenizer):
                observed.append(value)
                return original(value, tokenizer)

            with patch.object(encoding, function, side_effect=encode):
                actual = self.build(corpus, tokenizer, f"training-{objective}", objective=objective)
            try:
                self.assertEqual(len(actual["test"]), 0)
                self.assertEqual(len(observed), 3)
                reserved = rows[-1]["text"] if objective == "causal_lm" else rows[-1]["messages"]
                self.assertNotIn(reserved, observed)
                expected = training._encoded(rows, tokenizer, 32, "sliding_context", objective=objective)
                for split in ("train", "validation"):
                    self.assertEqual(list(actual[split]), expected[split])
            finally:
                self.close(actual)

    def test_cache_reopen_does_not_retokenize_and_missing_cache_rebuilds(self):
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        actual = self.build(corpus, tokenizer, objective="causal_lm")
        expected = {split: list(values) for split, values in actual.items()}
        self.close(actual)
        with patch.object(training, "_record_windows", side_effect=AssertionError("existing cache retokenized")):
            reopened = self.build(corpus, tokenizer, objective="causal_lm")
        try:
            self.assertEqual({split: list(values) for split, values in reopened.items()}, expected)
        finally:
            self.close(reopened)
        shutil.rmtree(self.root / "cache")
        rebuilt = self.build(corpus, tokenizer, objective="causal_lm")
        try:
            self.assertEqual({split: list(values) for split, values in rebuilt.items()}, expected)
        finally:
            self.close(rebuilt)

    def test_cache_payload_and_index_corruption_are_rejected_without_overwrite(self):
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        for suffix in ("windows.bin", "index.bin"):
            name = "cache-" + suffix
            actual = self.build(corpus, tokenizer, name, objective="causal_lm")
            self.close(actual)
            path = self.root / name / ("train." + suffix)
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 1
            path.write_bytes(damaged)
            with self.subTest(file=suffix), self.assertRaises(ValueError):
                self.build(corpus, tokenizer, name, objective="causal_lm")
            self.assertEqual(path.read_bytes(), damaged)

    def test_cache_identity_binds_tokenizer_corpus_policy_context_and_test_inclusion(self):
        from ia_propia.conversation.disk_data import build_or_load_windows
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        alternative = self.fit("causal_lm", rows_for("causal_lm"), 320)
        self.assertNotEqual(tokenizer.to_str(), alternative.to_str())
        actual = self.build(corpus, tokenizer, objective="causal_lm")
        self.close(actual)
        common = {"objective": "causal_lm", "corpus_sha256": corpus.sha256}
        changes = [dict(tokenizer=alternative), dict(context_length=16),
                   dict(window_policy="contiguous_nonoverlapping"),
                   dict(corpus_sha256="f" * 64), dict(include_test=True),
                   dict(objective="assistant_last_reply")]
        for change in changes:
            arguments = {"cache_dir": self.root / "cache", "records": corpus, "tokenizer": tokenizer,
                         "context_length": 32, "window_policy": "sliding_context", **common, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_or_load_windows(**arguments)

    def test_partition_gate_precedes_tokenization_and_missing_validation_is_rejected(self):
        from ia_propia.conversation.disk_data import build_or_load_windows
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        with patch.object(training, "_record_windows", side_effect=AssertionError("heldout corpus was encoded")):
            with self.assertRaises(ValueError):
                build_or_load_windows(self.root / "rejected", corpus, tokenizer, 32, "sliding_context",
                                      objective="causal_lm", partition_policy="train_validation_only",
                                      corpus_sha256=corpus.sha256)
        for missing, rows in (("validation", rows_for("causal_lm")[:2]),
                              ("test", rows_for("causal_lm")[:-1])):
            partial = self.corpus("causal_lm", rows, name="without-" + missing + ".jsonl")
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.build(partial, tokenizer, "missing-" + missing, objective="causal_lm")

    def test_source_mutation_after_validation_and_during_encoding_is_rejected_atomically(self):
        from ia_propia.conversation.disk_data import build_or_load_windows
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        original_bytes = corpus.path.read_bytes()
        corpus.path.write_bytes(original_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            list(corpus)
        with patch.object(training, "_record_windows", side_effect=AssertionError("mutated input encoded")):
            with self.assertRaisesRegex(ValueError, "changed"):
                build_or_load_windows(self.root / "changed", corpus, tokenizer, 32, "sliding_context",
                                      objective="causal_lm", corpus_sha256=corpus.sha256)
        self.assertFalse((self.root / "changed").exists())
        corpus.path.write_bytes(original_bytes)
        original = training._record_windows
        mutated = []

        def encode(*args, **kwargs):
            for window in original(*args, **kwargs):
                yield window
            if not mutated:
                mutated.append(True)
                corpus.path.write_bytes(original_bytes + b"\n")

        with patch.object(training, "_record_windows", side_effect=encode):
            with self.assertRaises(ValueError):
                self.build(corpus, tokenizer, "interrupted", objective="causal_lm")
        self.assertFalse((self.root / "interrupted").exists())
        self.assertEqual(list(self.root.glob(".interrupted-*")), [])

    def test_cache_index_bounds_and_targets_are_checked_beyond_file_hashes(self):
        import struct
        corpus = self.corpus("causal_lm")
        tokenizer = self.fit("causal_lm", rows_for("causal_lm"))
        for alteration in ("index-bounds", "no-target"):
            directory = self.root / alteration
            actual = self.build(corpus, tokenizer, alteration, objective="causal_lm")
            self.close(actual)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if alteration == "index-bounds":
                path = directory / "train.index.bin"
                damaged = bytearray(path.read_bytes())
                struct.pack_into("<Q", damaged, 0, 4)  # first window must start at byte zero
                field = "index"
            else:
                path = directory / "train.windows.bin"
                damaged = bytearray(path.read_bytes())
                _, length = struct.unpack_from("<QQ", (directory / "train.index.bin").read_bytes())
                struct.pack_into(f"<{length}i", damaged, length * 4, *([-100] * length))
                field = "data"
            path.write_bytes(damaged)
            # A matching digest does not make a structurally impossible window
            # valid: cache checks must cover the index and target contract too.
            manifest["splits"]["train"][field]["sha256"] = hashlib.sha256(damaged).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.build(corpus, tokenizer, alteration, objective="causal_lm")


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class DiskTrainingIntegrationTests(DiskFixture):
    MODEL = {"vocab_size": 300, "d_model": 8, "n_layers": 1, "n_heads": 2,
             "ffn_multiplier": 1, "context_length": 32, "dropout": .2, "seed": 9}
    CONFIG = {"batch_size": 2, "max_steps": 4, "max_epochs": 50, "validation_every": 2,
              "checkpoint_every": 1, "patience": 100, "min_delta": 0, "seed": 7,
              "warmup_steps": 2, "torch_threads": 1, "projection_mode": "supervised",
              "window_policy": "sliding_context"}

    def setUp(self):
        import torch
        super().setUp()
        self.previous_threads = torch.get_num_threads()
        self.previous_torch_rng = torch.get_rng_state().clone()
        self.previous_python_rng = random.getstate()

    def tearDown(self):
        import torch
        super().tearDown()
        torch.set_num_threads(self.previous_threads)
        torch.set_rng_state(self.previous_torch_rng)
        random.setstate(self.previous_python_rng)

    def assert_nested_equal(self, first, second, name="checkpoint"):
        import torch
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second), name)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second), name)
            for key in first:
                self.assert_nested_equal(first[key], second[key], f"{name}.{key}")
        elif isinstance(first, (tuple, list)):
            self.assertEqual(type(first), type(second), name)
            self.assertEqual(len(first), len(second), name)
            for index, value in enumerate(first):
                self.assert_nested_equal(value, second[index], f"{name}[{index}]")
        else:
            self.assertEqual(first, second, name)

    def compare_checkpoints(self, first_dir, first_result, second_dir, second_result):
        first, first_tokenizer, first_state, first_payload = training._load_checkpoint(
            first_dir, first_result["lastCheckpoint"])
        second, second_tokenizer, second_state, second_payload = training._load_checkpoint(
            second_dir, second_result["lastCheckpoint"])
        self.assert_nested_equal(first.state_dict(), second.state_dict(), "weights")
        self.assertEqual(first_tokenizer.to_str(), second_tokenizer.to_str())
        self.assert_nested_equal(first_payload, second_payload)
        for key in ("python_rng", "order", "cursor", "epoch", "step", "best_validation_nll",
                    "last_validation_step", "bad_validations", "assistant_targets", "window_policy"):
            self.assert_nested_equal(first_state[key], second_state[key], key)
        return first_state, second_state

    def test_memory_and_mmap_training_produce_exact_weights_adam_rng_and_validation(self):
        for objective in ("causal_lm", "assistant_last_reply"):
            with self.subTest(objective=objective):
                corpus = self.write(f"{objective}.jsonl", rows_for(objective))
                config = {**self.CONFIG, "objective": objective}
                memory_dir, disk_dir = self.root / (objective + "-memory"), self.root / (objective + "-mmap")
                memory = training.train(corpus, memory_dir, model_config=self.MODEL, training_config=config)
                disk = training.train(corpus, disk_dir, model_config=self.MODEL,
                                      training_config={**config, "data_loading": "mmap"})
                memory_state, disk_state = self.compare_checkpoints(memory_dir, memory, disk_dir, disk)
                self.assertNotIn("data_loading", memory_state["training_config"])
                self.assertEqual(disk_state["training_config"], {**memory_state["training_config"], "data_loading": "mmap"})

    def test_mmap_resume_rebuilds_missing_cache_and_restores_exact_training_state(self):
        for objective in ("causal_lm", "assistant_last_reply"):
            with self.subTest(objective=objective):
                corpus = self.write(f"{objective}.jsonl", rows_for(objective))
                config = {**self.CONFIG, "objective": objective, "data_loading": "mmap"}
                continuous_dir, paused_dir = self.root / (objective + "-continuous"), self.root / (objective + "-paused")
                continuous = training.train(corpus, continuous_dir, model_config=self.MODEL, training_config=config)
                original = training._checkpoint
                interrupted = []

                def checkpoint(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if args[5]["step"] == 2 and not interrupted:
                        interrupted.append(True)
                        os.kill(os.getpid(), signal.SIGTERM)
                    return result

                with patch.object(training, "_checkpoint", side_effect=checkpoint):
                    paused = training.train(corpus, paused_dir, model_config=self.MODEL, training_config=config)
                self.assertEqual(paused["step"], 2)
                # Cache files are derived data, unlike weights/Adam/RNG. A deleted
                # cache must be reconstructed from the checkpoint's fixed inputs.
                cached_manifest = [path for path in paused_dir.rglob("manifest.json")
                                   if "checkpoints" not in path.parts]
                self.assertEqual(len(cached_manifest), 1)
                shutil.rmtree(cached_manifest[0].parent)
                resumed = training.train(corpus, paused_dir, resume=True)
                self.compare_checkpoints(continuous_dir, continuous, paused_dir, resumed)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    training.train(corpus, paused_dir, resume=True,
                                   training_config={key: value for key, value in config.items() if key != "data_loading"})

    def test_training_does_not_tokenize_test_and_finalize_does_so_explicitly(self):
        from ia_propia.conversation import tokenizer as encoding
        corpus = self.write("text.jsonl", rows_for("causal_lm"))
        observed = []
        original = encoding.text_tokens

        def encode(text, tokenizer):
            observed.append(text)
            return original(text, tokenizer)

        config = {**self.CONFIG, "objective": "causal_lm", "data_loading": "mmap", "max_steps": 1}
        with patch.object(encoding, "text_tokens", side_effect=encode):
            training.train(corpus, self.root / "run", model_config=self.MODEL, training_config=config)
        reserved_text = rows_for("causal_lm")[-1]["text"]
        self.assertNotIn(reserved_text, observed)
        observed.clear()
        with patch.object(encoding, "text_tokens", side_effect=encode):
            result = training.finalize(self.root / "run")
        self.assertIn(reserved_text, observed)
        self.assertFalse(result["testUsedForSelection"])

    def test_own_foundation_to_sft_bf16_microbatches_preserves_lineage_and_exact_resume(self):
        import torch
        from ia_propia.conversation.artifacts import load_artifact
        text = self.write("foundation.jsonl", rows_for("causal_lm"))
        foundation_config = {**self.CONFIG, "objective": "causal_lm", "data_loading": "mmap",
                             "max_steps": 2, "validation_every": 1}
        training.train(text, self.root / "foundation", model_config=self.MODEL,
                       training_config=foundation_config)
        parent_dir = self.root / "own-parent"
        training.export_best(self.root / "foundation", parent_dir)
        parent, tokenizer, manifest = load_artifact(parent_dir)
        sft = self.write("conversation.jsonl", rows_for("assistant_last_reply"))
        config = {**self.CONFIG, "objective": "assistant_last_reply", "data_loading": "mmap",
                  "train_precision": "cpu_bfloat16", "micro_batch_size": 1}
        initial_states = []
        original_checkpoint = training._checkpoint

        def checkpoint(*args, **kwargs):
            if args[5]["step"] == 0:
                initial_states.append(({key: value.clone() for key, value in args[1].state_dict().items()},
                                       args[2].to_str(), copy.deepcopy(args[3].state_dict())))
            return original_checkpoint(*args, **kwargs)

        continuous_dir, paused_dir = self.root / "sft-continuous", self.root / "sft-paused"
        with patch.object(training, "_checkpoint", side_effect=checkpoint):
            continuous = training.train(sft, continuous_dir, initial_model=parent_dir, training_config=config)
        self.assertTrue(initial_states)
        for weights, serialized_tokenizer, optimizer in initial_states:
            self.assert_nested_equal(weights, parent.state_dict(), "inherited weights")
            self.assertEqual(serialized_tokenizer, tokenizer.to_str())
            self.assertEqual(optimizer["state"], {})
        original_log = training._log
        stopped = []

        def log(run_dir, event):
            result = original_log(run_dir, event)
            if event.get("event") == "train" and event["step"] == 2 and not stopped:
                training._atomic_json(Path(run_dir) / "stop-request.json",
                                      {"pid": os.getpid(), "processBirth": training._birth(os.getpid())})
                stopped.append(True)
            return result

        with patch.object(training, "_log", side_effect=log):
            paused = training.train(sft, paused_dir, initial_model=parent_dir, training_config=config)
        self.assertEqual(paused["step"], 2)
        self.assertEqual(paused["status"], "stopped")
        (paused_dir / "stop-request.json").unlink()
        resumed = training.train(sft, paused_dir, resume=True)
        self.assertEqual(resumed["step"], config["max_steps"])
        continuous_state, resumed_state = self.compare_checkpoints(continuous_dir, continuous, paused_dir, resumed)
        self.assertEqual(continuous_state["lineage"]["parentModelId"], manifest["modelId"])
        self.assertEqual(continuous_state["lineage"], resumed_state["lineage"])
        model, inherited_tokenizer, _, payload = training._load_checkpoint(paused_dir, resumed["lastCheckpoint"])
        self.assertEqual(inherited_tokenizer.to_str(), tokenizer.to_str())
        self.assertTrue(all(value.dtype == torch.float32 for value in model.state_dict().values()))
        for moments in payload["optimizer"]["state"].values():
            self.assertEqual(moments["exp_avg"].dtype, torch.float32)
            self.assertEqual(moments["exp_avg_sq"].dtype, torch.float32)

    def test_failed_training_closes_open_mmaps(self):
        from ia_propia.conversation import disk_data
        corpus = self.write("text.jsonl", rows_for("causal_lm"))
        original = disk_data.build_or_load_windows
        caches = []

        def build(*args, **kwargs):
            result = original(*args, **kwargs)
            caches.append(result)
            return result

        config = {**self.CONFIG, "objective": "causal_lm", "data_loading": "mmap"}
        with patch.object(disk_data, "build_or_load_windows", side_effect=build), patch.object(
                training, "_evaluate_configured", side_effect=RuntimeError("fixture validation interruption")):
            with self.assertRaisesRegex(RuntimeError, "fixture validation interruption"):
                training.train(corpus, self.root / "interrupted", model_config=self.MODEL, training_config=config)
        self.assertEqual(len(caches), 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            caches[0]["train"][0]


if __name__ == "__main__":
    unittest.main()
