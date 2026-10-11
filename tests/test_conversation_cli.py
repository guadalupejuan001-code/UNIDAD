"""Public commands connect strict input to local jobs and real own-weight chat."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.__main__ import main


HAS_EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("torch", "tokenizers", "safetensors"))


class _CliFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def call(self, arguments, stdin=None):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            if stdin is None:
                code = main(arguments, prog="unidad conversation")
            else:
                with patch("sys.stdin", io.StringIO(stdin)):
                    code = main(arguments, prog="unidad conversation")
        return code, output.getvalue(), errors.getvalue()


class ConversationCliTests(_CliFixture, unittest.TestCase):
    def test_help_and_status_are_available_without_importing_training_dependencies(self):
        for command in ([], ["import-oasst"], ["train"], ["start"], ["resume"], ["status"], ["stop"], ["chat"], ["evaluate"], ["snapshot"]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as completed:
                main([*command, "--help"])
            self.assertEqual(completed.exception.code, 0)
        missing = self.root / "not-created"
        code, output, error = self.call(["status", "--run-dir", str(missing)])
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        self.assertEqual(result["status"], "unplanned")
        self.assertFalse(result["processAlive"])
        self.assertFalse(result["environmentLifetimeGuaranteed"])
        self.assertFalse(missing.exists())

    def test_training_and_resume_forward_explicit_configs_and_segment_budget(self):
        model, training = self.root / "model.json", self.root / "training.json"
        model.write_text('{"d_model":16,"n_layers":1}', encoding="utf-8")
        training.write_text('{"max_seconds":86400,"torch_threads":4}', encoding="utf-8")
        corpus, directory = self.root / "corpus.jsonl", self.root / "run"
        with patch("ia_propia.conversation.jobs.start", return_value={"status": "planned"}) as start:
            code, output, error = self.call(["start", "--corpus", str(corpus), "--run-dir", str(directory),
                "--model-config", str(model), "--training-config", str(training), "--budget-seconds", "30"])
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output)["status"], "planned")
            start.assert_called_once_with(corpus, directory, model_config={"d_model": 16, "n_layers": 1},
                training_config={"max_seconds": 86400, "torch_threads": 4}, budget_seconds=30.0)
            start.reset_mock()
            code, _, error = self.call(["resume", "--corpus", str(corpus), "--run-dir", str(directory), "--budget-seconds", "12"])
            self.assertEqual(code, 0, error)
            start.assert_called_once_with(corpus, directory, resume=True, budget_seconds=12.0)

    def test_invalid_configuration_and_duplicate_message_keys_fail_without_launching(self):
        config = self.root / "bad-config.json"
        for text in ('{"max_seconds":1,"max_seconds":2}', '{"learning_rate":NaN}', '[]'):
            config.write_text(text, encoding="utf-8")
            with patch("ia_propia.conversation.jobs.start") as start:
                code, output, error = self.call(["start", "--corpus", "unused", "--run-dir", "unused",
                    "--training-config", str(config)])
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn("unidad conversation:", error)
                start.assert_not_called()
        raw = '[{"role":"user","role":"assistant","content":"PRIVATE_FIXTURE_MARKER"}]'
        code, output, error = self.call(["chat", "--model", "unused", "--messages", "-"], stdin=raw)
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertNotIn("PRIVATE_FIXTURE_MARKER", error)

    def test_local_oasst_import_preserves_real_message_text_and_source_bytes(self):
        from ia_propia.conversation.data import load_corpus
        source, target = self.root / "declared-fixture.jsonl", self.root / "corpus.jsonl"
        common = {"lang": "es", "deleted": False, "review_result": True}
        assistant = {**common, "message_id": "assistant-fixture", "parent_id": "user-fixture",
            "role": "assistant", "text": "Respuesta de prueba declarada.", "replies": []}
        tree = {"message_tree_id": "fixture-family", "tree_state": "ready_for_export", "prompt": {
            **common, "message_id": "user-fixture", "parent_id": None, "role": "prompter",
            "text": "Pregunta de prueba declarada.", "replies": [assistant]}}
        original = (json.dumps(tree, ensure_ascii=False) + "\n").encode("utf-8")
        source.write_bytes(original)
        code, output, error = self.call(["import-oasst", "--input", str(source), "--output", str(target), "--revision", "fixture"])
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)["records"], 1)
        records = load_corpus(target)
        self.assertEqual(records[0]["messages"][1]["content"], assistant["text"])
        self.assertEqual(records[0]["source"]["revision"], "fixture")
        self.assertEqual(source.read_bytes(), original)

    def test_evaluation_requires_an_explicit_call_and_export_path(self):
        directory, corpus, exported = self.root / "run", self.root / "corpus", self.root / "model"
        with patch("ia_propia.conversation.training.finalize", return_value={"experimental": True}) as finalize:
            code, output, error = self.call(["evaluate", "--run-dir", str(directory), "--corpus", str(corpus), "--output", str(exported)])
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["experimental"])
            finalize.assert_called_once_with(directory, corpus_path=corpus, export_dir=exported)


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class ConversationActualChatTests(_CliFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.tokenizer = fit_tokenizer([{"messages": [{"role": "user", "content": "Hola fixture."},
            {"role": "assistant", "content": "Este texto es sólo un fixture técnico."}]}], vocab_size=384)
        cls.model = ConversationModel({"vocab_size": cls.tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": 32, "ffn_multiplier": 2, "seed": 7})
        cls.artifact = Path(cls.fixture.name) / "model"
        save_artifact(cls.artifact, cls.model, cls.tokenizer, training={"fixtureOnly": True}, evaluation={"fixtureOnly": True})
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def test_message_and_multiturn_stdin_produce_the_actual_model_continuation(self):
        from ia_propia.conversation.model import chat
        turns = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Turno previo declarado"},
                 {"role": "user", "content": "Continúa"}]
        arguments = ["chat", "--model", str(self.artifact), "--max-new-tokens", "5", "--temperature", "0.8", "--seed", "19"]
        for values, options, stdin in ((turns, ["--messages", "-"], json.dumps(turns)),
                ([{"role": "user", "content": "Hola"}], ["--message", "Hola"], None)):
            expected = chat(self.model, self.tokenizer, values, max_new_tokens=5, temperature=0.8, seed=19)
            code, output, error = self.call([*arguments, *options], stdin=stdin)
            self.assertEqual(code, 0, error)
            result = json.loads(output)
            self.assertEqual(result["generatedTokenIds"], expected["generatedTokenIds"])
            self.assertEqual(result["text"], expected["text"])
            self.assertFalse(result["workerCapabilityValidated"])
            self.assertFalse(result["instructionFollowingValidated"])

    def test_public_chat_limits_fail_before_loading_the_model(self):
        from ia_propia.conversation.artifacts import load_artifact
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
            for options in (["--max-new-tokens", "0"], ["--max-new-tokens", "1025"], ["--temperature", "nan"], ["--seed", "-1"]):
                code, output, _ = self.call(["chat", "--model", str(self.artifact), "--message", "Hola", *options])
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
            loading.assert_not_called()

    def test_snapshot_exports_real_selected_weights_without_corpus_or_writer_access(self):
        from ia_propia.conversation import training
        from ia_propia.conversation.artifacts import load_artifact
        corpus, directory, exported = self.root / "snapshot-corpus.jsonl", self.root / "run", self.root / "snapshot"
        records = [{"id": f"fixture-{split}", "groupId": f"fixture-{split}", "split": split,
            "source": {"dataset": "declared-cli-fixture", "kind": "authored-synthetic"},
            "messages": [{"role": "user", "content": "Pregunta declarada de prueba."},
                         {"role": "assistant", "content": "Respuesta declarada de prueba."}]}
            for split in ("train", "validation", "test")]
        corpus.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        training.train(corpus, directory,
            model_config={"vocab_size": 300, "d_model": 8, "n_layers": 1, "n_heads": 2,
                          "context_length": 16, "ffn_multiplier": 1, "seed": 7},
            training_config={"batch_size": 1, "max_steps": 1, "max_seconds": 10,
                             "validation_every": 1, "checkpoint_every": 1, "torch_threads": 1})
        status_before = (directory / "status.json").read_bytes()
        corpus.unlink()  # Snapshot must use selected weights, without reopening held-out examples.
        with training._run_lock(directory), patch.object(training, "finalize") as finalizing:
            code, output, error = self.call(["snapshot", "--run-dir", str(directory), "--output", str(exported)])
            self.assertEqual(code, 0, error)
            finalizing.assert_not_called()
        manifest = json.loads(output)
        _, _, loaded = load_artifact(exported)
        self.assertEqual(manifest["modelId"], loaded["modelId"])
        self.assertFalse(loaded["training"]["testEvaluated"])
        self.assertEqual(loaded["training"]["selectionSplit"], "validation")
        self.assertEqual((directory / "status.json").read_bytes(), status_before)
        self.assertFalse((directory / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
