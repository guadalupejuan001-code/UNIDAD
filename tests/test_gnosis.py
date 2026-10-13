import copy
import importlib.util
import io
import json
import hashlib
import shutil
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from ia_propia.gnosis import GnosisProject, load_project
from ia_propia.gnosis.__main__ import main


ROOT = Path(__file__).resolve().parents[1]
HAS_EXTRAS = all(importlib.util.find_spec(p) for p in ("torch", "tokenizers", "safetensors"))


def configuration(root):
    value = json.loads((ROOT / "configs/gnosis.json").read_text())
    value["projectRoot"] = str(root)
    value["brain"].pop("conversation_ensemble", None)
    value["brain"].update(conversation_model=None, expected_model_id=None, language_backend="numpy",
                          conversation_decoder="torch")
    value["learning"]["profile"] = "small"
    # This fixture exercises a fresh three-stage lifecycle. The product default
    # instead continues the published current model at its conversation stage.
    foundation = value["learning"]["stages"]["foundation"]["run_dir"]
    value["learning"]["stages"]["conversation"]["initial_model"] = foundation + "/artifact"
    return value


class GnosisPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = configuration(self.directory)

    def project(self, config=None):
        return GnosisProject(config or self.config, config_dir=self.directory)

    def test_old_host_config_defaults_to_torch_and_new_controls_are_strict(self):
        old = copy.deepcopy(self.config)
        del old["brain"]["conversation_decoder"]
        del old["brain"]["conversation_cache"]
        project = self.project(old)
        self.assertEqual(project.config["brain"]["conversation_decoder"], "torch")
        self.assertTrue(project.config["brain"]["conversation_cache"])
        self.assertNotIn("conversation_decoder", old["brain"])
        for key, value in (("conversation_decoder", "unknown"), ("conversation_decoder", []),
                           ("conversation_cache", 1), ("conversation_cache", "false")):
            config = copy.deepcopy(self.config)
            config["brain"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.project(config)

    def test_plan_counts_scaling_without_loading_a_model_or_writing_weights(self):
        for profile, expected in (("small", 6_967_296), ("15m", 15_769_344), ("50m", 48_852_992), ("100m", 99_446_400), ("1b", 991_586_304)):
            with self.subTest(profile=profile):
                self.config["learning"]["profile"] = profile
                with patch("ia_propia.brain.Brain", side_effect=AssertionError("plan must not load models")):
                    report = self.project().plan()
                self.assertEqual(report["parameterCount"], expected)
                self.assertFalse(report["weightsCreated"])
                self.assertFalse(report["conversationQualityValidated"])
                self.assertFalse(report["canStart"])
                self.assertIn("corpus_not_configured_or_missing", report["blockers"])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_large_profile_fails_admission_before_any_training_mutation(self):
        self.config["learning"]["profile"] = "1b"
        project = self.project()
        with patch("ia_propia.conversation.training.train", side_effect=AssertionError("must not start")):
            with self.assertRaisesRegex(ValueError, "memory_budget_exceeded"):
                project.train("foundation", budget_seconds=1)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_corpus_identity_is_checked_and_detached_configuration_is_not_authority(self):
        path = self.directory / "corpus.jsonl"
        path.write_text("fictional data identity only\n")
        stage = self.config["learning"]["stages"]["foundation"]
        stage.update(corpus="corpus.jsonl", corpus_sha256="0" * 64)
        project = self.project()
        self.assertIn("corpus_identity_mismatch", project.plan()["blockers"])
        detached = project.config
        detached["learning"]["profile"] = "1b"
        self.assertEqual(project.plan()["profile"], "small")

    def test_stage_partition_and_objective_cannot_be_changed_silently(self):
        for field, value in (("objective", "assistant_last_reply"), ("partition_policy", None)):
            config = copy.deepcopy(self.config)
            config["learning"]["stages"]["foundation"]["training_config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.project(config)
        config = copy.deepcopy(self.config)
        config["learning"]["stages"]["adaptation"]["run_dir"] = config["learning"]["stages"]["foundation"]["run_dir"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.project(config)

    def test_run_cannot_overwrite_selected_reference_and_budget_is_explicit(self):
        self.config["brain"].update(conversation_model="private/gnosis/foundation", expected_model_id="own-reference")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.project()
        project = GnosisProject(configuration(self.directory), config_dir=self.directory)
        for value in (None, True, 0, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project.train("foundation", budget_seconds=value)

    def test_run_nesting_and_oversized_numeric_budgets_are_rejected(self):
        for first, second in (("run", "run/checkpoints/conversation"), ("run/checkpoints/foundation", "run")):
            config = copy.deepcopy(self.config)
            config["learning"]["stages"]["foundation"]["run_dir"] = first
            config["learning"]["stages"]["conversation"]["run_dir"] = second
            with self.subTest(first=first), self.assertRaisesRegex(ValueError, "non-overlapping"):
                self.project(config)
        config = copy.deepcopy(self.config)
        config["learning"]["memory_budget_gib"] = 10 ** 400
        with self.assertRaises(ValueError):
            self.project(config)

    def test_corpus_symlink_is_rejected_even_when_its_hash_matches(self):
        data = self.directory / "real.jsonl"
        data.write_text("Fictional data.\n")
        link = self.directory / "linked.jsonl"
        link.symlink_to(data)
        self.config["learning"]["stages"]["foundation"].update(
            corpus=link.name, corpus_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
        report = self.project().plan()
        self.assertIn("corpus_identity_mismatch", report["blockers"])

    def test_corpus_changed_after_admission_is_rejected_by_the_trainer(self):
        path = self.directory / "corpus.jsonl"
        path.write_text("original fictional bytes\n")
        self.config["learning"]["stages"]["foundation"].update(
            corpus=path.name, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        project = self.project()
        admitted = project.plan()
        self.assertTrue(admitted["canStart"])
        path.write_text("changed fictional bytes\n")
        with patch.object(project, "plan", return_value=admitted):
            with self.assertRaisesRegex(ValueError, "corpus.*identity|corpus.*hash|Corpus.*identity|Corpus.*hash"):
                project.train("foundation", budget_seconds=1)
        self.assertFalse(project.path("private/gnosis/foundation").exists())

    def test_resume_admission_reads_actual_checkpoint_size_before_allocating(self):
        from ia_propia.gnosis.project import PROFILES
        from ia_propia.conversation.configuration import validate_config
        run = self.directory / "private/gnosis/conversation"
        name = "step-000000001-" + "a" * 32
        checkpoint = run / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"lastCheckpoint": name}))
        model = validate_config(PROFILES["100m"])
        (checkpoint / "state.json").write_text(json.dumps({"model_config": model}))
        for name in ("weights.safetensors", "resume.pt", "tokenizer.json"):
            (checkpoint / name).write_text("fictional fixture bytes, never decoded")
        digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
        (checkpoint / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "files": digests}))
        project = self.project()
        report = project.plan("conversation", resume=True)
        self.assertEqual(report["parameterCount"], 99_446_400)
        self.assertIn("inherited_architecture_differs_from_profile", report["blockers"])
        with patch("ia_propia.conversation.training.train", side_effect=AssertionError("reject before allocation")):
            with self.assertRaises(ValueError):
                project.train("conversation", budget_seconds=1, resume=True)

    def test_strict_config_loading_and_cli_plan_status_do_not_start_workers(self):
        path = self.directory / "project.json"
        path.write_text(json.dumps(self.config))
        with patch("ia_propia.conversation.jobs.start", side_effect=AssertionError("no background worker")):
            for command in ("plan", "status"):
                output = io.StringIO()
                with patch("sys.stdout", output):
                    self.assertEqual(main([command, "--project", str(path)]), 0)
                self.assertFalse(json.loads(output.getvalue()).get("processAlive", False))
        path.write_text('{"schemaVersion":1,"schemaVersion":1}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_project(path)
        path.write_text('{"schemaVersion":NaN}')
        with self.assertRaises(ValueError):
            load_project(path)

    def test_all_bodies_use_one_brain_with_separate_memory_and_no_automatic_learning(self):
        project = self.project()
        first, second = project.connect("pc"), project.connect("android")
        self.assertIs(first.brain, second.brain)
        self.assertIs(first.models, second.models)
        request = {"schemaVersion": 1, "requestId": "memory", "operation": "memory.task.add", "input": {"goal": "Fictional task"}}
        self.assertNotIn("error", first.request(request))
        result = second.request({**request, "requestId": "listing", "operation": "memory.task.list", "input": {}})
        self.assertEqual(result["result"]["tasks"], [])
        self.assertFalse(second.capabilities()["automaticLearning"])
        self.assertFalse(second.capabilities()["naturalLanguagePlanner"])


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class GnosisInferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        rows = [{"messages": [{"role": "user", "content": "Pregunta ficticia."},
                                {"role": "assistant", "content": "Respuesta ficticia."}]}] * 3
        tokenizer = fit_tokenizer(rows, 300)
        model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
                                   "n_layers": 1, "n_heads": 2, "context_length": 128})
        self.manifest = save_artifact(self.directory / "reference", model, tokenizer,
                                      training={"trainedFrom": "random", "purpose": "software-test"})
        self.config = configuration(self.directory)
        self.config["brain"].update(conversation_model="reference", expected_model_id=self.manifest["modelId"])

    @unittest.skipUnless(shutil.which("node"), "NERVA requires Node.js 18+")
    def test_gnosis_config_connects_the_real_nerva_decoder_to_its_shared_brain(self):
        supplied = json.loads((ROOT / "configs/gnosis.json").read_text())
        self.assertEqual(supplied["brain"]["conversation_decoder"], "torch")
        self.config["brain"]["conversation_decoder"] = "nerva"
        project = GnosisProject(self.config, config_dir=self.directory)
        result = project.connect("pc").request({"schemaVersion": 1, "requestId": "nerva-gnosis",
            "operation": "conversation.reply", "input": {"messages": [{"role": "user", "content": "Hola"}],
                                                         "max_new_tokens": 4, "temperature": 0}})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["result"]["decoding"]["backend"], "nerva")
        self.assertEqual(result["result"]["modelId"], self.manifest["modelId"])

    def test_pinned_reference_and_same_engine_generation_for_all_bodies(self):
        from ia_propia.conversation import artifacts
        from ia_propia.conversation.runtime import ConversationEngine
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=artifacts.load_artifact) as loads:
            project = GnosisProject(self.config, config_dir=self.directory)
            first, second = project.connect("pc"), project.connect("android")
            self.assertEqual(loads.call_count, 1)
            engine = project.brain.models.conversation_engine
            self.assertIsInstance(engine, ConversationEngine)
            inputs = {"messages": [{"role": "user", "content": "Hola"}], "max_new_tokens": 8, "temperature": 0, "seed": 42}
            expected = engine.reply(**inputs)
            payload = {"schemaVersion": 1, "requestId": "reply", "operation": "conversation.reply", "input": inputs}
            self.assertEqual(first.request(payload)["result"], expected)
            self.assertEqual(second.request(payload)["result"], expected)
            self.assertEqual(loads.call_count, 1)
            rejected = first.request({**payload, "input": {**inputs, "model_path": "other"}})
            self.assertEqual(rejected["error"]["code"], "invalid_input")
        self.config["brain"]["expected_model_id"] = "different-model"
        with patch("ia_propia.conversation.artifacts.load_artifact", side_effect=AssertionError("reject before allocation")):
            project = GnosisProject(self.config, config_dir=self.directory)
            with self.assertRaisesRegex(ValueError, "identity"):
                project.connect("pc")

    def test_common_console_reuses_already_loaded_engine(self):
        from ia_propia.conversation.__main__ import _converse
        from types import SimpleNamespace
        project = GnosisProject(self.config, config_dir=self.directory)
        engine = project.brain.models.conversation_engine
        args = SimpleNamespace(model="not-an-artifact", max_new_tokens=4, temperature=0, seed=42)
        with patch("sys.stdin", io.StringIO("Hola\n/salir\n")), patch("sys.stdout", io.StringIO()) as output:
            with patch("ia_propia.conversation.artifacts.load_artifact", side_effect=AssertionError("no second load")):
                self.assertEqual(_converse(args, engine=engine), 0)
        self.assertIn(self.manifest["modelId"], output.getvalue())

    def test_concurrent_products_do_not_allocate_duplicate_brains(self):
        from ia_propia.conversation import artifacts
        project = GnosisProject(self.config, config_dir=self.directory)
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=artifacts.load_artifact) as loads:
            with ThreadPoolExecutor(max_workers=4) as pool:
                sessions = list(pool.map(project.connect, ("one", "two", "three", "four")))
            self.assertEqual(loads.call_count, 1)
            self.assertTrue(all(session.brain is sessions[0].brain for session in sessions))

    def test_real_foundation_to_conversation_keeps_tokenizer_and_own_lineage(self):
        from ia_propia.gnosis.project import PROFILES
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.runtime import ConversationEngine
        config = copy.deepcopy(self.config)
        config["brain"].update(conversation_model=None, expected_model_id=None)
        text = [{"id": f"doc-{i}", "groupId": f"family-{i}", "split": split,
                 "source": {"dataset": "gnosis-software-test", "kind": "authored-synthetic"},
                 "text": f"Documento ficticio número {i}: palabras para una prueba de software."}
                for i, split in enumerate(("train", "train", "validation"))]
        dialogues = [{"id": f"chat-{i}", "groupId": f"dialogue-{i}", "split": split,
                      "source": {"dataset": "gnosis-software-test", "kind": "authored-synthetic"},
                      "messages": [{"role": "user", "content": f"Pregunta ficticia {i}."},
                                   {"role": "assistant", "content": f"Respuesta ficticia {i}."}]}
                     for i, split in enumerate(("train", "train", "validation"))]
        for name, records in (("foundation", text), ("conversation", dialogues), ("adaptation", dialogues)):
            path = self.directory / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            stage = config["learning"]["stages"][name]
            stage.update(corpus=path.name, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            stage["training_config"].update(batch_size=2, micro_batch_size=1, max_steps=2,
                                            max_epochs=10, max_seconds=60, validation_every=1,
                                            checkpoint_every=1, warmup_steps=0, min_delta=0)
        # A real tiny model exercises orchestration, not linguistic capability.
        tiny = {"vocab_size": 300, "d_model": 16, "n_layers": 1, "n_heads": 2, "context_length": 128}
        with patch.dict(PROFILES, {"small": tiny}):
            project = GnosisProject(config, config_dir=self.directory)
            self.assertTrue(project.plan("foundation")["canStart"])
            self.assertEqual(project.train("foundation", budget_seconds=30)["step"], 2)
            first = project.export("foundation")
            initial = project.path(config["learning"]["stages"]["foundation"]["run_dir"]) / "artifact"
            first_weights = hashlib.sha256((initial / "weights.safetensors").read_bytes()).hexdigest()
            self.assertTrue(project.plan("conversation")["canStart"])
            self.assertEqual(project.train("conversation", budget_seconds=30)["step"], 2)
            second = project.export("conversation")
            target = project.path(config["learning"]["stages"]["conversation"]["run_dir"]) / "artifact"
            loaded, tokenizer, manifest = load_artifact(target)
            self.assertEqual(second["files"]["tokenizer.json"], first["files"]["tokenizer.json"])
            self.assertEqual(manifest["training"]["trainedFrom"], "own_checkpoint")
            self.assertEqual(hashlib.sha256((initial / "weights.safetensors").read_bytes()).hexdigest(), first_weights)
            self.assertEqual(ConversationEngine(target).model_id, second["modelId"])
            self.assertIsNone(project.config["brain"]["conversation_model"])
            self.assertFalse(second["evaluation"]["productionAccepted"])
            second_weights = hashlib.sha256((target / "weights.safetensors").read_bytes()).hexdigest()
            self.assertTrue(project.plan("adaptation")["canStart"])
            self.assertEqual(project.train("adaptation", budget_seconds=30)["step"], 2)
            third = project.export("adaptation")
            self.assertEqual(third["files"]["tokenizer.json"], first["files"]["tokenizer.json"])
            self.assertEqual(third["training"]["lineage"]["parentModelId"], second["modelId"])
            self.assertEqual(second["training"]["lineage"]["parentModelId"], first["modelId"])
            self.assertEqual(hashlib.sha256((target / "weights.safetensors").read_bytes()).hexdigest(), second_weights)
            self.assertIsNone(project.config["brain"]["conversation_model"])


if __name__ == "__main__":
    unittest.main()
