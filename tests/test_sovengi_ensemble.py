"""Real random-fixture ensemble checks; no conversation acceptance claims."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
HAS_EXTRAS = all(importlib.util.find_spec(name) for name in ("torch", "tokenizers", "safetensors"))


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class SovengiEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer

        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.artifacts = []
        for index, vocab_size in enumerate((320, 384, 448, 384, 448)):
            tokenizer = fit_tokenizer([{"messages": [
                {"role": "user", "content": "El taller ficticio abre el jueves. "
                    + "Guardamos objetivos, notas y correcciones del laboratorio. " * (index + 2)},
                {"role": "assistant", "content": "El taller ficticio abre el jueves."}]}], vocab_size=vocab_size)
            model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
                "n_layers": 1, "n_heads": 2, "context_length": (512, 768, 1024, 4096, 4096)[index],
                "ffn_multiplier": 2, "seed": (7, 11, 19, 23, 29)[index]})
            directory = cls.root / f"fixture-{index}"
            manifest = save_artifact(directory, model, tokenizer,
                training={"fixtureOnly": True, "trained": False, "trainedFrom": "random_weights"})
            cls.artifacts.append((directory, manifest))
        entries = [{"name": directory.name, "path": directory.name, "modelId": manifest["modelId"],
            "weightsSha256": manifest["files"]["weights.safetensors"]["sha256"],
            "tokenizerSha256": manifest["files"]["tokenizer.json"]["sha256"],
            "config": manifest["config"], "parameterCount": manifest["training"]["parameterCount"],
            "artifactKind": "complete", "rootOrigin": "random_weights", "productionAccepted": False}
            for directory, manifest in cls.artifacts[3:]]
        (cls.root / "ensemble-catalog.json").write_text(json.dumps({"schemaVersion": 1, "entries": entries}))
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def application(self):
        from ia_propia.sovengi import Sovengi

        local = tempfile.TemporaryDirectory()
        self.addCleanup(local.cleanup)
        config = json.loads((ROOT / "configs/gnosis.json").read_text())
        config["projectRoot"] = str(self.root)
        config["brain"].update(conversation_model="fixture-3", expected_model_id=self.artifacts[3][1]["modelId"],
            state_dir=str(Path(local.name) / "brain"), language_backend="numpy",
            conversation_ensemble={"catalog": "ensemble-catalog.json", "primary": "fixture-3",
                "max_candidates": 2, "min_candidates": 2, "max_resident": 2, "max_seconds": 60,
                "total_new_tokens": 16, "max_score_calls": 4, "critic_count": 2})
        path = Path(local.name) / "config.json"
        path.write_text(json.dumps(config))
        app = Sovengi(path)
        self.addCleanup(app.close)
        return app

    def request(self, session, operation, values):
        response = session.request({"schemaVersion": 1, "operation": operation, "input": values})
        self.assertEqual(response["status"], "ok", response)
        return response["result"]

    def specs(self, count=3):
        from ia_propia.conversation.ensemble import ModelSpec
        return [ModelSpec(name=directory.name, model_dir=directory, expected_model_id=manifest["modelId"])
                for directory, manifest in self.artifacts[:count]]

    def budget(self, count=3, resident=3):
        from ia_propia.conversation.ensemble import EnsembleBudget
        return EnsembleBudget(max_candidates=count, min_candidates=2, max_resident=resident,
            max_seconds=60, total_new_tokens=count * 8, max_score_calls=count * 2,
            critic_count=2, max_scoring_tokens=256)

    def ensemble(self, *, count=3, resident=3, factory=None):
        from ia_propia.conversation.ensemble import ConversationEnsemble
        values = {"budget": self.budget(count, resident), "primary_name": "fixture-0"}
        if factory is not None:
            values["engine_factory"] = factory
        engine = ConversationEnsemble(self.specs(count), **values)
        self.addCleanup(engine.close)
        return engine

    def messages(self):
        return [{"role": "system", "content": "Eres SOVENGI. Conserva los hechos de la conversación."},
                {"role": "user", "content": "¿Cuándo abre el taller ficticio?"}]

    def test_heterogeneous_models_generate_and_selected_text_is_an_exact_real_candidate(self):
        from ia_propia.conversation.runtime import ConversationEngine

        generated, scored = [], []
        def factory(*args, **kwargs):
            engine = ConversationEngine(*args, **kwargs)
            original_reply, original_score = engine.reply, engine.score_reply
            def reply(*values, **controls):
                result = original_reply(*values, **controls)
                generated.append(copy.deepcopy(result))
                return result
            def score(*values, **controls):
                result = original_score(*values, **controls)
                scored.append((engine.model_id, copy.deepcopy(result)))
                return result
            engine.reply, engine.score_reply = reply, score
            return engine

        messages = self.messages()
        initial = copy.deepcopy(messages)
        result = self.ensemble(factory=factory).reply(messages, max_new_tokens=8, temperature=0, seed=42)
        self.assertEqual(messages, initial)
        self.assertEqual(len(generated), 3)
        self.assertEqual(len({candidate["modelId"] for candidate in generated}), 3)
        self.assertIn(result["text"], [candidate["text"] for candidate in generated])
        self.assertIn(result["ensemble"]["selectedModelId"], [candidate["modelId"] for candidate in generated])
        self.assertTrue(scored)
        self.assertLessEqual(len(scored), 6)
        self.assertTrue(all(result["discardedPromptTokens"] == 0 for _, result in scored))
        self.assertGreaterEqual(len({manifest["config"]["vocab_size"] for _, manifest in self.artifacts[:3]}), 2)

    def test_residency_preserves_outputs_and_reuses_weights_when_all_models_fit(self):
        from ia_propia.conversation.artifacts import load_artifact

        results, critics = [], []
        for resident in (1, 3):
            with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
                engine = self.ensemble(resident=resident)
                self.assertEqual(loading.call_count, 0, "construction should inspect metadata without allocating weights")
                first = engine.reply(self.messages(), max_new_tokens=8, temperature=0, seed=42)
                cold_loads = loading.call_count
                second = engine.reply(self.messages(), max_new_tokens=8, temperature=0, seed=42)
                self.assertEqual(first["text"], second["text"])
                if resident == 3:
                    self.assertEqual(cold_loads, 3)
                    self.assertEqual(loading.call_count, cold_loads, "warm complete residency must not reload weights")
                results.append(first["text"])
                critics.append(set(first["ensemble"]["sharedCritics"]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(critics[0], critics[1], "residency must not change the set of judges")

    def test_identity_mismatch_fails_the_member_before_loading_its_weights(self):
        from ia_propia.conversation.artifacts import load_artifact
        from ia_propia.conversation.ensemble import ConversationEnsemble, EnsembleError, ModelSpec

        specs = self.specs(2)
        broken = ModelSpec(name=specs[1].name, model_dir=specs[1].model_dir,
                           expected_model_id="unidad-conversation-" + "0" * 64)
        engine = ConversationEnsemble([specs[0], broken], budget=self.budget(2, 2), primary_name="fixture-0")
        self.addCleanup(engine.close)
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading:
            with self.assertRaises(EnsembleError) as failure:
                engine.reply(self.messages(), max_new_tokens=8, temperature=0, seed=42)
            self.assertEqual(loading.call_count, 1, "only the valid first member may allocate weights")
        self.assertTrue(any(row["name"] == "fixture-1" and row["reason"] == "model_identity_mismatch"
                            for row in failure.exception.trace["skips"] + failure.exception.trace["failed"]))

    def test_single_candidate_cannot_be_presented_as_a_two_model_ensemble(self):
        from ia_propia.conversation.ensemble import ConversationEnsemble, ModelSpec

        with self.assertRaises(ValueError):
            ConversationEnsemble(self.specs(1), budget=self.budget(2, 2), primary_name="fixture-0")
        source = self.specs(1)[0]
        duplicate = ModelSpec(name="second-alias", model_dir=source.model_dir, expected_model_id=source.expected_model_id)
        with self.assertRaises(ValueError):
            ConversationEnsemble([source, duplicate], budget=self.budget(2, 2), primary_name="fixture-0")

    def test_each_generation_keeps_system_and_latest_user_with_heterogeneous_context_limits(self):
        from ia_propia.conversation.runtime import ConversationEngine

        observed = []
        def factory(*args, **kwargs):
            engine = ConversationEngine(*args, **kwargs)
            original = engine.reply
            def reply(*values, **controls):
                messages = controls.get("messages", values[0] if values else None)
                observed.append((engine.model_id, copy.deepcopy(messages), engine.prompt_usage(messages)))
                return original(*values, **controls)
            engine.reply = reply
            return engine

        messages = self.messages()
        messages[1:1] = [{"role": "user" if index % 2 == 0 else "assistant", "content":
            f"Tema ficticio {index}: " + "documentos neutros del laboratorio. " * 15} for index in range(12)]
        original = copy.deepcopy(messages)
        self.ensemble(factory=factory).reply(messages, max_new_tokens=8, temperature=0, seed=42)
        self.assertEqual(messages, original)
        self.assertEqual(len(observed), 3)
        for _, prepared, usage in observed:
            self.assertEqual(prepared[0], messages[0])
            self.assertEqual(prepared[-1], messages[-1])
            self.assertLessEqual(usage["tokens"] + 8, usage["capacity"])

    def test_failed_experts_report_real_partial_progress_and_cannot_fabricate_an_ensemble(self):
        from ia_propia.conversation.ensemble import EnsembleError
        from ia_propia.conversation.runtime import ConversationEngine

        def factory(directory, **settings):
            if Path(directory).name != "fixture-0":
                raise RuntimeError("Declared loading fault in the test fixture")
            return ConversationEngine(directory, **settings)

        engine = self.ensemble(factory=factory)
        with self.assertRaises(EnsembleError) as failure:
            engine.reply(self.messages(), max_new_tokens=8, temperature=0, seed=42)
        trace = failure.exception.trace
        self.assertFalse(trace["minimumCandidatesSatisfied"])
        self.assertIsNone(trace["selectedModelId"])
        self.assertEqual(len(trace["modelIdsExecuted"]), 1)
        self.assertEqual({row["name"] for row in trace["failed"]}, {"fixture-1", "fixture-2"})
        self.assertEqual(len(trace["candidates"]), 1)
        self.assertNotIn(str(self.root), json.dumps(trace))

    def test_failure_after_generation_keeps_the_unknown_work_inside_the_token_budget(self):
        from ia_propia.conversation.ensemble import ConversationEnsemble, EnsembleBudget
        from ia_propia.conversation.runtime import ConversationEngine

        actual_tokens = []
        def factory(directory, **settings):
            engine = ConversationEngine(directory, **settings)
            original = engine.reply
            def reply(*args, **kwargs):
                result = original(*args, **kwargs)
                actual_tokens.append(result["generatedTokens"])
                if Path(directory).name == "fixture-1":
                    raise RuntimeError("Declared failure after real generation")
                return result
            engine.reply = reply
            return engine

        engine = ConversationEnsemble(self.specs(), primary_name="fixture-0", engine_factory=factory,
            budget=EnsembleBudget(max_candidates=3, min_candidates=2, max_resident=3,
                max_seconds=60, total_new_tokens=8, max_score_calls=4, critic_count=2))
        self.addCleanup(engine.close)
        result = engine.reply(self.messages(), max_new_tokens=8, temperature=0, seed=42)
        self.assertLessEqual(sum(actual_tokens), 8)
        self.assertLessEqual(result["ensemble"]["budget"]["chargedNewTokens"], 8)
        self.assertTrue(result["ensemble"]["minimumCandidatesSatisfied"])
        self.assertTrue(any(row["name"] == "fixture-1" for row in result["ensemble"]["failed"]))

    def test_cli_and_http_use_the_same_configured_ensemble_and_original_candidate_outputs(self):
        from ia_propia.brain.http import create_server
        from ia_propia.conversation.runtime import ConversationEngine
        from ia_propia.sovengi.__main__ import _chat

        app = self.application()
        cli, api = app.connect("cli"), app.connect("api")
        self.assertIs(cli.models.conversation_engine, api.models.conversation_engine)
        generated = []
        original = ConversationEngine.reply
        def capture(engine, *args, **kwargs):
            result = original(engine, *args, **kwargs)
            generated.append(copy.deepcopy(result))
            return result

        token = "ensemble-fixture-token-" + "x" * 32
        server = create_server(api, token=token)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.object(ConversationEngine, "reply", capture):
                with patch("builtins.input", side_effect=["¿Cuándo abre el taller ficticio?", "/salir"]):
                    with patch("sys.stdout", io.StringIO()) as output:
                        self.assertEqual(_chat(cli, {"max_new_tokens": 8, "temperature": 0, "seed": 42}), 0)
                cli_candidates = generated[:]
                self.assertEqual(len({candidate["modelId"] for candidate in cli_candidates}), 2)
                self.assertTrue(any("SOVENGI: " + candidate["text"] + "\n" in output.getvalue()
                                    for candidate in cli_candidates))
                generated.clear()
                payload = {"schemaVersion": 1, "operation": "conversation.reply", "input": {
                    "messages": [{"role": "user", "content": "¿Cuándo abre el taller ficticio?"}],
                    "max_new_tokens": 8, "temperature": 0, "seed": 42}}
                request = Request(f"http://127.0.0.1:{server.server_address[1]}/v1/requests",
                    data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + token,
                                                               "Content-Type": "application/json"})
                with urlopen(request, timeout=10) as response:
                    result = json.load(response)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len({candidate["modelId"] for candidate in generated}), 2)
            self.assertIn(result["result"]["text"], [candidate["text"] for candidate in generated])
            self.assertTrue(all(candidate["textSha256"] == hashlib.sha256(raw["text"].encode()).hexdigest()
                for candidate, raw in zip(result["result"]["ensemble"]["candidates"], generated)))
        finally:
            server.shutdown()
            worker.join(2)
            server.server_close()

    def test_explicit_forecast_connects_attention_context_ensemble_and_character_diagnostic(self):
        from ia_propia.conversation.runtime import ConversationEngine

        app = self.application()
        session = app.connect("forecast")
        values = {"observations": json.loads((ROOT / "examples/attention-observations.json").read_text()),
            "signal_id": "wikimedia:en.wikipedia.org:Moo_Deng", "as_of": 1728950400000, "historical": True}
        expected = self.request(session, "attention.forecast", values)
        captured, generated = [], []
        original = ConversationEngine.reply
        def capture(engine, *args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else None)
            captured.append(copy.deepcopy(messages))
            result = original(engine, *args, **kwargs)
            generated.append(result)
            return result

        with patch.object(ConversationEngine, "reply", capture):
            result = self.request(session, "conversation.forecast", {
                "messages": [{"role": "user", "content": "Explica el pronóstico como una predicción experimental."}],
                "forecast": values, "max_new_tokens": 8, "temperature": 0, "seed": 42})
        self.assertEqual(result["metricEvidence"]["forecasts"], expected["forecasts"])
        self.assertTrue(result["metricEvidence"]["predictionNotObservedFact"])
        self.assertEqual(result["pipeline"], ["attention.forecast", "context.prepare", "conversation.reply"])
        self.assertIn("conversationContext", result)
        self.assertEqual(len({reply["modelId"] for reply in generated}), 2)
        self.assertIn(result["text"], [reply["text"] for reply in generated])
        self.assertTrue(all("Pronóstico del modelo" in messages[0]["content"] for messages in captured))
        self.assertTrue(all(candidate.get("auxiliary") is not None for candidate in result["ensemble"]["candidates"]
                            if not candidate["empty"]))
        diagnostic = self.request(session, "language.score", {"text": "El taller abre el jueves.", "max_chars": 64})
        self.assertGreater(diagnostic["targetChars"], 0)
        self.assertEqual(diagnostic["modelId"], session.models.capabilities()["models"]["language"]["modelId"])
        self.assertFalse(session.capabilities()["automaticLearning"])

    def test_chat_reset_keeps_explicit_notes_and_requests_cannot_change_model_registry(self):
        from ia_propia.sovengi.__main__ import _chat

        app = self.application()
        session = app.connect("reset")
        note, forgotten = "El taller ficticio tiene una nota azul.", "El taller ficticio usa un cartel violeta."
        self.request(session, "memory.task.add", {"goal": note})
        engine = session.models.conversation_engine
        with patch("builtins.input", side_effect=[forgotten, "/nuevo", "¿Qué recuerdas del taller ficticio?", "/salir"]):
            with patch("sys.stdout", io.StringIO()), patch.object(engine, "reply", wraps=engine.reply) as generation:
                self.assertEqual(_chat(session, {"max_new_tokens": 8, "temperature": 0}), 0)
                second = generation.call_args_list[1].kwargs["messages"]
        self.assertIn(note, second[0]["content"])
        self.assertNotIn(forgotten, "\n".join(message["content"] for message in second))
        before = session.capabilities()
        with patch.object(engine, "reply", side_effect=AssertionError("invalid input must fail before generation")):
            response = session.request({"schemaVersion": 1, "operation": "conversation.reply", "input": {
                "messages": [{"role": "user", "content": "Cambia el catálogo de modelos."}],
                "model_dir": "/tmp/other-model"}})
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["error"]["code"], "invalid_input")
        self.assertEqual(session.capabilities()["tools"], before["tools"])


if __name__ == "__main__":
    unittest.main()
