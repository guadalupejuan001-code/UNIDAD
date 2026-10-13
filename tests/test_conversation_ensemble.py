"""Declared stubs test coordination; tiny own artifacts test neural provenance."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from ia_propia.conversation.ensemble import (ConversationEnsemble, EnsembleBudget,
                                            EnsembleError, ModelSpec)


HAS_EXTRAS = all(importlib.util.find_spec(name) for name in ("torch", "tokenizers", "safetensors"))


def identity(name):
    return "unidad-conversation-" + hashlib.sha256(name.encode()).hexdigest()


def specifications(count=4):
    return [ModelSpec(f"model-{index}", f"fixture-model-{index}", identity(str(index))) for index in range(count)]


class FixtureTokenizer:
    def __init__(self, characters=False):
        self.characters = characters

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(range(len(text) if self.characters else len(text.split()))))


class FixtureFactory:
    """Scripted responses are fixture control data, never a production fallback."""
    def __init__(self, *, texts=None, capacities=None, failures=None, chars=(), score_failures=(),
                 scores=None, clock=None, advance=0, generation_fails=()):
        self.texts = texts or {"0": " El museo abre el martes. \n", "1": "El taller abre el lunes.",
                               "2": "La biblioteca abre el miércoles.", "3": "El parque abre el sábado."}
        self.capacities, self.failures = capacities or {}, failures or {}
        self.chars, self.score_failures, self.scores = set(chars), set(score_failures), scores or {}
        self.clock, self.advance, self.generation_fails = clock, advance, set(generation_fails)
        self.loads, self.generated, self.scored, self.alive = [], [], [], weakref.WeakSet()
        self.peak = 0

    def __call__(self, path, **kwargs):
        name = str(path).rsplit("-", 1)[-1]
        if name in self.failures:
            raise self.failures[name]
        engine = FixtureEngine(self, name)
        self.alive.add(engine)
        self.loads.append(name)
        self.peak = max(self.peak, len(self.alive))
        return engine


class FixtureEngine:
    def __init__(self, factory, name):
        self.factory, self.name = factory, name
        self.model_id = identity(name)
        self._tokenizer = FixtureTokenizer(name in factory.chars)
        self.manifest = {"modelId": self.model_id, "config": {"context_length": factory.capacities.get(name, 1024),
            "vocab_size": 512, "d_model": 8, "n_layers": 1, "n_heads": 2, "ffn_multiplier": 2,
            "dropout": 0., "seed": 1}}

    def prompt_usage(self, messages):
        count = sum(len(self._tokenizer.encode(row["content"]).ids) + 2 for row in messages)
        capacity = self.manifest["config"]["context_length"]
        return {"tokens": count, "capacity": capacity, "discarded": max(0, count - capacity)}

    def reply(self, **values):
        self.factory.generated.append((self.name, deepcopy(values)))
        if self.factory.clock is not None:
            self.factory.clock[0] += self.factory.advance
        if self.name in self.factory.generation_fails:
            raise RuntimeError("fixture private generation failure /private/path")
        text = self.factory.texts[self.name]
        count = min(values["max_new_tokens"], 4)
        return {"text": text, "modelId": self.model_id, "generatedTokens": count,
                "generatedTokenIds": list(range(count)), "stopReason": "eos", "seed": values["seed"],
                "temperature": values["temperature"], "workerCapabilityValidated": False}

    def score_reply(self, messages, text):
        self.factory.scored.append((self.name, text, deepcopy(messages)))
        if self.name in self.factory.score_failures:
            return {"meanNll": float("nan")}
        count = len(self._tokenizer.encode(text).ids) + 1
        cost = self.factory.scores.get((self.name, text), 1 if "lunes" in text else 3)
        total = cost * len(text)
        usage = self.prompt_usage(messages)
        return {"meanNll": total / count, "totalLogProbability": -total, "targetTokens": count,
                "targetsWithSlidingPrompt": max(0, usage["tokens"] + count - 1 - usage["capacity"]),
                "discardedPromptTokens": usage["discarded"]}


class EnsembleImportTests(unittest.TestCase):
    def test_facade_import_and_description_do_not_import_ml_or_load_weights(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','tokenizers','safetensors'}:
        raise AssertionError('unexpected eager ML import')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from ia_propia.conversation.ensemble import ConversationEnsemble,ModelSpec
engine=ConversationEnsemble([ModelSpec('a','absent-a','a'),ModelSpec('b','absent-b','b')])
assert engine.describe()['cache']['loads']==0
assert engine.describe()['memberCount']==2
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_bounds_reject_duplicate_identity_and_untrusted_member_fields(self):
        for values in ({"max_candidates": True}, {"min_candidates": 1}, {"max_resident": 0},
                       {"max_seconds": float("nan")}, {"total_new_tokens": 1}, {"max_score_calls": 65},
                       {"max_scoring_tokens": 1026}, {"max_sliding_score_tokens": -1}):
            with self.assertRaises(ValueError):
                EnsembleBudget(**values)
        with self.assertRaises(ValueError):
            ConversationEnsemble([ModelSpec("a", "a", "same"), ModelSpec("b", "b", "same")])
        with self.assertRaises(ValueError):
            ModelSpec("../../private", "fixture", "a")


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class EnsembleCoordinationTests(unittest.TestCase):
    messages = [{"role": "system", "content": "Habla español. Conserva los hechos."},
                {"role": "user", "content": "El taller abre el lunes. ¿Cuándo abre?"}]

    def ensemble(self, factory=None, *, count=4, **budget):
        factory = factory or FixtureFactory()
        engine = ConversationEnsemble(specifications(count), budget=EnsembleBudget(**budget),
                                      engine_factory=factory)
        self.addCleanup(engine.close)
        return engine, factory

    def test_four_candidates_cross_critic_ranks_preserve_one_original_and_input(self):
        factory = FixtureFactory(chars=("1",))
        engine, _ = self.ensemble(factory)
        before = deepcopy(self.messages)
        result = engine.reply(self.messages, max_new_tokens=16, temperature=0, seed=37)
        self.assertEqual(result["text"], factory.texts["1"])
        self.assertEqual(result["modelId"], identity("1"))
        self.assertEqual((result["generatedTokens"], result["generatedTokenIds"], result["seed"]), (4, [0, 1, 2, 3], 37))
        trace = result["ensemble"]
        self.assertEqual(len(trace["candidates"]), 4)
        self.assertEqual(trace["budget"]["scoreCalls"], 8)
        self.assertEqual(set(trace["sharedCritics"]), {identity("0"), identity("1")})
        self.assertFalse(trace["learnedSelector"])
        self.assertTrue(all(critic["complete"] for critic in trace["critics"]))
        self.assertEqual(self.messages, before)

    def test_lru_actual_live_engines_are_bounded_and_memory_limit_keeps_critic_set(self):
        results = []
        for maximum in (1, 4):
            engine, factory = self.ensemble(max_resident=maximum)
            result = engine.reply(self.messages, max_new_tokens=8, temperature=0)
            self.assertLessEqual(factory.peak, maximum)
            self.assertLessEqual(engine.cache_info()["residentCount"], maximum)
            results.append((result["text"], result["modelId"], set(result["ensemble"]["sharedCritics"])))
        self.assertEqual(results[0], results[1])

    def test_score_call_budget_admits_fixed_critics_before_cache_optimization(self):
        default_texts = FixtureFactory().texts
        scores = {(critic, text): 1 if name == critic else 3
                  for critic in ("0", "1") for name, text in default_texts.items()}
        results = []
        for maximum in (1, 4):
            engine, _ = self.ensemble(FixtureFactory(scores=scores), max_resident=maximum, max_score_calls=4)
            result = engine.reply([{ "role": "user", "content": "Háblame."}], max_new_tokens=8, temperature=0)
            self.assertEqual(result["ensemble"]["sharedCritics"], [identity("0")])
            results.append((result["text"], result["modelId"]))
        self.assertEqual(results[0], results[1])

    def test_primary_is_always_used_and_other_members_rotate_without_eager_load(self):
        texts = {str(index): f"Respuesta ficticia número {index}." for index in range(6)}
        engine, factory = self.ensemble(FixtureFactory(texts=texts), count=6, max_candidates=3, critic_count=1)
        self.assertEqual(factory.loads, [])
        for _ in range(3):
            engine.reply(self.messages, max_new_tokens=4, temperature=0)
        self.assertEqual([name for name, _ in factory.generated], ["0", "1", "2", "0", "1", "3", "0", "1", "4"])
        self.assertEqual(engine.describe()["memberCount"], 6)
        self.assertEqual(engine.describe()["cohortPolicy"]["anchors"], ["model-0", "model-1"])
        self.assertEqual(engine.catalog_state()[-1]["state"], "not_loaded")

    def test_two_large_anchors_keep_minimum_generations_when_rotating_small_models_cannot_fit(self):
        texts = {str(index): f"Respuesta ficticia del taller {index}." for index in range(5)}
        factory = FixtureFactory(texts=texts, capacities={"0": 512, "1": 512, "2": 64, "3": 64, "4": 64})
        engine, _ = self.ensemble(factory, count=5)
        messages = [{"role": "system", "content": "Conserva las restricciones del taller. " * 20}, self.messages[-1]]
        rotations = []
        for _ in range(3):
            result = engine.reply(messages, max_new_tokens=16)
            trace = result["ensemble"]
            self.assertTrue(trace["minimumCandidatesSatisfied"])
            self.assertEqual([candidate["name"] for candidate in trace["candidates"]], ["model-0", "model-1"])
            self.assertEqual(trace["cohortPolicy"]["anchors"], ["model-0", "model-1"])
            rotations.extend(trace["cohortPolicy"]["rotating"])
            self.assertTrue(all(row["reason"] == "minimum_context_exceeds_capacity" for row in trace["failed"]))
        self.assertEqual(set(rotations), {"model-2", "model-3", "model-4"})

    def test_missing_member_is_explicit_and_fewer_than_two_successes_raise_with_metadata(self):
        factory = FixtureFactory(failures={"1": FileNotFoundError("private path /secret")})
        engine, _ = self.ensemble(factory, count=2)
        with self.assertRaises(EnsembleError) as raised:
            engine.reply(self.messages)
        self.assertEqual(raised.exception.reason, "minimum_successful_candidates_not_met")
        trace = raised.exception.details
        self.assertEqual(len(trace["candidates"]), 1)
        self.assertFalse(trace["minimumCandidatesSatisfied"])
        self.assertEqual(trace["failed"][0]["errorType"], "FileNotFoundError")
        self.assertNotIn("/secret", json.dumps(trace))

    def test_failed_neural_calls_keep_reserved_token_cost_and_cannot_overspend(self):
        engine, factory = self.ensemble(FixtureFactory(generation_fails={"0", "1", "2", "3"}),
                                        total_new_tokens=8)
        with self.assertRaises(EnsembleError) as raised:
            engine.reply(self.messages, max_new_tokens=8)
        quotas = [values["max_new_tokens"] for _, values in factory.generated]
        self.assertEqual(quotas, [2, 2, 2, 2])
        self.assertEqual(raised.exception.details["budget"]["chargedNewTokens"], 8)
        self.assertEqual(raised.exception.details["budget"]["generatedTokens"], 0)

    def test_cooperative_time_limit_stops_future_operations_and_reports_overrun(self):
        clock = [0.0]
        engine, factory = self.ensemble(FixtureFactory(clock=clock, advance=2), max_seconds=1)
        with patch("ia_propia.conversation.ensemble.perf_counter", side_effect=lambda: clock[0]):
            with self.assertRaises(EnsembleError) as raised:
                engine.reply(self.messages)
        self.assertEqual(len(factory.generated), 1)
        self.assertEqual(factory.scored, [])
        self.assertTrue(raised.exception.trace["budget"]["timeBudgetExceeded"])
        self.assertEqual(len(raised.exception.trace["skips"]), 3)

    def test_whole_turn_history_reduction_preserves_system_and_current_for_each_model(self):
        engine, factory = self.ensemble(FixtureFactory(capacities={"0": 24, "1": 30}), count=2,
                                        total_new_tokens=8, max_candidates=2)
        history = [self.messages[0], {"role": "user", "content": "viejo " * 20},
                   {"role": "assistant", "content": "respuesta vieja " * 20}, self.messages[-1]]
        result = engine.reply(history, max_new_tokens=4)
        for _, values in factory.generated:
            self.assertEqual(values["messages"], self.messages)
        self.assertEqual([row["context"]["omittedMessages"] for row in result["ensemble"]["candidates"]], [2, 2])
        self.assertTrue(all(row["targetsWithSlidingPrompt"] == 0 for critic in result["ensemble"]["critics"] for row in critic["scores"]))

    def test_minimum_context_is_never_silently_truncated_and_long_targets_skip_scoring(self):
        engine, factory = self.ensemble(FixtureFactory(capacities={"0": 3}), count=3, max_scoring_tokens=1)
        result = engine.reply(self.messages, max_new_tokens=4)
        self.assertNotIn("0", [name for name, _ in factory.generated])
        self.assertTrue(any(row["reason"] == "minimum_context_exceeds_capacity" for row in result["ensemble"]["failed"]))
        self.assertEqual(factory.scored, [])
        self.assertTrue(result["ensemble"]["selectionRule"]["unscoredSelection"])

    def test_privacy_and_untrusted_error_reason_no_raw_text_or_paths_enter_trace(self):
        class PrivateError(RuntimeError):
            reason = "PRIVATE SOURCE /private/full/path"
        engine, _ = self.ensemble(count=2)
        engine._auxiliary = lambda text: (_ for _ in ()).throw(PrivateError("hidden message"))
        source = {"role": "assistant", "text": "PRIVATE ASSISTANT CLAIM"}
        result = engine.reply(self.messages, context_sources=[source])
        serialized = json.dumps(result["ensemble"], ensure_ascii=False)
        for private in ("PRIVATE", "/private", "hidden message", self.messages[-1]["content"], result["text"]):
            self.assertNotIn(private, serialized)
        self.assertTrue(all(row["reason"] == "operation_failed" for row in result["ensemble"]["failed"]))
        class UnstructuredError(RuntimeError):
            reason = {"text": "PRIVATE"}
        engine._auxiliary = lambda text: (_ for _ in ()).throw(UnstructuredError())
        self.assertTrue(all(row["reason"] == "operation_failed" for row in
                            engine.reply(self.messages)["ensemble"]["failed"]))

    def test_auxiliary_only_breaks_ties_and_preserves_original_whitespace(self):
        texts = {"0": " Una nota. \n", "1": "Otra nota."}
        equal_scores = {(critic, text): 2 for critic in ("0", "1") for text in texts.values()}
        engine, _ = self.ensemble(FixtureFactory(texts=texts, scores=equal_scores), count=2)
        engine._auxiliary = lambda text: {"meanNll": 0.1 if text == texts["0"] else 1,
                                          "unknownRatio": 0, "privateIgnored": "not traced"}
        result = engine.reply([{ "role": "user", "content": "Háblame."}])
        self.assertEqual(result["text"], texts["0"])
        self.assertEqual(result["ensemble"]["selectionRule"]["auxiliary"], "diagnostic_and_tie_break_only")

    def test_auxiliary_provenance_is_visible_and_abstention_has_no_fluency_tie_break(self):
        texts = {"0": "X", "1": "YZ"}
        scores = {(critic, text): 2 for critic in ("0", "1") for text in texts.values()}
        engine, _ = self.ensemble(FixtureFactory(texts=texts, scores=scores), count=2)
        rnn_id = "language-" + hashlib.sha256(b"declared-rnn-fixture").hexdigest()
        engine._auxiliary = lambda text: {
            "modelId": rnn_id, "source": "own_character_rnn", "task": "next_character_likelihood",
            "executionBackend": "numpy", "scoreable": len(text) > 1,
            "meanNll": 1.0 if len(text) > 1 else None, "unknownRatio": .5 if len(text) > 1 else 0,
            "targetChars": len(text) - 1, "scoredCharacters": len(text), "truncated": False,
            "privatePath": "/private/file", "text": text}
        result = engine.reply([{ "role": "user", "content": "Háblame."}])
        self.assertEqual(result["text"], "YZ")
        abstention = result["ensemble"]["candidates"][0]["auxiliary"]
        self.assertEqual(abstention["modelId"], rnn_id)
        self.assertFalse(abstention["scoreable"])
        self.assertFalse(abstention["usedForTieBreak"])
        self.assertIsNone(abstention["meanNll"])
        self.assertEqual(abstention["scoredCharacters"], 1)
        self.assertNotIn("privatePath", abstention)
        self.assertNotIn("text", abstention)

    def test_incomplete_or_nonfinite_critics_are_excluded_from_cross_model_vote(self):
        engine, _ = self.ensemble(FixtureFactory(score_failures={"0", "1"}), count=2)
        result = engine.reply(self.messages)
        self.assertEqual(result["ensemble"]["sharedCritics"], [])
        self.assertEqual(result["text"], "El taller abre el lunes.")
        self.assertTrue(result["ensemble"]["selectionRule"]["unscoredSelection"])


@unittest.skipUnless(HAS_EXTRAS, "conversation extras are optional")
class EnsembleOwnArtifactTests(unittest.TestCase):
    def test_two_real_transformers_run_and_selected_reply_matches_native_engine(self):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.runtime import ConversationEngine
        from ia_propia.conversation.tokenizer import fit_tokenizer
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = fit_tokenizer([{"messages": [{"role": "user", "content": "Hola fixture."},
                {"role": "assistant", "content": "Respuesta ficticia técnica."}]}], vocab_size=320)
            specs = []
            for seed in (7, 19):
                model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 8,
                    "n_layers": 1, "n_heads": 2, "context_length": 256,
                    "ffn_multiplier": 2, "seed": seed})
                path = Path(folder) / f"own-{seed}"
                manifest = save_artifact(path, model, tokenizer, training={"fixtureOnly": True})
                specs.append(ModelSpec(f"own-{seed}", path, manifest["modelId"]))
            coordinator = ConversationEnsemble(specs, budget=EnsembleBudget(max_candidates=2,
                total_new_tokens=8, max_seconds=20, max_score_calls=4))
            self.addCleanup(coordinator.close)
            result = coordinator.reply([{ "role": "user", "content": "Hola"}],
                                       max_new_tokens=4, temperature=.8, seed=31)
            self.assertEqual(len(result["ensemble"]["candidates"]), 2)
            self.assertEqual(set(result["ensemble"]["modelIdsExecuted"]), {spec.expected_model_id for spec in specs})
            selected = next(spec for spec in specs if spec.expected_model_id == result["modelId"])
            native = ConversationEngine(selected.model_dir).reply([{ "role": "user", "content": "Hola"}],
                max_new_tokens=4, temperature=.8, seed=31)
            self.assertEqual({key: value for key, value in result.items() if key != "ensemble"}, native)
            mismatch = [ModelSpec(specs[0].name, specs[0].model_dir, "wrong-pinned-id"), specs[1]]
            bad = ConversationEnsemble(mismatch, budget=EnsembleBudget(max_candidates=2))
            self.addCleanup(bad.close)
            with self.assertRaises(EnsembleError) as raised:
                bad.reply([{ "role": "user", "content": "Hola"}], max_new_tokens=4)
            self.assertTrue(any(row["reason"] == "model_identity_mismatch" for row in raised.exception.trace["skips"]))

    def test_native_catalog_keeps_missing_entries_visible_and_rotates_only_available_artifacts(self):
        import torch
        from ia_propia.conversation.artifacts import save_artifact
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = fit_tokenizer([{"messages": [{"role": "user", "content": "Hola fixture."},
                {"role": "assistant", "content": "Una prueba técnica ficticia."}]}], vocab_size=320)
            specs = []
            for seed in (7, 19, 31):
                model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 8,
                    "n_layers": 1, "n_heads": 2, "context_length": 256,
                    "ffn_multiplier": 2, "seed": seed})
                path = Path(folder) / f"own-{seed}"
                manifest = save_artifact(path, model, tokenizer, training={"fixtureOnly": True})
                specs.append(ModelSpec(f"own-{seed}", path, manifest["modelId"]))
            specs.extend(ModelSpec(f"missing-{index}", Path(folder) / f"missing-{index}", identity(f"missing-{index}"))
                         for index in range(32))
            coordinator = ConversationEnsemble(specs, budget=EnsembleBudget(max_candidates=2,
                total_new_tokens=8, max_score_calls=4, max_seconds=20))
            self.addCleanup(coordinator.close)
            report = coordinator.describe()
            self.assertEqual(report["cache"]["loads"], 0)
            self.assertEqual(report["memberCount"], 35)
            self.assertEqual(sum(member["available"] for member in report["members"]), 3)
            selected = []
            for _ in range(4):
                reply = coordinator.reply([{ "role": "user", "content": "Hola"}],
                                          max_new_tokens=4, temperature=.8, seed=31)
                selected.append([candidate["name"] for candidate in reply["ensemble"]["candidates"]])
                unavailable = [member for member in reply["ensemble"]["registryAvailability"] if not member["available"]]
                self.assertEqual(len(unavailable), 32)
                self.assertEqual(len(reply["ensemble"]["modelIdsExecuted"]), 2)
                self.assertTrue(all(member["reason"] == "artifact_unavailable" for member in unavailable))
                self.assertTrue(all(not model.startswith("missing") for model in reply["ensemble"]["cache"]["residentNames"]))
            self.assertEqual(selected, [["own-7", "own-19"]] * 4)
            # A malformed header is excluded before a fresh coordinator loads
            # neural weights; other available members still satisfy the minimum.
            weights = Path(specs[2].model_dir) / "weights.safetensors"
            with weights.open("r+b") as stream:
                stream.write(b"bad-head")
            fresh = ConversationEnsemble(specs, budget=EnsembleBudget(max_candidates=4,
                total_new_tokens=8, max_score_calls=4, max_seconds=20))
            self.addCleanup(fresh.close)
            reply = fresh.reply([{ "role": "user", "content": "Hola"}],
                                max_new_tokens=4, temperature=.8, seed=31)
            rejected = next(member for member in reply["ensemble"]["registryAvailability"] if member["name"] == "own-31")
            self.assertFalse(rejected["available"])
            self.assertEqual(rejected["reason"], "weights_header_invalid")
            self.assertEqual(reply["ensemble"]["cache"]["loads"], 2)


if __name__ == "__main__":
    unittest.main()
