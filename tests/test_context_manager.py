"""Context orchestration checks with explicit counting/ranking fixtures."""

import math
from pathlib import Path
import tempfile
import unittest

from ia_propia.context.manager import ContextManager
from ia_propia.context.store import ContextStore


class CountingEngine:
    model_id = "token-count-fixture"

    def prompt_usage(self, messages):
        tokens = 2 + sum(2 + math.ceil(len(row["content"]) / 4) for row in messages)
        return {"tokens": tokens, "capacity": 512, "discarded": max(0, tokens - 512)}


class RankingFixture:
    model_id = "ranking-fixture-not-trained"

    def score_candidates(self, query, candidates, *, thread=""):
        return [{**row, "score": 1.0 if "Lucerna" in row["text"] else 0.0} for row in candidates]


class ContextManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ContextStore(Path(self.directory.name) / "context.db", product_id="fixture", session_id="one")
        self.manager = ContextManager(self.store, RankingFixture())
        self.addCleanup(self.manager.close)
        self.engine = CountingEngine()

    def prepare(self, messages):
        return self.manager.prepare([{"role": "system", "content": "Instrucción original."}, *messages], self.engine, 32)

    def test_recovers_old_source_after_rolling_history_and_keeps_exact_request(self):
        original = "El taller Lucerna abre el jueves."
        self.store.append([{"role": "user", "content": original}])
        for batch in range(5):
            self.store.append([{"role": "user", "content": f"Distractor {batch}-{i}."} for i in range(64)])
        request = {"role": "user", "content": "¿Cuándo abre Lucerna?"}
        result = self.prepare([request])
        self.assertIn(original, result.messages[0]["content"])
        self.assertEqual(result.messages[-1], request)
        self.assertEqual(result.metadata["status"], "retrieved")
        self.assertEqual(result.metadata["candidateCount"] <= 48, True)
        self.assertLessEqual(result.metadata["promptTokens"] + 32, 512)

    def test_compacts_whole_turns_preserves_system_and_reports_sources(self):
        history = [{"role": "user", "content": "El taller Lucerna abre el jueves."}]
        history += [{"role": "user", "content": "Conversación extensa de prueba. " * 6} for _ in range(30)]
        history += [{"role": "user", "content": "¿Cuándo abre Lucerna?"}]
        result = self.prepare(history)
        self.assertTrue(result.messages[0]["content"].startswith("Instrucción original."))
        self.assertIn(history[0]["content"], result.messages[0]["content"])
        self.assertEqual(result.messages[-1], history[-1])
        self.assertEqual(result.metadata["status"], "compacted")
        self.assertGreater(result.metadata["omittedMessages"], 0)
        self.assertTrue(result.metadata["sources"])
        self.assertNotIn("text", result.metadata["sources"][0])
        self.assertLessEqual(result.metadata["promptTokens"] + 32, 512)

    def test_rolling_overlap_does_not_duplicate_and_repeated_new_turn_is_distinct(self):
        first = {"role": "user", "content": "Pregunta inicial."}
        self.prepare([first])
        self.manager.observe_reply("Respuesta realmente generada.")
        second = {"role": "user", "content": "Pregunta posterior."}
        self.prepare([first, {"role": "assistant", "content": "Respuesta realmente generada."}, second])
        self.assertEqual(self.store.count(), 3)
        self.manager.observe_reply("Otra respuesta.")
        self.prepare([second])
        self.assertEqual(self.store.count(), 5)

    def test_assistant_output_is_not_fact_and_reset_removes_conversation(self):
        self.manager.observe_reply("El taller Lucerna abre el sábado; esto es salida no verificada.")
        result = self.prepare([{"role": "user", "content": "¿Cuándo abre Lucerna?"}])
        self.assertNotIn("sábado", result.messages[0]["content"])
        self.manager.reset()
        self.assertEqual(self.store.count(), 0)

    def test_oversized_current_request_is_kept_whole_and_overflow_is_explicit(self):
        request = {"role": "user", "content": "texto largo " * 250}
        result = self.prepare([request])
        self.assertEqual(result.messages[-1], request)
        self.assertEqual(result.metadata["status"], "current_message_exceeds_budget")
        self.assertEqual(result.metadata["sourceIds"], [])


if __name__ == "__main__":
    unittest.main()
