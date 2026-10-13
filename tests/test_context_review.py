"""Regression checks from the independent context integration review.

All text is fictional. These tests verify source and dialogue handling, without
claiming that a generated answer understands the restored conversation.
"""

from pathlib import Path
import tempfile
import unittest

from ia_propia.context.manager import ContextManager
from ia_propia.context.store import ContextStore
from tests.test_context_manager import CountingEngine


SYSTEM = {"role": "system", "content": "Eres SOVENGI. Conserva esta instrucción."}


class ContextReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "context.sqlite3"
        self.managers = []
        self.addCleanup(self.close_managers)
        self.engine = CountingEngine()

    def close_managers(self):
        for manager in self.managers:
            manager.close()

    def manager(self):
        manager = ContextManager(ContextStore(self.path, product_id="review", session_id="fictional"))
        self.managers.append(manager)
        return manager

    def test_retrieves_unique_old_fact_for_natural_question_without_a_correction(self):
        manager = self.manager()
        original = "El taller Lucerna abre el jueves."
        manager.store.append([{"role": "user", "content": original}])
        for batch in range(3):
            manager.store.append([{"role": "user", "content": f"Distractor ficticio {batch}-{index}."}
                                  for index in range(64)])
        request = {"role": "user", "content": "¿Qué día abre el taller Lucerna?"}
        result = manager.prepare([SYSTEM, request], self.engine, 32)
        self.assertIn(original, result.messages[0]["content"])
        self.assertEqual(result.messages[-1], request)
        self.assertIn("context:1", result.metadata["sourceIds"])
        self.assertLessEqual(result.metadata["promptTokens"] + 32, 512)

    def test_full_snapshot_retry_does_not_make_old_sources_new_again(self):
        manager = self.manager()
        first = {"role": "user", "content": "El ensayo ficticio será el jueves."}
        answer = {"role": "assistant", "content": "Revisemos ese ensayo ficticio."}
        correction = {"role": "user", "content": "Corrijo: el ensayo ficticio será el lunes."}
        manager.prepare([SYSTEM, first], self.engine, 32)
        manager.observe_reply(answer["content"])
        snapshot = [SYSTEM, first, answer, correction]
        manager.prepare(snapshot, self.engine, 32)
        manager.observe_reply("Entendido, conservaré la corrección.")
        before = manager.store.recent(limit=64, include_assistant=True)
        retry = manager.prepare(snapshot, self.engine, 32)
        manager.observe_reply("Entendido, conservaré la corrección.")
        self.assertEqual(manager.store.recent(limit=64, include_assistant=True), before)
        self.assertTrue(retry.metadata["replayedSnapshot"])

    def test_retry_after_reopening_preserves_source_ids_and_original_ordinals(self):
        manager = self.manager()
        first = {"role": "user", "content": "Mi objetivo es revisar el ensayo ficticio."}
        answer = {"role": "assistant", "content": "Podemos revisar el ensayo mañana."}
        current = {"role": "user", "content": "Mejor revisemos el ensayo hoy."}
        manager.prepare([SYSTEM, first], self.engine, 32)
        manager.observe_reply(answer["content"])
        snapshot = [SYSTEM, first, answer, current]
        manager.prepare(snapshot, self.engine, 32)
        manager.observe_reply("Entendido, hoy.")
        before = manager.store.recent(limit=64, include_assistant=True)
        manager.close()
        resumed = self.manager()
        result = resumed.prepare(snapshot, self.engine, 32)
        resumed.observe_reply("Entendido, hoy.")
        self.assertEqual(resumed.store.recent(limit=64, include_assistant=True), before)
        self.assertTrue(result.metadata["replayedSnapshot"])

    def test_incremental_reopening_restores_assistant_as_dialogue_without_fact_promotion(self):
        manager = self.manager()
        original = {"role": "user", "content": "Busquemos un día para el ensayo ficticio."}
        generated = "Podríamos considerar el martes para ese ensayo ficticio."
        manager.prepare([SYSTEM, original], self.engine, 32)
        manager.observe_reply(generated)
        manager.close()
        resumed = self.manager()
        question = {"role": "user", "content": "¿Por qué sugeriste ese día?"}
        result = resumed.prepare([SYSTEM, question], self.engine, 32)
        self.assertIn({"role": "assistant", "content": generated}, result.messages)
        self.assertNotIn(generated, result.messages[0]["content"])
        self.assertEqual(result.messages[-1], question)
        self.assertEqual(result.metadata["restoredMessages"], 2)
        self.assertTrue(all(source["role"] != "assistant" for source in result.metadata["sources"]))


if __name__ == "__main__":
    unittest.main()
