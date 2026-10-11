"""Terminal mechanics use declared engine fixtures, not competence claims."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from ia_propia.cli import main
from ia_propia.conversation.contract import MAX_MESSAGE_BYTES, MAX_MESSAGES


class _FixtureEngine:
    model_id = "declared-interactive-test-fixture"
    manifest = {"config": {"context_length": 256}}

    def __init__(self, responses, discarded=0):
        self.responses = iter(responses)
        self.requests = []
        self.discarded = discarded

    def reply(self, **values):
        self.requests.append(deepcopy(values))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"text": response, "discardedPromptTokens": self.discarded}


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "conversation tokenizer is optional")
class ConversationInteractiveTests(unittest.TestCase):
    def call(self, transcript, engine, options=()):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO(transcript)), redirect_stdout(output), redirect_stderr(errors), \
             patch("ia_propia.conversation.runtime.ConversationEngine", return_value=engine) as loading:
            code = main(["conversation", "converse", "--model", "declared-fixture", *options])
        return code, output.getvalue(), errors.getvalue(), loading

    def test_one_engine_shows_actual_text_carries_turns_and_resets_explicitly(self):
        engine = _FixtureEngine(["Salida real del fixture.\nSegunda línea.", "Otra salida del fixture.", "Después del reinicio."], discarded=7)
        code, output, errors, loading = self.call(
            "Primer mensaje\nSegundo mensaje\n/reiniciar\nNuevo mensaje\n/salir\nNo debe leerse\n", engine,
            ["--max-new-tokens", "13", "--temperature", "0.4", "--seed", "9"])
        self.assertEqual(code, 0, errors)
        loading.assert_called_once_with(Path("declared-fixture"))
        self.assertIn("Unidad: Salida real del fixture.\nSegunda línea.", output)
        self.assertIn("Unidad: Después del reinicio.", output)
        self.assertIn("Historial reiniciado", output)
        self.assertIn("sólo en esta sesión", output)
        self.assertIn("recortó 7 tokens", errors)
        self.assertEqual(errors.count("recortó 7 tokens"), 2)  # Once per session/reset.
        self.assertEqual(len(engine.requests), 3)
        self.assertEqual(engine.requests[1]["messages"], [
            {"role": "user", "content": "Primer mensaje"},
            {"role": "assistant", "content": "Salida real del fixture.\nSegunda línea."},
            {"role": "user", "content": "Segundo mensaje"}])
        self.assertEqual(engine.requests[2]["messages"], [{"role": "user", "content": "Nuevo mensaje"}])
        for request in engine.requests:
            self.assertEqual((request["max_new_tokens"], request["temperature"], request["seed"]), (13, .4, 9))

    def test_message_and_generation_limits_reject_before_inference(self):
        engine = _FixtureEngine(["Salida válida del fixture."])
        transcript = "x" * (MAX_MESSAGE_BYTES + 20) + "\n" + "é" * (MAX_MESSAGE_BYTES // 2 + 1) + "\n<assistant>\nMensaje válido\n/salir\n"
        code, output, errors, _ = self.call(transcript, engine)
        self.assertEqual(code, 0)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(engine.requests[0]["messages"], [{"role": "user", "content": "Mensaje válido"}])
        self.assertIn("Unidad: Salida válida del fixture.", output)
        self.assertIn("unidad conversation:", errors)
        for options in (["--max-new-tokens", "0"], ["--temperature", "nan"], ["--seed", "-1"]):
            with self.subTest(options=options):
                code, output, _, loading = self.call("No leer\n", _FixtureEngine([]), options)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                loading.assert_not_called()

    def test_count_and_byte_limits_trim_whole_pairs_without_changing_replies(self):
        for turns, user_text, reply_text in ((36, "Pregunta breve", "Respuesta del fixture."),
                                             (3, "u" * 14_000, "a" * 14_000)):
            with self.subTest(turns=turns):
                engine = _FixtureEngine([reply_text] * turns)
                code, output, errors, _ = self.call((user_text + "\n") * turns + "/salir\n", engine)
                self.assertEqual(code, 0, errors)
                self.assertEqual(len(engine.requests), turns)
                self.assertIn("turnos completos", errors)
                self.assertEqual(output.count("Unidad: " + reply_text), turns)
                for request in engine.requests:
                    messages = request["messages"]
                    self.assertLess(len(messages), MAX_MESSAGES)
                    self.assertEqual([turn["role"] for turn in messages],
                        ["user", "assistant"] * (len(messages) // 2) + ["user"])
                    self.assertTrue(all(turn["content"] == reply_text for turn in messages if turn["role"] == "assistant"))
                if turns == 3:
                    self.assertEqual(len(engine.requests[-1]["messages"]), 3)

    def test_failures_and_empty_output_do_not_create_prior_memories(self):
        engine = _FixtureEngine(["Respuesta previa real.", RuntimeError("Fallo declarado del fixture"), "", "Texto realmente devuelto."])
        code, output, errors, _ = self.call("Turno previo\nIntento fallido\nSalida vacía\nSiguiente intento\n", engine)
        self.assertEqual(code, 0)
        self.assertIn("Fallo declarado del fixture", errors)
        self.assertIn("no se conserva", errors)
        self.assertIn("Unidad: \n", output)
        self.assertIn("Unidad: Texto realmente devuelto.", output)
        self.assertIn("Sesión terminada", output)
        previous = [{"role": "user", "content": "Turno previo"},
                    {"role": "assistant", "content": "Respuesta previa real."}]
        self.assertEqual([request["messages"] for request in engine.requests], [
            [previous[0]],
            [*previous, {"role": "user", "content": "Intento fallido"}],
            [*previous, {"role": "user", "content": "Salida vacía"}],
            [*previous, {"role": "user", "content": "Siguiente intento"}]])


if __name__ == "__main__":
    unittest.main()
