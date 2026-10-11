"""Pruebas de la integración del CLI con un espacio temporal de ejemplo."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.cli import main


class CliTests(unittest.TestCase):
    def call(self, arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = main(arguments)
        return result, output.getvalue(), error.getvalue()

    def json_call(self, arguments):
        result, output, error = self.call(arguments)
        self.assertEqual(result, 0, error)
        return json.loads(output)

    def test_help_shows_the_complete_public_command(self):
        for prefix in ([], ["language"], ["trend"], ["brain"], ["conversation"]):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                main([*prefix, "--help"])
            self.assertEqual(stopped.exception.code, 0)
            command = " ".join(["unidad", *prefix])
            self.assertIn(f"usage: {command} ", output.getvalue())

    def test_goal_attempt_correction_export_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            flags = ["--workspace", str(directory / "private")]
            self.json_call(["init", *flags])
            task = self.json_call(["task", "add", *flags,
                                   "--goal", "Preparar una propuesta de ejemplo",
                                   "--context", "Datos ficticios usados solo para pruebas"])
            attempt = self.json_call(["task", "attempt", *flags, "--id", task["id"],
                                      "--text", "Propuesta de prueba con una referencia",
                                      "--evidence", "https://example.org/fixture"])
            self.json_call(["task", "feedback", *flags, "--id", task["id"],
                            "--attempt", attempt["id"], "--text", "Aprobado como ejemplo",
                            "--accepted"])
            listing = self.json_call(["task", "list", *flags])
            self.assertEqual(listing["tasks"][0]["id"], task["id"])
            self.assertEqual(listing["tasks"][0]["attemptCount"], 1)
            self.assertNotIn("attempts", listing["tasks"][0])
            self.assertNotIn("context", listing["tasks"][0])
            shown = self.json_call(["task", "show", *flags, "--id", task["id"]])
            self.assertEqual(shown["id"], task["id"])
            destination = directory / "corpus.txt"
            exported = self.json_call(["task", "export", *flags, "--output", str(destination)])
            self.assertEqual(exported["sampleCount"], 1)
            self.assertIn("Propuesta de prueba", destination.read_text(encoding="utf-8"))
            self.json_call(["task", "feedback", *flags, "--id", task["id"],
                            "--attempt", attempt["id"], "--text", "Cambio de criterio",
                            "--rejected"])
            result, _, error = self.call(["task", "export", *flags,
                                          "--output", str(directory / "rejected.txt")])
            self.assertEqual(result, 2)
            self.assertTrue(error)
            self.assertFalse((directory / "rejected.txt").exists())

    def test_missing_workspace_returns_error_without_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            result, _, error = self.call(["task", "list", "--workspace", str(missing)])
            self.assertEqual(result, 2)
            self.assertTrue(error)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
