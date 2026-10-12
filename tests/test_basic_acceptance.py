"""Acceptance-policy regressions using fabricated records, never reserved cases."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/run_basic_acceptance.py"
spec = importlib.util.spec_from_file_location("basic_acceptance", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BasicAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cases = {"suiteId": "synthetic-regression-only", "scenarios": []}
        for i in range(10):
            self.cases["scenarios"].append({"id": f"dialog-{i}", "steps": [
                {"id": f"d{i}-t{j}", "user": f"Entrada ficticia {i}, {j}.", "critical": False}
                for j in range(4)]})
        self.rubric = {"suiteId": self.cases["suiteId"], "acceptance": {
            "expectedTurns": 40, "minimumFullyMetTurns": 36,
            "minimumFullyMetDialogs": 8}, "allowedClaimIfAccepted": "Synthetic fixture only",
            "limits": ["Not a model evaluation"]}
        self.failed = set()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def fixture(self):
        self.write("cases.json", self.cases)
        self.write("rubric.json", self.rubric)
        frozen = {"suiteId": self.cases["suiteId"], "files": {
            name: {"sha256": runner.digest(self.root / name)}
            for name in ("cases.json", "rubric.json")}}
        self.write("freeze.json", frozen)
        outputs, turns = [], []
        for scenario in self.cases["scenarios"]:
            history = []
            for step in scenario["steps"]:
                history.append({"role": "user", "content": step["user"]})
                reply = {"modelId": "synthetic-model", "text": f"Texto ficticio {step['id']}."}
                outputs.append({"id": step["id"], "scenarioId": scenario["id"],
                    "modelId": "synthetic-model", "inputMessages": deepcopy(history), "reply": reply,
                    "effectiveContext": {"discardedPromptTokens": 0}, "error": None})
                history.append({"role": "assistant", "content": reply["text"]})
                ok = step["id"] not in self.failed
                turns.append({"id": step["id"], "relevance": ok, "contextGrounding": True,
                    "spanish": True, "criticalCorrect": True if step["critical"] else None,
                    "note": "Fallo ficticio de prueba." if not ok else ""})
        self.outputs = {"suiteId": self.cases["suiteId"], "modelId": "synthetic-model",
            "suiteFreeze": frozen, "outputs": outputs}
        self.review = {"suiteId": self.cases["suiteId"], "modelId": "synthetic-model",
            "reviewer": "unit-test", "method": "fabricated booleans", "turns": turns}
        self.save_evidence()

    def save_evidence(self):
        path = self.write("outputs.json", self.outputs)
        self.review["outputsSha256"] = runner.digest(path)
        self.write("review.json", self.review)

    def assess(self):
        with redirect_stdout(io.StringIO()):
            runner.assess(self.root, self.root / "outputs.json", self.root / "review.json",
                          self.root / "assessment.json")
        return json.loads((self.root / "assessment.json").read_text())

    def test_current_36_turns_and_eight_whole_dialogs_can_pass(self):
        self.failed = {"d8-t0", "d8-t1", "d9-t0", "d9-t1"}
        self.fixture()
        result = self.assess()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["fullyMetTurns"], 36)
        self.assertEqual(result["fullyMetDialogs"], 8)
        self.assertEqual(result["totalDialogs"], 10)

    def test_36_turns_scattered_across_six_whole_dialogs_must_fail(self):
        self.failed = {f"d{i}-t0" for i in range(4)}
        self.fixture()
        result = self.assess()
        self.assertEqual(result["fullyMetTurns"], 36)
        self.assertEqual(result["fullyMetDialogs"], 6)
        self.assertFalse(result["accepted"])

    def test_one_critical_failure_blocks_acceptance(self):
        self.cases["scenarios"][0]["steps"][0].update(critical=True, criticalExpected="Dato ficticio")
        self.fixture()
        self.review["turns"][0].update(criticalCorrect=False, note="Fallo crítico ficticio")
        self.save_evidence()
        result = self.assess()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["criticalFailures"], ["d0-t0"])
        self.assertEqual(result["fullyMetTurns"], 39)
        self.assertEqual(result["fullyMetDialogs"], 9)

    def test_historical_30_turn_suite_cannot_claim_current_acceptance(self):
        for row in self.cases["scenarios"]:
            row["steps"].pop()
        self.rubric["acceptance"].update(expectedTurns=30, minimumFullyMetTurns=27)
        self.fixture()
        with self.assertRaises(ValueError):
            runner.read_suite(self.root)

    def test_both_cases_and_rubric_must_be_frozen(self):
        self.fixture()
        frozen = json.loads((self.root / "freeze.json").read_text())
        frozen["files"].pop("rubric.json")
        self.write("freeze.json", frozen)
        with self.assertRaisesRegex(ValueError, "freeze"):
            runner.read_suite(self.root)

    def test_duplicate_scenario_ids_rejected(self):
        self.cases["scenarios"][1]["id"] = "dialog-0"
        self.fixture()
        with self.assertRaisesRegex(ValueError, "scenario"):
            runner.read_suite(self.root)

    def test_turn_id_cannot_escape_output_directory(self):
        self.cases["scenarios"][0]["steps"][0]["id"] = "../escaped"
        self.fixture()
        with self.assertRaisesRegex(ValueError, "identifier"):
            runner.read_suite(self.root)

    def test_frozen_paths_cannot_escape_suite(self):
        self.fixture()
        frozen = json.loads((self.root / "freeze.json").read_text())
        frozen["files"]["../outside.json"] = {"sha256": "0" * 64}
        self.write("freeze.json", frozen)
        with self.assertRaisesRegex(ValueError, "filename"):
            runner.read_suite(self.root)

    def test_empty_reply_cannot_be_marked_success(self):
        self.fixture()
        # Last turn avoids relying on history validation to detect the blank.
        self.outputs["outputs"][-1]["reply"]["text"] = "  "
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "reply"):
            self.assess()

    def test_missing_reply_cannot_be_marked_success(self):
        self.fixture()
        self.outputs["outputs"][-1]["reply"] = None
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "reply"):
            self.assess()

    def test_foreign_scenario_assignment_rejected(self):
        self.fixture()
        self.outputs["outputs"][-1]["scenarioId"] = "dialog-0"
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "scenario"):
            self.assess()

    def test_tampered_real_history_rejected(self):
        self.fixture()
        self.outputs["outputs"][1]["inputMessages"][1]["content"] = "Invented ideal reply"
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, "actual generated"):
            self.assess()

    def test_weakened_threshold_rejected(self):
        self.rubric["acceptance"]["minimumFullyMetTurns"] = 1
        self.fixture()
        with self.assertRaisesRegex(ValueError, "threshold"):
            runner.read_suite(self.root)

    def test_truncated_context_blocks_acceptance(self):
        self.fixture()
        self.outputs["outputs"][-1]["effectiveContext"]["discardedPromptTokens"] = 1
        self.save_evidence()
        result = self.assess()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["technicalFailures"], ["d9-t3"])

    def test_existing_assessment_is_never_overwritten(self):
        self.fixture()
        (self.root / "assessment.json").write_text("KEEP", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.assess()
        self.assertEqual((self.root / "assessment.json").read_text(), "KEEP")


if __name__ == "__main__":
    unittest.main()
