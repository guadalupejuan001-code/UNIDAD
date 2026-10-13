"""No ML imports: prevent gold leakage and exercise bounded real subprocess stop."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location(
    "conversation_pattern_probe", Path(__file__).resolve().parents[1] / "examples/run_conversation_pattern_probe.py")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def fixtures():
    records, cases = {}, []
    for split in ("train", "validation"):
        for branch, value in (("a", "Veldoria"), ("b", "Oltavia")):
            case_id = split + "-" + branch
            initial = [{"role": "user", "content": "Mi lugar es " + value},
                       {"role": "assistant", "content": "Anotado."}]
            turns = ["¿Cuál es mi lugar?", "Repítelo."]
            expected = [value, value]
            ids, history = [], deepcopy(initial)
            for index, (user, gold) in enumerate(zip(turns, expected)):
                identifier = case_id + "-" + str(index)
                ids.append(identifier)
                history += [{"role": "user", "content": user}, {"role": "assistant", "content": gold}]
                records[identifier] = {"id": identifier, "groupId": split + "-family", "split": split,
                                       "source": {"dataset": "declared-unit-fixture", "kind": "authored-synthetic"},
                                       "messages": deepcopy(history)}
            cases.append({"id": case_id, "groupId": split + "-family", "split": split,
                          "competency": "facts", "contrastPairId": split + "-pair", "branch": branch,
                          "initialMessages": initial, "userTurns": turns, "expectedResponses": expected,
                          "recordIds": ids})
    return records, cases


class FixtureEngine:
    """Declared test double; this is never wired into the CLI's runtime."""
    manifest = {"config": {"context_length": 2048}}
    def __init__(self):
        self.inputs, self.scored = [], []
    def reply(self, messages, **generation):
        self.inputs.append(deepcopy(messages))
        return {"text": "REAL WRONG OUTPUT " + str(len(self.inputs)), "stopReason": "eos"}
    def score_reply(self, messages, text):
        self.scored.append((deepcopy(messages), text))
        return {"totalLogProbability": -3, "meanLogProbability": -1, "includesEos": True}


class PatternProbeTests(unittest.TestCase):
    def setUp(self):
        self.records, self.cases = fixtures()
        self.engine = FixtureEngine()
    def run_mode(self, selected, mode, **options):
        return HELPER.generate_cases(self.engine, None, self.records, selected, self.cases, mode,
            {"max_new_tokens": 8, "temperature": 0, "seed": 42},
            encode_prompt=lambda *values: ([1, 4, 5], 0), **options)

    def test_last_target_is_not_sent_to_generation_but_likelihood_scores_it_separately(self):
        case = self.cases[2]
        original = deepcopy(self.records)
        rows = self.run_mode([case], "validation-prefix")
        self.assertEqual(self.engine.inputs[0], self.records[case["recordIds"][-1]]["messages"][:-1])
        self.assertEqual(self.engine.inputs[0][-1], {"role": "user", "content": "Repítelo."})
        self.assertEqual(self.engine.scored[0][1], "Veldoria")
        self.assertEqual(self.engine.scored[1][1], "Oltavia")
        self.assertEqual(rows[0]["historyPolicy"], "ideal_prefix_diagnostic")
        self.assertEqual(self.records, original)

    def test_free_history_preserves_raw_wrong_reply_instead_of_correct_target(self):
        rows = self.run_mode([self.cases[2]], "validation-history", likelihood=False)
        self.assertIn({"role": "assistant", "content": "REAL WRONG OUTPUT 1"}, self.engine.inputs[1])
        self.assertNotIn({"role": "assistant", "content": "Veldoria"}, self.engine.inputs[1])
        self.assertEqual(rows[1]["historyPolicy"], "actual_generated_after_declared_initial_context")
        self.assertEqual(rows[1]["initialContextAssistantCount"], 1)

    def test_same_query_different_history_is_actually_sent_to_model(self):
        rows = self.run_mode(self.cases[2:], "validation-prefix", likelihood=False)
        self.assertEqual(self.engine.inputs[0][-1], self.engine.inputs[1][-1])
        self.assertNotEqual(self.engine.inputs[0], self.engine.inputs[1])
        self.assertNotEqual(rows[0]["inputSha256"], rows[1]["inputSha256"])
        self.assertFalse(any(row["narrowMetric"]["exactTextAndEos"] for row in rows))

    def test_contrast_mode_skips_equal_gold_at_final_turn(self):
        for case in self.cases[2:]:
            case["expectedResponses"][-1] = "Same final answer"
            self.records[case["recordIds"][-1]]["messages"][-1]["content"] = "Same final answer"
        rows = self.run_mode(self.cases[2:], "validation-prefix", likelihood=False)
        self.assertEqual([row["turn"] for row in rows], [0, 0])
        self.assertNotEqual(rows[0]["expectedText"], rows[1]["expectedText"])

    def test_split_selection_cannot_include_train_in_generalization(self):
        selected = HELPER.select_cases(self.cases, "validation-prefix", max_turns=2)
        self.assertTrue(all(case["split"] == "validation" for case in selected))
        with self.assertRaises(ValueError):
            HELPER.select_cases(self.cases, "validation-prefix", case_ids=["train-a"])
        with self.assertRaises(ValueError):
            HELPER.select_cases(self.cases, "validation-prefix", max_turns=25)

    def test_corpus_and_cases_must_correspond_and_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, cases, manifest = (root / name for name in ("corpus.jsonl", "cases.json", "manifest.json"))
            corpus.write_text("".join(json.dumps(row) + "\n" for row in self.records.values()))
            cases.write_text(json.dumps({"schemaVersion": 1, "cases": self.cases}))
            manifest.write_text(json.dumps({"corpusSha256": HELPER.digest(corpus)}))
            HELPER.read_inputs(corpus, cases, manifest)
            changed = deepcopy(self.cases)
            changed[0]["expectedResponses"][1] = "Leaked replacement"
            cases.write_text(json.dumps({"schemaVersion": 1, "cases": changed}))
            with self.assertRaisesRegex(ValueError, "canonical"):
                HELPER.read_inputs(corpus, cases, manifest)
            test = deepcopy(next(iter(self.records.values())))
            test["split"] = "test"
            corpus.write_text(json.dumps(test) + "\n")
            with self.assertRaisesRegex(ValueError, "TEST"):
                HELPER.read_inputs(corpus, cases, manifest)

    def test_output_budget_and_exclusive_creation_preserve_old_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            HELPER.write_new(root, "record.json", {"actual": "kept"}, 1024)
            original = (root / "record.json").read_bytes()
            with self.assertRaises(FileExistsError):
                HELPER.write_new(root, "record.json", {"actual": "changed"}, 1024)
            with self.assertRaisesRegex(ValueError, "budget"):
                HELPER.write_new(root, "extra.json", {"actual": "x" * 2000}, 1024)
            self.assertEqual((root / "record.json").read_bytes(), original)
            self.assertFalse((root / "extra.json").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux bounded worker contract")
    def test_real_worker_stops_on_wall_limit_and_keeps_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "partial.json"
            command = [sys.executable, "-c", "from pathlib import Path; import time; Path(" + repr(str(marker)) + ").write_text('actual partial'); time.sleep(30)"]
            limits = SimpleNamespace(max_seconds=1, max_ram_mib=512, max_output_mib=4)
            receipt = HELPER.supervise(command, root, limits)
            self.assertEqual(receipt["termination"], "wall_time_budget")
            self.assertLess(receipt["elapsedSeconds"], 5)
            self.assertLess(receipt["returnCode"], 0)
            self.assertEqual(marker.read_text(), "actual partial")


if __name__ == "__main__":
    unittest.main()
