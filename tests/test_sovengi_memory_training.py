"""Dataset invariants; these tests make no claims about learned conversation."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sovengi_memory_training", ROOT / "examples/train_sovengi_memory.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class MemoryTrainingDataTests(unittest.TestCase):
    def test_families_are_disjoint_and_all_records_validate(self):
        rows, probes = experiment.authored_cases()
        groups, ids = {}, set()
        for row in rows:
            experiment.validate_record(row)
            self.assertNotIn(row["id"], ids)
            ids.add(row["id"])
            self.assertEqual(groups.setdefault(row["groupId"], row["split"]), row["split"])
        self.assertTrue(all(p["groupId"] not in groups for p in probes))
        names = [set(value) for value in experiment.NAMES.values()]
        self.assertTrue(all(not a & b for i, a in enumerate(names) for b in names[i+1:]))
        self.assertEqual({r["split"] for r in rows}, {"train", "validation"})

    def test_memory_queries_have_counterfactual_answers_and_no_gold_in_user_prompt(self):
        rows, probes = experiment.authored_cases()
        pairs = {}
        for row in rows:
            if row["id"].endswith("/0"):
                pairs.setdefault(row["groupId"], []).append(row)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["messages"][-2], pair[1]["messages"][-2])
            self.assertNotEqual(pair[0]["messages"][-1], pair[1]["messages"][-1])
        for probe in probes:
            self.assertEqual(len(probe["messages"]), 1)
            self.assertNotIn(probe["expected"], probe["messages"][0]["content"])

    def test_prompt_matches_actual_retrieval_contract_and_is_deterministic(self):
        note, query = "El taller Arce abre el lunes.", "¿Cuándo abre el taller Arce?"
        prompt = experiment.memory_prompt(note, query, "fixture")
        self.assertEqual(prompt, experiment.memory_prompt(note, query, "fixture"))
        self.assertIn(experiment.MEMORY_NOTICE, prompt)
        self.assertIn(note, prompt)
        self.assertIn('"field": "goal"', prompt)


if __name__ == "__main__":
    unittest.main()
