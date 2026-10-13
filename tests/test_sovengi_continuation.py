"""Protect the stage's data boundary; these checks do not grade learned replies."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
spec = importlib.util.spec_from_file_location("sovengi_continuation", ROOT / "examples/train_sovengi_continuation.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ContinuationDataTests(unittest.TestCase):
    def test_retention_stays_in_its_original_fold_and_probes_stay_out(self):
        import json
        config = json.loads((ROOT / "configs/sovengi-dialogue-stage.json").read_text())
        original = list(experiment.load_corpus(ROOT / config["retentionCorpus"]))
        rows = original + experiment.authored_rows()
        probes = json.loads((ROOT / config["retentionProbes"]).read_text()) + experiment.development_probes()
        experiment.validate_isolation(rows, probes, original)
        self.assertEqual({r["split"] for r in rows}, {"train", "validation"})
        names = [set(value) for value in experiment.FAMILIES.values()]
        self.assertTrue(all(not a & b for i, a in enumerate(names) for b in names[i + 1:]))

    def test_rejects_changed_retention_and_leaked_development_prompt(self):
        rows = experiment.authored_rows()
        original = [deepcopy(rows[0])]
        altered = deepcopy(rows)
        altered[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            experiment.validate_isolation(altered, [], original)
        probes = experiment.development_probes()
        altered = deepcopy(rows)
        altered[0]["messages"][1] = deepcopy(probes[0]["messages"][0])
        with self.assertRaisesRegex(ValueError, "held-out"):
            experiment.validate_isolation(altered, probes, [])

    def test_declared_parent_is_the_published_trained_candidate(self):
        import json
        config = json.loads((ROOT / "configs/sovengi-dialogue-stage.json").read_text())
        previous = json.loads((ROOT / "configs/sovengi-memory-candidate.json").read_text())
        self.assertEqual(config["initialModel"], previous["brain"]["conversation_model"])
        self.assertEqual(config["expectedModelId"], previous["brain"]["expected_model_id"])
        self.assertLessEqual(config["training"]["max_seconds"], 900)
        with self.assertRaises(ValueError):
            experiment.paths({"experimentId": "../old-stage"})


if __name__ == "__main__":
    unittest.main()
