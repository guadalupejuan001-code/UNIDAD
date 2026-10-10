import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from ia_propia.trend.data import DataError, chronological_split, load_observations, make_samples
from ia_propia.trend.model import (ArtifactError, digest, load_artifact, model_id,
                                        predict, train, validate_artifact, write_json)


ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000
BASE = 1_700_000_000_000
SOURCE = "fixture:views"


def observation(signal, day, value, source=SOURCE):
    at = BASE + day * DAY
    return {"id": f"{source}:{signal}:{day}", "signalId": signal, "observedAt": at,
            "text": f"Measured {value} views", "metrics": {"views": value},
            "provenance": {"source": source, "uri": "https://example.org/measured",
                           "fetchedAt": at, "contentHash": hashlib.sha256(str(value).encode()).hexdigest()},
            "privacy": "PUBLIC"}


def synthetic(days=50):
    # Three different growth rates; no pretrained weights or model-produced
    # targets. A useful learner must outperform a no-change count forecast.
    return [observation(f"signal-{signal}", day, math.expm1(2 + signal + (0.03 + signal * 0.03) * day))
            for day in range(days) for signal in range(3)]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def load(self, rows, **kwargs):
        path = self.directory / "observations.json"
        write_json(path, rows)
        return load_observations(path, source=SOURCE, metric="views", **kwargs)

    def train_fixture(self, rows=None, **kwargs):
        observations, metadata = self.load(synthetic() if rows is None else rows)
        return train(observations, metadata, source=SOURCE, metric="views", epochs=500, patience=80, **kwargs)

    def parity(self):
        return json.loads((ROOT / "tests/fixtures/parity.json").read_text())

    def test_global_split_keeps_all_signals_of_same_target_date_together(self):
        observations, _ = self.load(synthetic())
        samples, skipped = make_samples(observations)
        splits = chronological_split(samples)
        self.assertEqual(skipped, 0)
        dates = [{s.target.observed_at for s in split} for split in splits.values()]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertTrue(dates[a].isdisjoint(dates[b]))
        self.assertLess(max(dates[0]), min(dates[1]))
        self.assertLess(max(dates[1]), min(dates[2]))

    def test_horizon_embargo_keeps_future_labels_out_of_earlier_origins(self):
        observations, _ = self.load(synthetic())
        samples, _ = make_samples(observations, horizon_steps=3)
        splits = chronological_split(samples)
        for earlier, later in (("train", "validation"), ("validation", "test")):
            self.assertLessEqual(max(s.target.observed_at for s in splits[earlier]),
                                 min(s.inputs[-1].observed_at for s in splits[later]))
        self.assertLess(sum(map(len, splits.values())), len(samples))

    def test_heldout_outcomes_do_not_change_scaler_weights_or_checkpoint(self):
        rows = synthetic()
        model, _ = self.train_fixture(rows)
        changed = copy.deepcopy(rows)
        changed[-1]["metrics"]["views"] = 1e100
        changed_model, _ = self.train_fixture(changed)
        for key in ("featureMean", "featureScale", "weightsInput", "biasHidden", "weightsOutput", "biasOutput", "modelId"):
            self.assertEqual(model[key], changed_model[key])
        self.assertEqual(model["training"]["bestEpoch"], changed_model["training"]["bestEpoch"])
        self.assertNotEqual(model["evaluation"]["splits"]["test"], changed_model["evaluation"]["splits"]["test"])

    def test_training_learns_growth_and_changes_random_weights(self):
        model, _ = self.train_fixture()
        self.assertGreater(model["training"]["bestEpoch"], 0)
        final_parameters = [model["weightsInput"], model["biasHidden"], model["weightsOutput"], [model["biasOutput"]]]
        self.assertNotEqual(digest(final_parameters), model["training"]["initialWeightsSha256"])
        for split in ("train", "test"):
            metrics = model["evaluation"]["splits"][split]
            self.assertLess(metrics["model"]["rmseLogGrowth"], metrics["persistence"]["rmseLogGrowth"] / 4)
        self.assertFalse(model["evaluation"]["productionAccepted"])

    def test_roundtrip_artifact_preserves_prediction(self):
        model, _ = self.train_fixture()
        observations, _ = self.load(synthetic())
        origin = max(o.observed_at for o in observations)
        before = predict(model, observations, as_of=origin, historical=True)
        path = self.directory / "model.json"
        write_json(path, model)
        after = predict(load_artifact(path), observations, as_of=origin, historical=True)
        self.assertEqual(before, after)

    def test_exported_numerical_fixture_matches_shared_contract(self):
        fixture = self.parity()
        observations, _ = self.load(fixture["observations"], now_ms=400_000_000)
        result = predict(fixture["artifact"], observations, as_of=fixture["asOf"])[0]
        for key, expected in fixture["expected"].items():
            if isinstance(expected, float):
                self.assertAlmostEqual(result[key], expected, places=12)
            else:
                self.assertEqual(result[key], expected)

    def test_rejects_private_missing_privacy_and_inherited_private_signal(self):
        for privacy in ("PRIVATE", None):
            rows = synthetic()
            rows[0]["privacy"] = privacy
            with self.assertRaises(DataError):
                self.load(rows)
        rows = synthetic()
        state = {"schemaVersion": 1, "observations": {r["id"]: r for r in rows},
                 "signals": {f"signal-{i}": {"id": f"signal-{i}", "privacy": "PRIVATE" if i == 0 else "PUBLIC"} for i in range(3)}}
        with self.assertRaises(DataError):
            self.load(state)
        state["signals"]["signal-0"]["privacy"] = "PUBLIC"
        observations, _ = self.load(state)
        self.assertEqual(len(observations), len(rows))
        del state["signals"]["signal-0"]
        with self.assertRaises(DataError):
            self.load(state)

    def test_private_other_source_elevates_same_signal(self):
        rows = synthetic()
        private = observation("signal-0", 100, 100, "different:source")
        private["privacy"] = "PRIVATE"
        with self.assertRaises(DataError):
            self.load(rows + [private])

    def test_source_and_metric_selection_does_not_mix_semantics(self):
        rows = synthetic()
        wrong_source = observation("unrelated", 0, 999, "other:source")
        wrong_metric = observation("another", 0, 999)
        wrong_metric["metrics"] = {"shares": 999}
        observations, metadata = self.load(rows + [wrong_source, wrong_metric])
        self.assertEqual(len(observations), len(rows))
        self.assertEqual(metadata["excludedSourceOrMetric"], 2)
        samples, _ = make_samples(observations)
        self.assertTrue(all(s.target.source == SOURCE for s in samples))

    def test_duplicate_ids_and_ambiguous_timestamps_are_rejected(self):
        rows = synthetic(8)
        observations, metadata = self.load(rows + [copy.deepcopy(rows[0])])
        self.assertEqual(metadata["deduplicatedIdenticalRows"], 1)
        self.assertEqual(len(observations), len(rows))
        conflict = copy.deepcopy(rows[0])
        conflict["metrics"]["views"] = 999
        with self.assertRaises(DataError):
            self.load(rows + [conflict])
        conflict["id"] = "new-revision-same-time"
        with self.assertRaises(DataError):
            self.load(rows + [conflict])

    def test_future_incomplete_provenance_and_invalid_numbers_are_rejected(self):
        for field, value in (("observedAt", True), ("observedAt", 10**30), ("metrics", {"views": True}),
                             ("metrics", {"views": -1}), ("privacy", "UNKNOWN")):
            rows = synthetic(8)
            rows[0][field] = value
            with self.assertRaises(DataError):
                self.load(rows)
        for field, value in (("contentHash", "not-a-hash"), ("uri", ""), ("fetchedAt", int(time.time() * 1000) + DAY)):
            rows = synthetic(8)
            rows[0]["provenance"][field] = value
            with self.assertRaises(DataError):
                self.load(rows)

    def test_missing_cadence_never_becomes_nearest_future_target(self):
        rows = [observation("single", d, d + 10) for d in range(10) if d != 4]
        observations, _ = self.load(rows)
        samples, skipped = make_samples(observations)
        self.assertEqual(skipped, 4)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].target.observed_at, BASE + 9 * DAY)

    def test_causal_prediction_ignores_later_observation(self):
        fixture = self.parity()
        observations, _ = self.load(fixture["observations"], now_ms=400_000_000)
        before = predict(fixture["artifact"], observations, as_of=fixture["asOf"])
        later = copy.deepcopy(fixture["observations"][-1])
        later.update(id="later", observedAt=432_000_000)
        later["provenance"]["fetchedAt"] = 432_000_000
        later["metrics"]["views"] = 999999
        observations, _ = self.load(fixture["observations"] + [later], now_ms=500_000_000)
        self.assertEqual(before, predict(fixture["artifact"], observations, as_of=fixture["asOf"]))

    def test_late_fetch_requires_explicit_historical_replay(self):
        fixture = self.parity()
        for row in fixture["observations"]:
            row["provenance"]["fetchedAt"] = fixture["asOf"] + DAY
        observations, _ = self.load(fixture["observations"], now_ms=500_000_000)
        with self.assertRaises(DataError):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"])
        result = predict(fixture["artifact"], observations, as_of=fixture["asOf"], historical=True)
        self.assertTrue(result[0]["historicalReplay"])
        self.assertFalse(result[0]["liveForecastValidated"])

    def test_model_fitting_outcomes_cannot_appear_in_earlier_forecast(self):
        fixture = self.parity()
        observations, _ = self.load(fixture["observations"], now_ms=400_000_000)
        fixture["artifact"]["training"]["fitCutoffAt"] = fixture["asOf"] + DAY
        fixture["artifact"]["training"]["trainedAt"] = fixture["asOf"] + 2 * DAY
        with self.assertRaisesRegex(DataError, "fit/select"):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"], historical=True)
        # A late asOf does not justify using a stale origin preceding fitted
        # outcomes; historical replay must also respect the actual window origin.
        with self.assertRaisesRegex(DataError, "Window origin"):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"] + 3 * DAY, historical=True)
        fixture["artifact"]["training"]["fitCutoffAt"] = 0
        fixture["artifact"]["training"]["trainedAt"] = int(time.time() * 1000) + DAY
        with self.assertRaisesRegex(DataError, "training timestamp"):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"], historical=True)

    def test_stale_live_forecast_is_rejected(self):
        fixture = self.parity()
        observations, _ = self.load(fixture["observations"], now_ms=400_000_000)
        with self.assertRaisesRegex(DataError, "stale"):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"] + DAY)

    def test_invalid_artifacts_and_weight_tampering_are_rejected(self):
        artifact = self.parity()["artifact"]
        duplicate_key = self.directory / "duplicate-key-model.json"
        duplicate_key.write_text(json.dumps(artifact)[:-1] + ',"metric":"views"}', encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "Duplicate JSON"):
            load_artifact(duplicate_key)
        for key, value in (("featureVersion", "unknown"), ("featureScale", [0] * 4),
                           ("weightsInput", [[1]]), ("weightsOutput", [True] * 8),
                           ("biasOutput", float("nan")), ("schemaVersion", True)):
            changed = copy.deepcopy(artifact)
            changed[key] = value
            with self.assertRaises(ArtifactError):
                validate_artifact(changed)
        artifact["weightsOutput"][0] += 0.01
        with self.assertRaisesRegex(ArtifactError, "modelId"):
            validate_artifact(artifact)

    def test_overflow_is_rejected_before_tanh_saturation_or_value_conversion(self):
        fixture = self.parity()
        observations, _ = self.load(fixture["observations"], now_ms=400_000_000)
        for change in ("normalization", "hidden", "value"):
            artifact = copy.deepcopy(fixture["artifact"])
            if change == "normalization":
                artifact["featureScale"][0] = 1e-320
            elif change == "hidden":
                artifact["weightsInput"][0] = [1e308, -1e308, 1e308, 1e308]
            else:
                artifact["biasOutput"] = 1000
            artifact["modelId"] = model_id(artifact)
            with self.assertRaises(DataError):
                predict(artifact, observations, as_of=fixture["asOf"])

    def test_direct_api_cannot_reuse_values_loaded_as_another_metric(self):
        fixture = self.parity()
        rows = fixture["observations"]
        for row in rows:
            row["metrics"]["shares"] = 1
        path = self.directory / "rows.json"
        write_json(path, rows)
        observations, _ = load_observations(path, source=SOURCE, metric="shares", now_ms=400_000_000)
        with self.assertRaises(DataError):
            predict(fixture["artifact"], observations, as_of=fixture["asOf"])

    def test_real_archived_observations_smoke_cli_and_baseline(self):
        model_path = self.directory / "real-model.json"
        report_path = self.directory / "real-report.json"
        result = subprocess.run([sys.executable, "-m", "ia_propia.trend", "train",
                                 "--input", "examples/attention-observations.json",
                                 "--source", "wikimedia:en.wikipedia.org", "--metric", "views",
                                 "--output", str(model_path), "--report", str(report_path)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        model = load_artifact(model_path)
        self.assertEqual(model["training"]["sampleCount"], 22)
        self.assertEqual([s["sampleCount"] for s in model["training"]["split"].values()], [12, 4, 6])
        self.assertGreater(model["training"]["bestEpoch"], 0)
        self.assertTrue(model["evaluation"]["retrospectiveEvaluation"])
        report = json.loads(report_path.read_text())
        self.assertIn("persistence", report["evaluation"]["splits"]["test"])
        result = subprocess.run([sys.executable, "-m", "ia_propia.trend", "predict",
                                 "--model", str(model_path), "--input", "examples/attention-observations.json",
                                 "--as-of", "1729036800000", "--historical"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["forecasts"]), 2)


if __name__ == "__main__":
    unittest.main()
