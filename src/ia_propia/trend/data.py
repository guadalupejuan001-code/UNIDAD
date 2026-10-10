"""Observation ingestion and causal windows; no trainable statistics live here."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np


class DataError(ValueError):
    """Evidence is invalid or cannot support the requested temporal experiment."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_json(value):
    def distinct_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise DataError(f"Duplicate JSON object key: {key}")
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=distinct_keys)


def timestamp(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 9_223_372_036_854_775_807:
        raise DataError(f"{name} must be a nonnegative integer timestamp in milliseconds")
    return value


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{name} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise DataError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise DataError(f"{name} must be finite")
    return result


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Observation:
    id: str
    signal_id: str
    observed_at: int
    value: float
    provenance: dict
    raw: dict

    @property
    def source(self):
        return self.provenance["source"]

    def manifest(self):
        return {"id": self.id, "signalId": self.signal_id, "observedAt": self.observed_at,
                "value": self.value, "provenance": self.provenance, "privacy": "PUBLIC"}


@dataclass(frozen=True)
class Sample:
    inputs: tuple
    target: Observation
    intermediate: tuple

    @property
    def features(self):
        return features([o.value for o in self.inputs])

    @property
    def label(self):
        return math.log1p(self.target.value) - math.log1p(self.inputs[-1].value)

    def manifest(self):
        return {"signalId": self.target.signal_id, "source": self.target.source,
                "inputIds": [o.id for o in self.inputs],
                "originAt": self.inputs[-1].observed_at,
                "targetId": self.target.id, "targetAt": self.target.observed_at,
                "intermediateIds": [o.id for o in self.intermediate]}


def features(values):
    """window-1 successive log deltas followed by the last log level."""
    logs = np.log1p(np.asarray(values, dtype=np.float64))
    return np.concatenate((np.diff(logs), logs[-1:]))


def load_observations(path, *, source, metric, now_ms=None):
    """Read existing Observation lists or schema-1 SystemState exports.

    Only selected PUBLIC observations enter the experiment. Selected private or
    unclassified observations are rejected, including inherited signal privacy.
    Conflicting identities are rejected even when one row would be filtered out.
    """
    nonempty(source, "source")
    nonempty(metric, "metric")
    now_ms = timestamp(int(time.time() * 1000) if now_ms is None else now_ms, "now")
    try:
        raw = parse_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read observations: {exc}") from exc
    return observations_from_payload(raw, source=source, metric=metric, now_ms=now_ms)


def observations_from_payload(raw, *, source, metric, now_ms=None):
    """Validate in-memory Observation JSON with the same file-import guarantees.

    This performs no file reads. Callers must validate the complete payload before
    selecting an individual signal, retaining scope privacy and ID checks.
    """
    nonempty(source, "source")
    nonempty(metric, "metric")
    now_ms = timestamp(int(time.time() * 1000) if now_ms is None else now_ms, "now")
    signals = None
    if isinstance(raw, dict):
        if type(raw.get("schemaVersion")) is not int or raw["schemaVersion"] != 1:
            raise DataError("SystemState schemaVersion must be 1")
        rows = raw.get("observations")
        if not isinstance(rows, dict):
            raise DataError("SystemState observations must be a map")
        for key, row in rows.items():
            if not isinstance(row, dict) or row.get("id") != key:
                raise DataError("SystemState observation map keys must equal observation IDs")
        signals = raw.get("signals", {})
        if not isinstance(signals, dict):
            raise DataError("SystemState signals must be a map")
        rows = list(rows.values())
    elif isinstance(raw, list):
        rows = raw
    else:
        raise DataError("Input must be an Observation list or schema-1 SystemState")
    # Detached lists have no Signal record. A private/unknown observation of a
    # selected signal still elevates that signal, even in another metric/source.
    tainted_signals = {row.get("signalId") for row in rows if isinstance(row, dict)
                       and isinstance(row.get("signalId"), str) and row.get("privacy") != "PUBLIC"}
    by_id = {}
    selected = []
    by_time = {}
    excluded = 0
    duplicate_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise DataError("Each observation must be an object")
        observation_id = nonempty(row.get("id"), "observation.id")
        # Compare complete original rows: a repeated ID cannot conceal a changed
        # metric, provenance, privacy, or timestamp by selection/filtering.
        try:
            encoded = canonical_json(row)
        except (ValueError, TypeError) as exc:
            raise DataError("Observation contains nonfinite or non-JSON values") from exc
        if observation_id in by_id:
            if by_id[observation_id] != encoded:
                raise DataError(f"Conflicting observation ID: {observation_id}")
            duplicate_count += 1
            continue
        by_id[observation_id] = encoded
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise DataError("Observation provenance is required")
        row_source = nonempty(provenance.get("source"), "provenance.source")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise DataError("Observation metrics must be an object")
        if row_source != source or metric not in metrics:
            excluded += 1
            continue
        if row.get("privacy") != "PUBLIC":
            raise DataError("Selected observation must be explicitly PUBLIC")
        signal_id = nonempty(row.get("signalId"), "observation.signalId")
        if signal_id in tainted_signals:
            raise DataError("Selected signal contains private or unclassified observations")
        if signals is not None:
            signal = signals.get(signal_id)
            if not isinstance(signal, dict) or signal.get("privacy") != "PUBLIC":
                raise DataError("Selected SystemState signal must be explicitly PUBLIC")
            if signal.get("id") != signal_id:
                raise DataError("SystemState signal map key must equal signal ID")
        observed_at = timestamp(row.get("observedAt"), "observedAt")
        fetched_at = timestamp(provenance.get("fetchedAt"), "fetchedAt")
        if observed_at > now_ms or fetched_at > now_ms:
            raise DataError("Future observations or fetch timestamps are not accepted")
        if fetched_at < observed_at:
            raise DataError("fetchedAt cannot precede the end of the measured interval")
        nonempty(row.get("text"), "observation.text")
        nonempty(provenance.get("uri"), "provenance.uri")
        content_hash = nonempty(provenance.get("contentHash"), "provenance.contentHash")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", content_hash):
            raise DataError("provenance.contentHash must be a SHA-256 hex digest")
        value = finite_number(metrics[metric], "metric value")
        if value < 0:
            raise DataError("The log-count specialist requires nonnegative metric values")
        for name, metric_value in metrics.items():
            nonempty(name, "metric name")
            if finite_number(metric_value, "metric value") < 0:
                raise DataError("Selected observation metrics must be nonnegative, as in the existing domain")
        key = (signal_id, row_source, observed_at)
        if key in by_time:
            raise DataError("Ambiguous observation timestamp within a signal/source series")
        by_time[key] = observation_id
        selected.append(Observation(observation_id, signal_id, observed_at, value, dict(provenance), row))
    if not selected:
        raise DataError("No PUBLIC observations match the requested source and metric")
    selected.sort(key=lambda o: (o.observed_at, o.signal_id, o.id))
    metadata = {"inputRows": len(rows), "selectedObservations": len(selected),
                "excludedSourceOrMetric": excluded, "deduplicatedIdenticalRows": duplicate_count,
                "datasetSha256": digest([o.raw for o in selected]), "validatedAt": now_ms}
    return selected, metadata


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataError(f"{name} must be an integer >= {minimum}")
    return value


def group_series(observations):
    series = {}
    for observation in observations:
        series.setdefault((observation.signal_id, observation.source), []).append(observation)
    return {key: sorted(rows, key=lambda o: o.observed_at) for key, rows in sorted(series.items())}


def make_samples(observations, *, window=4, horizon_steps=1, cadence_ms=86_400_000):
    """Only a complete contiguous span can supply a window and future target."""
    positive_int(window, "window", 2)
    positive_int(horizon_steps, "horizonSteps")
    positive_int(cadence_ms, "cadenceMs")
    samples = []
    skipped = 0
    for rows in group_series(observations).values():
        span_size = window + horizon_steps
        for start in range(len(rows) - span_size + 1):
            span = rows[start:start + span_size]
            if any(b.observed_at - a.observed_at != cadence_ms for a, b in zip(span, span[1:])):
                skipped += 1
                continue
            samples.append(Sample(tuple(span[:window]), span[-1], tuple(span[window:-1])))
    samples.sort(key=lambda sample: (sample.target.observed_at, sample.target.signal_id, sample.target.id))
    return samples, skipped


def chronological_split(samples, train_fraction=0.6, validation_fraction=0.2):
    """GLOBAL chronological target dates, kept indivisible across all signals.

    Historical input overlap is allowed: earlier measured values are known at
    later forecast origins. Targets unavailable at the next split's earliest
    origin are purged (a horizon embargo, important for horizonSteps > 1).
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise DataError("Train and validation fractions must leave a nonempty test fraction")
    dates = sorted({sample.target.observed_at for sample in samples})
    if len(dates) < 3:
        raise DataError("At least three distinct target dates are required for train/validation/test")
    train_count = max(1, min(len(dates) - 2, int(len(dates) * train_fraction)))
    validation_count = max(1, min(len(dates) - train_count - 1, int(len(dates) * validation_fraction)))
    train_end = dates[train_count - 1]
    validation_end = dates[train_count + validation_count - 1]
    splits = {
        "train": [sample for sample in samples if sample.target.observed_at <= train_end],
        "validation": [sample for sample in samples if train_end < sample.target.observed_at <= validation_end],
        "test": [sample for sample in samples if sample.target.observed_at > validation_end],
    }
    # Checkpoints must exist at every validation forecast origin; validation
    # outcomes used for model selection must exist at every test forecast origin.
    # Equality is causal: observedAt is the end of the measured interval.
    test_origin = min(sample.inputs[-1].observed_at for sample in splits["test"])
    splits["validation"] = [sample for sample in splits["validation"] if sample.target.observed_at <= test_origin]
    if not splits["validation"]:
        raise DataError("Not enough dates for a causal horizon embargo before test")
    validation_origin = min(sample.inputs[-1].observed_at for sample in splits["validation"])
    splits["train"] = [sample for sample in splits["train"] if sample.target.observed_at <= validation_origin]
    if not splits["train"]:
        raise DataError("Not enough dates for a causal horizon embargo before validation")
    return splits
