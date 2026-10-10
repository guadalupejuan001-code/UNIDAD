"""NumPy tanh MLP, manual backpropagation and Adam, starting with random weights."""

import json
import math
import platform
from pathlib import Path
import time

import numpy as np

from .data import (DataError, chronological_split, digest, features, finite_number,
                   group_series, make_samples, nonempty, parse_json, positive_int, timestamp)


class ArtifactError(ValueError):
    """The specialist artifact cannot be safely evaluated."""


ARTIFACT_KEYS = {"schemaVersion", "modelType", "modelId", "metric", "source", "privacy",
                 "window", "horizonSteps", "cadenceMs", "featureVersion", "hiddenSize",
                 "featureMean", "featureScale", "weightsInput", "biasHidden",
                 "weightsOutput", "biasOutput", "training", "evaluation"}
IDENTITY_KEYS = ARTIFACT_KEYS - {"modelId", "training", "evaluation"}


def model_id(artifact):
    return "hormiga-trend-" + digest({key: artifact[key] for key in sorted(IDENTITY_KEYS)})


def initialize(window, hidden_size, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, math.sqrt(2 / (window + hidden_size)), (hidden_size, window)),
            np.zeros(hidden_size, dtype=np.float64),
            rng.normal(0, math.sqrt(2 / (hidden_size + 1)), hidden_size),
            np.zeros(1, dtype=np.float64)]


def forward(x, params):
    if not np.all(np.isfinite(x)):
        raise DataError("Normalized features are nonfinite")
    # Accumulate explicitly in feature order, as the Kotlin reader does. BLAS
    # can hide intermediate overflow with cancellation before a finite tanh.
    preactivation = np.broadcast_to(params[1], (len(x), len(params[1]))).copy()
    with np.errstate(over="raise", invalid="raise"):
        try:
            for feature in range(x.shape[1]):
                preactivation += x[:, feature, None] * params[0][None, :, feature]
                if not np.all(np.isfinite(preactivation)):
                    raise DataError("Hidden computation overflows numeric range")
        except FloatingPointError as exc:
            raise DataError("Hidden computation overflows numeric range") from exc
    hidden = np.tanh(preactivation)
    output = np.full(len(x), params[3][0], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for unit in range(hidden.shape[1]):
                output += hidden[:, unit] * params[2][unit]
                if not np.all(np.isfinite(output)):
                    raise DataError("Output computation overflows numeric range")
        except FloatingPointError as exc:
            raise DataError("Output computation overflows numeric range") from exc
    return output, hidden


def arrays(samples):
    return np.asarray([s.features for s in samples]), np.asarray([s.label for s in samples])


def fit(train_x, train_y, validation_x, validation_y, *, hidden_size, seed, epochs,
        learning_rate, patience):
    """The test split is deliberately absent from this fitting function."""
    params = initialize(train_x.shape[1], hidden_size, seed)
    initial_hash = digest([p.tolist() for p in params])
    best = [p.copy() for p in params]
    initial_prediction, _ = forward(validation_x, params)
    best_loss = float(np.mean((initial_prediction - validation_y) ** 2))
    initial_validation = best_loss
    best_epoch = 0
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    stale = 0
    completed_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        prediction, hidden = forward(train_x, params)
        derivative = 2 * (prediction - train_y) / len(train_y)
        hidden_derivative = derivative[:, None] * params[2][None, :] * (1 - hidden ** 2)
        gradients = [hidden_derivative.T @ train_x, hidden_derivative.sum(axis=0),
                     hidden.T @ derivative, np.asarray([derivative.sum()])]
        for i, gradient in enumerate(gradients):
            m[i] = 0.9 * m[i] + 0.1 * gradient
            v[i] = 0.999 * v[i] + 0.001 * gradient ** 2
            params[i] -= learning_rate * (m[i] / (1 - 0.9 ** epoch)) / (np.sqrt(v[i] / (1 - 0.999 ** epoch)) + 1e-8)
        if not all(np.all(np.isfinite(p)) for p in params):
            raise DataError("Training produced nonfinite weights; reduce learning rate")
        training_prediction, _ = forward(train_x, params)
        validation_prediction, _ = forward(validation_x, params)
        training_loss = float(np.mean((training_prediction - train_y) ** 2))
        validation_loss = float(np.mean((validation_prediction - validation_y) ** 2))
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
            raise DataError("Training produced a nonfinite loss")
        history.append({"epoch": epoch, "trainMseLogGrowth": training_loss,
                        "validationMseLogGrowth": validation_loss})
        completed_epoch = epoch
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best = [p.copy() for p in params]
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, {"initialWeightsSha256": initial_hash, "initialValidationMseLogGrowth": initial_validation,
                  "bestEpoch": best_epoch, "epochsCompleted": completed_epoch,
                  "checkpointSelection": "minimum validation MSE of log growth; test never used",
                  "bestValidationMseLogGrowth": best_loss, "history": history}


def predicted_value(last_value, log_growth):
    last_value = finite_number(last_value, "lastValue")
    log_growth = finite_number(log_growth, "predictedLogGrowth")
    if last_value < 0:
        raise DataError("lastValue must be nonnegative")
    try:
        result = math.expm1(math.log1p(last_value) + log_growth)
    except OverflowError as exc:
        raise DataError("Prediction overflows numeric range") from exc
    if not math.isfinite(result):
        raise DataError("Prediction is nonfinite")
    return max(0.0, result)


def metrics(samples, predictions):
    target = np.asarray([s.label for s in samples])
    error = np.asarray(predictions) - target
    mse = float(np.mean(error ** 2))
    values = np.asarray([predicted_value(s.inputs[-1].value, p) for s, p in zip(samples, predictions)])
    actual = np.asarray([s.target.value for s in samples])
    result = {"sampleCount": len(samples), "maeLogGrowth": float(np.mean(np.abs(error))),
              "mseLogGrowth": mse, "rmseLogGrowth": math.sqrt(mse),
              "maeValue": float(np.mean(np.abs(values - actual))),
              "directionAccuracy": float(np.mean(np.sign(predictions) == np.sign(target)))}
    if not all(math.isfinite(value) for value in result.values()):
        raise DataError("Evaluation produced nonfinite metrics")
    return result


def train(observations, metadata, *, source, metric, window=4, horizon_steps=1,
          cadence_ms=86_400_000, hidden_size=8, seed=42, epochs=2000,
          learning_rate=0.01, patience=150, train_fraction=0.6, validation_fraction=0.2):
    positive_int(hidden_size, "hiddenSize")
    positive_int(epochs, "epochs")
    positive_int(patience, "patience")
    positive_int(seed, "seed", 0)
    learning_rate = finite_number(learning_rate, "learningRate")
    if learning_rate <= 0:
        raise DataError("learningRate must be positive")
    if any(o.source != source or o.raw.get("privacy") != "PUBLIC" or metric not in o.raw.get("metrics", {})
           or o.value != finite_number(o.raw["metrics"][metric], "metric value") for o in observations):
        raise DataError("Training requires only selected PUBLIC source/metric observations")
    samples, skipped = make_samples(observations, window=window, horizon_steps=horizon_steps, cadence_ms=cadence_ms)
    splits = chronological_split(samples, train_fraction, validation_fraction)
    x_train, y_train = arrays(splits["train"])
    mean = x_train.mean(axis=0)
    raw_scale = x_train.std(axis=0)
    # Constant (or numerically constant) train features have unit scale. This
    # definition is serialized and reused verbatim at inference time.
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
    x_validation, y_validation = arrays(splits["validation"])
    params, fit_report = fit((x_train - mean) / scale, y_train,
                             (x_validation - mean) / scale, y_validation,
                             hidden_size=hidden_size, seed=seed, epochs=epochs,
                             learning_rate=learning_rate, patience=patience)
    evaluated = {}
    for name, split in splits.items():
        x, _ = arrays(split)
        prediction, _ = forward((x - mean) / scale, params)
        evaluated[name] = {"model": metrics(split, prediction),
                           "persistence": metrics(split, np.zeros(len(split)))}
    limitations = ["Experimental count-growth regressor; no production acceptance.",
                   "Historical observations were recovered later; this is retrospective evaluation, not live forecasting evidence.",
                   "One declared source and metric only; pageviews describe attention, not cultural replication or creativity.",
                   "Small or narrow datasets cannot demonstrate generalization to unseen cultural signals.",
                   "No calibrated confidence, probabilities, or language-generation capability."]
    evaluation = {"experimental": True, "productionAccepted": False,
                  "retrospectiveEvaluation": True, "liveForecastValidated": False,
                  "baseline": "persistence: predicted log growth = 0 (last observed value)",
                  "splits": evaluated, "limitations": limitations,
                  "testRmseBelowPersistence": evaluated["test"]["model"]["rmseLogGrowth"] < evaluated["test"]["persistence"]["rmseLogGrowth"]}
    split_manifest = {name: {"sampleCount": len(split),
                           "firstTargetAt": min(s.target.observed_at for s in split),
                           "lastTargetAt": max(s.target.observed_at for s in split),
                           "samples": [s.manifest() for s in split]} for name, split in splits.items()}
    training = {**metadata, "initializedFrom": "random weights; no pretrained parameters",
                "initialization": "NumPy default_rng(seed); scaled normal input/output weights; zero biases",
                "seed": seed, "optimizer": "manual backpropagation and full-batch Adam",
                "numpyVersion": np.__version__, "pythonVersion": platform.python_version(),
                "learningRate": learning_rate, "epochsRequested": epochs, "patience": patience,
                "trainedAt": int(time.time() * 1000),
                "fitCutoffAt": max(s.target.observed_at for name in ("train", "validation") for s in splits[name]),
                "sampleCount": len(samples), "signalCount": len(group_series(observations)),
                "skippedNoncontiguousWindows": skipped, "target": "log1p(future value) - log1p(last input value)",
                "splitPolicy": "GLOBAL chronological indivisible target observedAt dates, purged targets after next split origin",
                "purgedForHorizonEmbargo": len(samples) - sum(len(split) for split in splits.values()),
                "featureScalerFit": "train samples only", "split": split_manifest,
                "provenance": [o.manifest() for o in observations],
                **{key: value for key, value in fit_report.items() if key != "history"}}
    artifact = {"schemaVersion": 1, "modelType": "hormiga-trend-mlp", "modelId": "pending",
                "metric": metric, "source": source, "privacy": "PUBLIC", "window": window,
                "horizonSteps": horizon_steps, "cadenceMs": cadence_ms,
                "featureVersion": "log-deltas-level-v1", "hiddenSize": hidden_size,
                "featureMean": mean.tolist(), "featureScale": scale.tolist(),
                "weightsInput": params[0].tolist(), "biasHidden": params[1].tolist(),
                "weightsOutput": params[2].tolist(), "biasOutput": float(params[3][0]),
                "training": training, "evaluation": evaluation}
    artifact["modelId"] = model_id(artifact)
    validate_artifact(artifact)
    report = {"schemaVersion": 1, "modelId": artifact["modelId"], "training": training,
              "evaluation": evaluation, "learningHistory": fit_report["history"]}
    return artifact, report


def validate_artifact(artifact):
    try:
        if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
            raise DataError("Artifact must match schema-1 specialist keys exactly")
        if type(artifact["schemaVersion"]) is not int or artifact["schemaVersion"] != 1:
            raise DataError("Unsupported specialist schemaVersion")
        if artifact["modelType"] != "hormiga-trend-mlp" or artifact["featureVersion"] != "log-deltas-level-v1":
            raise DataError("Unsupported specialist model/feature version")
        if artifact["privacy"] != "PUBLIC":
            raise DataError("Only PUBLIC specialist models are supported")
        for name in ("metric", "source", "modelId"):
            nonempty(artifact[name], name)
        window = positive_int(artifact["window"], "window", 2)
        hidden = positive_int(artifact["hiddenSize"], "hiddenSize")
        positive_int(artifact["horizonSteps"], "horizonSteps")
        positive_int(artifact["cadenceMs"], "cadenceMs")
        for name, length in (("featureMean", window), ("featureScale", window),
                             ("biasHidden", hidden), ("weightsOutput", hidden)):
            values = artifact[name]
            if not isinstance(values, list) or len(values) != length:
                raise DataError(f"{name} has the wrong dimensions")
            for value in values:
                number = finite_number(value, name)
                if name == "featureScale" and number <= 0:
                    raise DataError("featureScale must be positive")
        weights = artifact["weightsInput"]
        if not isinstance(weights, list) or len(weights) != hidden:
            raise DataError("weightsInput has the wrong hidden dimension")
        for row in weights:
            if not isinstance(row, list) or len(row) != window:
                raise DataError("weightsInput has the wrong feature dimension")
            for value in row:
                finite_number(value, "weightsInput")
        finite_number(artifact["biasOutput"], "biasOutput")
        for name in ("training", "evaluation"):
            if not isinstance(artifact[name], dict):
                raise DataError(f"{name} must be an object")
        timestamp(artifact["training"].get("fitCutoffAt"), "training.fitCutoffAt")
        timestamp(artifact["training"].get("trainedAt"), "training.trainedAt")
        if artifact["training"]["fitCutoffAt"] > artifact["training"]["trainedAt"]:
            raise DataError("fitCutoffAt cannot exceed actual trainedAt")
        if artifact["modelId"] != model_id(artifact):
            raise DataError("modelId does not identify the current weights/configuration")
        # Validate nonfinite JSON values also in nested provenance/evaluation.
        digest(artifact)
    except (DataError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ArtifactError(str(exc)) from exc
    return artifact


def load_artifact(path):
    try:
        artifact = parse_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read specialist artifact: {exc}") from exc
    return validate_artifact(artifact)


def predict(artifact, observations, *, as_of=None, historical=False):
    validate_artifact(artifact)
    now_ms = int(time.time() * 1000)
    as_of = timestamp(now_ms if as_of is None else as_of, "asOf")
    if as_of > now_ms:
        raise DataError("asOf cannot be in the future")
    if artifact["training"]["trainedAt"] > now_ms:
        raise DataError("Model training timestamp cannot be in the future")
    if as_of < artifact["training"]["fitCutoffAt"]:
        raise DataError("asOf precedes outcomes used to fit/select this model")
    if not historical and as_of < artifact["training"]["trainedAt"]:
        raise DataError("Model had not been trained at the requested live asOf")
    tainted_signals = {o.signal_id for o in observations if o.observed_at <= as_of
                       and (historical or o.provenance["fetchedAt"] <= as_of)
                       and o.raw.get("privacy") != "PUBLIC"}
    result = []
    for (signal_id, source), rows in group_series(observations).items():
        if source != artifact["source"]:
            continue
        if signal_id in tainted_signals:
            raise DataError("Prediction signal contains private or unclassified evidence")
        if any(o.observed_at > now_ms or o.provenance["fetchedAt"] > now_ms for o in rows):
            raise DataError("Future evidence cannot be used, including historical replay")
        rows = [o for o in rows if o.observed_at <= as_of and (historical or o.provenance["fetchedAt"] <= as_of)]
        if len(rows) < artifact["window"]:
            continue
        inputs = rows[-artifact["window"]:]
        if inputs[-1].observed_at < artifact["training"]["fitCutoffAt"]:
            raise DataError("Window origin precedes outcomes used to fit/select this model")
        if any(o.raw.get("privacy") != "PUBLIC" or artifact["metric"] not in o.raw.get("metrics", {})
               or o.value != finite_number(o.raw["metrics"][artifact["metric"]], "metric value") for o in inputs):
            raise DataError("Prediction inputs must match PUBLIC artifact metric")
        if any(b.observed_at - a.observed_at != artifact["cadenceMs"] for a, b in zip(inputs, inputs[1:])):
            raise DataError(f"Latest window is not contiguous for signal {signal_id}")
        if len({o.id for o in inputs}) != len(inputs):
            raise DataError("Prediction has duplicate evidence IDs")
        target_at = inputs[-1].observed_at + artifact["cadenceMs"] * artifact["horizonSteps"]
        if target_at > 9_223_372_036_854_775_807:
            raise DataError("Prediction target timestamp overflows int64")
        if not historical and target_at <= as_of:
            raise DataError("Latest evidence is stale for the requested live horizon; use explicit historical replay for archived data")
        params = [np.asarray(artifact["weightsInput"]), np.asarray(artifact["biasHidden"]),
                  np.asarray(artifact["weightsOutput"]), np.asarray([artifact["biasOutput"]])]
        with np.errstate(over="raise", invalid="raise"):
            try:
                x = (features([o.value for o in inputs]) - np.asarray(artifact["featureMean"])) / np.asarray(artifact["featureScale"])
                output, _ = forward(x[None, :], params)
            except FloatingPointError as exc:
                raise DataError("Forward computation overflows numeric range") from exc
        growth = finite_number(float(output[0]), "predictedLogGrowth")
        result.append({"modelId": artifact["modelId"], "signalId": signal_id,
                       "metric": artifact["metric"], "source": source, "privacy": "PUBLIC",
                       "asOf": as_of, "lastObservedAt": inputs[-1].observed_at, "targetAt": target_at,
                       "lastValue": inputs[-1].value, "predictedLogGrowth": growth,
                       "predictedValue": predicted_value(inputs[-1].value, growth),
                       "evidenceIds": [o.id for o in inputs],
                       "evidence": [{"observationId": o.id, "provenance": o.provenance} for o in inputs],
                       "epistemicKind": "HYPOTHESIS", "experimental": True,
                       "historicalReplay": historical, "liveForecastValidated": False,
                       "retrospectiveWarning": "Replay may use evidence retrieved after asOf; it does not demonstrate a live forecast." if historical else None})
    if not result:
        raise DataError("No complete compatible window is available at asOf")
    return result


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
