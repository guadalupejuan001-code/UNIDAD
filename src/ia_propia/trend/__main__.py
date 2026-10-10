"""Run from the repository root: python3 -m ia_propia.trend."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from .data import DataError, load_observations, timestamp
from .model import ArtifactError, load_artifact, predict, train, write_json


def as_of_value(value):
    try:
        if value.isdecimal():
            return timestamp(int(value), "asOf")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("ISO timestamp must include its timezone")
        return timestamp(int(parsed.timestamp() * 1000), "asOf")
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("as-of must be epoch milliseconds or an ISO-8601 timestamp with timezone") from exc


def parser(prog=None):
    root = argparse.ArgumentParser(prog=prog, description="Train Unidad's optional experimental count-growth specialist from random weights")
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train, select checkpoint on validation, evaluate once on held-out test")
    training.add_argument("--input", required=True, help="Observation JSON list or schema-1 SystemState")
    training.add_argument("--source", required=True)
    training.add_argument("--metric", required=True)
    training.add_argument("--output", required=True, help="Exported schema-1 model JSON")
    training.add_argument("--report", required=True, help="Evaluation, provenance and learning history JSON")
    training.add_argument("--window", type=int, default=4)
    training.add_argument("--horizon-steps", type=int, default=1)
    training.add_argument("--cadence-ms", type=int, default=86_400_000, help="Exact required cadence; default daily 86400000 ms")
    training.add_argument("--hidden-size", type=int, default=8)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=2000)
    training.add_argument("--learning-rate", type=float, default=0.01)
    training.add_argument("--patience", type=int, default=150)
    training.add_argument("--train-fraction", type=float, default=0.6)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    prediction = commands.add_parser("predict", help="Predict from a complete compatible window; always experimental")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--source", help="Optional check that this matches the artifact source")
    prediction.add_argument("--metric", help="Optional check that this matches the artifact metric")
    prediction.add_argument("--signal-id", help="Restrict output to this signal")
    prediction.add_argument("--as-of", type=as_of_value, help="Availability timestamp, default current time")
    prediction.add_argument("--historical", action="store_true", help="Explicit retrospective replay; does not prove a live forecast")
    prediction.add_argument("--output", help="Optional JSON output file; otherwise stdout")
    return root


def main(argv=None, *, prog=None):
    args = parser(prog).parse_args(argv)
    try:
        if args.command == "train":
            if Path(args.output).resolve() == Path(args.report).resolve():
                raise DataError("Model and report must have different destinations")
            if Path(args.input).resolve() in (Path(args.output).resolve(), Path(args.report).resolve()):
                raise DataError("Output cannot replace original observations")
            observations, metadata = load_observations(args.input, source=args.source, metric=args.metric)
            artifact, report = train(observations, metadata, source=args.source, metric=args.metric,
                                     window=args.window, horizon_steps=args.horizon_steps,
                                     cadence_ms=args.cadence_ms, hidden_size=args.hidden_size,
                                     seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
                                     patience=args.patience, train_fraction=args.train_fraction,
                                     validation_fraction=args.validation_fraction)
            write_json(args.output, artifact)
            write_json(args.report, report)
            print(json.dumps({"modelId": artifact["modelId"], "model": args.output,
                              "report": args.report, "bestEpoch": artifact["training"]["bestEpoch"],
                              "splitCounts": {name: split["sampleCount"] for name, split in artifact["training"]["split"].items()},
                              "test": artifact["evaluation"]["splits"]["test"],
                              "experimental": True, "productionAccepted": False}, allow_nan=False))
        else:
            artifact = load_artifact(args.model)
            for key in ("source", "metric"):
                supplied = getattr(args, key)
                if supplied is not None and supplied != artifact[key]:
                    raise DataError(f"Requested {key} does not match the artifact")
            observations, _ = load_observations(args.input, source=artifact["source"], metric=artifact["metric"])
            if args.signal_id is not None:
                observations = [o for o in observations if o.signal_id == args.signal_id]
            forecasts = predict(artifact, observations, as_of=args.as_of, historical=args.historical)
            payload = {"schemaVersion": 1, "forecasts": forecasts}
            if args.output:
                if Path(args.output).resolve() in (Path(args.input).resolve(), Path(args.model).resolve()):
                    raise DataError("Prediction output cannot replace evidence or model")
                write_json(args.output, payload)
            print(json.dumps(payload, allow_nan=False))
        return 0
    except (DataError, ArtifactError, OSError) as exc:
        print(f"unidad trend: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
