"""Local train/generate commands for the experimental language model."""
import argparse
import json
from pathlib import Path
import sys

from .model import LanguageError, generate, load_artifact, train, write_json


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Train a small character RNN from random weights; no pretrained models or APIs")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--corpus", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--report", type=Path, required=True)
    training.add_argument("--hidden-size", type=int, default=24)
    training.add_argument("--sequence-length", type=int, default=48)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--learning-rate", type=float, default=0.015)
    training.add_argument("--clip-norm", type=float, default=1.0)
    training.add_argument("--patience", type=int, default=6)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--train-fraction", type=float, default=0.6)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    generation = commands.add_parser("generate")
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--prompt", default="")
    generation.add_argument("--length", type=int, default=200)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--backend", choices=("numpy", "nerva"), default="numpy",
                            help="Motor local; NERVA requiere Node.js 18+ y --temperature 0")
    try:
        args = parser.parse_args(argv)
        if args.command == "train":
            corpus, output, report = (path.resolve() for path in (args.corpus, args.output, args.report))
            if len({corpus, output, report}) != 3:
                raise LanguageError("Corpus, model, and report must have different paths")
            artifact, result = train(corpus, hidden_size=args.hidden_size, sequence_length=args.sequence_length,
                                     epochs=args.epochs, learning_rate=args.learning_rate, clip_norm=args.clip_norm,
                                     patience=args.patience, seed=args.seed, train_fraction=args.train_fraction,
                                     validation_fraction=args.validation_fraction)
            write_json(output, artifact)
            write_json(report, result)
            print(json.dumps({"modelId": artifact["modelId"], "model": str(output), "report": str(report),
                              "bestEpoch": artifact["training"]["bestEpoch"], "evaluation": artifact["evaluation"],
                              "experimental": True}, ensure_ascii=False, allow_nan=False))
        else:
            selected_generate = generate
            if args.backend == "nerva":
                from .nerva_backend import generate_nerva
                selected_generate = generate_nerva
            print(json.dumps(selected_generate(load_artifact(args.model), args.prompt, length=args.length,
                                      temperature=args.temperature, seed=args.seed), ensure_ascii=False, allow_nan=False))
        return 0
    except (LanguageError, OSError) as error:
        print(f"unidad language: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
