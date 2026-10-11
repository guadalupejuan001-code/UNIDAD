"""Explicit local data, training-job and conversation commands."""

import argparse
import json
from pathlib import Path
import sys

from ..brain.protocol import MAX_REQUEST_BYTES, decode_json


def parser(prog=None):
    root = argparse.ArgumentParser(prog=prog or "unidad conversation",
        description="Entrenar conversación desde pesos aleatorios, sin descargar otros modelos")
    commands = root.add_subparsers(dest="command", required=True)
    importing = commands.add_parser("import-oasst", help="Convertir un corpus OASST local a conversaciones revisadas")
    importing.add_argument("--input", type=Path, required=True)
    importing.add_argument("--output", type=Path, required=True)
    importing.add_argument("--language", default="es")
    importing.add_argument("--revision")
    importing.add_argument("--max-records", type=int)
    for name in ("train", "start"):
        training = commands.add_parser(name, help="Entrenar localmente" if name == "train" else "Iniciar un trabajo local en segundo plano")
        training.add_argument("--corpus", type=Path, required=True)
        training.add_argument("--run-dir", type=Path, required=True)
        training.add_argument("--model-config", type=Path, help="Objeto JSON de configuración del Transformer y tokenizador")
        training.add_argument("--training-config", type=Path, help="Objeto JSON del entrenamiento; presupuesto total por defecto 86400 segundos")
        training.add_argument("--budget-seconds", type=float, help="Presupuesto de este segmento; el presupuesto total se conserva")
    resuming = commands.add_parser("resume", help="Continuar en segundo plano desde un checkpoint local")
    resuming.add_argument("--corpus", type=Path, required=True)
    resuming.add_argument("--run-dir", type=Path, required=True)
    resuming.add_argument("--budget-seconds", type=float)
    for name in ("status", "stop"):
        command = commands.add_parser(name, help="Consultar el trabajo y su liveness" if name == "status" else "Solicitar parada cooperativa y checkpoint")
        command.add_argument("--run-dir", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluar explícitamente el mejor checkpoint y exportarlo")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--corpus", type=Path)
    evaluation.add_argument("--output", type=Path, help="Directorio nuevo para el modelo exportado")
    snapshot = commands.add_parser("snapshot", help="Exportar el mejor checkpoint disponible sin evaluar test ni detener el entrenamiento")
    snapshot.add_argument("--run-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    chatting = commands.add_parser("chat", help="Generar una respuesta real con un modelo local configurado")
    chatting.add_argument("--model", type=Path, required=True)
    inputs = chatting.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--message", help="Texto de un turno del usuario")
    inputs.add_argument("--messages", help="Archivo JSON de mensajes de varios turnos, o - para la entrada estándar")
    chatting.add_argument("--max-new-tokens", type=int, default=96)
    chatting.add_argument("--temperature", type=float, default=0.7)
    chatting.add_argument("--seed", type=int, default=42)
    return root


def _read_json(location):
    if str(location) == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    else:
        with Path(location).open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
    return decode_json(raw)


def _config(path):
    if path is None:
        return None
    value = _read_json(path)
    if type(value) is not dict:
        raise ValueError("A configuration file must contain a strict JSON object")
    return value


def main(argv=None, *, prog=None):
    args = parser(prog).parse_args(argv)
    try:
        if args.command == "import-oasst":
            from .data import import_oasst
            result = import_oasst(args.input, args.output, language=args.language,
                revision=args.revision, max_records=args.max_records)
        elif args.command in ("train", "start"):
            model_config, training_config = _config(args.model_config), _config(args.training_config)
            if args.command == "train":
                from .training import train as run
            else:
                from .jobs import start as run
            result = run(args.corpus, args.run_dir, model_config=model_config,
                training_config=training_config, budget_seconds=args.budget_seconds)
        elif args.command == "resume":
            from .jobs import start
            result = start(args.corpus, args.run_dir, resume=True, budget_seconds=args.budget_seconds)
        elif args.command in ("status", "stop"):
            from . import jobs
            result = getattr(jobs, args.command)(args.run_dir)
        elif args.command == "evaluate":
            from .training import finalize
            result = finalize(args.run_dir, corpus_path=args.corpus, export_dir=args.output)
        elif args.command == "snapshot":
            from .training import export_best
            result = export_best(args.run_dir, args.output)
        else:
            messages = [{"role": "user", "content": args.message}] if args.message is not None else _read_json(args.messages)
            from .tokenizer import integer, number, validate_messages
            messages = validate_messages(messages)
            if messages[-1]["role"] != "user":
                raise ValueError("Conversation messages must end with an explicit user turn")
            integer(args.max_new_tokens, "max_new_tokens", 1, 1024)
            number(args.temperature, "temperature", 0, 2)
            integer(args.seed, "seed")
            from .artifacts import load_artifact
            from .model import chat
            model, tokenizer, manifest = load_artifact(args.model)
            result = {**chat(model, tokenizer, messages, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, seed=args.seed), "modelId": manifest["modelId"],
                "workerCapabilityValidated": False}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except ImportError:
        print("unidad conversation: faltan dependencias opcionales; instala unidad[conversation] explícitamente", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"unidad conversation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
