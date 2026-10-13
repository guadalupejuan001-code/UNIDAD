"""Explicit local data, training-job and conversation commands."""

import argparse
import json
from pathlib import Path
import sys

from ..brain.protocol import MAX_REQUEST_BYTES, decode_json
from .contract import (DEFAULT_SYSTEM_PROMPT, MAX_DIALOGUE_BYTES,
                       MAX_MESSAGE_BYTES, MAX_MESSAGES)
from .runtime import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, DEFAULT_TEMPERATURE


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
        training.add_argument("--initial-model", type=Path, help="Artefacto propio local cuyos pesos y tokenizador inicia esta nueva etapa")
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
    assessment = commands.add_parser("assess", help="Revisar conversación y contexto sin entrenar ni evaluar el test final")
    assessment.add_argument("--model", type=Path, required=True)
    assessment.add_argument("--cases", type=Path, required=True)
    assessment.add_argument("--output", type=Path, help="Archivo JSON nuevo para conservar respuestas y puntuaciones")
    assessment.add_argument("--max-new-tokens", type=int, default=64)
    assessment.add_argument("--temperature", type=float, default=0)
    assessment.add_argument("--seed", type=int, default=42)
    assessment.add_argument("--torch-threads", type=int, default=2)
    chatting = commands.add_parser("chat", help="Generar una respuesta real con un modelo local configurado")
    chatting.add_argument("--model", type=Path, required=True)
    inputs = chatting.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--message", help="Texto de un turno del usuario")
    inputs.add_argument("--messages", help="Archivo JSON de mensajes de varios turnos, o - para la entrada estándar")
    chatting.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    chatting.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    chatting.add_argument("--seed", type=int, default=DEFAULT_SEED)
    converse = commands.add_parser("converse", help="Conversar en terminal; historial temporal, /salir y /reiniciar")
    converse.add_argument("--model", type=Path, required=True)
    converse.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    converse.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    converse.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for command in (chatting, converse, assessment):
        command.add_argument("--decoder-backend", choices=("torch", "nerva"), default="torch")
        command.add_argument("--no-cache", action="store_true", help="Recalcular contexto para comparar con la referencia")
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--top-p", type=float, default=1.0)
        command.add_argument("--repetition-penalty", type=float, default=1.0)
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


def _trim_history(messages):
    """Remove whole completed turns, preserving room for the default system."""
    messages = list(messages)
    removed = 0
    while (len(messages) + 1 > MAX_MESSAGES or
           sum(len(turn["content"].encode("utf-8")) for turn in messages) +
           len(DEFAULT_SYSTEM_PROMPT.encode("utf-8")) > MAX_DIALOGUE_BYTES):
        messages = messages[2:]
        removed += 1
    if removed:
        print(f"[Aviso: se retiraron {removed} turnos completos del historial por sus límites.]", file=sys.stderr)
    return messages


def _converse(args, *, engine=None):
    """Keep only actual successful turns in RAM, using the common engine."""
    from .runtime import ConversationEngine, validate_reply_request
    from .tokenizer import validate_messages
    settings = validate_reply_request({"max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "top_k": getattr(args, "top_k", 0), "top_p": getattr(args, "top_p", 1.0),
        "repetition_penalty": getattr(args, "repetition_penalty", 1.0)}, partial=True)
    if engine is None:
        engine = ConversationEngine(args.model, decoder_backend=getattr(args, "decoder_backend", "torch"),
                                    use_cache=not getattr(args, "no_cache", False))
    elif not isinstance(engine, ConversationEngine):
        raise ValueError("The console must use the common ConversationEngine")
    print(f"[Modelo local: {engine.model_id}; contexto: {engine.manifest['config']['context_length']} tokens.]\n"
          "[Conversación experimental. Historial sólo en esta sesión; no se guarda ni entrena.]\n"
          "[Un mensaje por línea. /salir termina; /reiniciar borra el historial.]", flush=True)
    history = []
    context_warning_shown = False
    try:
        while True:
            print("Tú: ", end="", flush=True)
            line = sys.stdin.readline(MAX_MESSAGE_BYTES + 2)
            if not line:
                print("\n[Sesión terminada.]", flush=True)
                return 0
            if len(line) > MAX_MESSAGE_BYTES and not line.endswith("\n"):
                while line and not line.endswith("\n"):
                    line = sys.stdin.readline(MAX_MESSAGE_BYTES + 2)
                print("unidad conversation: el mensaje excede el límite de texto", file=sys.stderr)
                continue
            text = line.rstrip("\r\n")
            if text.strip() == "/salir":
                print("[Sesión terminada.]", flush=True)
                return 0
            if text.strip() == "/reiniciar":
                history = []
                context_warning_shown = False
                print("[Historial reiniciado.]", flush=True)
                continue
            if not text.strip():
                continue
            user = {"role": "user", "content": text}
            try:
                # Validate this input before discarding any previous turn.
                validate_reply_request({"messages": [user], **settings})
                messages = _trim_history([*history, user])
                values = validate_reply_request({"messages": messages, **settings})
                result = engine.reply(**values)
            except (ValueError, OSError, RuntimeError) as error:
                print(f"unidad conversation: {error}", file=sys.stderr)
                continue
            print(f"Unidad: {result['text']}", flush=True)
            discarded = result.get("discardedPromptTokens", 0)
            if discarded and not context_warning_shown:
                print(f"[Aviso: el modelo recortó {discarded} tokens del contexto anterior.]", file=sys.stderr)
                context_warning_shown = True
            assistant = {"role": "assistant", "content": result["text"]}
            try:
                validate_messages([user, assistant])
            except ValueError:
                # Empty or oversized generations are shown intact, never invented
                # or silently shortened to manufacture a valid remembered reply.
                history = messages[:-1]
                print("[Aviso: esta respuesta no se conserva en el historial porque no cumple sus límites de texto.]", file=sys.stderr)
                continue
            history = _trim_history([*messages, assistant])
    except KeyboardInterrupt:
        print("\n[Sesión terminada.]", flush=True)
        return 0


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
            options = {"model_config": model_config, "training_config": training_config,
                       "budget_seconds": args.budget_seconds}
            if args.initial_model is not None:
                options["initial_model"] = args.initial_model
            result = run(args.corpus, args.run_dir, **options)
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
        elif args.command == "assess":
            from .assessment import assess
            result = assess(args.model, args.cases, output=args.output,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                seed=args.seed, torch_threads=args.torch_threads, decoder_backend=args.decoder_backend,
                use_cache=not args.no_cache, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty)
        elif args.command == "converse":
            return _converse(args)
        else:
            messages = [{"role": "user", "content": args.message}] if args.message is not None else _read_json(args.messages)
            from .runtime import ConversationEngine, validate_reply_request
            values = {"messages": messages, "max_new_tokens": args.max_new_tokens,
                      "temperature": args.temperature, "seed": args.seed,
                      "top_k": args.top_k, "top_p": args.top_p, "repetition_penalty": args.repetition_penalty}
            validate_reply_request(values)
            result = ConversationEngine(args.model, decoder_backend=args.decoder_backend, use_cache=not args.no_cache).reply(**values)
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
