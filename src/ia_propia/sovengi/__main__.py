"""Single SOVENGI entry point for conversation, memory, tools and learning."""

import argparse
import json
import os
import sys
from pathlib import Path

from .application import Sovengi
from ..conversation.contract import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED
from ..conversation.runtime import validate_reply_request


def _print(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _request(session, operation, values):
    response = session.request({"schemaVersion": 1, "operation": operation, "input": values})
    if response["status"] != "ok":
        raise ValueError(response["error"]["message"])
    return response["result"]


def _generation_settings(args):
    return validate_reply_request({"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "seed": args.seed, "top_k": args.top_k, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty}, partial=True)


def _chat(session, settings):
    history = []
    print("SOVENGI · modelo experimental propio. /recordar texto · /memoria palabra · /nuevo · /salir")
    while True:
        try:
            text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text == "/salir":
            return 0
        if text == "/nuevo":
            session.reset_context()
            history.clear()
            print("Conversación reiniciada; la memoria guardada se conserva.")
            continue
        try:
            if text.startswith("/recordar "):
                note = _request(session, "memory.task.add", {"goal": text[10:].strip()})
                print("Guardado: " + note["id"])
                continue
            if text.startswith("/memoria "):
                _print(_request(session, "memory.task.search", {"query": text[9:].strip()}))
                continue
            messages = history + [{"role": "user", "content": text}]
            result = _request(session, "conversation.reply", {"messages": messages, **settings})
            print("SOVENGI: " + (result["text"] or "[El modelo no generó texto.]"))
            if result["text"].strip():
                history = (messages + [{"role": "assistant", "content": result["text"]}])[-60:]
            if result.get("discardedPromptTokens", 0):
                print("[El historial superó el contexto del modelo.]")
        except ValueError as error:
            print("SOVENGI: " + str(error), file=sys.stderr)


def parser(prog="sovengi"):
    root = argparse.ArgumentParser(prog=prog, description="SOVENGI: conversación, memoria y herramientas con pesos propios")
    root.add_argument("--config", default="configs/gnosis.json", help="Configuración común de SOVENGI/GNOSIS")
    root.add_argument("--body", default="sovengi", help="Producto y memoria privada seleccionados por el anfitrión")
    root.add_argument("--context-session", help="Identificador de conversación para recuperar su contexto local")
    root.add_argument("--no-context", action="store_true", help="Desactivar selección de contexto para comparación")
    root.add_argument("--input-root", help="Carpeta existente de archivos autorizados")
    root.add_argument("--report-root", help="Carpeta existente de informes")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("chat", "ask", "forecast"):
        command = commands.add_parser(name, help="Conversar" if name == "chat" else "Hacer una pregunta")
        if name in ("ask", "forecast"):
            command.add_argument("text")
        if name == "forecast":
            command.add_argument("--observations", type=Path, required=True,
                                 help="JSON de observaciones explícitas del dominio del modelo de métricas")
            command.add_argument("--signal-id", help="Señal concreta para pronosticar y explicar")
        command.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
        command.add_argument("--temperature", type=float, default=0)
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--top-p", type=float, default=1.0)
        command.add_argument("--repetition-penalty", type=float, default=1.0)
    commands.add_parser("status", help="Ver identidad, servicios y límites reales")
    commands.add_parser("models", help="Ver modelos coordinados, disponibilidad y estado de carga")
    score = commands.add_parser("score", help="Evaluar texto con el modelo auxiliar de caracteres")
    score.add_argument("text")
    score.add_argument("--max-chars", type=int, default=512)
    remember = commands.add_parser("remember", help="Guardar texto explícitamente en memoria")
    remember.add_argument("text")
    memory = commands.add_parser("memory", help="Buscar en la memoria guardada")
    memory.add_argument("query")
    call = commands.add_parser("request", help="Enviar una petición JSON del protocolo común")
    call.add_argument("json", nargs="?", help="JSON literal; se lee de stdin si se omite")
    work = commands.add_parser("work", help="Intentar una tarea acotada con el modelo y las herramientas de archivos")
    work.add_argument("goal")
    work.add_argument("--files", nargs="+", required=True)
    work.add_argument("--report", required=True)
    serve = commands.add_parser("serve", help="API HTTP del mismo producto y memoria")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token-env", default="SOVENGI_API_TOKEN", help="Variable con un token de al menos 32 caracteres")
    for name in ("plan", "train"):
        command = commands.add_parser(name, help="Planificar aprendizaje" if name == "plan" else "Entrenar con datos y presupuesto explícitos")
        command.add_argument("--stage", choices=("foundation", "conversation", "adaptation"), default="conversation")
        command.add_argument("--resume", action="store_true")
        if name == "train":
            command.add_argument("--budget-seconds", type=float, required=True)
    for name in ("learning-status", "export"):
        command = commands.add_parser(name, help="Consultar o exportar una etapa explícita de aprendizaje")
        command.add_argument("--stage", choices=("foundation", "conversation", "adaptation"), default="conversation")
    return root


def main(argv=None, *, prog="sovengi"):
    args = parser(prog).parse_args(argv)
    application = None
    try:
        settings = _generation_settings(args) if args.command in ("chat", "ask", "forecast") else {}
        application = Sovengi(args.config)
        if args.command in ("plan", "train"):
            kwargs = {"resume": args.resume}
            if args.command == "train":
                kwargs["budget_seconds"] = args.budget_seconds
            _print(getattr(application.project, args.command)(args.stage, **kwargs))
            return 0
        if args.command in ("learning-status", "export"):
            _print(getattr(application.project, "status" if args.command == "learning-status" else "export")(args.stage))
            return 0
        session = application.connect(args.body, input_root=args.input_root, report_root=args.report_root,
            operator_token=os.environ.get("SOVENGI_OPERATOR_TOKEN") if args.input_root else None,
            context_enabled=not args.no_context, context_session_id=args.context_session)
        if args.command == "chat":
            return _chat(session, settings)
        if args.command == "status":
            _print(session.capabilities())
        elif args.command == "models":
            _print(session.capabilities()["models"]["models"]["conversation"])
        elif args.command == "score":
            _print(_request(session, "language.score", {"text": args.text, "max_chars": args.max_chars}))
        elif args.command == "forecast":
            from ..brain.protocol import decode_json
            with args.observations.open("rb") as stream:
                observations = decode_json(stream.read(262145))
            request = {"messages": [{"role": "user", "content": args.text}],
                       "forecast": {"observations": observations}, **settings}
            if args.signal_id is not None:
                request["forecast"]["signal_id"] = args.signal_id
            _print(_request(session, "conversation.forecast", request))
        elif args.command == "ask":
            _print(_request(session, "conversation.reply", {"messages": [{"role": "user", "content": args.text}],
                **settings}))
        elif args.command == "remember":
            _print(_request(session, "memory.task.add", {"goal": args.text}))
        elif args.command == "memory":
            _print(_request(session, "memory.task.search", {"query": args.query}))
        elif args.command == "request":
            raw = args.json if args.json is not None else sys.stdin.read(262145)
            if len(raw.encode("utf-8")) > 262144:
                raise ValueError("La petición supera el límite de bytes")
            response = session.request(json.loads(raw))
            _print(response)
            return 0 if response["status"] == "ok" else 2
        elif args.command == "work":
            started = _request(session, "autonomy.start", {"goal": args.goal,
                "task_inputs": {"files": args.files, "report": args.report}})
            result = _request(session, "autonomy.run", {"task_id": started["task_id"]})
            _print(result)
            return 0 if result["status"] == "completed" else 2
        elif args.command == "serve":
            from ..brain.http import create_server
            server = create_server(session, token=os.environ.get(args.token_env), host=args.host, port=args.port)
            print(f"SOVENGI: http://{server.server_address[0]}:{server.server_address[1]} · Ctrl+C para cerrar", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(f"SOVENGI: {error}", file=sys.stderr)
        return 2
    finally:
        if application is not None:
            application.close()


if __name__ == "__main__":
    raise SystemExit(main())
