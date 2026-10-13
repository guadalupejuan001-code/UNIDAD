"""Local CLI adapter for the same brain used by product bodies and HTTP."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from . import Brain
from .http import create_server, error_envelope, validate_token
from .protocol import BrainError, MAX_REQUEST_BYTES, decode_json


def parser(prog=None):
    root = argparse.ArgumentParser(prog=prog or "unidad brain", description="Unidad: operaciones explícitas del mismo cerebro para un producto")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("capabilities", "Consultar las capacidades reales del producto conectado"),
                            ("call", "Ejecutar una operación explícita en JSON"),
                            ("serve", "Conectar un producto por HTTP con autenticación")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--body", default="product-cli", help="Identificador del producto fijado al conectarlo")
        command.add_argument("--state-dir", type=Path, default=Path("private/brain"), help="Directorio privado del cerebro")
        command.add_argument("--memory-scope", help="Memoria compartida explícitamente por configuración, no por petición")
        command.add_argument("--language-model", type=Path, help="Ruta del modelo de lenguaje fijada al iniciar")
        command.add_argument("--language-backend", choices=("numpy", "nerva"), default="numpy",
                             help="Motor del char-RNN fijado por el host; NERVA usa greedy y requiere Node.js 18+")
        command.add_argument("--attention-model", type=Path, help="Ruta del modelo de atención fijada al iniciar")
        command.add_argument("--conversation-model", type=Path, help="Directorio local del modelo conversacional fijado al iniciar")
        command.add_argument("--conversation-decoder", choices=("torch", "nerva", "nerva-fast"), default="torch",
                             help="Selector del Transformer; NERVA requiere Node.js 18+")
        command.add_argument("--no-conversation-cache", action="store_true", help="Recalcular el contexto para comparación de referencia")
        command.add_argument("--autonomy-input-root", type=Path,
                             help="Activar autonomía experimental sobre archivos de este directorio existente")
        command.add_argument("--autonomy-report-root", type=Path,
                             help="Directorio existente y separado para informes nuevos; requiere input-root")
        command.add_argument("--autonomy-max-steps", type=int, default=16,
                             help="Máximo de pasos autónomos fijado por el host (1 a 64)")
        command.add_argument("--autonomy-max-seconds", type=int, default=120,
                             help="Presupuesto finito de autonomía fijado por el host (1 a 3600 segundos)")
        command.add_argument("--autonomy-max-argument-repairs", type=int, default=0,
                             help="Nuevas propuestas tras argumentos rechazados de lectura, sin efecto (0 a 64; opt-in del host)")
        command.add_argument("--autonomy-operator-token-env",
                             help="Variable con una credencial separada para aprobar efectos sensibles")
        if name == "call":
            command.add_argument("--input", required=True, help="Archivo JSON de petición, o - para la entrada estándar")
        if name == "serve":
            command.add_argument("--token-env", default="UNIDAD_API_TOKEN", help="Variable de entorno con la credencial Bearer")
            command.add_argument("--host", default="127.0.0.1", help="Interfaz HTTP; por defecto sólo localhost")
            command.add_argument("--port", type=int, default=8765, help="Puerto HTTP; 0 elige un puerto disponible")
    return root


def _read_request(location):
    if location == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    else:
        with Path(location).open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
    return decode_json(raw, max_bytes=MAX_REQUEST_BYTES)


def _emit(payload):
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def main(argv=None, *, prog=None):
    args = parser(prog=prog).parse_args(argv)
    try:
        token = None
        if args.command == "serve":
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env) is None:
                raise ValueError("Configure a valid API token environment variable name")
            token = os.environ.get(args.token_env)
            validate_token(token)  # Fail before creating any private state/listener.
        autonomy_options = {}
        if args.autonomy_input_root is not None or args.autonomy_report_root is not None:
            if args.autonomy_input_root is None or args.autonomy_report_root is None:
                raise ValueError("Autonomy requires both host-configured input-root and report-root")
            if args.conversation_model is None:
                raise ValueError("Autonomy requires a host-configured local conversation model")
            if not 1 <= args.autonomy_max_steps <= 64 or not 1 <= args.autonomy_max_seconds <= 3600:
                raise ValueError("Autonomy requires bounded host step and time limits")
            if not 0 <= args.autonomy_max_argument_repairs <= 64:
                raise ValueError("Argument repairs require a bounded host limit between zero and 64")
            operator_token = None
            if args.autonomy_operator_token_env is not None:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.autonomy_operator_token_env) is None:
                    raise ValueError("Configure a valid operator token environment variable name")
                operator_token = os.environ.get(args.autonomy_operator_token_env)
                validate_token(operator_token)
            from .autonomy import AutonomyLimits
            from .file_tools import SafeFileTools
            files = SafeFileTools(args.autonomy_input_root, args.autonomy_report_root)
            autonomy_options = {"autonomy_tools": files.tools,
                                "autonomy_limits": AutonomyLimits(max_steps=args.autonomy_max_steps,
                                                                   max_seconds=args.autonomy_max_seconds,
                                                                   max_argument_repairs=args.autonomy_max_argument_repairs),
                                "autonomy_completion_verifier": files.completion_verifier,
                                "autonomy_input_validator": files.validate_task_inputs,
                                "autonomy_operator_token": operator_token}
        elif args.autonomy_operator_token_env is not None:
            raise ValueError("Operator approval requires configured autonomy roots")
        brain = Brain(args.state_dir, language_model=args.language_model, attention_model=args.attention_model,
                      conversation_model=args.conversation_model, language_backend=args.language_backend,
                      conversation_decoder=args.conversation_decoder, conversation_cache=not args.no_conversation_cache)
        session = brain.connect(args.body, memory_scope=args.memory_scope, **autonomy_options)
        if args.command == "capabilities":
            _emit(session.capabilities())
            return 0
        if args.command == "call":
            try:
                payload = _read_request(args.input)
            except (BrainError, ValueError, UnicodeError, RecursionError):
                _emit(error_envelope(args.body, "invalid_json", "Input must be strict bounded UTF-8 JSON"))
                return 2
            response = session.request(payload)
            _emit(response)
            return 0 if response.get("status") == "ok" else 2
        with create_server(session, token, host=args.host, port=args.port) as server:
            # Startup emits the bound endpoint/body, never a credential or payload.
            print(json.dumps({"schemaVersion": 1, "status": "ready", "bodyId": session.body_id,
                              "host": server.server_address[0], "port": server.server_address[1]}),
                  file=sys.stderr, flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return 0
    except (BrainError, ValueError, OSError, ImportError, RuntimeError) as error:
        if args.command == "call":
            _emit(error_envelope(args.body, "operation_failed", "Configured request could not be completed"))
        else:
            message = "faltan dependencias opcionales; instala unidad[conversation] explícitamente" if isinstance(error, ImportError) else str(error)
            print(f"unidad brain: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
