"""GNOSIS commands reuse Unidad's inference, jobs and assessment contracts."""

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

from .project import load_project


def parser(prog=None):
    root = argparse.ArgumentParser(prog=prog or "unidad gnosis",
                                   description="Un proyecto para el cerebro, NERVA y las etapas de aprendizaje")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "capabilities", "call", "chat", "serve", "train", "resume", "status", "export", "assess"):
        command = commands.add_parser(name)
        command.add_argument("--project", required=True, help="Configuración local fijada por el host")
        if name in ("plan", "train", "resume", "status", "export", "assess"):
            command.add_argument("--stage", choices=("foundation", "conversation", "adaptation"), default="foundation")
        if name in ("capabilities", "call", "serve"):
            command.add_argument("--body", default="gnosis-cli")
        if name == "call":
            command.add_argument("--input", required=True, help="Petición JSON local o - para stdin")
        if name in ("train", "resume"):
            command.add_argument("--budget-seconds", required=True, type=float)
        if name == "serve":
            command.add_argument("--token-env", default="UNIDAD_API_TOKEN")
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", default=8765, type=int)
        if name == "chat":
            command.add_argument("--max-new-tokens", default=96, type=int)
            command.add_argument("--temperature", default=0.7, type=float)
            command.add_argument("--seed", default=42, type=int)
        if name == "assess":
            command.add_argument("--cases", required=True)
            command.add_argument("--output", required=True)
    return root


def main(argv=None, *, prog=None):
    args = parser(prog).parse_args(argv)
    try:
        project = load_project(args.project)
        if args.command == "plan":
            result = project.plan(args.stage)
        elif args.command in ("train", "resume"):
            result = project.train(args.stage, budget_seconds=args.budget_seconds, resume=args.command == "resume")
        elif args.command == "status":
            result = project.status(args.stage)
        elif args.command == "export":
            result = project.export(args.stage)
        elif args.command == "capabilities":
            result = project.connect(args.body).capabilities()
        elif args.command == "call":
            from ..brain.__main__ import _read_request
            result = project.connect(args.body).request(_read_request(args.input))
        elif args.command == "chat":
            from ..conversation.__main__ import _converse
            selected = project.config["brain"]["conversation_model"]
            if selected is None:
                raise ValueError("No conversation reference is selected")
            # The common console owns history; every body shares this loaded engine.
            settings = SimpleNamespace(model=project.path(selected), max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature, seed=args.seed)
            return _converse(settings, engine=project.brain.models.conversation_engine)
        elif args.command == "serve":
            from ..brain.http import create_server, validate_token
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env) is None:
                raise ValueError("Invalid token environment variable")
            token = os.environ.get(args.token_env)
            validate_token(token)
            server = create_server(project.connect(args.body), token, host=args.host, port=args.port)
            print(json.dumps({"schemaVersion": 1, "host": server.server_address[0],
                              "port": server.server_address[1], "bodyId": args.body}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        elif args.command == "assess":
            from ..conversation.assessment import assess
            artifact = project.path(project.config["learning"]["stages"][args.stage]["run_dir"]) / "artifact"
            result = assess(artifact, args.cases, output=args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"unidad gnosis: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
