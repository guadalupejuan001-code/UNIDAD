"""CLI de Unidad: modelos, memoria y núcleo conectable a productos."""

import argparse
import json
import sys

from . import __version__


def add_workspace(command):
    command.add_argument("--workspace", default="private/default",
                         help="Directorio local de encargos (por defecto private/default)")


def parser():
    root = argparse.ArgumentParser(
        prog="unidad",
        description="Unidad: modelos propios y un núcleo conectable a distintos productos")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Crear un espacio local privado")
    add_workspace(init)
    tasks = commands.add_parser("task", help="Registrar encargos, intentos y correcciones")
    actions = tasks.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add", help="Guardar un encargo libre; no lo ejecuta")
    add_workspace(add)
    add.add_argument("--goal", required=True)
    add.add_argument("--context", default="")
    listing = actions.add_parser("list", help="Ver los encargos registrados")
    add_workspace(listing)
    show = actions.add_parser("show", help="Ver un encargo y su historial")
    add_workspace(show)
    show.add_argument("--id", required=True)
    attempt = actions.add_parser("attempt", help="Guardar un intento realizado para el encargo")
    add_workspace(attempt)
    attempt.add_argument("--id", required=True, help="ID del encargo")
    attempt.add_argument("--text", required=True)
    attempt.add_argument("--evidence", action="append", default=[],
                         help="Referencia de evidencia, repetible; no se abre automáticamente")
    feedback = actions.add_parser("feedback", help="Aprobar o rechazar un intento y explicar por qué")
    add_workspace(feedback)
    feedback.add_argument("--id", required=True, help="ID del encargo")
    feedback.add_argument("--attempt", required=True, help="ID del intento")
    feedback.add_argument("--text", required=True)
    verdict = feedback.add_mutually_exclusive_group(required=True)
    verdict.add_argument("--accepted", action="store_true")
    verdict.add_argument("--rejected", action="store_true")
    export = actions.add_parser("export", help="Exportar solamente intentos actualmente aprobados")
    add_workspace(export)
    export.add_argument("--output", required=True)
    for name, help_text in (("language", "Entrenar o generar con el modelo de lenguaje propio"),
                            ("trend", "Entrenar o predecir con el especialista de atención"),
                            ("brain", "Conectar el núcleo desde CLI o HTTP")):
        command = commands.add_parser(name, help=help_text, add_help=False)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("language", "trend", "brain"):
        if argv[0] == "language":
            from .language.__main__ import main as model_main
        elif argv[0] == "trend":
            from .trend.__main__ import main as model_main
        else:
            from .brain.__main__ import main as model_main
        return model_main(argv[1:], prog=f"unidad {argv[0]}")
    args = parser().parse_args(argv)
    from . import workspace
    try:
        if args.command == "init":
            result = workspace.init_workspace(args.workspace)
        elif args.action == "add":
            result = workspace.add_task(args.workspace, args.goal, args.context)
        elif args.action == "list":
            result = workspace.list_tasks(args.workspace)
        elif args.action == "show":
            result = workspace.show(args.workspace, args.id)
        elif args.action == "attempt":
            result = workspace.record_attempt(args.workspace, args.id, args.text, args.evidence)
        elif args.action == "feedback":
            result = workspace.feedback(args.workspace, args.id, args.attempt, args.text,
                                        accepted=args.accepted)
        elif args.action == "export":
            result = workspace.export_training(args.workspace, args.output)
        else:
            raise ValueError("Comando no reconocido")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"unidad: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
