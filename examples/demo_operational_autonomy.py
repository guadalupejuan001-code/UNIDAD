"""Real local API/file execution, distinguishing fixtures from own-weight plans.

No fixture is used as a fallback for the model. Outputs include raw proposals,
observations, independent file hashes and safe cancellation evidence.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import resource
import secrets
import threading
import time

from ia_propia.brain import Brain
from ia_propia.brain.autonomy import AutonomyLimits
from ia_propia.brain.autonomy_model import ModelActionProposer
from ia_propia.brain.file_tools import SafeFileTools
from ia_propia.brain.http import create_server


class DeclaredFixture:
    origin = "test_fixture"
    proposer_id = "demonstration-software-fixture-not-learned"

    def __init__(self, actions):
        self.actions = iter(actions)

    def __call__(self, _context):
        return json.dumps(next(self.actions), ensure_ascii=False)


def tool(name, arguments):
    return {"action": "tool", "tool": name, "arguments": arguments,
            "reason": "Paso explícito de la prueba del controlador, no decisión aprendida"}


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.is_file()}


@contextmanager
def local_api(session):
    token = secrets.token_urlsafe(32)
    server = create_server(session, token, port=0)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    records = []

    def request(operation, values):
        payload = {"schemaVersion": 1, "requestId": f"demo-{len(records)}",
                   "operation": operation, "input": values}
        connection = http.client.HTTPConnection(*server.server_address, timeout=90)
        try:
            connection.request("POST", "/v1/requests", body=json.dumps(payload).encode(),
                               headers={"Authorization": f"Bearer {token}",
                                        "Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.loads(response.read())
            records.append({"request": payload, "httpStatus": response.status, "response": result})
            if result.get("status") != "ok":
                raise RuntimeError(f"API operation failed: {operation}")
            return result["result"]
        finally:
            connection.close()

    try:
        yield request, records
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("Local API did not stop")


def demonstrate(output_dir, model_dir=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, reports = output / "inputs", output / "reports"
    inputs.mkdir()
    reports.mkdir()
    (inputs / "uno.txt").write_text("La reunión será el martes.\nParticipan Lucía y Omar.\n", encoding="utf-8")
    (inputs / "dos.txt").write_text("El informe sólo analiza estos archivos.\nNo modifica los originales.\n", encoding="utf-8")
    before = hashes(inputs)
    started = time.monotonic()
    evidence = {"schemaVersion": 1, "atUtc": datetime.now(timezone.utc).isoformat(),
                "scope": "Local authenticated API, real files, declared fixtures and optional raw own-model attempt",
                "originalHashesBefore": before, "experiencesRetrainModel": False}
    brain = Brain(output / "state", conversation_model=model_dir)
    limits = AutonomyLimits(max_steps=8, max_failures=2, max_seconds=60)
    with SafeFileTools(inputs, reports) as files:
        bindings = {"autonomy_tools": files.tools, "autonomy_limits": limits,
                    "autonomy_completion_verifier": files.completion_verifier,
                    "autonomy_input_validator": files.validate_task_inputs}
        actions = [tool("inspect_file", {"filename": "uno.txt"}),
                   tool("read_file", {"filename": "dos.txt"}),
                   tool("save_report", {"filename": "informe.json", "files": list(before)})]
        session = brain.connect("declared-fixture", autonomy_proposer=DeclaredFixture(actions), **bindings)
        task_inputs = {"files": list(before), "report": "informe.json"}
        with local_api(session) as (request, records):
            task = request("autonomy.start", {"goal": "Analizar dos archivos y guardar un informe verificable.",
                                               "task_inputs": task_inputs})
            result = request("autonomy.run", {"task_id": task["task_id"]})
            assert result["status"] == "completed" and result["source"] == "test_fixture"
            evidence["controllerExecution"] = {"learnedPlanningProven": False, "transcript": records}
        cancel = brain.connect("cancel-fixture", autonomy_proposer=DeclaredFixture(actions), **bindings)
        with local_api(cancel) as (request, records):
            task = request("autonomy.start", {"goal": "Demostrar cancelación antes de guardar.",
                                               "task_inputs": {**task_inputs, "report": "cancelado.json"}})
            request("autonomy.step", {"task_id": task["task_id"]})
            request("autonomy.cancel", {"task_id": task["task_id"]})
            result = request("autonomy.run", {"task_id": task["task_id"]})
            assert result["status"] == "cancelled" and not (reports / "cancelado.json").exists()
            evidence["safeCancellation"] = {"reportAbsent": True, "transcript": records}
        if model_dir is not None:
            proposer = ModelActionProposer(brain.models, max_new_tokens=128, temperature=0, seed=42)
            own = brain.connect("own-model", autonomy_proposer=proposer, **bindings)
            with local_api(own) as (request, records):
                task = request("autonomy.start", {"goal": "Analiza uno.txt y dos.txt y guarda un informe sin alterar originales.",
                                                   "task_inputs": {**task_inputs, "report": "modelo.json"}})
                result = request("autonomy.run", {"task_id": task["task_id"]})
                evidence["ownModelExecution"] = {"modelId": proposer.proposer_id,
                    "status": result["status"], "transcript": records,
                    "rawMetadata": proposer.last_metadata,
                    "reportPresent": (reports / "modelo.json").exists(),
                    "fixtureFallback": False, "planningAccepted": result["status"] == "completed"}
                # Generation occurs on an HTTP worker; its metadata is thread-local
                # and the authoritative raw prompt/reply remains in the task journal.
            # A separate smaller task isolates action-format learning from the
            # model's context capacity. This is another real model invocation,
            # with one registered read tool and no scripted fallback.
            minimal_proposer = ModelActionProposer(brain.models, max_new_tokens=128,
                                                   temperature=0, seed=42)
            minimal = brain.connect("own-model-minimal", autonomy_proposer=minimal_proposer,
                                    autonomy_tools=[files.tools[0]],
                                    autonomy_limits=AutonomyLimits(max_steps=1, max_failures=1,
                                                                  max_seconds=60))
            with local_api(minimal) as (request, records):
                task = request("autonomy.start", {"goal": "Inspecciona uno.txt sin modificarlo."})
                result = request("autonomy.run", {"task_id": task["task_id"]})
                evidence["ownMinimalProposal"] = {"modelId": minimal_proposer.proposer_id,
                    "status": result["status"], "transcript": records,
                    "purpose": "Separate own-model JSON/action ability from multi-tool context overflow",
                    "fixtureFallback": False, "planningAccepted": False}
        evidence["originalHashesAfter"] = hashes(inputs)
        assert evidence["originalHashesAfter"] == before
        evidence["reportHashes"] = hashes(reports)
    evidence["originalsUnchanged"] = True
    evidence["elapsedSeconds"] = time.monotonic() - started
    evidence["peakRSSKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "originalsUnchanged": True,
            "controllerFixtureCompleted": True, "ownModelStatus": evidence.get("ownModelExecution", {}).get("status"),
            "learnedPlanningClaim": False, "evidenceSha256": hashlib.sha256((output / "evidence.json").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New output directory; never overwrite an earlier execution")
    parser.add_argument("--model-dir", help="An existing UNIDAD own-weight conversation artifact")
    arguments = parser.parse_args()
    if arguments.model_dir:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    print(json.dumps(demonstrate(arguments.output_dir, arguments.model_dir), ensure_ascii=False))
