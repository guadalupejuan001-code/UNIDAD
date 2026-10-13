"""Execute raw own-model proposals with one bounded READ argument repair.

The fixed fictional files/goal repeat an exposed development demonstration.
This is a controller comparison, not a new acceptance test or retraining.
No action fixture, JSON repair, substituted arguments or model fallback is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

from demo_operational_autonomy import local_api, hashes
from ia_propia.brain import Brain
from ia_propia.brain.autonomy import AutonomyLimits
from ia_propia.brain.autonomy_model import ModelActionProposer
from ia_propia.brain.file_tools import SafeFileTools


def run(model_dir, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(exist_ok=False, mode=0o700)
    inputs, reports = output / "inputs", output / "reports"
    inputs.mkdir()
    reports.mkdir()
    (inputs / "uno.txt").write_text("La reunión será el martes.\nParticipan Lucía y Omar.\n", encoding="utf-8")
    (inputs / "dos.txt").write_text("El informe sólo analiza estos archivos.\nNo modifica los originales.\n", encoding="utf-8")
    before = hashes(inputs)
    started = time.monotonic()
    brain = Brain(output / "state", conversation_model=model_dir)
    proposer = ModelActionProposer(brain.models, max_new_tokens=128, temperature=0, seed=42)
    limits = AutonomyLimits(max_steps=8, max_failures=3, max_seconds=120, max_argument_repairs=1)
    with SafeFileTools(inputs, reports) as files:
        session = brain.connect("own-model-read-repair", autonomy_proposer=proposer,
                                autonomy_tools=files.tools, autonomy_limits=limits,
                                autonomy_completion_verifier=files.completion_verifier,
                                autonomy_input_validator=files.validate_task_inputs)
        with local_api(session) as (request, records):
            task = request("autonomy.start", {
                "goal": "Analiza uno.txt y dos.txt y guarda un informe sin alterar originales.",
                "task_inputs": {"files": list(before), "report": "modelo.json"}})
            result = request("autonomy.run", {"task_id": task["task_id"]})
            persisted = request("autonomy.status", {"task_id": task["task_id"]})
        if persisted != result:
            raise RuntimeError("Persisted task differs from actual execution receipt")
        verification = files.completion_verifier(result)
    after = hashes(inputs)
    if before != after:
        raise RuntimeError("Original file bytes changed")
    evidence = {
        "schemaVersion": 1, "atUtc": datetime.now(timezone.utc).isoformat(),
        "modelId": proposer.proposer_id, "modelManifest": brain.models.conversation_engine.manifest,
        "sourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Exposed fictional development task; same own weights, host opt-in READ repair; actual authenticated API/file execution",
        "limits": result["limits"], "status": result["status"], "steps": result["steps"],
        "failures": result["failures"], "argumentRepairs": result.get("argument_repairs", 0),
        "transcript": records, "persistedExecutionVerified": True,
        "completionEvidence": verification, "reportPresent": (reports / "modelo.json").is_file(),
        "originalHashesBefore": before, "originalHashesAfter": after, "originalsUnchanged": True,
        "reportHashes": hashes(reports), "fixtureFallback": False, "weightsModified": False,
        "experiencesRetrainModel": False, "workerCapabilityAccepted": False,
        "elapsedSeconds": time.monotonic() - started,
        "selfPeakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rssScope": "Linux demonstration-process high-water RSS; excludes trainer"}
    destination = output / "evidence.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "argumentRepairs": evidence["argumentRepairs"], "reportPresent": evidence["reportPresent"],
                      "originalsUnchanged": True, "fixtureFallback": False,
                      "evidenceSha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Existing own UNIDAD artifact")
    parser.add_argument("--output-dir", required=True, help="New output; never overwrite prior evidence")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.model_dir, args.output_dir)
