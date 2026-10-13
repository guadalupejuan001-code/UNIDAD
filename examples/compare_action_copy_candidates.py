"""Freeze new DEV tasks, then compare fixed own candidates through the real API.

No fixture, argument repair, replacement action, decoding filter or training.
Every examined task remains DEV and is excluded from future training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import threading
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "docs/learning-repair-20261013/action-copy-v2-development-tasks-v1"
PRIVATE = ROOT / "private/learning-repair-20261013/action-copy-v2-development-tasks-v1"
MODELS = {
    "treatment": ("models/learning-repair-20261013/autonomy/action-copy-v2-treatment-final-best256",
                  "unidad-conversation-e70ea28e8d5dfb07481bcf966058f022ed3eebd392fc4afe773102a6e2e88088"),
    "control": ("models/learning-repair-20261013/autonomy/action-copy-v2-control-final-best256",
                "unidad-conversation-734c92f9440d2d03bf1a6372e74335818e8ed65f00b69b24c2038d0ffffc0ac5")}
CORPORA = (
    "examples/learning-repair-20261013/autonomy-action-sft-v1/autonomy-action-sft.train-validation.jsonl",
    "examples/learning-repair-20261013/autonomy-action-copy-v2/autonomy-action-copy.train-validation.jsonl")
SOURCE_FILES = ("src/ia_propia/brain/core.py", "src/ia_propia/brain/autonomy.py", "src/ia_propia/brain/autonomy_model.py",
                "src/ia_propia/brain/file_tools.py", "src/ia_propia/brain/http.py", "src/ia_propia/conversation/runtime.py",
                "src/ia_propia/conversation/model.py", "src/ia_propia/conversation/tokenizer.py")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return sha(path)


def norm(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(c))


def task_specs():
    return [
        {"id": "generic-material-review", "family": "development-new-world:Zelvoria", "world": "Zelvoria",
         "goal": "Analiza las fuentes registradas y guarda un informe verificable sin cambiarlas.",
         "files": {"cobre claro.csv": "sector,unidades\nZelvoria,17\n",
                   "letra tenue.md": "# Zelvoria\nMaterial ficticio: 23 unidades.\n"},
         "report": "cotejo-lago.json", "goalStyle": "brief-natural-generic",
         "requiredCommittedToolSequence": None},
        {"id": "ordered-archive-review", "family": "development-new-world:Orsental", "world": "Orsental",
         "goal": "Inspecciona primero «turno blanco.txt», lee después «pauta violeta.md» y guarda «nota-faro.json» sin alterar las entradas.",
         "files": {"pauta violeta.md": "# Orsental\nInventario ficticio: 31 unidades.\n",
                   "turno blanco.txt": "Sector ficticio Orsental: lote de 29 unidades.\n"},
         "report": "nota-faro.json", "goalStyle": "explicit-tool-order",
         "requiredCommittedToolSequence": [
             {"tool": "inspect_file", "filename": "turno blanco.txt"},
             {"tool": "read_file", "filename": "pauta violeta.md"},
             {"tool": "save_report", "filename": "nota-faro.json"}]}]


def freeze():
    from tokenizers import Tokenizer
    tasks = task_specs()
    names, goals, families, worlds = set(), set(), set(), set()
    corpus_pins, split_counts = {}, {}
    for relative in CORPORA:
        path = ROOT / relative
        corpus_pins[relative] = sha(path)
        counts = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["split"] not in ("train", "validation"):
                    raise ValueError("DEV freeze must not read a TEST corpus")
                counts[record["split"]] = counts.get(record["split"], 0) + 1
                task = json.loads(record["messages"][1]["content"])["task"]
                names.update(norm(name) for name in [*task["task_inputs"]["files"], task["task_inputs"]["report"]])
                goals.add(task["goal"])
                families.add(record["source"]["family"])
                worlds.add(record["source"]["world"])
        split_counts[relative] = counts
    names.update(norm(n) for n in ("uno.txt", "dos.txt", "modelo.json", "informe.json", "k.txt"))
    model_pins = {}
    for arm, (relative, model_id) in MODELS.items():
        path = ROOT / relative
        manifest = json.loads((path / "manifest.json").read_bytes())
        if manifest["modelId"] != model_id or manifest["training"]["selectedStep"] != 256:
            raise ValueError("Candidates must be the previously validation-selected published BEST256")
        files = {name: {"bytes": (path / name).stat().st_size, "sha256": sha(path / name)}
                 for name in ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")}
        for name, expected in manifest["files"].items():
            if files[name] != expected:
                raise ValueError("Candidate artifact manifest hash mismatch")
        model_pins[arm] = {"path": relative, "modelId": model_id, "files": files}
    tokenizer = Tokenizer.from_file(str(ROOT / MODELS["treatment"][0] / "tokenizer.json"))
    for task in tasks:
        task_names = [*task["files"], task["report"]]
        if (any(norm(name) in names for name in task_names) or task["goal"] in goals
                or task["family"] in families or task["world"] in worlds):
            raise ValueError("New DEV task overlaps prior TRAIN/validation or demo")
        if any(not 6 <= len(name) <= 24 for name in task_names):
            raise ValueError("New task basename exceeds6–24-character bounds")
        task["filenameBpeLengths"] = {name: len(tokenizer.encode(name, add_special_tokens=False).ids) for name in task_names}
        task["inputSha256"] = {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in task["files"].items()}
    frozen = {"schemaVersion": 1, "kind": "pre-generation-fixed-candidate-new-development-task-freeze",
              "atUtc": datetime.now(timezone.utc).isoformat(), "tasks": tasks, "models": model_pins,
              "previousTrainingValidationCorpora": corpus_pins, "previousSplitCountsRead": split_counts,
              "sourceFilesSha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
              "helperSha256": sha(Path(__file__)), "candidateSelection": "Native validation-selected BEST256 fixed before these tasks or responses",
              "options": {"max_steps": 8, "max_failures": 2, "max_seconds": 90, "max_argument_repairs": 0,
                          "max_new_tokens": 128, "temperature": 0, "seed": 42},
              "budgets": {"cpuCount": 1, "taskInvocationSeconds": 150, "globalSeconds": 600,
                          "rssMiB": 6144, "diskMiB": 512, "network": "localhost authenticated API only"},
              "successCriteria": ["Own-model source and raw proposals", "Host current-task verified report at exact requested basename",
                                  "Task status completed", "Original bytes unchanged", "Explicit task follows committed tool order when requested",
                                  "No argument repair, fixture fallback, action substitution or decoding filter"],
              "originalConservationAloneIsSuccess": False, "acceptance": "DEV only; not fresh conversation or production acceptance",
              "excludeExaminedTasksFromFutureTraining": True, "testOrReservedConversationRead": False,
              "exactBasenameGoalFamilyAndWorldOverlap": False, "modelGenerationExecuted": False,
              "author_AI": True, "independentHumanReview": False}
    PUBLIC.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False, mode=0o700)
    digest = write(PUBLIC / "task-freeze.json", frozen)
    write(PRIVATE / "task-freeze.json", frozen)
    write(PUBLIC / "future-training-exclusions.json", {"schemaVersion": 1, "reason": "Frozen DEV tasks; retain exclusion after results examined",
          "taskIds": [t["id"] for t in tasks], "families": [t["family"] for t in tasks],
          "basenames": [name for t in tasks for name in [*t["files"], t["report"]]], "freezeSha256": digest})
    return {"freezePath": str((PUBLIC / "task-freeze.json").relative_to(ROOT)), "freezeSha256": digest,
            "taskCount": len(tasks), "modelGenerationExecuted": False}


def read_freeze():
    frozen = json.loads((PUBLIC / "task-freeze.json").read_bytes())
    if (PRIVATE / "task-freeze.json").read_bytes() != (PUBLIC / "task-freeze.json").read_bytes():
        raise ValueError("Private/public freeze differs")
    if frozen["helperSha256"] != sha(Path(__file__)):
        raise ValueError("Frozen helper changed")
    for path, expected in frozen["sourceFilesSha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen runtime source changed: " + path)
    for path, expected in frozen["previousTrainingValidationCorpora"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Original training corpus changed")
    return frozen


@contextmanager
def local_api(session):
    from ia_propia.brain.http import create_server
    token = secrets.token_urlsafe(32)
    server = create_server(session, token, host="127.0.0.1", port=0)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    transcript = []

    def request(operation, values):
        payload = {"schemaVersion": 1, "requestId": "copy-dev-" + str(len(transcript)), "operation": operation, "input": values}
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=105)
        try:
            connection.request("POST", "/v1/requests", body=compact(payload).encode("utf-8"),
                               headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.loads(response.read())
            transcript.append({"request": payload, "httpStatus": response.status, "response": result})
            if result.get("status") != "ok":
                raise RuntimeError("Actual API failed: " + compact(result))
            return result["result"]
        finally:
            connection.close()

    try:
        yield request, transcript
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        if worker.is_alive():
            raise RuntimeError("Local API server did not close")


def worker_case(arm, case_id):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from ia_propia.brain import Brain
    from ia_propia.brain.autonomy import AutonomyLimits
    from ia_propia.brain.autonomy_model import ModelActionProposer
    from ia_propia.brain.file_tools import SafeFileTools
    frozen = read_freeze()
    task = next(t for t in frozen["tasks"] if t["id"] == case_id)
    pin = frozen["models"][arm]
    model = ROOT / pin["path"]
    for name, expected in pin["files"].items():
        if (model / name).stat().st_size != expected["bytes"] or sha(model / name) != expected["sha256"]:
            raise ValueError("Frozen candidate bytes changed")
    output = PRIVATE / arm / case_id
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    inputs, reports = output / "inputs", output / "reports"
    inputs.mkdir()
    reports.mkdir()
    for filename, text in task["files"].items():
        (inputs / filename).write_text(text, encoding="utf-8")
    before = {filename: sha(inputs / filename) for filename in task["files"]}
    started = time.monotonic()
    brain = Brain(output / "state", conversation_model=model)
    if brain.models.conversation_engine.model_id != pin["modelId"]:
        raise ValueError("Loaded own model differs from candidate pin")
    options = frozen["options"]
    proposer = ModelActionProposer(brain.models, max_new_tokens=options["max_new_tokens"], temperature=options["temperature"], seed=options["seed"])
    with SafeFileTools(inputs, reports, filenames=list(task["files"])) as files:
        session = brain.connect("copy-dev-" + arm + "-" + case_id, autonomy_proposer=proposer,
            autonomy_tools=files.tools, autonomy_limits=AutonomyLimits(**{k: options[k] for k in ("max_steps", "max_failures", "max_seconds", "max_argument_repairs")}),
            autonomy_completion_verifier=files.completion_verifier, autonomy_input_validator=files.validate_task_inputs)
        with local_api(session) as (request, transcript):
            initial = request("autonomy.start", {"goal": task["goal"], "task_inputs": {"files": list(task["files"]), "report": task["report"]}})
            final = request("autonomy.run", {"task_id": initial["task_id"]})
            persisted = request("autonomy.status", {"task_id": initial["task_id"]})
            if persisted != final:
                raise ValueError("Persistent task snapshot differs after real API execution")
        host_verified = files.completion_verifier(final).get("verified") is True
        after = {filename: sha(inputs / filename) for filename in task["files"]}
        report_hashes = {p.name: sha(p) for p in reports.iterdir() if p.is_file()}
        report_path = reports / task["report"]
        report_document = json.loads(report_path.read_bytes()) if report_path.exists() else None
    journal = final["journal"]
    proposals = [event for event in journal if event["kind"] == "proposal"]
    committed = [event for event in journal if event["kind"] == "tool_result"]
    sequence = [{"tool": event["tool"], "filename": event["arguments"].get("filename")} for event in committed]
    order_met = task["requiredCommittedToolSequence"] is None or sequence == task["requiredCommittedToolSequence"]
    unchanged = before == after == task["inputSha256"]
    success = final["source"] == "own_model" and final["status"] == "completed" and host_verified and report_document is not None and unchanged and order_met
    evidence = {"schemaVersion": 1, "atUtc": datetime.now(timezone.utc).isoformat(), "arm": arm, "caseId": case_id,
                "scope": "Frozen new DEV task; actual own model, authenticated localhost API and SafeFileTools",
                "freezeSha256": sha(PUBLIC / "task-freeze.json"), "modelId": pin["modelId"], "modelFiles": pin["files"],
                "goal": task["goal"], "taskInputs": {"files": list(task["files"]), "report": task["report"]},
                "options": options, "source": final["source"], "status": final["status"], "finalTask": final,
                "apiTranscript": transcript, "rawProposals": proposals, "actualCommittedToolResults": committed,
                "committedToolSequence": sequence, "explicitToolOrderMet": order_met, "hostCompletionVerified": host_verified,
                "originalHashesBefore": before, "originalHashesAfter": after, "originalsUnchanged": unchanged,
                "requestedReportPresent": report_document is not None, "reportHashes": report_hashes, "requestedReportDocument": report_document,
                "taskSuccess": success, "originalConservationAloneIsSuccess": False,
                "fixtureFallback": False, "argumentRepair": False, "argumentSubstitution": False, "decodingFilters": False,
                "modelGenerationExecuted": True, "trainingExecuted": False, "experiencesRetrainModel": False,
                "excludeThisExaminedTaskFromFutureTraining": True, "freshConversationAcceptance": False,
                "elapsedSeconds": time.monotonic() - started, "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cpuCount": 1, "network": "localhost only", "privatePersistence": True}
    if any(event["kind"] == "argument_repair_requested" for event in journal):
        raise ValueError("Argument repair occurred in an explicitly disabled comparison")
    if evidence["peakRssBytes"] > 6144 * 1024**2:
        raise MemoryError("Task RSS ceiling exceeded")
    digest = write(output / "evidence.json", evidence)
    return {"arm": arm, "caseId": case_id, "status": final["status"], "taskSuccess": success,
            "evidencePath": str((output / "evidence.json").relative_to(ROOT)), "evidenceSha256": digest}


def run_all():
    frozen = read_freeze()
    started = time.monotonic()
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(ROOT / "src")
    for case in frozen["tasks"]:
        for arm in ("treatment", "control"):
            if time.monotonic() - started >= 450:
                raise TimeoutError("Global600-second ceiling cannot admit another150-second task")
            output = PRIVATE / arm / case["id"]
            command = [sys.executable, str(Path(__file__)), "worker", "--arm", arm, "--case", case["id"]]
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=150)
            if process.returncode:
                raise RuntimeError("Actual candidate worker failed: " + process.stderr[-4096:])
            result = json.loads(process.stdout)
            evidence = json.loads((ROOT / result["evidencePath"]).read_bytes())
            public_case = PUBLIC / arm / case["id"]
            public_case.mkdir(parents=True, exist_ok=False)
            raw = (ROOT / result["evidencePath"]).read_bytes()
            if str(ROOT).encode() in raw:
                raise ValueError("Absolute host path in evidence; private result retained for review")
            (public_case / "evidence.json").write_bytes(raw)
            result["publicEvidencePath"] = str((public_case / "evidence.json").relative_to(ROOT))
            result["originalsUnchanged"] = evidence["originalsUnchanged"]
            result["requestedReportPresent"] = evidence["requestedReportPresent"]
            result["hostCompletionVerified"] = evidence["hostCompletionVerified"]
            result["proposalCount"] = len(evidence["rawProposals"])
            result["committedToolCount"] = len(evidence["actualCommittedToolResults"])
            results.append(result)
            print(compact(result), flush=True)
    total_disk = sum(p.stat().st_size for folder in (PRIVATE, PUBLIC) for p in folder.rglob("*") if p.is_file())
    if total_disk > 512 * 1024**2:
        raise ValueError("Task evidence disk ceiling exceeded")
    summary = {"schemaVersion": 1, "kind": "fixed-candidate-two-new-dev-task-comparison",
               "freezeSha256": sha(PUBLIC / "task-freeze.json"), "results": results,
               "completedTasksByArm": {arm: sum(r["taskSuccess"] for r in results if r["arm"] == arm) for arm in MODELS},
               "caseCountPerArm": 2, "originalsUnchangedEveryRun": all(r["originalsUnchanged"] for r in results),
               "argumentRepair": False, "fixtureFallback": False, "decodingFilters": False,
               "elapsedSeconds": time.monotonic() - started, "totalPrivateAndPublicBytes": total_disk,
               "excludeExaminedTasksFromFutureTraining": True, "freshConversationAcceptance": False,
               "productionAccepted": False, "humanReview": False,
               "limitation": "Two examined DEV tasks compare these fixed candidates; no final reserved acceptance, universal generalization or conversation-competence claim."}
    digest = write(PUBLIC / "summary.json", summary)
    write(PRIVATE / "summary.json", summary)
    write(PUBLIC / "manifest.json", {"schemaVersion": 1, "freezeSha256": summary["freezeSha256"], "summarySha256": digest,
          "helperSha256": frozen["helperSha256"], "realResults": [{"path": r["publicEvidencePath"], "sha256": r["evidenceSha256"]} for r in results],
          "developmentOnly": True, "excludedFromFutureTraining": True, "trainingExecuted": False})
    return {"summaryPath": str((PUBLIC / "summary.json").relative_to(ROOT)), "summarySha256": digest,
            "completedTasksByArm": summary["completedTasksByArm"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run", "worker"))
    parser.add_argument("--arm", choices=tuple(MODELS))
    parser.add_argument("--case")
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (600 if args.operation == "run" else 150,) * 2)
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3,) * 2)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Finite comparison invocation wall ceiling")))
    signal.alarm(600 if args.operation == "run" else 150)
    if args.operation == "freeze":
        result = freeze()
    elif args.operation == "run":
        result = run_all()
    else:
        if not args.arm or not args.case:
            parser.error("worker needs an arm and frozen case")
        result = worker_case(args.arm, args.case)
    print(compact(result), flush=True)


if __name__ == "__main__":
    main()
