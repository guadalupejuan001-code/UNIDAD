"""Bounded fixture supervision from real file-tool/controller executions.

This program loads a tokenizer, never model weights. Labels are an explicitly
declared project-authored fixture, not raw output from a learned planner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import time
from types import SimpleNamespace


DATASET = "unidad-autonomy-action-sft-v1"
SOURCE = "project-authored synthetic fixture supervision with actual filesystem measurements"
DEFAULT_TOKENIZER = Path("models/learning-repair-20261013/conversation-15m/intermediate-step004096/tokenizer.json")


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "min": values[0], "max": values[-1],
            "median": values[len(values) // 2], "sum": sum(values)}


def compact_projection(context):
    """Pure inspection view, checked against the real adapter's captured request."""
    from ia_propia.brain.autonomy_model import ACTION_INSTRUCTION, _observations, _schema_constraints
    task = context["task"]
    view = {key: deepcopy(task[key]) for key in ("goal", "task_inputs") if key in task}
    tools = [{key: (_schema_constraints(tool[key]) if key == "schema" else deepcopy(tool[key]))
              for key in ("name", "schema", "effect") if key in tool} for tool in context["tools"]]
    observations = _observations(context.get("journal", task.get("journal", [])))
    return [{"role": "system", "content": ACTION_INSTRUCTION},
            {"role": "user", "content": compact({"task": view, "tools": tools, "observations": observations})}]


class DeclaredActionFixture:
    """Exercise the actual prompt adapter using a clearly named fixture engine."""
    origin = "test_fixture"
    proposer_id = "project-authored-action-label-fixture-not-model-generation"

    def __init__(self, tokenizer, actions):
        from ia_propia.brain.autonomy_model import ModelActionProposer
        self.actions = iter(deepcopy(actions))
        self.captures = []
        self.last_metadata = {}
        self._next_raw = None

        class FixtureEngine:
            model_id = "declared-test-fixture-no-weights"
            _model = SimpleNamespace(config={"context_length": 2048})
            _tokenizer = tokenizer

            def reply(engine, **request):
                self._request = deepcopy(request)
                return {"text": self._next_raw, "source": "test_fixture", "modelGeneration": False}

        self._adapter = ModelActionProposer(SimpleNamespace(conversation_engine=FixtureEngine()),
                                           max_new_tokens=128, temperature=0, seed=42)

    def __call__(self, context):
        self._next_raw = compact(next(self.actions))
        raw = self._adapter(context)
        messages = self._request["messages"]
        if messages != compact_projection(context) or raw != self._next_raw:
            raise ValueError("Actual adapter differs from pure projection or changed fixture label")
        self.captures.append({"messages": deepcopy(messages), "rawAction": raw, "context": deepcopy(context)})
        self.last_metadata = {"source": "test_fixture", "modelGeneration": False,
                              "request": deepcopy(self._request), "rawReply": raw,
                              "promptTokens": self._adapter.last_metadata["promptTokens"],
                              "contextTokens": 2048, "discardedPromptTokens": 0}
        return raw


def check_small_context_rejections(captures, tokenizer):
    """Replay overflowing real contexts through the adapter, with reply forbidden."""
    from ia_propia.brain.autonomy_model import ModelActionProposer
    from ia_propia.brain.protocol import BrainError
    from ia_propia.conversation.tokenizer import encode_prompt
    checks = []
    for step, capture in enumerate(captures, 1):
        _encoded, discarded = encode_prompt(capture["messages"], tokenizer, 1024)
        if not discarded:
            continue
        calls = []

        def forbidden_reply(**request):
            calls.append(request)
            raise AssertionError("An overflowing context must never call generation")

        engine = SimpleNamespace(model_id="declared-fixture-context-control-no-weights", reply=forbidden_reply,
                                 _tokenizer=tokenizer, _model=SimpleNamespace(config={"context_length": 1024}))
        adapter = ModelActionProposer(SimpleNamespace(conversation_engine=engine), max_new_tokens=128)
        try:
            adapter(capture["context"])
        except BrainError as error:
            if error.code != "context_too_large" or calls:
                raise
        else:
            raise ValueError("Expected actual adapter context rejection")
        if adapter.last_metadata["request"]["messages"] != capture["messages"]:
            raise ValueError("Smaller-context control changed the actual observations")
        checks.append({"step": step, "contextLength": 1024, "code": "context_too_large",
                       "actualAdapterInvoked": True, "generationCalls": len(calls),
                       "promptTokens": adapter.last_metadata["promptTokens"],
                       "discardedPromptTokens": adapter.last_metadata["discardedPromptTokens"],
                       "contextSha256": adapter.last_metadata["contextSha256"],
                       "modelWeightsLoaded": False, "positiveTrainingRecordCreated": False})
    return checks


def tool(name, arguments, reason):
    return {"action": "tool", "tool": name, "arguments": arguments, "reason": reason}


def build_task_specs():
    """Whole tasks and fictional worlds are disjoint across TRAIN and validation."""
    train_worlds = ("Dralvi", "Senzuk", "Belxor", "Nuvrek", "Kelzir", "Femdul", "Ravnit", "Zophel")
    val_worlds = ("Qutema", "Yobrin", "Wexal", "Husvek", "Tazmur", "Pildor", "Cazfen", "Gomsel")
    specs = []
    for split, count in (("train", 64), ("validation", 8)):
        for index in range(count):
            world = (train_worlds[index // 8] + str(index % 8)) if split == "train" else val_worlds[index]
            # Held-out basenames do not share the training filename prefix.
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            lead = "jxzvk"[index // 13] if split == "train" else "q"
            first = lead + alphabet[(index % 13) * 2] + ".txt"
            second = lead + alphabet[(index % 13) * 2 + 1] + ".txt"
            report = lead + alphabet[(index % 13) * 2] + ".json"
            mode = index % 4
            first_tool = "read_file" if mode == 0 else "inspect_file"
            second_tool = "read_file" if mode == 1 else "inspect_file"
            order = [first, second] if mode != 3 else [second, first]
            if split == "train":
                phrases = (
                    "Lee {a}; mide {b}; guarda {r}; preserva.",
                    "Mide {a}; lee {b}; crea {r}; conserva.",
                    "Mide {a}, luego {b}; escribe {r}; preserva.",
                    "Primero mide {a}, después {b}; guarda {r}; conserva.")
            else:
                phrases = (
                    "Consulta {a}, revisa medidas de {b}, entrega {r}; sin cambios.",
                    "Revisa medidas de {a}, consulta {b}, entrega {r}; sin cambios.",
                    "Revisa medidas de {a} y {b}; entrega {r}; sin cambios.",
                    "Revisa medidas de {a}, continúa con {b}; entrega {r}; sin cambios.")
            goal = phrases[mode].format(a=order[0], b=order[1], r=report)
            ident = "autonomy-fixture:" + split + ":" + world.lower()
            specs.append({"split": split, "groupId": ident, "family": "fictional-world:" + world,
                          "world": world, "goal": goal, "report": report,
                          "files": {first: f"{world} lote {index + 3}.\n",
                                    second: f"{world} cajas {index + 7}.\n"},
                          "actions": [tool(first_tool, {"filename": order[0]}, "Leer" if first_tool == "read_file" else "Medir"),
                                      tool(second_tool, {"filename": order[1]}, "Leer" if second_tool == "read_file" else "Medir"),
                                      tool("save_report", {"filename": report, "files": [first, second]},
                                           "Fuentes observadas")]})
    return specs


def execute_task(task_spec, tokenizer, workspace_parent=None):
    """Execute labels through the real controller; retain receipts, destroy roots."""
    from ia_propia.brain.autonomy import AutonomyController, AutonomyLimits
    from ia_propia.brain.file_tools import SafeFileTools
    from ia_propia.conversation.data import validate_record
    spec = deepcopy(task_spec)
    with tempfile.TemporaryDirectory(prefix="unidad-action-fixture-", dir=workspace_parent) as temporary:
        root = Path(temporary)
        inputs, reports = root / "inputs", root / "reports"
        inputs.mkdir()
        reports.mkdir()
        for filename, content in spec["files"].items():
            (inputs / filename).write_text(content, encoding="utf-8")
        before = {name: sha((inputs / name).read_bytes()) for name in spec["files"]}
        fixture = DeclaredActionFixture(tokenizer, spec["actions"])
        with SafeFileTools(inputs, reports, filenames=list(spec["files"])) as files:
            task_inputs = files.validate_task_inputs({"files": list(spec["files"]), "report": spec["report"]})
            controller = AutonomyController(root / "state", fixture, files.tools,
                                            AutonomyLimits(max_steps=3, max_failures=1, max_seconds=5),
                                            files.completion_verifier, policy_id="action-sft-real-fixture-v1")
            state = controller.start(spec["goal"], task_inputs=task_inputs)
            final = controller.run(state["task_id"])
            persisted = controller.status(state["task_id"])
            after = {name: sha((inputs / name).read_bytes()) for name in spec["files"]}
            if final["status"] != "completed" or before != after or persisted != final:
                raise ValueError("Fixture task did not persist verified completion with unchanged originals: "
                                 + compact({"status": final["status"], "steps": final["steps"],
                                            "lastEvents": final["journal"][-2:]}))
            report_bytes = (reports / spec["report"]).read_bytes()
            report_document = json.loads(report_bytes)
            if not files.completion_verifier(final)["verified"]:
                raise ValueError("Current task report not independently verified")
            records = []
            for index, capture in enumerate(fixture.captures):
                source = {"dataset": DATASET, "kind": "authored-synthetic", "description": SOURCE,
                          "author_AI": True, "independentHumanReview": False, "license": "CC0-1.0",
                          "modelResponsesUsed": False, "fixtureOrigin": "test_fixture",
                          "supervisionPolicy": "assistant_last_reply", "actualToolExecution": True,
                          "family": spec["family"], "world": spec.get("world", spec["family"]),
                          "step": index + 1, "tool": json.loads(capture["rawAction"]).get("tool"),
                          "privateConversationUsed": False}
                record = {"id": spec["groupId"] + ":step:" + str(index + 1),
                          "groupId": spec["groupId"], "split": spec["split"], "source": source,
                          "messages": [*capture["messages"], {"role": "assistant", "content": capture["rawAction"]}]}
                validate_record(record)
                if temporary in compact(record) or str(root) in compact(record):
                    raise ValueError("Temporary root entered model context")
                records.append(record)
            receipt = {"groupId": spec["groupId"], "family": spec["family"], "split": spec["split"],
                       "taskId": final["task_id"], "terminalStatus": final["status"], "source": final["source"],
                       "journal": final["journal"], "taskInputs": task_inputs, "goal": spec["goal"],
                       "inputContents": spec["files"], "originalHashesBefore": before, "originalHashesAfter": after,
                       "originalsUnchanged": True, "reportBytesSha256": sha(report_bytes),
                       "reportDocument": report_document, "persistedTaskJournalVerified": True,
                       "learnedPlanningProven": False, "privateRootsDestroyedAfterReceipt": True,
                       "recordIds": [r["id"] for r in records],
                       "context1024NegativeControls": check_small_context_rejections(fixture.captures, tokenizer)}
            if temporary in compact(receipt):
                raise ValueError("Private temporary path leaked in public receipt")
    return records, receipt


def audit_tokens(records, tokenizer):
    """Use the frozen native encoder; require full prompt and targets in one window."""
    from ia_propia.conversation.contract import ASSISTANT_ID, EOS_ID, IGNORE_INDEX
    from ia_propia.conversation.tokenizer import dialogue_tokens, encode_prompt
    from ia_propia.conversation.training import _record_windows
    maximum, prompt_maximum, target_count, eos_count = 0, 0, 0, 0
    step_measures = {}
    controls = []
    for record in records:
        ids, _mask = dialogue_tokens(record["messages"], tokenizer)
        prompt, discarded = encode_prompt(record["messages"][:-1], tokenizer, 2048)
        target = tokenizer.encode(record["messages"][-1]["content"], add_special_tokens=False).ids + [EOS_ID]
        if discarded or len(ids) - 1 > 2048:
            raise ValueError(f"Full prompt plus action exceeds 2048: {record['id']} ({len(ids)-1})")
        for policy in ("contiguous_nonoverlapping", "sliding_context"):
            windows = list(_record_windows(record, tokenizer, 2048, policy, "assistant_last_reply"))
            if len(windows) != 1:
                raise ValueError("Action supervision requires one complete native window")
            inputs, labels = windows[0]
            assistant = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
            if (inputs != ids[:-1] or [token for token in labels if token != IGNORE_INDEX] != target
                    or any(token != IGNORE_INDEX for token in labels[:assistant])
                    or sum(token == EOS_ID for token in labels) != 1):
                raise ValueError("Native lastreply masking or EOS mismatch")
        maximum = max(maximum, len(ids) - 1)
        prompt_maximum = max(prompt_maximum, len(prompt))
        target_count += len(target)
        eos_count += 1
        step = str(record.get("source", {}).get("step", 0))
        measures = step_measures.setdefault(step, {"promptTokens": [], "trainingInputTokens": []})
        measures["promptTokens"].append(len(prompt))
        measures["trainingInputTokens"].append(len(ids) - 1)
        for limit in (512, 1024):
            _cropped, dropped = encode_prompt(record["messages"][:-1], tokenizer, limit)
            controls.append({"id": record["id"], "step": step, "contextLength": limit,
                             "promptTokens": len(prompt), "discardedPromptTokens": dropped,
                             "adapterWouldBlockBeforeGeneration": dropped > 0,
                             "fullTrainingPromptAndTargetFit": len(ids) - 1 <= limit})
    return {"recordCount": len(records), "contextLength": 2048, "maxSequenceInputs": maximum,
            "maxPromptTokens": prompt_maximum, "targetTokensIncludingEos": target_count,
            "supervisedEos": eos_count, "eachFinalReplyTargetExactlyOnce": True,
            "historicalAssistantTargetsIgnored": True, "promptAndTargetFit": True,
            "nativeEncoderPoliciesChecked": ["contiguous_nonoverlapping", "sliding_context"],
            "torchLoaded": "torch" in sys.modules, "modelWeightsLoaded": False, "testRowsRead": 0,
            "stepTokenDistributions": {step: {key: distribution(values) for key, values in measures.items()}
                                       for step, measures in step_measures.items()},
            "smallerContextControlAudit": controls}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(output, tokenizer_path=DEFAULT_TOKENIZER):
    from tokenizers import Tokenizer
    from ia_propia.conversation.tokenizer import validate_tokenizer
    from ia_propia.conversation.data import load_corpus
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = validate_tokenizer(Tokenizer.from_file(str(tokenizer_path)), strict=True)
    if tokenizer.get_vocab_size() != 2048:
        raise ValueError("Require own parent BPE2048, without re-fitting")
    specs = build_task_specs()
    design = {"schemaVersion": 1, "dataset": DATASET, "status": "preregistered-before-fixture-executions",
              "hypothesis": "Own weights lack structured-action supervision; full three-tool context also exceeds 512.",
              "author_AI": True, "independentHumanReview": False,
              "source": SOURCE, "labelOrigin": "declared-test-fixture-not-model-output",
              "modelWeightsLoaded": False, "testUsedForTraining": False,
              "splitUnit": "whole task / fictional world / family / filename / goal phrase template",
              "registry": "Untouched three-tool SafeFileTools registry and schemas; actual handlers/verifiers",
              "genericRegistryLimit": "Untouched three-tool schemas with real observations exceed1024; future2048 extension required and not trained in preparation",
              "taskCounts": {"train": 64, "validation": 8, "test": 0},
              "trainingObjective": "assistant_last_reply",
              "tokenizerSha256": sha(tokenizer_path.read_bytes()),
              "futureExperiment": {"status": "proposed-not-executed", "ownParentModelIdPrefix": "072eedd2",
                  "contextLength": 2048, "preserveParentWeights": True,
                  "first": "Tiny overfit control first; no generalization claim from memorization",
                  "second": "Separate unseen filenames/tasks/groups DEV; measure raw JSON validity and actual tools with no fixture fallback",
                  "trainingUpdates": [256], "wallSecondsMaximum": 600,
                  "matchedControl": "same own parent, unextended 512 single-tool contexts; separate identical budget",
                  "success": "Actual raw model JSON plus completed current-task verified report and unchanged originals",
                  "failure": "Block malformed actions, loss without action improvement, context overflow; retain raw evidence"},
              "budgets": {"wallSeconds": 120, "cpuThreads": 1, "rssMiB": 256, "diskMiB": 128,
                          "networkUsed": False, "trainingStarted": False},
              "taskSpecSha256": sha(compact(specs).encode("utf-8"))}
    write_json(output / "preregister.json", design)
    all_records, receipts = [], []
    for spec in specs:
        records, receipt = execute_task(spec, tokenizer)
        all_records.extend(records)
        receipts.append(receipt)
        if time.monotonic() - started > 115:
            raise TimeoutError("Preparation exhausted preregistered wall budget")
    train = [r for r in all_records if r["split"] == "train"]
    validation = [r for r in all_records if r["split"] == "validation"]
    audit = {"train": audit_tokens(train, tokenizer), "validation": audit_tokens(validation, tokenizer)}
    for field in ("groupId",):
        if {r[field] for r in train} & {r[field] for r in validation}:
            raise ValueError("Whole-task split leakage")
    for field in ("family", "world"):
        if {r["source"][field] for r in train} & {r["source"][field] for r in validation}:
            raise ValueError("World/family split leakage")
    corpus_path = output / "autonomy-action-sft.train-validation.jsonl"
    corpus_path.write_text("".join(compact(record) + "\n" for record in all_records), encoding="utf-8")
    if load_corpus(corpus_path) != all_records:
        raise ValueError("Canonical corpus parser changed records")
    (output / "execution-receipts.jsonl").write_text("".join(compact(receipt) + "\n" for receipt in receipts), encoding="utf-8")
    # Content identity excludes provenance and serialization, matching supervisor semantics.
    content_sha = sha("".join(json.dumps(r["messages"], ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False) + "\n"
                              for r in all_records).encode("utf-8"))
    corpus_sha = sha(corpus_path.read_bytes())
    audit.update({"schemaVersion": 1, "fixtureTasksActuallyExecuted": len(receipts),
                  "tasksCompleted": sum(r["terminalStatus"] == "completed" for r in receipts),
                  "originalsUnchanged": all(r["originalsUnchanged"] for r in receipts),
                  "actualToolResults": sum(e["kind"] == "tool_result" for r in receipts for e in r["journal"]),
                  "actualAdapter1024Rejections": sum(len(r["context1024NegativeControls"]) for r in receipts),
                  "splitCounts": dict(Counter(r["split"] for r in all_records)),
                  "wholeFamilySplitsDisjoint": True, "learnedPlanningProven": False,
                  "elapsedSeconds": time.monotonic() - started,
                  "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "corpusSha256": corpus_sha,
                  "executionReceiptsSha256": sha((output / "execution-receipts.jsonl").read_bytes()),
                  "review": "Project AI/code measurements, not independent human evaluation"})
    if audit["peakRssMiB"] > 256 or audit["train"]["torchLoaded"] or audit["validation"]["torchLoaded"]:
        raise ValueError("Resource/no-weight preparation contract violated")
    write_json(output / "qa.json", audit)
    source_names = ("src/ia_propia/brain/autonomy_model.py", "src/ia_propia/brain/autonomy.py",
                    "src/ia_propia/brain/file_tools.py", "src/ia_propia/conversation/tokenizer.py",
                    "src/ia_propia/conversation/training.py")
    manifest = {"schemaVersion": 1, "dataset": DATASET, "status": "prepared-not-trained-not-autonomy-accepted",
                "corpusSha256": corpus_sha, "contentSha256": content_sha,
                "contentSerialization": "sorted compact UTF-8 messages JSON followed by LF, in corpus order",
                "file": {"path": corpus_path.name, "bytes": corpus_path.stat().st_size, "sha256": corpus_sha},
                "splitCounts": {"train": len(train), "validation": len(validation), "test": 0},
                "taskCounts": {"train": 64, "validation": 8, "test": 0},
                "testUsedForTraining": False, "testUsed": False,
                "sources": [{"dataset": DATASET, "kind": "authored-synthetic", "description": SOURCE,
                             "author_AI": True, "independentHumanReview": False, "license": "CC0-1.0",
                             "licensePath": "LICENSE-DATA.md", "networkUsed": False,
                             "actualFilesystemMeasurements": True, "modelGeneratedLabels": False}],
                "objective": "assistant_last_reply", "contextLength": 2048,
                "tokenizer": {"path": str(tokenizer_path), "sha256": sha(tokenizer_path.read_bytes()), "vocabSize": 2048},
                "preregisterSha256": sha((output / "preregister.json").read_bytes()),
                "qaSha256": sha((output / "qa.json").read_bytes()),
                "executionReceiptsSha256": audit["executionReceiptsSha256"],
                "helperSha256": sha(Path(__file__).read_bytes()),
                "sourceSha256": {name: sha(Path(name).read_bytes()) for name in source_names},
                "learnedPlanningProven": False, "modelWeightsLoaded": False, "trainingStarted": False}
    write_json(output / "manifest.json", manifest)
    (output / "LICENSE-DATA.md").write_text("# Dataset license\n\nProject-authored fictional task text, curated action labels, and local file measurements are dedicated under CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/ .\n\nAuthoring/review: project AI; no independent human certification. No private conversations, remote source records, model-generated labels, or original third-party text were included. Existing source code/tokenizer retain their repository license; this dedication applies to this new dataset only.\n", encoding="utf-8")
    (output / "README.md").write_text("# Supervisión de acciones con archivos reales\n\n192 TRAIN y 24 VALIDATION de 72 tareas ficticias separadas por tarea, familia, nombres y plantillas de objetivo. Cada etiqueta JSON fue escrita por una fixture declarada del proyecto. El controlador y SafeFileTools ejecutaron 216 acciones sobre archivos temporales reales, verificaron informes y conservaron originales por SHA-256; las raíces se destruyeron después del recibo.\n\nNo se cargaron pesos ni se generaron respuestas con modelos. Estos datos no demuestran planificación aprendida ni autonomía fiable. El experimento futuro registrado propone un control pequeño de memorización y luego DEV con tareas y nombres nuevos, usando pesos propios y contexto 2048. La preparación no entrena ni encola etapas.\n\n[Manifiesto](manifest.json), [diseño previo](preregister.json), [QA](qa.json), [recibos reales](execution-receipts.jsonl), [licencia](LICENSE-DATA.md). La auditoría usa el BPE2048 propio y ambos codificadores de ventanas nativos congelados; supervisa sólo el último JSON y su EOS, con todo el contexto y target dentro de 2048. TEST no se leyó ni se incluyó.\n\nReproducción desde la raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python3 examples/prepare_autonomy_action_sft.py --output-dir /tmp/unidad-action-sft-new`. Cada ejecución escribe una carpeta nueva y no sustituye evidencia anterior.\n", encoding="utf-8")
    disk = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if disk > 128 * 1024 ** 2:
        raise ValueError("Preparation output exceeded 128 MiB")
    return {"output": str(output), "corpusSha256": corpus_sha, "splitCounts": manifest["splitCounts"],
            "maxTrainInputs": audit["train"]["maxSequenceInputs"], "peakRssMiB": audit["peakRssMiB"],
            "elapsedSeconds": audit["elapsedSeconds"], "trainingStarted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 ** 2, 128 * 1024 ** 2))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Preparation wall ceiling")))
    signal.alarm(120)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    print(compact(prepare(args.output_dir, args.tokenizer)))


if __name__ == "__main__":
    main()
