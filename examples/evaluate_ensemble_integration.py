"""Actual own-model ensemble execution, bounded coverage and Spanish probes.

No training or generated-answer substitution occurs here. Tracing wrappers pass
through to the shared engine and the actual auxiliary RNN scorer. New fictional
prompts measure execution and retain outputs for semantic inspection, without
inventing an automatic semantic grader or asserting conversational acceptance.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
from unittest.mock import patch

from ia_propia.brain.models import ModelRuntime
from ia_propia.conversation.catalog import load_catalog
from ia_propia.conversation.ensemble import ConversationEnsemble, EnsembleBudget
from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.sovengi import Sovengi


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
SYSTEM = "Eres SOVENGI. Responde en español."
QUALITY_TOKEN_CAP = 24


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fictional_cases():
    rows = [
        ("opening", ["El taller Velofresa abre el jueves.", "¿Qué día abre el taller Velofresa?"],
         "Debe indicar jueves; se proporciona el dato en la historia."),
        ("correction", ["El taller Miraflojo abre el martes.",
                        "Corrijo: abre el sábado. ¿Qué día abre el taller Miraflojo?"],
         "Debe usar sábado y conservar la corrección."),
        ("short-antecedent", ["El taller Espiralta abre el domingo.",
                              "Retomemos el día de apertura del taller Espiralta.", "¿Y qué día era?"],
         "Debe resolver el antecedente como la apertura del taller: domingo."),
        ("goal-constraint", ["Quiero organizar una visita al taller Oriluz.",
                            "No publiques datos personales.", "¿Qué quiero hacer y qué debo evitar?"],
         "Debe conservar visita y prohibición de publicar datos personales."),
        ("missing-information", ["No te he dado el nombre del farero. ¿Cómo se llama?"],
         "Debe reconocer que no tiene ese nombre, sin inventarlo."),
        ("greeting", ["Hola, ¿cómo estás?"],
         "Debe contestar al saludo con español comprensible."),
        ("location", ["La libreta Nacarín está en la caja verde. ¿Dónde está la libreta?"],
         "Debe indicar la caja verde."),
        ("next-step", ["Quiero preparar una visita. Primero debo confirmar horario. ¿Qué hago primero?"],
         "Debe indicar confirmar el horario, sin inventar acciones ejecutadas."),
    ]
    return [{"id": "ensemble-heldout-v1/" + name,
             "messages": [{"role": "system", "content": SYSTEM}] +
                         [{"role": "user", "content": text} for text in texts],
             "semanticReviewCriterion": criterion} for name, texts, criterion in rows]


class ActualCallTrace:
    """Observe common-engine calls without replacing any computation or output."""

    def __init__(self):
        self.generations, self.critics, self.rnn = [], [], []
        self._stack = ExitStack()

    def __enter__(self):
        original_reply = ConversationEngine.reply
        original_score = ConversationEngine.score_reply
        original_rnn = ModelRuntime._score_language

        def generation(engine, *args, **kwargs):
            started = time.perf_counter()
            result = original_reply(engine, *args, **kwargs)
            messages = kwargs.get("messages", args[0] if args else None)
            self.generations.append({"modelId": engine.model_id,
                "inputMessages": deepcopy(messages), "inputParameters": deepcopy({
                    key: value for key, value in kwargs.items() if key != "messages"}),
                "rawReply": deepcopy(result), "textSha256": hashlib.sha256(result["text"].encode("utf-8")).hexdigest(),
                "generatedTokenIdsSha256": value_sha(result["generatedTokenIds"]),
                "elapsedSeconds": time.perf_counter() - started,
                "verifiedArtifactFiles": deepcopy(engine.manifest["files"]),
                "weightsVerifiedByCommonArtifactLoader": True})
            return result

        def critic(engine, messages, text):
            started = time.perf_counter()
            result = original_score(engine, messages, text)
            self.critics.append({"criticModelId": engine.model_id, "inputMessages": deepcopy(messages),
                "candidateText": text, "candidateTextSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "rawScore": deepcopy(result), "elapsedSeconds": time.perf_counter() - started})
            return result

        def rnn(runtime, text):
            started = time.perf_counter()
            result = original_rnn(runtime, text)
            self.rnn.append({"modelId": runtime._language["modelId"], "candidateText": text,
                "candidateTextSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "rawScore": deepcopy(result), "elapsedSeconds": time.perf_counter() - started,
                "use": "auxiliary diagnostic and tie-break; not semantic correctness"})
            return result

        self._stack.enter_context(patch.object(ConversationEngine, "reply", new=generation))
        self._stack.enter_context(patch.object(ConversationEngine, "score_reply", new=critic))
        self._stack.enter_context(patch.object(ModelRuntime, "_score_language", new=rnn))
        return self

    def __exit__(self, *arguments):
        return self._stack.__exit__(*arguments)

    def mark(self):
        return len(self.generations), len(self.critics), len(self.rnn)

    def since(self, mark):
        return {"generationCalls": deepcopy(self.generations[mark[0]:]),
                "criticCalls": deepcopy(self.critics[mark[1]:]),
                "rnnCalls": deepcopy(self.rnn[mark[2]:])}


def winner_evidence(result, calls):
    matched = [call for call in calls["generationCalls"] if (
        call["rawReply"]["modelId"] == result["modelId"] and
        call["rawReply"]["text"] == result["text"] and
        call["rawReply"]["generatedTokenIds"] == result["generatedTokenIds"])]
    return {"exactOriginalGeneratedTextAndTokenIds": bool(matched),
            "modelId": result["modelId"], "textSha256": hashlib.sha256(result["text"].encode("utf-8")).hexdigest(),
            "generatedTokenIdsSha256": value_sha(result["generatedTokenIds"]),
            "matchingActualGenerationCalls": len(matched)}


def quality(config, directory, traces):
    outputs, apps = [], {}
    try:
        for arm in ("single_primary", "default_ensemble"):
            local = deepcopy(config)
            # Catalog artifact paths stay rooted at the real, read-only project.
            # Every mutable session, note and context file has a private root.
            local["projectRoot"] = str(ROOT)
            local["brain"]["state_dir"] = str(directory / arm / "brain")
            if arm == "single_primary":
                local["brain"].pop("conversation_ensemble", None)
            path = directory / (arm + ".json")
            write_json(path, local)
            apps[arm] = Sovengi(path)
        for index, case in enumerate(fictional_cases()):
            pair = {**deepcopy(case), "inputMessagesSha256": value_sha(case["messages"]), "arms": {}}
            for arm, app in apps.items():
                session = app.connect("heldout-" + str(index), context_session_id="heldout-" + str(index))
                mark = traces.mark()
                started = time.perf_counter()
                response = session.request({"schemaVersion": 1, "operation": "conversation.reply", "input": {
                    "messages": deepcopy(case["messages"]), "max_new_tokens": QUALITY_TOKEN_CAP,
                    "temperature": 0, "seed": SEED}})
                calls = traces.since(mark)
                elapsed = time.perf_counter() - started
                record = {"rawResponse": response, "elapsedSeconds": elapsed, **calls,
                    "semanticAssessment": {"status": "requires_inspection_of_raw_output",
                        "automaticSemanticGrader": False,
                        "criterion": case["semanticReviewCriterion"]}}
                if response["status"] == "ok":
                    record["winnerEvidence"] = winner_evidence(response["result"], calls)
                pair["arms"][arm] = record
                print(json.dumps({"event": "actual_ensemble_quality", "caseId": case["id"], "arm": arm,
                    "status": response["status"], "text": response.get("result", {}).get("text"),
                    "generatedCandidates": len(calls["generationCalls"]),
                    "critics": len(calls["criticCalls"]), "rnn": len(calls["rnnCalls"]),
                    "elapsedSeconds": elapsed}, ensure_ascii=False), flush=True)
            outputs.append(pair)
        capabilities = {arm: app.project.brain.models.capabilities() for arm, app in apps.items()}
        return {"cases": outputs, "capabilitiesAfterExecution": capabilities,
                "generation": {"maxNewTokensPerCandidate": QUALITY_TOKEN_CAP, "temperature": 0, "seed": SEED},
                "comparisonScope": "Different real prompts are fixed before generation. Same primary weights, input histories, temperature and seed. The ensemble adds bounded generation and scoring work; selection is not a trained semantic verifier."}
    finally:
        for app in apps.values():
            app.close()


def coverage(catalog, config, traces):
    descriptors = [{"name": entry["name"], "model_dir": entry["artifactDirectory"],
                    "expected_model_id": entry["modelId"], "decoder_backend": "torch", "use_cache": True}
                   for entry in catalog]
    primary = config["brain"]["conversation_ensemble"]["primary"]
    selected = next(entry for entry in catalog if entry["name"] == primary)
    budget = EnsembleBudget(max_candidates=4, min_candidates=2, max_resident=4,
        max_seconds=60, total_new_tokens=8, max_score_calls=2, critic_count=1,
        max_scoring_tokens=64, max_sliding_score_tokens=0)
    runtime = ModelRuntime(conversation_model=selected["artifactDirectory"],
        conversation_models=descriptors, ensemble_budget=asdict(budget), ensemble_primary=primary,
        language_backend=config["brain"]["language_backend"])
    requests, actual_ids = [], set()
    try:
        for index in range(17):
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "El faro ficticio Trazul está junto al lago. ¿Dónde está?"}]
            mark = traces.mark()
            started = time.perf_counter()
            result, error = None, None
            try:
                result = runtime.execute("conversation.reply", {"messages": deepcopy(messages),
                    "max_new_tokens": 2, "temperature": 0, "seed": SEED})
            except Exception as exception:
                error = {"type": type(exception).__name__, "reason": str(exception),
                         "trace": deepcopy(getattr(exception, "trace", None))}
            calls = traces.since(mark)
            actual_ids.update(call["modelId"] for call in calls["generationCalls"])
            row = {"requestIndex": index, "inputMessages": messages,
                "elapsedSeconds": time.perf_counter() - started, "rawResult": result, "error": error, **calls}
            if result is not None:
                row["winnerEvidence"] = winner_evidence(result, calls)
            requests.append(row)
            print(json.dumps({"event": "actual_ensemble_coverage", "request": index,
                "generatedModelIds": [call["modelId"] for call in calls["generationCalls"]],
                "distinctGeneratedIds": len(actual_ids), "error": error is not None,
                "elapsedSeconds": row["elapsedSeconds"]}, ensure_ascii=False), flush=True)
            if len(actual_ids) == len(catalog):
                break
        cache = runtime.conversation_engine.cache_info()
        missing = [entry["modelId"] for entry in catalog if entry["modelId"] not in actual_ids]
        return {"budget": asdict(budget), "maxNewTokensPerCandidate": 2, "requests": requests,
                "distinctModelsActuallyGenerated": len(actual_ids), "expectedModels": len(catalog),
                "missingModelIds": missing, "allCatalogModelsActuallyGenerated": not missing,
                "generatedModelIds": sorted(actual_ids), "cacheAfterExecution": cache,
                "scoringBudgetLimitation": "Two score calls cannot score all four nonempty candidates under a complete critic block. Such requests explicitly skip critic scoring; this section establishes actual neural execution coverage, not all-model semantic review or cross-critic coverage."}
    finally:
        runtime.close()


def run(config_path, output_path):
    import torch

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble_config = config["brain"]["conversation_ensemble"]
    catalog_path = ROOT / ensemble_config["catalog"]
    catalog = load_catalog(catalog_path, ROOT)
    if len(catalog) != 35 or not all(entry["available"] for entry in catalog):
        raise ValueError("This finite evaluation expects the 35 materialized own models")
    # Freeze all probes and provenance before the first reply.
    cases = fictional_cases()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ActualCallTrace() as traces, tempfile.TemporaryDirectory(prefix="sovengi-ensemble-eval-") as temporary:
            quality_result = quality(config, Path(temporary), traces)
            coverage_result = coverage(catalog, config, traces)
    finally:
        torch.set_num_threads(previous_threads)
    all_winners = [row["arms"][arm].get("winnerEvidence", {}).get("exactOriginalGeneratedTextAndTokenIds", False)
                   for row in quality_result["cases"] for arm in row["arms"]]
    report = {"schemaVersion": 1, "experiment": "sovengi-own-ensemble-real-evaluation-v1",
        "startedAtUtc": started_utc, "completedAtUtc": datetime.now(timezone.utc).isoformat(),
        "elapsedSeconds": time.perf_counter() - started,
        "generatorSha256": sha(__file__), "configSha256": sha(config_path),
        "catalogSha256": sha(catalog_path), "frozenPromptsSha256": value_sha(cases),
        "componentSha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "src/ia_propia/conversation/ensemble.py", ROOT / "src/ia_propia/conversation/catalog.py",
            ROOT / "src/ia_propia/brain/models.py", ROOT / "src/ia_propia/sovengi/application.py")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "cpuCountReported": os.cpu_count(), "torchThreads": 1,
            "maxProcessRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "execution": "local_cpu", "paidServicesUsed": False, "trainingExecuted": False,
            "networkCallsRequiredForEvaluation": False},
        "source": {"kind": "new-authored-fictional-conversations-by-Codex", "personalData": False,
            "promptsFrozenBeforeGeneration": True, "usedForTraining": False,
            "heldOutScope": "New complete authored prompts; no claim of unseen vocabulary in all historical pretraining documents.",
            "applicationStateIsolatedInTemporaryDirectories": True,
            "generatedTextAndTokensUnmodified": True},
        "catalog": catalog, "qualityComparison": quality_result, "modelCoverage": coverage_result,
        "exactOriginalWinnerForEverySuccessfulQualityResponse": all(all_winners),
        "semanticReview": {"automaticSemanticGrader": False, "status": "raw_outputs_require_inspection",
            "notHumanIndependentEvaluation": True, "generalConversationValidated": False},
        "limitations": ["Execution and likelihood ranking do not prove understanding, truth, specialized expertise or improved conversational quality.",
            "Reply caps of 24 tokens for quality and two for coverage bound runtime and limit conclusions.",
            "The time budget is cooperative between operations; a running neural call can exceed it.",
            "Thirty-five distinct weight identities are historical own models and are not independent training replications.",
            "Source support is lexical; the auxiliary RNN is a diagnostic and tie-break, not a semantic critic.",
            "The coverage pass uses a cheaper scoring budget than the default quality configuration."]}
    write_json(output_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gnosis.json")
    parser.add_argument("--output", default="docs/sovengi-ensemble-20261013/real-evaluation.json")
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.output)
    print(json.dumps({"event": "ensemble_real_evaluation_complete",
        "modelsActuallyGenerated": result["modelCoverage"]["distinctModelsActuallyGenerated"],
        "allCatalogModelsActuallyGenerated": result["modelCoverage"]["allCatalogModelsActuallyGenerated"],
        "exactOriginalQualityWinners": result["exactOriginalWinnerForEverySuccessfulQualityResponse"],
        "elapsedSeconds": result["elapsedSeconds"]}, ensure_ascii=False), flush=True)
