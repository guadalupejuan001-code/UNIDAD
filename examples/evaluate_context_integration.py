"""Paired, held-out context probes through SOVENGI's actual ConversationEngine.

Fictional source text is conditioning, never a generated answer substitute. This
script does not train, change weights, or write in the application's private
memory. A narrow day-name diagnostic is separate from source-selection checks
and does not establish general conversational acceptance.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from unittest.mock import patch

from ia_propia.conversation.contract import MAX_DIALOGUE_BYTES, MAX_MESSAGES
from ia_propia.conversation.tokenizer import encode_prompt, validate_messages
from ia_propia.sovengi import Sovengi


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
MAX_REPLY_TOKENS = 24
DAY_NAMES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
# These entities do not occur in the ranker's entity families or the memory
# candidate's authored training/validation/development families.
ENTITIES = (
    "Asterión", "Bellavista", "Crisálida", "Dromedario", "Escarlata", "Fulgor",
    "Glicina", "Hemisferio", "Isóbara", "Jubiloso", "Kermés", "Luciérnaga",
)
SYSTEM = (
    "Eres SOVENGI. Contesta en español a la última pregunta usando los datos de "
    "esta conversación. Si hay una corrección posterior, usa el dato corregido. "
    "No inventes información que no se haya proporcionado."
)


def normalized(text):
    return "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(char) != "Mn")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def distractors():
    """Unrelated complete user turns; no fabricated assistant replies in history."""
    return [{"role": "user", "content": (
        f"Apartado ficticio independiente {index}: "
        + "En otra historia imaginaria revisamos fotografías de nubes, etiquetas "
          "de carpetas, mapas de montañas y registros de colores. Es material "
          "de prueba sin relación con la pregunta pendiente. " * 5)} for index in range(20)]


def authored_cases():
    cases = []
    for index, name in enumerate(ENTITIES):
        entity = "taller " + name
        kind = ("distant_fact", "correction", "short_antecedent", "goal_constraint")[index // 3]
        day = DAY_NAMES[(index * 3 + 1) % len(DAY_NAMES)]
        old_day = DAY_NAMES[(index * 3 + 2) % len(DAY_NAMES)]
        note = f"El {entity} abre el {day}."
        messages = [{"role": "system", "content": SYSTEM}]
        if kind == "goal_constraint":
            sources = [f"Mi objetivo es organizar una visita al {entity}.",
                       f"No publiques datos personales del {entity}; esa es la condición."]
            messages += [{"role": "user", "content": text} for text in sources]
            messages += distractors()
            query = f"¿Cuál es mi objetivo para el {entity} y qué condición debo conservar?"
            expected = {"goal": sources[0], "constraint": sources[1]}
        else:
            sources = [note]
            if kind == "correction":
                messages += [{"role": "user", "content": f"El {entity} abre el {old_day}."},
                             {"role": "user", "content": (
                                 f"Corrección: el {entity} abre el {day}; el dato anterior era incorrecto.")}]
                sources = [messages[-1]["content"]]
            else:
                messages += [{"role": "user", "content": note}]
            messages += distractors()
            query = f"¿Qué día abre el {entity}?"
            if kind == "short_antecedent":
                messages += [{"role": "user", "content": (
                    f"Volvamos al día de apertura del {entity}; me interesa ese dato.")}]
                query = "¿Y qué día era?"
            expected = {"day": day, "fact": note}
        messages.append({"role": "user", "content": query})
        validate_messages(messages)
        assert len(messages) <= MAX_MESSAGES
        assert sum(len(message["content"].encode("utf-8")) for message in messages) <= MAX_DIALOGUE_BYTES
        cases.append({"id": f"context-integration-heldout/{kind}/{name}",
                      "entityFamily": name, "kind": kind, "messages": messages,
                      "requiredOriginalSources": sources, "expected": expected})
    return cases


def check_entity_separation(cases):
    """Reject entity leakage against both authored corpora before generation."""
    paths = (ROOT / "models/context-reference-v1/corpus.json",
             ROOT / "examples/sovengi-memory-20261013/corpus.jsonl",
             ROOT / "examples/sovengi-memory-20261013/development-cases.json")
    available = [path for path in paths if path.is_file()]
    corpora = "\n".join(normalized(path.read_text(encoding="utf-8")) for path in available)
    for case in cases:
        # Whole words: Glicina and Luciérnaga cannot accidentally match fragments.
        name = re.escape(normalized(case["entityFamily"]))
        if re.search(r"(?<!\w)" + name + r"(?!\w)", corpora):
            raise ValueError("Held-out entity occurs in an existing authored corpus")
    return [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in available]


def day_diagnostic(case, result):
    """Names present in output, not an independent semantic conversation grader."""
    if "day" not in case["expected"]:
        return {"applicable": False, "semanticReviewRequired": True}
    words = set(re.findall(r"\w+", normalized(result["text"])))
    names = sorted(normalized(day) for day in DAY_NAMES if normalized(day) in words)
    return {"applicable": True, "expectedDay": case["expected"]["day"],
            "daysPresent": names,
            "onlyExpectedDayPresent": names == [normalized(case["expected"]["day"])],
            "semanticReviewRequired": True}


def run(config_path, output_path):
    import torch

    config_path = Path(config_path).resolve()
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    original_root = (config_path.parent / original_config["projectRoot"]).resolve()
    model_path = (original_root / original_config["brain"]["conversation_model"]).resolve()
    cases = authored_cases()
    separation = check_entity_separation(cases)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="sovengi-context-evaluation-") as temporary:
            # GNOSIS resolves both model and memory relative to its project root.
            # The selected artifact is read-only; all runtime state stays temporary.
            config = deepcopy(original_config)
            # Preserve this context-only comparison of one pinned generator.
            config["brain"].pop("conversation_ensemble", None)
            config["projectRoot"] = temporary
            config["brain"]["conversation_model"] = str(model_path)
            local_config = Path(temporary) / "config.json"
            write_json(local_config, config)
            app = Sovengi(local_config)
            try:
                for index, case in enumerate(cases):
                    pair = {"caseId": case["id"], "entityFamily": case["entityFamily"],
                            "kind": case["kind"], "inputMessages": deepcopy(case["messages"]),
                            "requiredOriginalSources": case["requiredOriginalSources"],
                            "expected": case["expected"], "arms": {}}
                    for enabled, label in ((False, "without_context"), (True, "with_context")):
                        session = app.connect(f"probe-{index}-{label}", context_enabled=enabled,
                                              context_session_id=f"heldout-{index}")
                        engine = session.models.conversation_engine
                        usage = engine.prompt_usage(case["messages"])
                        if usage["tokens"] <= usage["capacity"]:
                            raise ValueError("Long-history probe must exceed the model's input window")
                        request = {"schemaVersion": 1, "operation": "conversation.reply", "input": {
                            "messages": deepcopy(case["messages"]), "max_new_tokens": MAX_REPLY_TOKENS,
                            "temperature": 0, "seed": SEED}}
                        arm_started = time.monotonic()
                        # wraps preserves real generation while recording the exact
                        # public-engine inputs; it supplies no answers or mock tokens.
                        with patch.object(engine, "reply", wraps=engine.reply) as generation:
                            response = session.request(request)
                            elapsed = time.monotonic() - arm_started
                            if response["status"] != "ok":
                                raise RuntimeError(f"Real generation failed: {response}")
                            prepared = deepcopy(generation.call_args.kwargs["messages"])
                        result = response["result"]
                        prepared_usage = engine.prompt_usage(prepared)
                        text = "\n".join(message["content"] for message in prepared)
                        effective_ids, discarded = encode_prompt(
                            prepared, engine._tokenizer, prepared_usage["capacity"])
                        assert discarded == result["discardedPromptTokens"]
                        effective_text = engine._tokenizer.decode(effective_ids, skip_special_tokens=True)
                        supplied = [source in text for source in case["requiredOriginalSources"]]
                        retained = [source in effective_text for source in case["requiredOriginalSources"]]
                        actual_sources = []
                        if enabled:
                            source_ids = set(result["conversationContext"]["sourceIds"])
                            actual_sources = [source.as_dict() for source in
                                session._context.store.recent(limit=64, include_assistant=False)
                                if source.id in source_ids]
                        pair["arms"][label] = {
                            "contextEnabled": enabled, "elapsedSeconds": elapsed,
                            "rawResponse": response, "engineInputMessages": prepared,
                            "effectivePromptTokenIds": effective_ids,
                            "effectivePromptDecodedText": effective_text,
                            "engineInputParameters": {key: value for key, value in
                                generation.call_args.kwargs.items() if key != "messages"},
                            "selectedSourceRecords": actual_sources,
                            "selectionAndBudget": {
                                "originalPromptUsage": usage, "preparedPromptUsage": prepared_usage,
                                "requiredSourcePresentInEngineMessages": supplied,
                                "requiredSourceRetained": retained, "allRequiredSourcesRetained": all(retained),
                                "systemPresentInEngineMessages": prepared[0]["content"].startswith(SYSTEM),
                                "systemPreservedAfterEngineTrim": SYSTEM in effective_text,
                                "currentMessagePreserved": (prepared[-1] == case["messages"][-1]
                                    and case["messages"][-1]["content"] in effective_text),
                                "fitsWithReplyReserve": prepared_usage["tokens"] + MAX_REPLY_TOKENS <= prepared_usage["capacity"],
                                "engineDiscardedPromptTokens": result["discardedPromptTokens"]},
                            "dayNameDiagnostic": day_diagnostic(case, result)}
                        print(json.dumps({"event": "context_generation", "caseId": case["id"],
                            "arm": label, "modelId": result["modelId"], "text": result["text"],
                            "sourcesRetained": retained, "elapsedSeconds": elapsed}, ensure_ascii=False), flush=True)
                    outputs.append(pair)
                engine = app.project.brain.models.conversation_engine
                metadata = {"modelId": engine.model_id, "modelManifest": engine.manifest,
                            "execution": engine.execution, "parameterCount": engine.parameter_count}
                context_identity = app._context_rankers[None].model_id
            finally:
                app.close()
    finally:
        torch.set_num_threads(previous_threads)
    summary = {}
    for label in ("without_context", "with_context"):
        arms = [pair["arms"][label] for pair in outputs]
        diagnostics = [arm["dayNameDiagnostic"] for arm in arms if arm["dayNameDiagnostic"]["applicable"]]
        summary[label] = {"cases": len(arms), "allRequiredSourcesRetained": sum(
                arm["selectionAndBudget"]["allRequiredSourcesRetained"] for arm in arms),
            "systemPreservedAfterEngineTrim": sum(
                arm["selectionAndBudget"]["systemPreservedAfterEngineTrim"] for arm in arms),
            "currentMessagePreserved": sum(arm["selectionAndBudget"]["currentMessagePreserved"] for arm in arms),
            "fitsWithReplyReserve": sum(arm["selectionAndBudget"]["fitsWithReplyReserve"] for arm in arms),
            "noDiscardedPromptTokens": sum(arm["selectionAndBudget"]["engineDiscardedPromptTokens"] == 0 for arm in arms),
            "dayNameDiagnosticApplicable": len(diagnostics),
            "onlyExpectedDayPresent": sum(row["onlyExpectedDayPresent"] for row in diagnostics),
            "totalElapsedSeconds": sum(arm["elapsedSeconds"] for arm in arms)}
    report = {"schemaVersion": 1, "experiment": "context-integration-heldout-v1",
        "startedAtUtc": started_utc, "completedAtUtc": datetime.now(timezone.utc).isoformat(),
        "elapsedSeconds": time.monotonic() - started, "generatorSha256": digest(__file__),
        "configPath": str(config_path.relative_to(ROOT)), "configSha256": digest(config_path),
        "modelDirectory": str(model_path.relative_to(ROOT)), "weightsSha256": digest(model_path / "weights.safetensors"),
        "tokenizerSha256": digest(model_path / "tokenizer.json"), "contextModelId": context_identity,
        "generation": {"seed": SEED, "temperature": 0, "maxNewTokens": MAX_REPLY_TOKENS, "torchThreads": 1},
        "resourceDecision": "Both arms use 24 generated tokens after an initial 64-token resource probe exposed repeated full-window cache prefills in the baseline. The token cap changed to bound runtime, without tuning weights or source selection on these replies. Initial observed stdout is preserved in generation64-partial.json; this shorter output cap limits semantic assessment.",
        "source": {"kind": "authored-fictional-by-Codex", "personalData": False,
                   "entityFamilySeparationCheckedAgainst": separation,
                   "dialoguesNotUsedForTraining": True, "generatedAnswersUnmodified": True,
                   "runtimeMemoryIsolatedInTemporaryProject": True},
        "metricScope": "Source retention and token-budget checks are technical. Day-name presence is a narrow factual diagnostic requiring inspection of the unmodified output; it is not a full semantic grader.",
        "generalConversationValidated": False, "finalConversationAcceptanceTest": False,
        **metadata, "summary": summary, "outputs": outputs}
    write_json(output_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gnosis.json")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run(arguments.config, arguments.output)
    print(json.dumps({"event": "context_evaluation_complete", "modelId": report["modelId"],
                      "summary": report["summary"]}, ensure_ascii=False), flush=True)
