"""Generate a sealed reserved suite from one fixed own-weight artifact.

No training, response editing, candidate selection, semantic grading, or API calls.
This runner is prepared before seeing the new suite; verify its declared SHA first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


PURPOSE = "reserved-acceptance-fixed-candidate-no-training"
GENERATION = {"maxNewTokens": 160, "temperature": 0, "seed": 42, "torchThreads": 1}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-purpose", choices=[PURPOSE], required=True)
    parser.add_argument("--expected-case-sha256", required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous raw outputs; use a new output path")
    if digest(args.cases) != args.expected_case_sha256:
        raise ValueError("Suite bytes differ from the writer's declared final SHA256")
    if digest(args.model / "weights.safetensors") != args.expected_weights_sha256:
        raise ValueError("Weights differ from the fixed candidate chosen before suite access")
    suite = json.loads(args.cases.read_text())
    if suite.get("purpose") != args.suite_purpose or suite.get("schemaVersion") != 1:
        raise ValueError("Only a sealed reserved acceptance suite is allowed")
    episodes = suite.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise ValueError("Reserved acceptance requires exactly ten dialogues")
    ids = set()
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("id"), str) or episode["id"] in ids:
            raise ValueError("Each dialogue needs a unique string identifier")
        ids.add(episode["id"])
        if len(episode.get("turns", [])) != 4 or any(not isinstance(t.get("user"), str) or not t["user"] for t in episode["turns"]):
            raise ValueError("Each reserved dialogue requires exactly four nonempty user turns")
    import torch
    from ia_propia.conversation.runtime import ConversationEngine
    from ia_propia.conversation import runtime
    from ia_propia.conversation.contract import DEFAULT_SYSTEM_PROMPT
    from ia_propia.conversation.tokenizer import encode_prompt
    torch.set_num_threads(1)
    source_root = Path(runtime.__file__).parent
    source_hashes = {p.name: digest(p) for p in source_root.glob("*.py")}
    artifact_hashes = {name: digest(args.model / name)
                       for name in ("manifest.json", "weights.safetensors", "tokenizer.json", "config.json")}
    start = time.perf_counter()
    engine = ConversationEngine(args.model)
    load_seconds = time.perf_counter() - start
    if engine.model_id != args.expected_model_id:
        raise ValueError("Loaded model identity differs from the fixed candidate")
    result = {"schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
              "purpose": PURPOSE, "caseSha256": args.expected_case_sha256,
              "suiteFrozenBeforeRead": True, "fixedModelId": args.expected_model_id,
              "fixedWeightsSha256": args.expected_weights_sha256,
              "generation": GENERATION, "defaultSystemPrompt": DEFAULT_SYSTEM_PROMPT,
              "modelId": engine.model_id, "parameterCount": engine.parameter_count,
              "modelManifest": engine.manifest, "artifactSha256": artifact_hashes,
              "conversationSourceSha256": source_hashes, "runnerSha256": digest(Path(__file__)),
              "loadSeconds": load_seconds, "episodes": [],
              "trainingPerformed": False, "candidateSelectionPerformed": False,
              "testEvaluated": True, "existingCorpusTestSplitEvaluated": False,
              "freshReservedAcceptanceEvaluated": True,
              "rawOutputsEdited": False, "gradesAssigned": False,
              "historyUsesActualGeneratedReplies": True, "goldRepliesSentToModel": False,
              "rubricsSentToModel": False, "projectAIReview": True,
              "independentHumanReview": False, "productionAccepted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    save()
    timings, errors, blocked = [], 0, 0
    for episode in episodes:
        history = []
        record = {"id": episode["id"], "turns": []}
        result["episodes"].append(record)
        for index, turn in enumerate(episode["turns"], 1):
            history.append({"role": "user", "content": turn["user"]})
            before = time.perf_counter()
            try:
                effective_ids, discarded = encode_prompt(history, engine._tokenizer, engine._model.config["context_length"])
                actual = engine.reply(history, max_new_tokens=160, temperature=0, seed=42)
                if actual["discardedPromptTokens"] != discarded:
                    raise ValueError("Actual context discard differs from shared prompt encoding")
            except Exception as exc:
                errors += 1
                record["turns"].append({"number": index, "user": turn["user"],
                                         "inputMessages": [dict(message) for message in history],
                                         "status": "error", "error": f"{type(exc).__name__}: {exc}",
                                         "actual": None, "seconds": time.perf_counter() - before})
                for rest_index, rest_turn in enumerate(episode["turns"][index:], index + 1):
                    blocked += 1
                    record["turns"].append({"number": rest_index, "user": rest_turn["user"],
                                             "status": "blocked", "actual": None,
                                             "reason": "Previous actual generation failed; no invented history inserted."})
                save()
                break
            duration = time.perf_counter() - before
            timings.append(duration)
            record["turns"].append({"number": index, "user": turn["user"],
                                     "status": "generated",
                                     "inputMessages": [dict(message) for message in history],
                                     "actual": actual, "seconds": duration,
                                     "effectivePromptTokenIds": effective_ids,
                                     "effectiveContextReconstructedBySharedEncoding": True,
                                     "actualUtf8Sha256": hashlib.sha256(actual["text"].encode()).hexdigest()})
            history.append({"role": "assistant", "content": actual["text"]})
            save()
            print(json.dumps({"episode": episode["id"], "turn": index,
                              "text": actual["text"], "generatedTokens": actual["generatedTokens"],
                              "stopReason": actual["stopReason"]}, ensure_ascii=False), flush=True)
    generated = [t for ep in result["episodes"] for t in ep["turns"] if t["actual"]]
    result["summary"] = {"requestedTurns": 40, "recordedTurns": sum(len(e["turns"]) for e in result["episodes"]),
                         "generatedTurns": len(generated), "generationSeconds": sum(timings),
                         "errors": errors, "blockedTurns": blocked,
                         "emptyReplies": sum(not t["actual"]["text"] for t in generated),
                         "budgetStops": sum(t["actual"]["stopReason"] == "token_budget" for t in generated),
                         "truncatedPrompts": sum(t["actual"]["discardedPromptTokens"] > 0 for t in generated)}
    result["artifactUnmodified"] = artifact_hashes == {name: digest(args.model / name) for name in artifact_hashes}
    result["sourceUnmodifiedDuringGeneration"] = source_hashes == {name: digest(source_root / name) for name in source_hashes}
    result["suiteUnmodifiedDuringGeneration"] = digest(args.cases) == args.expected_case_sha256
    save()


if __name__ == "__main__":
    main()
