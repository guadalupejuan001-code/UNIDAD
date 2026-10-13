"""Reproduce a small exposed development subset with real generated history.

No grades, training, edits of responses, reserved tests or hosted AI calls.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ia_propia.conversation.runtime import ConversationEngine
    from ia_propia.conversation import runtime
    torch.set_num_threads(1)
    suite = json.loads(args.cases.read_text())
    assert suite["purpose"] == "exposed-development-baseline-nevertrain-notfinalacceptance"
    start = time.perf_counter()
    engine = ConversationEngine(args.model)
    load_seconds = time.perf_counter() - start
    artifact_hashes = {p.name: sha256(p) for p in args.model.iterdir()
                       if p.name in ("manifest.json", "weights.safetensors", "tokenizer.json", "config.json")}
    result = {"schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
              "purpose": suite["purpose"], "caseSha256": sha256(args.cases),
              "sourceHead": suite["sourceHead"], "sources": suite["sources"],
              "generation": suite["generation"], "modelId": engine.model_id,
              "conversationSourceSha256": {p.name: sha256(p) for p in Path(runtime.__file__).parent.glob("*.py")},
              "parameterCount": engine.parameter_count, "artifactSha256": artifact_hashes,
              "loadSeconds": load_seconds, "episodes": [],
              "trainingPerformed": False, "testEvaluated": False,
              "rawOutputsEdited": False, "gradesAssigned": False,
              "historyUsesActualGeneratedReplies": True,
              "projectAIReview": True, "independentHumanReview": False,
              "productionAccepted": False}
    timings = []
    errors = 0
    blocked = 0
    for episode in suite["episodes"]:
        history = []
        record = {"id": episode["id"], "source": episode["source"], "turns": []}
        result["episodes"].append(record)
        for index, turn in enumerate(episode["turns"], 1):
            history.append({"role": "user", "content": turn["user"]})
            before = time.perf_counter()
            try:
                actual = engine.reply(history, max_new_tokens=160, temperature=0, seed=42)
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
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                break
            duration = time.perf_counter() - before
            timings.append(duration)
            record["turns"].append({"number": index, "user": turn["user"],
                                     "status": "generated",
                                     "inputMessages": [dict(message) for message in history],
                                     "actual": actual, "seconds": duration,
                                     "actualUtf8Sha256": hashlib.sha256(actual["text"].encode()).hexdigest(),
                                     "constraints": turn.get("constraints"),
                                     "criticalIf": turn.get("criticalIf")})
            history.append({"role": "assistant", "content": actual["text"]})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"episode": episode["id"], "turn": index,
                              "text": actual["text"], "seconds": duration,
                              "generatedTokens": actual["generatedTokens"],
                              "stopReason": actual["stopReason"]}, ensure_ascii=False), flush=True)
    result["summary"] = {"turns": len(timings), "generationSeconds": sum(timings),
                         "errors": errors, "blockedTurns": blocked,
                         "emptyReplies": sum(not turn["actual"]["text"] for ep in result["episodes"] for turn in ep["turns"] if turn["actual"]),
                         "budgetStops": sum(turn["actual"]["stopReason"] == "token_budget" for ep in result["episodes"] for turn in ep["turns"] if turn["actual"]),
                         "truncatedPrompts": sum(turn["actual"]["discardedPromptTokens"] > 0 for ep in result["episodes"] for turn in ep["turns"] if turn["actual"])}
    result["artifactUnmodified"] = artifact_hashes == {name: sha256(args.model / name) for name in artifact_hashes}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
