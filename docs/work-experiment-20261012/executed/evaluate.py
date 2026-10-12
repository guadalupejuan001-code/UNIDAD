"""Generate paired development evidence with the native ConversationEngine."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from ia_propia.conversation.runtime import ConversationEngine


def evaluate(model, cases, output):
    torch.set_num_threads(1)
    engine = ConversationEngine(model)
    source = Path(cases).read_bytes()
    suite = json.loads(source)
    result = {"schemaVersion": 1, "purpose": suite["purpose"],
              "modelId": engine.model_id, "suiteSha256": hashlib.sha256(source).hexdigest(),
              "generation": {"temperature": 0, "seed": 42, "maxNewTokens": 32, "torchThreads": 1},
              "trainingPerformedByEvaluator": False, "rubricsSentToModel": False,
              "historyPolicy": "actual_generated_assistant_replies", "episodes": []}
    start = time.monotonic()
    for episode in suite["episodes"]:
        history = []
        observed = {"id": episode["id"], "turns": []}
        for turn in episode["turns"]:
            history.append({"role": "user", "content": turn["user"]})
            actual = engine.reply(history, max_new_tokens=32, temperature=0, seed=42)
            observed["turns"].append({"id": turn["id"], "user": turn["user"],
                                      "inputMessages": [dict(m) for m in history], "actual": actual,
                                      "matchesArchivedBaseline": actual["text"] == turn.get("baselineText")})
            if actual["text"].strip():
                history.append({"role": "assistant", "content": actual["text"]})
            else:
                observed["invalidEmptyReply"] = True
                break
        result["episodes"].append(observed)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(episode["id"], [(t["actual"]["text"], t["actual"]["stopReason"]) for t in observed["turns"]], flush=True)
    result["elapsedSeconds"] = time.monotonic() - start
    result["baselineExactMatches"] = sum(t["matchesArchivedBaseline"] for e in result["episodes"] for t in e["turns"])
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"modelId": engine.model_id, "baselineExactMatches": result["baselineExactMatches"],
                      "elapsedSeconds": result["elapsedSeconds"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.model, a.cases, a.output)
