"""Record own-model replies to a separately frozen LATAM development probe.

This diagnostic does not train or accept a model. Expected meanings remain in
the rubric; generation receives only user messages and actual earlier replies.
Use the same frozen probe and generation settings for each compared artifact.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import resource
import time

from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import encode_prompt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_probe(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if not {"cases.json", "rubric.json"} <= frozen.get("files", {}).keys():
        raise ValueError("A prospective freeze must cover cases and rubric")
    for name, expected in frozen["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise ValueError("Frozen paths must remain in the probe directory")
        if digest(directory / name) != expected["sha256"]:
            raise ValueError("Frozen probe changed: " + name)
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    rubric = json.loads((directory / "rubric.json").read_text(encoding="utf-8"))
    if cases["suiteId"] != frozen["suiteId"] or cases["suiteId"] != rubric["suiteId"]:
        raise ValueError("Probe identity differs")
    steps = [step for scenario in cases["scenarios"] for step in scenario["steps"]]
    if not 1 <= len(steps) <= 32:
        raise ValueError("This development diagnostic is bounded to 32 turns")
    identifiers = [step["id"] for step in steps]
    if len(set(identifiers)) != len(identifiers) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name)
        for name in identifiers
    ):
        raise ValueError("Turn IDs must be unique and safe filenames")
    if any(not isinstance(step.get("user"), str) or not step["user"].strip() for step in steps):
        raise ValueError("Each turn must contain an explicit user request")
    return cases, frozen


def run(model_dir, suite_dir, output_dir):
    cases, frozen = read_probe(suite_dir)
    destination = Path(output_dir)
    destination.mkdir(exist_ok=False, mode=0o700)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    engine = ConversationEngine(model_dir)
    if engine.manifest["training"].get("trainedFrom") not in ("random_weights", "own_checkpoint"):
        raise ValueError("Probe requires recorded own-weight lineage")
    outputs = []
    for scenario in cases["scenarios"]:
        history = []
        for step in scenario["steps"]:
            history.append({"role": "user", "content": step["user"]})
            ids, discarded = encode_prompt(history, engine._tokenizer, engine.manifest["config"]["context_length"])
            record = {"id": step["id"], "scenarioId": scenario["id"],
                      "modelId": engine.model_id, "inputMessages": deepcopy(history),
                      "effectiveContext": {"tokenIds": ids, "discardedPromptTokens": discarded},
                      "reply": None, "error": None}
            reply_started = time.monotonic()
            try:
                record["reply"] = engine.reply(history, max_new_tokens=80, temperature=0, seed=42)
                actual = record["reply"]["text"]
                if actual.strip():
                    history.append({"role": "assistant", "content": actual})
                else:
                    record["error"] = {"kind": "empty_model_reply"}
            except Exception as error:
                record["error"] = {"kind": type(error).__name__, "message": str(error)}
            record["generationSeconds"] = time.monotonic() - reply_started
            write_new(destination / (step["id"] + ".json"), record)
            outputs.append(record)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {"schemaVersion": 1, "suiteId": cases["suiteId"], "suiteFreeze": frozen,
              "modelId": engine.model_id, "modelDir": str(Path(model_dir).resolve()),
              "modelManifest": engine.manifest, "outputs": outputs,
              "generation": {"maxNewTokens": 80, "temperature": 0, "seed": 42, "torchThreads": 1},
              "historyPolicy": "Fresh scenario; user requests and only actual prior assistant replies; no rubric or gold answers in model input",
              "elapsedSeconds": time.monotonic() - started,
              "selfPeakRssKiB": usage.ru_maxrss, "rssScope": "Linux evaluation-process kernel peak; excludes trainer",
              "semanticReview": "pending", "diagnosticOnly": True, "conversationAccepted": False,
              "weightsModified": False, "sourceSha256": digest(__file__)}
    write_new(destination / "outputs.json", report)
    print(json.dumps({"outputDir": str(destination), "modelId": engine.model_id,
                      "turns": len(outputs), "generationErrors": sum(row["error"] is not None for row in outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="A new directory; earlier evidence is never overwritten")
    args = parser.parse_args()
    run(args.model_dir, args.suite_dir, args.output_dir)
