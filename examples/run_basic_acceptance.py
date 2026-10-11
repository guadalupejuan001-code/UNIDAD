"""Generate a prospective basic-conversation acceptance record with real history.

The engine is loaded once from a verified local own-model artifact. This runner
does not grade replies, call tools, insert ideal replies, or alter model weights.
It uses the frozen private cases/rubric and exclusively creates a new run folder.
Do not run until the selected model is ready for the prospective acceptance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import encode_prompt, prepare_messages


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    Path(path).chmod(0o600)


def read_suite(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in frozen["files"].items():
        if digest(directory / name) != expected["sha256"]:
            raise ValueError(f"Frozen acceptance file changed: {name}")
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    rubric = json.loads((directory / "rubric.json").read_text(encoding="utf-8"))
    if cases["suiteId"] != rubric["suiteId"] or cases["suiteId"] != frozen["suiteId"]:
        raise ValueError("Acceptance suite identity differs")
    steps = [step for scenario in cases["scenarios"] for step in scenario["steps"]]
    if len(cases["scenarios"]) != 10 or len(steps) != 30 or any(len(s["steps"]) != 3 for s in cases["scenarios"]):
        raise ValueError("The frozen suite requires ten scenarios and thirty turns")
    if len({step["id"] for step in steps}) != len(steps):
        raise ValueError("Acceptance turn IDs must be unique")
    if rubric["acceptance"]["expectedTurns"] != len(steps):
        raise ValueError("Acceptance rubric turn count differs")
    for step in steps:
        if not isinstance(step["user"], str) or not step["user"].strip() or type(step["critical"]) is not bool:
            raise ValueError("Acceptance turn requires explicit text and critical status")
        if step["critical"] and not step.get("criticalExpected"):
            raise ValueError("Critical acceptance turn lacks its frozen criterion")
    return cases, rubric, frozen


def run(model_dir, suite_dir, output_dir, *, torch_threads=1):
    cases, rubric, frozen = read_suite(suite_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=False, exist_ok=False, mode=0o700)
    import torch
    torch.set_num_threads(torch_threads)
    engine = ConversationEngine(model_dir)
    model_manifest = engine.manifest
    if model_manifest["training"].get("trainedFrom") not in ("random_weights", "own_checkpoint"):
        raise ValueError("Acceptance requires recorded random-weight or own-checkpoint origin")
    started = time.time()
    outputs, reviews = [], []
    for scenario in cases["scenarios"]:
        history = []
        blocked_by = None
        for step in scenario["steps"]:
            history.append({"role": "user", "content": step["user"]})
            record = {"id": step["id"], "scenarioId": scenario["id"], "critical": step["critical"],
                      "inputMessages": deepcopy(history), "modelId": engine.model_id,
                      "reply": None, "effectiveContext": None, "error": None}
            if blocked_by is not None:
                record["error"] = {"kind": "blocked_by_prior_failed_reply", "priorTurnId": blocked_by}
            else:
                try:
                    context_length = model_manifest["config"]["context_length"]
                    # Inspect the same tokenizer already loaded by the one engine.
                    tokenizer = engine._tokenizer
                    effective_ids, discarded = encode_prompt(history, tokenizer, context_length)
                    record["effectiveContext"] = {
                        "configuredLength": context_length,
                        "tokenIds": effective_ids,
                        "decodedWithRoleTokens": tokenizer.decode(effective_ids, skip_special_tokens=False),
                        "discardedPromptTokens": discarded,
                        "preparedMessages": prepare_messages(history),
                    }
                    record["reply"] = engine.reply(history, max_new_tokens=24, temperature=0, seed=109)
                    # Append only actual model text; no expected answer or cleanup.
                    actual = record["reply"]["text"]
                    if actual.strip():
                        history.append({"role": "assistant", "content": actual})
                    else:
                        record["error"] = {"kind": "empty_model_reply"}
                        blocked_by = step["id"]
                except Exception as error:
                    record["error"] = {"kind": type(error).__name__, "message": str(error)}
                    blocked_by = step["id"]
            outputs.append(record)
            reviews.append({"id": step["id"], "relevance": None, "contextGrounding": None,
                            "spanish": None, "criticalCorrect": None, "note": ""})
            write_new(destination / f"{step['id']}.json", record)
    report = {"schemaVersion": 1, "suiteId": cases["suiteId"], "suiteFreeze": frozen,
              "modelDir": str(Path(model_dir).resolve()), "modelId": engine.model_id,
              "modelManifest": model_manifest, "parameterCount": engine.parameter_count,
              "startedAtUnix": started, "finishedAtUnix": time.time(),
              "generation": {"temperature": 0, "seed": 109, "maxNewTokens": 24, "torchThreads": torch_threads},
              "historyPolicy": "Only actual generated assistant text; each scenario starts fresh; no gold assistant messages.",
              "outputs": outputs, "reviewStatus": "pending", "accepted": None}
    write_new(destination / "outputs.json", report)
    write_new(destination / "review-template.json", {"suiteId": cases["suiteId"], "modelId": engine.model_id,
                                                    "outputsSha256": digest(destination / "outputs.json"),
                                                    "reviewer": "", "method": "", "turns": reviews})
    print(json.dumps({"outputDir": str(destination.resolve()), "modelId": engine.model_id,
                      "turns": len(outputs), "generationErrors": sum(row["error"] is not None for row in outputs),
                      "reviewStatus": "pending"}, ensure_ascii=False))


def assess(suite_dir, outputs_path, review_path, output_path):
    cases, rubric, _frozen = read_suite(suite_dir)
    outputs = json.loads(Path(outputs_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (outputs["suiteId"] != cases["suiteId"] or review["suiteId"] != cases["suiteId"]
            or review["modelId"] != outputs["modelId"] or review["outputsSha256"] != digest(outputs_path)):
        raise ValueError("Review must identify this suite, artifact and exact output file")
    if not review.get("reviewer") or not review.get("method"):
        raise ValueError("Semantic review must identify reviewer and method")
    expected = {step["id"]: step for scenario in cases["scenarios"] for step in scenario["steps"]}
    judged = {row["id"]: row for row in review["turns"]}
    generated = {row["id"]: row for row in outputs["outputs"]}
    if (len(judged) != len(review["turns"]) or set(judged) != set(expected)
            or len(generated) != len(outputs["outputs"]) or set(generated) != set(expected)):
        raise ValueError("Semantic review requires every frozen turn exactly once")
    if outputs["suiteFreeze"] != _frozen:
        raise ValueError("Outputs must preserve the exact prospective suite freeze")
    # Check actual histories independently of the runner's descriptive policy.
    for scenario in cases["scenarios"]:
        history = []
        for case in scenario["steps"]:
            actual = generated[case["id"]]
            history.append({"role": "user", "content": case["user"]})
            if actual["inputMessages"] != history or actual["modelId"] != outputs["modelId"]:
                raise ValueError("Every turn must preserve user text and the prior actual generated replies")
            if actual["reply"] is not None:
                if actual["reply"]["modelId"] != outputs["modelId"]:
                    raise ValueError("Replies must all use the selected model artifact")
                if actual["error"] is None:
                    history.append({"role": "assistant", "content": actual["reply"]["text"]})
    full, critical_failures, technical_failures = 0, [], []
    for identifier, case in expected.items():
        row, actual = judged[identifier], generated[identifier]
        if any(type(row[key]) is not bool for key in ("relevance", "contextGrounding", "spanish")):
            raise ValueError("Every dimension requires an explicit true or false review")
        if case["critical"] and type(row["criticalCorrect"]) is not bool:
            raise ValueError("Every critical turn requires an explicit criticalCorrect review")
        if actual["error"] is not None or actual["effectiveContext"] is None or actual["effectiveContext"]["discardedPromptTokens"]:
            technical_failures.append(identifier)
        fully_met = all(row[key] for key in ("relevance", "contextGrounding", "spanish"))
        if actual["error"] is not None and (fully_met or (case["critical"] and row["criticalCorrect"])):
            raise ValueError("Empty or blocked replies cannot be scored as successful")
        full += int(fully_met)
        if (not fully_met or (case["critical"] and not row["criticalCorrect"])) and not row["note"].strip():
            raise ValueError("Each failed semantic criterion needs a short review note")
        if case["critical"] and not row["criticalCorrect"]:
            critical_failures.append(identifier)
    accepted = full >= rubric["acceptance"]["minimumFullyMetTurns"] and not critical_failures and not technical_failures
    result = {"schemaVersion": 1, "suiteId": cases["suiteId"], "modelId": outputs["modelId"],
              "outputsSha256": digest(outputs_path), "reviewSha256": digest(review_path),
              "reviewer": review["reviewer"], "reviewMethod": review["method"],
              "fullyMetTurns": full, "totalTurns": len(expected), "fullyMetFraction": full / len(expected),
              "criticalFailures": critical_failures, "technicalFailures": technical_failures,
              "accepted": accepted, "allowedClaim": rubric["allowedClaimIfAccepted"] if accepted else None,
              "limits": rubric["limits"]}
    write_new(output_path, result)
    print(json.dumps({"accepted": accepted, "fullyMetTurns": full, "totalTurns": len(expected),
                      "criticalFailures": len(critical_failures), "output": str(Path(output_path).resolve())}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=Path("private/conversation/basic-acceptance-v1"))
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--torch-threads", type=int, choices=range(1, 5), default=1)
    score = sub.add_parser("assess")
    score.add_argument("--outputs", type=Path, required=True)
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        run(args.model_dir, args.suite_dir, args.output_dir, torch_threads=args.torch_threads)
    else:
        assess(args.suite_dir, args.outputs, args.review, args.output)


if __name__ == "__main__":
    main()
