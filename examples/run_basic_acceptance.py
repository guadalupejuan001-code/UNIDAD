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
import re
import time

from ia_propia.conversation.runtime import ConversationEngine

EXPECTED_TURNS = 40
EXPECTED_DIALOGS = 10
TURNS_PER_DIALOG = 4
MINIMUM_FULLY_MET_TURNS = 36
MINIMUM_FULLY_MET_DIALOGS = 8


def safe_identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value) is None:
        raise ValueError("Acceptance requires a bounded, filename-safe identifier")
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    Path(path).chmod(0o600)


def read_suite(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    files = frozen.get("files")
    if type(files) is not dict or not {"cases.json", "rubric.json"} <= files.keys():
        raise ValueError("Acceptance freeze must cover both cases.json and rubric.json")
    for name, expected in frozen["files"].items():
        if type(name) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
            raise ValueError("Frozen acceptance filename must stay inside the suite directory")
        if (type(expected) is not dict or type(expected.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None):
            raise ValueError("Acceptance freeze requires explicit SHA-256 digests")
        if digest(directory / name) != expected["sha256"]:
            raise ValueError(f"Frozen acceptance file changed: {name}")
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    rubric = json.loads((directory / "rubric.json").read_text(encoding="utf-8"))
    if cases["suiteId"] != rubric["suiteId"] or cases["suiteId"] != frozen["suiteId"]:
        raise ValueError("Acceptance suite identity differs")
    steps = [step for scenario in cases["scenarios"] for step in scenario["steps"]]
    if (len(cases["scenarios"]) != EXPECTED_DIALOGS or len(steps) != EXPECTED_TURNS
            or any(len(s["steps"]) != TURNS_PER_DIALOG for s in cases["scenarios"])):
        raise ValueError("Current acceptance requires ten scenarios of four turns (40 total); "
                         "preserve historical 30-turn evidence and use a separately frozen suite")
    scenario_ids = [safe_identifier(s["id"]) for s in cases["scenarios"]]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Acceptance scenario IDs must be unique")
    for step in steps:
        safe_identifier(step["id"])
    if len({step["id"] for step in steps}) != len(steps):
        raise ValueError("Acceptance turn IDs must be unique")
    acceptance = rubric["acceptance"]
    if type(acceptance["expectedTurns"]) is not int or acceptance["expectedTurns"] != len(steps):
        raise ValueError("Acceptance rubric turn count differs")
    for name, minimum, maximum in (
        ("minimumFullyMetTurns", MINIMUM_FULLY_MET_TURNS, EXPECTED_TURNS),
        ("minimumFullyMetDialogs", MINIMUM_FULLY_MET_DIALOGS, EXPECTED_DIALOGS),
    ):
        value = acceptance.get(name, minimum)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Acceptance threshold {name} must be in [{minimum}, {maximum}]")
    for step in steps:
        if not isinstance(step["user"], str) or not step["user"].strip() or type(step["critical"]) is not bool:
            raise ValueError("Acceptance turn requires explicit text and critical status")
        if step["critical"] and not step.get("criticalExpected"):
            raise ValueError("Critical acceptance turn lacks its frozen criterion")
    return cases, rubric, frozen


def run(model_dir, suite_dir, output_dir, *, torch_threads=1):
    from ia_propia.conversation.tokenizer import encode_prompt, prepare_messages

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
    manifest = outputs.get("modelManifest")
    if (type(manifest) is not dict or manifest.get("modelId") != outputs["modelId"]
            or type(manifest.get("training")) is not dict
            or manifest["training"].get("trainedFrom") not in ("random_weights", "own_checkpoint")):
        raise ValueError("Outputs must identify the same recorded own-model artifact as its manifest")
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
            if actual.get("scenarioId") != scenario["id"]:
                raise ValueError("Every turn must belong to its frozen scenario")
            history.append({"role": "user", "content": case["user"]})
            if actual["inputMessages"] != history or actual["modelId"] != outputs["modelId"]:
                raise ValueError("Every turn must preserve user text and the prior actual generated replies")
            if actual["error"] is None and (type(actual["reply"]) is not dict
                    or type(actual["reply"].get("text")) is not str
                    or not actual["reply"]["text"].strip()):
                raise ValueError("A successful record requires a nonempty actual reply")
            if actual["reply"] is not None:
                if actual["reply"]["modelId"] != outputs["modelId"]:
                    raise ValueError("Replies must all use the selected model artifact")
                if actual["error"] is None:
                    history.append({"role": "assistant", "content": actual["reply"]["text"]})
    full, critical_failures, technical_failures, full_by_id = 0, [], [], {}
    for identifier, case in expected.items():
        row, actual = judged[identifier], generated[identifier]
        if any(type(row[key]) is not bool for key in ("relevance", "contextGrounding", "spanish")):
            raise ValueError("Every dimension requires an explicit true or false review")
        if case["critical"] and type(row["criticalCorrect"]) is not bool:
            raise ValueError("Every critical turn requires an explicit criticalCorrect review")
        context = actual["effectiveContext"]
        if context is not None and (type(context) is not dict
                or type(context.get("discardedPromptTokens")) is not int
                or context["discardedPromptTokens"] < 0):
            raise ValueError("Effective context must record a nonnegative discarded-token count")
        if actual["error"] is None:
            reply_discarded = actual["reply"].get("discardedPromptTokens")
            if type(reply_discarded) is not int or reply_discarded < 0:
                raise ValueError("A successful reply must record its nonnegative discarded-token count")
            if context is not None and reply_discarded != context["discardedPromptTokens"]:
                raise ValueError("Reply and effective context must agree on discarded prompt tokens")
        if actual["error"] is not None or context is None or context["discardedPromptTokens"]:
            technical_failures.append(identifier)
        dimensions_met = all(row[key] for key in ("relevance", "contextGrounding", "spanish"))
        fully_met = dimensions_met and (not case["critical"] or row["criticalCorrect"])
        if actual["error"] is not None and (dimensions_met or (case["critical"] and row["criticalCorrect"])):
            raise ValueError("Empty or blocked replies cannot be scored as successful")
        full += int(fully_met)
        full_by_id[identifier] = fully_met
        if (not fully_met or (case["critical"] and not row["criticalCorrect"])) and not row["note"].strip():
            raise ValueError("Each failed semantic criterion needs a short review note")
        if case["critical"] and not row["criticalCorrect"]:
            critical_failures.append(identifier)
    dialog_results = [{"id": scenario["id"], "fullyMet": all(full_by_id[step["id"]]
                       for step in scenario["steps"])} for scenario in cases["scenarios"]]
    full_dialogs = sum(row["fullyMet"] for row in dialog_results)
    minimum_turns = rubric["acceptance"].get("minimumFullyMetTurns", MINIMUM_FULLY_MET_TURNS)
    minimum_dialogs = rubric["acceptance"].get("minimumFullyMetDialogs", MINIMUM_FULLY_MET_DIALOGS)
    accepted = (full >= minimum_turns and full_dialogs >= minimum_dialogs
                and not critical_failures and not technical_failures)
    result = {"schemaVersion": 1, "suiteId": cases["suiteId"], "modelId": outputs["modelId"],
              "outputsSha256": digest(outputs_path), "reviewSha256": digest(review_path),
              "reviewer": review["reviewer"], "reviewMethod": review["method"],
              "fullyMetTurns": full, "totalTurns": len(expected), "fullyMetFraction": full / len(expected),
              "fullyMetDialogs": full_dialogs, "totalDialogs": len(dialog_results), "dialogs": dialog_results,
              "acceptanceThresholds": {"minimumFullyMetTurns": minimum_turns,
                                       "minimumFullyMetDialogs": minimum_dialogs},
              "criticalFailures": critical_failures, "technicalFailures": technical_failures,
              "accepted": accepted, "allowedClaim": rubric["allowedClaimIfAccepted"] if accepted else None,
              "limits": rubric["limits"]}
    write_new(output_path, result)
    print(json.dumps({"accepted": accepted, "fullyMetTurns": full, "totalTurns": len(expected),
                      "fullyMetDialogs": full_dialogs, "totalDialogs": len(dialog_results),
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
