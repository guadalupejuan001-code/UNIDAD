"""Bounded diagnostic of learned conversation patterns with UNIDAD's own weights.

Prefix modes measure train memorization or validation generalization with an
explicitly ideal earlier history. History mode uses only actual generated replies.
Gold answers are used after generation for narrow exact-text/EOS accounting and
optional native teacher-forced likelihood; they never replace a model reply.
Neither this diagnostic nor its exact matching is the 40-turn acceptance test.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import unicodedata

from ia_propia.conversation.data import load_corpus

MIB = 1024**2
MODES = ("train-prefix", "validation-prefix", "validation-history")


def bounded_integer(minimum, maximum):
    def parse(text):
        value = int(text)
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError("Expected an integer in [%s, %s]" % (minimum, maximum))
        return value
    return parse


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(MIB):
            result.update(block)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def read_json(path, maximum=4 * MIB):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Expected a bounded regular input file: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(directory, name, value, budget):
    """Reserve total output space before an exclusive write, including logs."""
    if Path(name).name != name:
        raise ValueError("Output filename must stay inside its directory")
    raw = encoded(value)
    used = sum(path.stat().st_size for path in Path(directory).iterdir() if path.is_file())
    if used + len(raw) > budget:
        raise ValueError("Diagnostic output byte budget exhausted")
    with (Path(directory) / name).open("xb") as stream:
        stream.write(raw)


def normalized(text):
    # Accents, case, punctuation and word order remain meaningful.
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_inputs(corpus, cases_path, manifest_path):
    if Path(corpus).stat().st_size > 64 * MIB:
        raise ValueError("Pattern corpus must be at most 64 MiB")
    records = {row["id"]: row for row in load_corpus(corpus)}
    if any(row["split"] == "test" for row in records.values()):
        raise ValueError("This development diagnostic does not read TEST records")
    cases, manifest = read_json(cases_path), read_json(manifest_path)
    rows = cases.get("cases")
    if cases.get("schemaVersion") != 1 or not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("Expected a bounded schemaVersion 1 cases object")
    if manifest.get("corpusSha256") != digest(corpus):
        raise ValueError("Corpus differs from its manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Case IDs must be unique")
    pairs = defaultdict(list)
    covered = set()
    for case in rows:
        if case.get("split") not in ("train", "validation"):
            raise ValueError("Case split must be explicit TRAIN or VALIDATION")
        turns, expected, ids = (case.get(key) for key in ("userTurns", "expectedResponses", "recordIds"))
        if not all(isinstance(value, list) for value in (turns, expected, ids)) or not 1 <= len(turns) <= 4:
            raise ValueError("Each diagnostic dialogue needs 1 to 4 turns")
        if not len(turns) == len(expected) == len(ids) or any(not isinstance(x, str) or not x.strip() for x in turns + expected):
            raise ValueError("User turns, answers and record IDs must correspond")
        history = deepcopy(case["initialMessages"])
        for index, identifier in enumerate(ids):
            record = records.get(identifier)
            history.append({"role": "user", "content": turns[index]})
            if (record is None or record["split"] != case["split"] or record["groupId"] != case["groupId"]
                    or record["messages"] != history + [{"role": "assistant", "content": expected[index]}]):
                raise ValueError("Case differs from its canonical last-reply record")
            history.append({"role": "assistant", "content": expected[index]})
            covered.add(identifier)
        pairs[(case["split"], case["contrastPairId"])].append(case)
    if covered != set(records):
        raise ValueError("Cases must cover the entire diagnostic corpus exactly")
    for pair in pairs.values():
        if (len(pair) != 2 or pair[0]["competency"] != pair[1]["competency"]
                or pair[0]["userTurns"] != pair[1]["userTurns"]
                or pair[0]["initialMessages"] == pair[1]["initialMessages"]
                or pair[0]["expectedResponses"] == pair[1]["expectedResponses"]):
            raise ValueError("Minimal pairs need the same query and different history and answers")
    return records, rows, manifest


def select_cases(cases, mode, max_turns=12, max_dialogues=2, case_ids=(), prefix_turn="contrast"):
    if mode not in MODES or type(max_turns) is not int or not 1 <= max_turns <= 24:
        raise ValueError("Mode and finite 1..24 turn budget are required")
    split = "train" if mode == "train-prefix" else "validation"
    eligible = [case for case in cases if case["split"] == split]
    if case_ids:
        requested = set(case_ids)
        eligible = [case for case in eligible if case["id"] in requested]
        if len(eligible) != len(requested):
            raise ValueError("Requested cases must exist in the selected split")
    # Round-robin complete pairs keeps a small prefix sample across competencies.
    pools = defaultdict(list)
    for case in eligible:
        pools[case["competency"]].append(case)
    ordered = []
    while any(pools.values()):
        for competency in sorted(pools):
            if pools[competency]:
                first = pools[competency].pop(0)
                ordered.append(first)
                if mode != "validation-history":
                    mate = next((x for x in pools[competency] if x["contrastPairId"] == first["contrastPairId"]), None)
                    if mate is not None:
                        pools[competency].remove(mate)
                        ordered.append(mate)
    selected, total = [], 0
    for case in ordered:
        cost = len(case["userTurns"]) if mode == "validation-history" or prefix_turn == "all" else 1
        if total + cost > max_turns or (mode == "validation-history" and len(selected) >= max_dialogues):
            break
        selected.append(case)
        total += cost
    if not selected:
        raise ValueError("No cases selected within the finite budget")
    return selected


def generate_cases(engine, tokenizer, records, selected, all_cases, mode, generation, *,
                   encode_prompt, likelihood=True, prefix_turn="contrast", emit=None):
    """Run the real engine; injected engine/encoder are only for declared unit fixtures."""
    outputs = []
    opposite = {(case["split"], case["contrastPairId"], case["branch"]): case for case in all_cases}
    for case in selected:
        history, blocked = deepcopy(case["initialMessages"]), False
        indexes = range(len(case["userTurns"]))
        mate = next(value for (split, pair, branch), value in opposite.items()
                    if split == case["split"] and pair == case["contrastPairId"] and branch != case["branch"])
        if mode != "validation-history" and prefix_turn != "all":
            index = len(case["userTurns"]) - 1
            if prefix_turn == "contrast":
                index = next(i for i in reversed(range(len(case["userTurns"])))
                             if normalized(case["expectedResponses"][i]) != normalized(mate["expectedResponses"][i]))
            indexes = [index]
        for index in indexes:
            if mode == "validation-history":
                history.append({"role": "user", "content": case["userTurns"][index]})
                messages = deepcopy(history)
            else:
                messages = deepcopy(records[case["recordIds"][index]]["messages"][:-1])
            expected = case["expectedResponses"][index]
            row = {"id": case["recordIds"][index], "caseId": case["id"], "split": case["split"],
                   "competency": case["competency"], "contrastPairId": case["contrastPairId"],
                   "branch": case["branch"], "turn": index, "inputMessages": messages,
                   "inputSha256": hashlib.sha256(encoded(messages)).hexdigest(),
                   "expectedText": expected, "alternativeText": mate["expectedResponses"][index],
                   "historyPolicy": "actual_generated_after_declared_initial_context" if mode == "validation-history" else "ideal_prefix_diagnostic",
                   "initialContextSupplied": bool(case["initialMessages"]),
                   "initialContextMessageCount": len(case["initialMessages"]),
                   "initialContextAssistantCount": sum(x["role"] == "assistant" for x in case["initialMessages"]),
                   "reply": None, "error": None, "likelihood": None, "likelihoodError": None,
                   "generationSeconds": None, "scoringSeconds": None, "effectiveContext": None,
                   "narrowMetric": {"exactText": False, "exactTextAndEos": False}}
            started = time.monotonic()
            if blocked:
                row["error"] = {"kind": "blocked_by_prior_failed_reply"}
            else:
                try:
                    ids, dropped = encode_prompt(messages, tokenizer, engine.manifest["config"]["context_length"])
                    row["effectiveContext"] = {"tokenIds": ids, "discardedPromptTokens": dropped}
                    # Generation receives no gold, alternate, source metadata or rubric.
                    generation_started = time.monotonic()
                    row["reply"] = engine.reply(messages, **generation)
                    row["generationSeconds"] = time.monotonic() - generation_started
                    actual = row["reply"]["text"]
                    exact = normalized(actual) == normalized(expected)
                    row["narrowMetric"] = {"exactText": exact,
                                          "exactTextAndEos": exact and row["reply"]["stopReason"] == "eos" and dropped == 0}
                    if mode == "validation-history":
                        if actual.strip():
                            history.append({"role": "assistant", "content": actual})
                        else:
                            row["error"] = {"kind": "empty_model_reply"}
                            blocked = True
                    if likelihood:
                        scoring_started = time.monotonic()
                        try:
                            correct = engine.score_reply(messages, expected)
                            alternate = engine.score_reply(messages, row["alternativeText"])
                            row["likelihood"] = {"correct": correct, "alternate": alternate,
                                "totalLogProbabilityMargin": correct["totalLogProbability"] - alternate["totalLogProbability"],
                                "meanLogProbabilityMargin": correct["meanLogProbability"] - alternate["meanLogProbability"],
                                "scope": "native_teacher_forced_gold_and_alternate_including_EOS; not generation quality"}
                        except Exception as error:
                            row["likelihoodError"] = {"kind": type(error).__name__, "message": str(error)}
                        row["scoringSeconds"] = time.monotonic() - scoring_started
                except Exception as error:
                    row["error"] = {"kind": type(error).__name__, "message": str(error)}
                    blocked = mode == "validation-history"
            row["generationAndScoringSeconds"] = time.monotonic() - started
            outputs.append(row)
            if emit is not None:
                emit(row)
    return outputs


def source_pins():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / "src/ia_propia/conversation").glob("*.py"))]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def limits_child(args):
    os.nice(19)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (args.max_ram_mib * MIB,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (min(MIB, args.max_output_mib * MIB // 2),) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (args.max_seconds + 1,) * 2)


def supervise(command, destination, args):
    """Kill only this diagnostic process group on finite wall/disk limits."""
    started, termination = time.monotonic(), None
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "OPENBLAS_NUM_THREADS": "1"}
    with (Path(destination) / "worker.log").open("xb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                   start_new_session=True, preexec_fn=lambda: limits_child(args))
        try:
            while process.poll() is None:
                used = sum(path.stat().st_size for path in Path(destination).iterdir() if path.is_file())
                if time.monotonic() - started > args.max_seconds:
                    termination = "wall_time_budget"
                elif used > (args.max_output_mib * MIB) - MIB:
                    termination = "output_byte_budget"
                if termination:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.1)
            process.wait(timeout=5)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
    return {"returnCode": process.returncode, "termination": termination,
            "elapsedSeconds": time.monotonic() - started}


def worker(args):
    request = read_json(args.output_dir / "request.json")
    current_inputs = {"corpusSha256": digest(args.corpus), "casesSha256": digest(args.cases),
                      "manifestSha256": digest(args.manifest)}
    if current_inputs != request["inputs"] or source_pins() != request["sourceSha256"]:
        raise ValueError("Frozen inputs or diagnostic source changed before worker load")
    import torch
    from ia_propia.conversation.runtime import ConversationEngine
    from ia_propia.conversation.tokenizer import encode_prompt
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records, cases, _manifest = read_inputs(args.corpus, args.cases, args.manifest)
    selected = select_cases(cases, args.mode, args.max_turns, args.max_dialogues, args.case_id, args.prefix_turn)
    engine = ConversationEngine(args.model_dir)
    if engine.manifest["training"].get("trainedFrom") not in ("own_checkpoint", "random_weights"):
        raise ValueError("Own-weight lineage is required")
    budget = args.max_output_mib * MIB - MIB
    counter = [0]
    def emit(row):
        counter[0] += 1
        write_new(args.output_dir, "turn-%02d.json" % counter[0], row, budget)
    outputs = generate_cases(engine, engine._tokenizer, records, selected, cases, args.mode,
                             {"max_new_tokens": args.max_new_tokens, "temperature": 0, "seed": 42},
                             encode_prompt=encode_prompt, likelihood=not args.no_likelihood,
                             prefix_turn=args.prefix_turn, emit=emit)
    pairs = defaultdict(list)
    for row in outputs:
        pairs[(row["contrastPairId"], row["turn"])].append(row)
    paired = [{"contrastPairId": pair, "turn": turn,
               "differentInputs": rows[0]["inputSha256"] != rows[1]["inputSha256"],
               "informativeContrast": normalized(rows[0]["expectedText"]) != normalized(rows[1]["expectedText"]),
               "bothExactAndEos": all(row["narrowMetric"]["exactTextAndEos"] for row in rows),
               "differentRawReplies": all(row["reply"] is not None for row in rows) and rows[0]["reply"]["text"] != rows[1]["reply"]["text"]}
              for (pair, turn), rows in pairs.items() if len(rows) == 2]
    report = {"schemaVersion": 1, "mode": args.mode, "outputs": outputs,
              "modelId": engine.model_id, "modelManifest": engine.manifest,
              "parameterCount": engine.parameter_count, "selectedCaseIds": [case["id"] for case in selected],
              "exactTextAndEos": sum(row["narrowMetric"]["exactTextAndEos"] for row in outputs),
              "totalTurns": len(outputs), "minimalPairs": paired,
              "successfulInformativeMinimalPairs": sum(pair["informativeContrast"] and pair["differentInputs"] and pair["bothExactAndEos"] for pair in paired),
              "fullyExactDialogs": sum(all(row["narrowMetric"]["exactTextAndEos"] for row in outputs if row["caseId"] == case["id"]) for case in selected) if args.mode == "validation-history" else None,
              "selfPeakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "memoryScope": "Linux diagnostic-process peak; excludes active trainer",
              "diagnosticOnly": True, "conversationAccepted": False, "weightsModified": False,
              "humanReviewPerformed": False, "metricScope": "NFC/whitespace-normalized exact text, EOS and no prompt drops on fictional bounded tasks; failure is not an independent semantic judgment"}
    write_new(args.output_dir, "outputs.json", report, budget)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "cases", "manifest", "model-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--max-turns", type=bounded_integer(1, 24), default=12)
    parser.add_argument("--max-dialogues", type=bounded_integer(1, 12), default=2)
    parser.add_argument("--max-new-tokens", type=bounded_integer(1, 64), default=64)
    parser.add_argument("--prefix-turn", choices=("contrast", "final", "all"), default="contrast",
                        help="contrast selects the last turn whose paired gold differs; final may have equal gold")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-seconds", type=bounded_integer(1, 900), default=300)
    parser.add_argument("--max-ram-mib", type=bounded_integer(512, 8192), default=6144)
    parser.add_argument("--max-output-mib", type=bounded_integer(2, 128), default=32)
    parser.add_argument("--no-likelihood", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        worker(args)
        return
    records, cases, _manifest = read_inputs(args.corpus, args.cases, args.manifest)
    selected = select_cases(cases, args.mode, args.max_turns, args.max_dialogues, args.case_id, args.prefix_turn)
    model_manifest = read_json(args.model_dir / "manifest.json")
    model_files = {name: digest(args.model_dir / name) for name in ("manifest.json", "config.json", "tokenizer.json", "weights.safetensors")}
    args.output_dir.mkdir(exist_ok=False, mode=0o700)
    pins = source_pins()
    request = {"schemaVersion": 1, "mode": args.mode, "createdAtUnix": time.time(),
               "inputs": {"corpusSha256": digest(args.corpus), "casesSha256": digest(args.cases), "manifestSha256": digest(args.manifest)},
               "modelId": model_manifest["modelId"], "parentOwnModelId": model_manifest["training"].get("sourceModelId"),
               "modelFilesSha256": model_files, "sourceSha256": pins,
               "selectedCaseIds": [case["id"] for case in selected],
               "generation": {"max_new_tokens": args.max_new_tokens, "temperature": 0, "seed": 42, "torchThreads": 1},
               "limits": {"wallSeconds": args.max_seconds, "ramMiB": args.max_ram_mib, "outputMiB": args.max_output_mib, "maxTurns": args.max_turns, "nice": 19, "cpuAffinityCount": 1},
               "prefixTurn": args.prefix_turn, "nativeLikelihoodEnabled": not args.no_likelihood,
               "diagnosticOnly": True, "conversationAccepted": False}
    write_new(args.output_dir, "request.json", request, args.max_output_mib * MIB)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv), "--_worker"]
    receipt = supervise(command, args.output_dir, args)
    receipt.update({"sourceUnchanged": source_pins() == pins,
                    "inputsUnchanged": request["inputs"] == {"corpusSha256": digest(args.corpus),
                        "casesSha256": digest(args.cases), "manifestSha256": digest(args.manifest)},
                    "modelFilesUnchanged": all(digest(args.model_dir / name) == value for name, value in model_files.items()),
                    "outputsComplete": (args.output_dir / "outputs.json").is_file(),
                    "weightsModified": False, "conversationAccepted": False})
    write_new(args.output_dir, "execution.json", receipt, args.max_output_mib * MIB)
    print(json.dumps({"outputDir": str(args.output_dir.resolve()), **receipt}, ensure_ascii=False))
    if receipt["returnCode"] or not receipt["outputsComplete"] or not receipt["sourceUnchanged"] or not receipt["inputsUnchanged"] or not receipt["modelFilesUnchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
