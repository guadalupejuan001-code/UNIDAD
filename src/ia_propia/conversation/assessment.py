"""Development probes: real output and counterfactual candidate preferences.

This operation never reads a training run, seals it, evaluates its test split,
updates weights, or awards a conversational competence label.
"""

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ..brain.protocol import MAX_REQUEST_BYTES, decode_json


def _identifier(value, name):
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty identifier of at most 256 characters")
    return value


def load_cases(path):
    """Require paired, literally swapped candidates before loading any weights."""
    from .tokenizer import validate_messages

    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    suite = decode_json(raw)
    if (type(suite) is not dict or type(suite.get("schemaVersion")) is not int
            or suite["schemaVersion"] != 1 or suite.get("purpose") != "development"):
        raise ValueError("Assessment requires a schemaVersion 1 development suite")
    _identifier(suite.get("evaluationId"), "evaluationId")
    source = suite.get("source", {})
    if type(source) is not dict or any(source.get(key, False) is not False for key in ("finalTest", "trainingCorpus")):
        raise ValueError("Development probes cannot be labelled final test or training corpus")
    cases = suite.get("cases")
    if type(cases) is not list or not 2 <= len(cases) <= 128:
        raise ValueError("A development suite must contain 2 to 128 cases")
    ids, pairs, families = set(), defaultdict(list), set()
    case_keys = {"id", "familyId", "pairId", "messages", "referenceGoodReply",
                 "plausibleDecoyResponse", "candidateOrder", "expectedBehaviors"}
    for case in cases:
        if type(case) is not dict or set(case) != case_keys:
            raise ValueError("Each development case must contain the documented fields")
        for name in ("id", "familyId", "pairId"):
            _identifier(case[name], name)
        if case["id"] in ids:
            raise ValueError("Development case IDs must be unique")
        ids.add(case["id"])
        families.add(case["familyId"])
        messages = validate_messages(case["messages"])
        if messages[-1]["role"] != "user":
            raise ValueError("Development messages must end with a user turn")
        for field in ("referenceGoodReply", "plausibleDecoyResponse"):
            validate_messages([{"role": "assistant", "content": case[field]}])
        if case["referenceGoodReply"] == case["plausibleDecoyResponse"]:
            raise ValueError("A candidate and its decoy must differ")
        if case["candidateOrder"] not in (["referenceGoodReply", "plausibleDecoyResponse"],
                                           ["plausibleDecoyResponse", "referenceGoodReply"]):
            raise ValueError("candidateOrder must contain the two candidate fields once")
        behaviors = case["expectedBehaviors"]
        if type(behaviors) is not list or not behaviors or any(type(x) is not str or not x.strip() for x in behaviors):
            raise ValueError("Expected behaviors must be explicit nonempty review instructions")
        pairs[case["pairId"]].append(case)
    for pair in pairs.values():
        if len(pair) != 2 or pair[0]["familyId"] != pair[1]["familyId"]:
            raise ValueError("Each pair must contain exactly two cases from one family")
        a, b = pair
        if (a["referenceGoodReply"] != b["plausibleDecoyResponse"]
                or b["referenceGoodReply"] != a["plausibleDecoyResponse"]
                or a["messages"][:-1] != b["messages"][:-1]
                or a["messages"][-1] == b["messages"][-1]):
            raise ValueError("Each pair must swap literal candidates and change only the final user turn")
    expected = {"cases": len(cases), "pairs": len(pairs), "families": len(families)}
    counts = suite.get("counts")
    if (type(counts) is not dict or any(type(v) is not int for v in counts.values()) or counts != expected):
        raise ValueError("Suite counts do not match its cases, pairs and families")
    if "pairs" in suite:
        declared = suite["pairs"]
        if type(declared) is not list or len(declared) != len(pairs):
            raise ValueError("Declared pairs do not match the development cases")
        seen = set()
        for row in declared:
            if type(row) is not dict:
                raise ValueError("Declared pair must be an object")
            _identifier(row.get("id"), "pair id")
            if row["id"] not in pairs or row["id"] in seen:
                raise ValueError("Declared pair IDs must match the development cases once")
            seen.add(row["id"])
            actual = pairs[row["id"]]
            if (row.get("familyId") != actual[0]["familyId"] or type(row.get("caseIds")) is not list
                    or any(type(x) is not str for x in row["caseIds"])
                    or sorted(row["caseIds"]) != sorted(c["id"] for c in actual)):
                raise ValueError("Declared pair membership does not match its cases")
    if "families" in suite:
        declared = suite["families"]
        if type(declared) is not list or len(declared) != len(families):
            raise ValueError("Declared families do not match the development cases")
        seen = set()
        for row in declared:
            if type(row) is not dict:
                raise ValueError("Declared family must be an object")
            _identifier(row.get("id"), "family id")
            if row["id"] not in families or row["id"] in seen:
                raise ValueError("Declared family IDs must match the development cases once")
            seen.add(row["id"])
            actual = [key for key, members in pairs.items() if members[0]["familyId"] == row["id"]]
            if (type(row.get("pairIds")) is not list or any(type(x) is not str for x in row["pairIds"])
                    or sorted(row["pairIds"]) != sorted(actual)):
                raise ValueError("Declared family membership does not match its pairs")
    return suite, hashlib.sha256(raw).hexdigest()


def _validate_output(path):
    if os.path.lexists(path):
        raise FileExistsError("Assessment output already exists; choose a new report path")
    # evaluation.json is a finalization sentinel. Never publish development
    # reports anywhere inside a training run, even when the destination is new.
    parent = Path(path).resolve().parent
    for directory in (parent, *parent.parents):
        if (directory / "status.json").exists() and (directory / "checkpoints").is_dir():
            raise ValueError("Save development assessment outside training run directories")


def _write_new_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".assessment-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive publish: do not overwrite evidence.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assess(model_dir, cases_path, *, output=None, max_new_tokens=64, temperature=0, seed=42, torch_threads=2,
           decoder_backend="torch", use_cache=True, top_k=0, top_p=1.0, repetition_penalty=1.0):
    from .runtime import ConversationEngine, validate_reply_request
    from .tokenizer import integer

    suite, digest = load_cases(cases_path)
    settings = {"max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed,
                "top_k": top_k, "top_p": top_p, "repetition_penalty": repetition_penalty}
    validate_reply_request({"messages": suite["cases"][0]["messages"], **settings})
    integer(torch_threads, "torch_threads", 1, 8)
    if output is not None:
        _validate_output(output)
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads)
    try:
        engine = ConversationEngine(model_dir, decoder_backend=decoder_backend, use_cache=use_cache)
        results, pair_cases = [], defaultdict(list)
        for case in suite["cases"]:
            good = engine.score_reply(case["messages"], case["referenceGoodReply"])
            decoy = engine.score_reply(case["messages"], case["plausibleDecoyResponse"])
            margin = good["totalLogProbability"] - decoy["totalLogProbability"]
            normalized = good["meanLogProbability"] - decoy["meanLogProbability"]
            result = {**case, "generation": engine.reply(case["messages"], **settings),
                      "referenceScore": good, "decoyScore": decoy,
                      "totalLogProbabilityMargin": margin, "meanLogProbabilityMargin": normalized,
                      "referencePreferred": margin > 0, "humanReview": None}
            results.append(result)
            pair_cases[case["pairId"]].append(result)
        pair_results, family_pairs = [], defaultdict(list)
        for pair_id, cases in pair_cases.items():
            row = {"pairId": pair_id, "familyId": cases[0]["familyId"],
                   "caseIds": [c["id"] for c in cases],
                   "bothReferencesPreferred": all(c["referencePreferred"] for c in cases),
                   "contextSensitivity": math.fsum(c["totalLogProbabilityMargin"] for c in cases),
                   "meanTokenContextSensitivity": math.fsum(c["meanLogProbabilityMargin"] for c in cases)}
            pair_results.append(row)
            family_pairs[row["familyId"]].append(row)
        family_results = [{"familyId": name, "pairs": len(rows),
                           "bothReferenceWins": sum(r["bothReferencesPreferred"] for r in rows),
                           "pairWinRate": sum(r["bothReferencesPreferred"] for r in rows) / len(rows),
                           "meanContextSensitivity": math.fsum(r["contextSensitivity"] for r in rows) / len(rows)}
                          for name, rows in sorted(family_pairs.items())]
        report = {"schemaVersion": 1, "purpose": "development", "evaluationId": suite["evaluationId"],
                  "suiteSha256": digest, "modelId": engine.model_id, "modelManifest": engine.manifest,
                  "suiteSource": suite.get("source", {}),
                  "generationSettings": settings, "execution": engine.execution, "counts": suite["counts"], "cases": results,
                  "pairs": pair_results, "families": family_results,
                  "familyMacroPairWinRate": math.fsum(r["pairWinRate"] for r in family_results) / len(family_results),
                  "humanRubric": suite.get("humanRubric"), "humanReviewCompleted": False,
                  "generalConversationValidated": False, "testEvaluated": False,
                  "weightsUpdated": False, "limitations": suite.get("limitations", []),
                  "interpretation": "Candidate likelihood and context sensitivity are development diagnostics. "
                    "Read the real generations separately; these scores do not establish competence."}
    finally:
        torch.set_num_threads(previous_threads)
    if output is not None:
        _write_new_json(output, report)
    return report
