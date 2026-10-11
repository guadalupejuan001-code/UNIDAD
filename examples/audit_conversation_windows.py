"""Audit training windows without forwarding a model or scoring held-out data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from ia_propia.conversation.artifacts import load_artifact
from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import ASSISTANT_ID, USER_ID, dialogue_tokens
from ia_propia.conversation.training import _window_ranges


def _distribution(values):
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "p90": ordered[int(.90 * (len(ordered) - 1))],
            "p99": ordered[int(.99 * (len(ordered) - 1))], "max": ordered[-1]}


def audit(corpus_path, artifact_path):
    corpus = Path(corpus_path)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    records = [record for record in load_corpus(corpus) if record["split"] == "train"]
    if not records or hashlib.sha256(corpus.read_bytes()).hexdigest() != digest:
        raise ValueError("Training corpus is empty or changed while loading")
    model, tokenizer, manifest = load_artifact(artifact_path)
    context = model.config["context_length"]
    counters = ("targetTokens", "windows", "records", "multiwindowRecords",
                "replyLongerThanContextRecords", "questionLongerThanContextRecords",
                "targetsWithoutCompleteLatestQuestion", "targetsWithoutAnyLatestQuestionText",
                "firstReplyTargetWithoutAnyQuestionRecords")
    policies = {policy: Counter({key: 0 for key in counters})
                for policy in ("contiguous_nonoverlapping", "sliding_context")}
    reply_lengths, question_lengths = [], []
    source_records, source_targets = Counter(), Counter()
    for record in records:
        ids, supervised = dialogue_tokens(record["messages"], tokenizer)
        assistant = max(index for index, token in enumerate(ids) if token == ASSISTANT_ID)
        user = max(index for index, token in enumerate(ids[:assistant]) if token == USER_ID)
        positions = [index for index in range(assistant + 1, len(ids)) if supervised[index]]
        reply_lengths.append(len(positions))
        question_lengths.append(assistant - user - 2)
        source = record["source"]["kind"]
        source_records[source] += 1
        source_targets[source] += len(positions)
        for policy, counts in policies.items():
            emitted_positions = []
            windows = 0
            for start, end, previous_end in _window_ranges(len(ids) - 1, context, policy):
                targets = [index + 1 for index in range(max(start, previous_end, assistant), end)
                           if supervised[index + 1]]
                if not targets:
                    continue
                windows += 1
                emitted_positions.extend(targets)
                counts["targetTokens"] += len(targets)
                if start > user + 1:
                    counts["targetsWithoutCompleteLatestQuestion"] += len(targets)
                if start > assistant - 2:
                    counts["targetsWithoutAnyLatestQuestionText"] += len(targets)
                if targets[0] == assistant + 1 and start > assistant - 2:
                    counts["firstReplyTargetWithoutAnyQuestionRecords"] += 1
            if emitted_positions != positions:
                raise ValueError("Window policy omitted or repeated assistant targets")
            counts["windows"] += windows
            counts["records"] += 1
            counts["multiwindowRecords"] += windows > 1
            counts["replyLongerThanContextRecords"] += len(positions) > context
            counts["questionLongerThanContextRecords"] += assistant - user - 2 > context
    results = {}
    for policy, counts in policies.items():
        results[policy] = {**dict(counts), "eachFinalReplyTargetExactlyOnce": True,
                           "percentTargetsWithoutAnyLatestQuestionText": 100 * counts["targetsWithoutAnyLatestQuestionText"] / counts["targetTokens"],
                           "percentTargetsWithoutCompleteLatestQuestion": 100 * counts["targetsWithoutCompleteLatestQuestion"] / counts["targetTokens"]}
    return {"schemaVersion": 1, "scope": "training_split_only", "corpusSha256": digest,
            "tokenizerSha256": manifest["files"]["tokenizer.json"]["sha256"],
            "contextLength": context, "parameterCount": model.parameter_count,
            "modelForwardCalls": 0, "heldoutEvaluated": False,
            "policies": results, "replyTokensIncludingEos": _distribution(reply_lengths),
            "latestQuestionTokens": _distribution(question_lengths),
            "sourceRecordCounts": dict(source_records), "sourceTargetTokenCounts": dict(source_targets),
            "interpretation": "Input context coverage only; no NLL, response-quality or comprehension measurement. "
                              "Overlapping windows do not keep questions beyond the finite context indefinitely."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.corpus, args.model)
    raw = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(raw)
    print(raw, end="")


if __name__ == "__main__":
    main()
