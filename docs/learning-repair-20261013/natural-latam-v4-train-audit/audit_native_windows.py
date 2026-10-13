#!/usr/bin/env python3
"""Finite TRAIN-only native BPE/window audit; never imports Torch or weights.

Validation and TEST JSON bodies are skipped via an unescaped canonical split
header before JSON decoding. Uses the project's actual pure native encoder.
Outputs only aggregates, record IDs and hashes, never message text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

EXPECTED = {
    "natural-latam-sft.train-validation.jsonl": "cf52daaaa0bfbd6dd539977ebaabb1b4b265d0ad192d9d6cea6c49e18da5515f",
    "natural-control.train-validation.jsonl": "48cc11848568c03a32727660968e47695d8eeb43c3302cbf7055eefe2e449491",
    "regional-sft.train-validation.jsonl": "520510f77e26626b66af6c1e7ee0c8492f50c381518a4877e2696841bb70138f",
}
TOKENIZER_SHA = "848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720"
HEADER = re.compile(rb'(?<!\\)"split"\s*:\s*"(train|validation|test)"')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    v = sorted(values)
    if not v:
        return {"count": 0}
    out = {"count": len(v), "min": v[0], "max": v[-1], "sum": sum(v), "mean": sum(v) / len(v)}
    out.update({f"p{p}": v[max(0, math.ceil(p * len(v) / 100) - 1)] for p in (10, 25, 50, 75, 90, 95, 99)})
    out["above512"] = sum(x > 512 for x in v)
    return out


def train_rows(path, skipped):
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            matches = list(HEADER.finditer(raw))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous canonical split header: {path.name}:{number}")
            split = matches[0].group(1).decode("ascii")
            if split != "train":
                skipped[split] += 1
                continue
            record = json.loads(raw)
            if record["split"] != "train":
                raise ValueError("Header mismatch")
            yield record, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("No frozen audit receipt may be overwritten")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 ** 2, 32 * 1024 ** 2))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second audit wall limit")))
    signal.alarm(120)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    started = time.monotonic()
    hashes = {name: sha(args.package / name) for name in EXPECTED}
    if hashes != EXPECTED or sha(args.tokenizer) != TOKENIZER_SHA:
        raise ValueError("Frozen TRAIN input/tokenizer identity mismatch")
    src = args.repo / "src/ia_propia/conversation"
    source_hashes = {name: sha(src / name) for name in ("contract.py", "configuration.py", "tokenizer.py", "training.py")}
    sys.path.insert(0, str(args.repo / "src"))
    from tokenizers import Tokenizer
    from ia_propia.conversation.tokenizer import dialogue_tokens, prepare_messages, validate_tokenizer
    from ia_propia.conversation.training import _record_windows, _window_ranges
    from ia_propia.conversation.contract import ASSISTANT_ID, BOS_ID, EOS_ID, IGNORE_INDEX, ROLE_IDS
    tok = validate_tokenizer(Tokenizer.from_file(str(args.tokenizer)), strict=True)
    if tok.get_vocab_size() != 2048 or "torch" in sys.modules:
        raise ValueError("Audit must use own BPE2048 without Torch")
    skipped = defaultdict(Counter)
    control = {r["id"]: h for r, h in train_rows(args.package / "natural-control.train-validation.jsonl", skipped["control"])}
    regional = {r["id"]: h for r, h in train_rows(args.package / "regional-sft.train-validation.jsonl", skipped["regional"])}
    if set(control) & set(regional) or len(control) != 9951 or len(regional) != 408:
        raise ValueError("Unexpected control/regional TRAIN union")
    measures = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(Counter)
    source_counts, behavior_counts = Counter(), Counter()
    seen = set()
    long_records = []
    for r, raw_hash in train_rows(args.package / "natural-latam-sft.train-validation.jsonl", skipped["treatment"]):
        ident = r["id"]
        if ident in seen:
            raise ValueError("Duplicate TRAIN identity")
        seen.add(ident)
        cohort = "NEW_REGIONAL_TRAIN" if ident in regional else "ORIGINAL_CONTROL_TRAIN"
        expected = regional if cohort == "NEW_REGIONAL_TRAIN" else control
        if expected.get(ident) != raw_hash:
            raise ValueError("Treatment TRAIN row differs from frozen component")
        groups = ("ALL_TRAIN", cohort)
        source_counts[r["source"]["dataset"]] += 1
        if cohort == "NEW_REGIONAL_TRAIN":
            behavior_counts[r["source"]["behavior"]] += 1
        prepared = prepare_messages(r["messages"])
        tokens, teach = dialogue_tokens(r["messages"], tok)
        inputs = tokens[:-1]
        labels = [t if s else IGNORE_INDEX for t, s in zip(tokens[1:], teach[1:])]
        last_assistant = max(i for i, t in enumerate(inputs) if t == ASSISTANT_ID)
        labels[:last_assistant] = [IGNORE_INDEX] * last_assistant
        target = tok.encode(prepared[-1]["content"], add_special_tokens=False).ids + [EOS_ID]
        if [x for x in labels if x != IGNORE_INDEX] != target:
            raise ValueError("Native lastreply label content mismatch")
        cursor = 1
        spans = []
        for m in prepared:
            n = len(tok.encode(m["content"], add_special_tokens=False).ids)
            spans.append({"role": m["role"], "roleIndex": cursor, "contentStart": cursor + 1, "contentEnd": cursor + 1 + n, "end": cursor + n + 2})
            cursor += n + 2
        if tokens[0] != BOS_ID or cursor != len(tokens) or spans[-1]["roleIndex"] != last_assistant:
            raise ValueError("Explicit role/content span disagreement")
        current_user = spans[-2]
        history_assistants = [s for s in spans[:-1] if s["role"] == "assistant"]
        if any(labels[s["contentStart"] - 1:s["end"] - 1] != [IGNORE_INDEX] * (s["end"] - s["contentStart"]) for s in history_assistants):
            raise ValueError("Historical assistant text/EOS became positive supervision")
        native = list(_record_windows(r, tok, 512, "sliding_context", "assistant_last_reply"))
        expected_windows = []
        starts = []
        retained_targets = []
        target_context_lengths = []
        target_dropped_tokens = []
        for start, end, previous_end in _window_ranges(len(inputs), 512, "sliding_context"):
            wl = labels[start:end]
            overlap = max(0, previous_end - start)
            wl[:overlap] = [IGNORE_INDEX] * overlap
            if not any(x != IGNORE_INDEX for x in wl):
                continue
            expected_windows.append((inputs[start:end], wl))
            starts.append(start)
            for offset, label in enumerate(wl):
                if label != IGNORE_INDEX:
                    if start + offset < last_assistant:
                        raise ValueError("Earlier assistant label appeared in native window")
                    retained_targets.append(label)
                    target_context_lengths.append(offset + 1)
                    target_dropped_tokens.append(start)
        if native != expected_windows or retained_targets != target or retained_targets.count(EOS_ID) != 1:
            raise ValueError("Native target/EOS lost, repeated, or window logic mismatch")
        first_start = starts[0]
        prompt_tokens = last_assistant + 1
        dropped_prompt = min(first_start, prompt_tokens)
        current_user_content_dropped = max(0, min(first_start, current_user["contentEnd"]) - current_user["contentStart"])
        historical_assistant_tokens = sum(s["contentEnd"] - s["contentStart"] for s in history_assistants)
        erroneous = bool(r["source"].get("erroneousAssistantContext"))
        for group in groups:
            c, m = counts[group], measures[group]
            c["records"] += 1
            c["nativeWindows"] += len(native)
            c["supervisedTargetsIncludingEos"] += len(target)
            c["supervisedEosExactlyOnce"] += 1
            c["allTargetTokensRetainedExactlyOnce"] += 1
            c["historicalAssistantTextTokensIgnored"] += historical_assistant_tokens
            c["historicalAssistantEosIgnored"] += len(history_assistants)
            c["recordsWithHistoricalAssistant"] += bool(history_assistants)
            c["recordsWithAuthoredWrongAssistantContext"] += erroneous
            c["authoredWrongContextRowsWithVerifiedLastreplyOnly"] += erroneous
            c["recordsWithFullInputAbove512"] += len(inputs) > 512
            c["recordsWithPromptAbove512"] += prompt_tokens > 512
            c["recordsWithDroppedPromptAtFirstReplyToken"] += dropped_prompt > 0
            c["recordsWithTruncatedCurrentUserAtFirstReplyToken"] += current_user_content_dropped > 0
            c["recordsWithCurrentUserEntirelyLostAtFirstReplyToken"] += first_start >= current_user["contentEnd"]
            c["windowsWithAnyPrefixDropped"] += sum(x > 0 for x in starts)
            c["targetTokensPredictedAfterDroppingPrefix"] += sum(x > 0 for x in target_dropped_tokens)
            c["recordsWithMultipleNativeWindows"] += len(native) > 1
            for key, value in {
                "fullDialogueTokensIncludingBosAndEos": len(tokens),
                "fullInputTokens": len(inputs),
                "promptTokensIncludingAssistantRole": prompt_tokens,
                "lastReplyTextTokens": len(target) - 1,
                "lastReplyTargetsIncludingEos": len(target),
                "nativeWindowsPerRecord": len(native),
                "droppedPromptTokensAtFirstReplyToken": dropped_prompt,
                "retainedPromptTokensAtFirstReplyToken": max(0, prompt_tokens - first_start),
                "currentUserTextTokens": current_user["contentEnd"] - current_user["contentStart"],
                "currentUserTextTokensDroppedAtFirstReplyToken": current_user_content_dropped,
            }.items():
                m[key].append(value)
            m["nativeWindowInputTokens"].extend(len(x[0]) for x in native)
            m["actualCausalContextTokensPerSupervisedTarget"].extend(target_context_lengths)
            m["prefixTokensDroppedPerSupervisedTarget"].extend(target_dropped_tokens)
        if len(inputs) > 512:
            long_records.append({"id": ident, "cohort": cohort, "inputTokens": len(inputs), "promptTokens": prompt_tokens, "replyTargets": len(target), "nativeWindows": len(native), "firstWindowStart": first_start, "currentUserTextTokensDropped": current_user_content_dropped})
    if seen != set(control) | set(regional):
        raise ValueError("Treatment differs from control plus regional TRAIN union")
    if "torch" in sys.modules or any(source_hashes[n] != sha(src / n) for n in source_hashes):
        raise ValueError("Pinned source changed during audit, or Torch loaded")
    if hashes != {name: sha(args.package / name) for name in EXPECTED} or sha(args.tokenizer) != TOKENIZER_SHA:
        raise ValueError("Pinned input changed during audit")
    report = {
        "schemaVersion": 1, "status": "completed_native_train_only_audit",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "corpusPackage": str(args.package), "inputSha256": hashes,
        "tokenizer": {"path": str(args.tokenizer), "sha256": TOKENIZER_SHA, "vocabularySize": tok.get_vocab_size()},
        "nativeEncoding": {"objective": "assistant_last_reply", "windowPolicy": "sliding_context", "contextLength": 512, "stride": 256, "actualNativeFunctionCalled": "ia_propia.conversation.training._record_windows", "sourceSha256": source_hashes, "labels": "Only final assistant text plus EOS; role markers and historical assistant responses ignored; overlapped targets masked"},
        "cohorts": {g: {"counts": dict(counts[g]), "lengths": {k: stats(v) for k, v in measures[g].items()}} for g in counts},
        "sourceTrainRecordCounts": dict(source_counts), "newRegionalBehaviorTrainRecordCounts": dict(behavior_counts),
        "fullInputAbove512Records": long_records,
        "componentTrainRowsByteExactInTreatment": True,
        "nonTrainHeadersSkippedWithoutJsonBodyDecoding": {k: dict(v) for k, v in skipped.items()},
        "validationBodiesDecoded": 0, "testBodiesDecoded": 0, "reservedEvaluationFilesOpened": 0,
        "modelWeightsLoaded": False, "torchImported": False, "tokenizerFitted": False,
        "inputsAndSourceUnchanged": True, "helperSha256": sha(Path(__file__)),
        "limits": {"cpuAffinityCount": len(os.sched_getaffinity(0)), "cpuSeconds": 120, "wallSeconds": 120, "addressSpaceBytes": 1024 ** 3, "networkBytes": 0},
        "elapsedSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "interpretation": "Each supervised target is retained exactly once even when a dialogue exceeds512. Window prefix loss measures dropped conditioning/history, not target truncation. Causal context counts include earlier reply tokens, while retained prompt isolates history before the reply. TRAIN-only encoding evidence does not establish semantic truth or trained conversational quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": str(args.output), "elapsedSeconds": report["elapsedSeconds"], "peakRssKiB": report["peakRssKiB"], "cohortCounts": report["cohorts"]["ALL_TRAIN"]["counts"], "newRegionalCounts": report["cohorts"]["NEW_REGIONAL_TRAIN"]["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
