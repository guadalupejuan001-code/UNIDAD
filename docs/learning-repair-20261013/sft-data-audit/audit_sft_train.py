"""Finite, TRAIN-only token/data audit. No Torch, weights or model generation.

Uses the native pure tokenizer helper. Mirrors only the last-reply masking in
training._record_windows, whose source hash is recorded; training is not imported.
Published output contains aggregate counts and no message text.
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


EXPECTED_CORPUS = "4f51ffdf549439707ddee9ea75127264fdd33bcaadc52d585d82a93e52825510"
EXPECTED_TOKENIZER = "848a8d7693a2527911f2e9ef4723b6b417e586e0f80d44390e057cf3e7944720"
SPLIT_HEADER = re.compile(rb'(?<!\\)"split"\s*:\s*"(train|validation|test)"')
WORDS = re.compile(r"\b\w+\b", flags=re.UNICODE)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                    sort_keys=True).encode()).digest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).digest()


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    result = {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
              "sum": sum(ordered), "mean": sum(ordered) / len(ordered)}
    result.update({f"p{percent}": ordered[max(0, math.ceil(percent * len(ordered) / 100) - 1)]
                   for percent in (10, 25, 50, 75, 90, 95, 99)})
    return result


def repetition(counter):
    total = sum(counter.values())
    repeated = [count for count in counter.values() if count > 1]
    return {"occurrences": total, "uniqueExactValues": len(counter),
            "valuesSeenMoreThanOnce": len(repeated),
            "occurrencesInRepeatedValues": sum(repeated),
            "extraOccurrencesBeyondFirst": total - len(counter),
            "largestValueOccurrences": max(counter.values(), default=0),
            "largestTenOccurrenceCounts": sorted(counter.values(), reverse=True)[:10]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing audit receipt")
    # Independent CPU-only data job, never more than one tokenizer worker.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 ** 2, 32 * 1024 ** 2))
    def wall_timeout(_signum, _frame):
        raise TimeoutError("Finite 120-second wall budget exhausted")
    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(120)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    start = time.monotonic()
    corpus_hash, tokenizer_hash = sha256(args.corpus), sha256(args.tokenizer)
    if corpus_hash != EXPECTED_CORPUS or tokenizer_hash != EXPECTED_TOKENIZER:
        raise ValueError("Frozen corpus/tokenizer identity mismatch")
    source_paths = [args.repo / "src/ia_propia/conversation" / name
                    for name in ("contract.py", "configuration.py", "tokenizer.py", "training.py")]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    sys.path.insert(0, str(args.repo / "src"))
    from tokenizers import Tokenizer
    from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer
    from ia_propia.conversation.contract import (ASSISTANT_ID, DEFAULT_SYSTEM_PROMPT,
                                               EOS_ID, IGNORE_INDEX)
    tokenizer = validate_tokenizer(Tokenizer.from_file(str(args.tokenizer)), strict=True)
    metrics = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    source_families = defaultdict(Counter)
    template_families = defaultdict(set)
    template_counts = Counter()
    prefix_counts = defaultdict(Counter)
    targets = defaultdict(Counter)
    normalized_targets = defaultdict(Counter)
    prompts = defaultdict(Counter)
    last_users = defaultdict(Counter)
    first_users = defaultdict(Counter)
    prompt_targets = defaultdict(set)
    target_token_counts = defaultdict(Counter)
    target_words = defaultdict(Counter)
    user_words = defaultdict(Counter)
    message_rehearsal = Counter()
    unique_family_turns = set()
    worlds = set()
    world_values = defaultdict(set)
    change_kinds = Counter()
    full_dialogues = set()
    ids = set()
    skipped_headers = Counter()
    mask_checks = Counter()
    default_system_count = 0
    # The frozen canonical file has one unescaped split field per row. This
    # header check skips validation without JSON-decoding its messages.
    with args.corpus.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if time.monotonic() - start > 118:
                raise TimeoutError("Finite 120-second audit budget exhausted")
            matches = list(SPLIT_HEADER.finditer(line))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous split header at line {line_number}")
            split = matches[0].group(1).decode("ascii")
            if split != "train":
                skipped_headers[split] += 1
                continue
            record = json.loads(line)
            if record["split"] != "train" or record["id"] in ids:
                raise ValueError(f"Invalid TRAIN row identity at line {line_number}")
            ids.add(record["id"])
            messages = record["messages"]
            source = record["source"]["dataset"]
            keys = ("ALL_TRAIN", source)
            counts[source] += 1
            source_families[source][record["groupId"]] += 1
            pairs = sum(message["role"] == "assistant" for message in messages)
            prefix_counts[source][pairs] += 1
            tokens, supervised = dialogue_tokens(messages, tokenizer)
            inputs = tokens[:-1]
            labels = [token if teach else IGNORE_INDEX
                      for token, teach in zip(tokens[1:], supervised[1:])]
            last_assistant = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
            labels[:last_assistant] = [IGNORE_INDEX] * last_assistant
            payload = tokenizer.encode(messages[-1]["content"], add_special_tokens=False).ids
            learned = [label for label in labels if label != IGNORE_INDEX]
            if messages[-1]["role"] != "assistant" or learned != [*payload, EOS_ID]:
                raise ValueError(f"Non-last-reply supervision at line {line_number}")
            if len(inputs) > 512:
                raise ValueError(f"Full context exceeds 512 at line {line_number}")
            mask_checks["rowsWithExactlyLastReplyAndOneEos"] += 1
            mask_checks["supervisedEosTokens"] += learned.count(EOS_ID)
            mask_checks["ignoredInputPositions"] += len(labels) - len(learned)
            default_system_count += messages[0]["role"] != "system"
            digest = exact_hash(messages[-1]["content"])
            normalized = normalized_hash(messages[-1]["content"])
            prompt = exact_hash(messages[:-1])
            prompt_targets[prompt].add(digest)
            full_dialogues.add(exact_hash(messages))
            target_word_list = WORDS.findall(messages[-1]["content"].casefold())
            user_word_list = [word for message in messages if message["role"] == "user"
                              for word in WORDS.findall(message["content"].casefold())]
            for key in keys:
                targets[key][digest] += 1
                normalized_targets[key][normalized] += 1
                prompts[key][prompt] += 1
                last_users[key][exact_hash(messages[-2]["content"])] += 1
                first_users[key][exact_hash(next(m["content"] for m in messages if m["role"] == "user"))] += 1
                target_token_counts[key].update(payload)
                target_words[key].update(target_word_list)
                user_words[key].update(user_word_list)
                values = {
                    "completeSequenceTokens": len(tokens), "teacherForcedInputTokens": len(inputs),
                    "generationPromptTokens": last_assistant + 1,
                    "lastAssistantPayloadTokens": len(payload),
                    "lastAssistantSupervisedTokensIncludingEos": len(learned),
                    "lastAssistantWords": len(target_word_list),
                    "userWordsAllHistory": len(user_word_list), "assistantPairs": pairs,
                }
                for name, value in values.items():
                    metrics[key][name].append(value)
            for index, message in enumerate(messages):
                message_rehearsal[message["role"]] += 1
                unique_family_turns.add((record["groupId"], index, message["role"], exact_hash(message["content"])))
            if "templateId" in record["source"]:
                template_id = record["source"]["templateId"]
                template_counts[template_id] += 1
                template_families[template_id].add(record["groupId"])
                worlds.add(record["source"]["compositionSha256"])
                change_kinds[record["source"]["world"]["changeKind"]] += 1
                world = record["source"]["world"]
                for name in ("objects", "owners", "places"):
                    world_values[name].update(world[name])
                world_values["owners"].add(world["newOwner"])
                world_values["places"].add(world["newPlace"])

    if len(ids) != 18588 or "torch" in sys.modules:
        raise ValueError("Unexpected TRAIN count or forbidden Torch import")
    target_total = sum(metrics["ALL_TRAIN"]["lastAssistantSupervisedTokensIncludingEos"])
    dataset_report = {}
    for source, count in sorted(counts.items()):
        dataset_report[source] = {
            "records": count, "recordShare": count / len(ids),
            "families": len(source_families[source]),
            "recordsPerFamily": quantiles(list(source_families[source].values())),
            "assistantPairsPerRecord": dict(sorted(prefix_counts[source].items())),
            "supervisedTargetTokenShareIncludingEos": sum(metrics[source]["lastAssistantSupervisedTokensIncludingEos"]) / target_total,
            "lengths": {name: quantiles(values) for name, values in metrics[source].items()},
            "exactLastTargets": repetition(targets[source]),
            "normalizedLastTargets": repetition(normalized_targets[source]),
            "exactCompletePrompts": repetition(prompts[source]),
            "exactLastUserMessages": repetition(last_users[source]),
            "exactFirstUserMessagesAcrossPrefixes": repetition(first_users[source]),
            "lexicalCounts": {"lastTargetWords": sum(target_words[source].values()),
                              "lastTargetDistinctCasefoldedWords": len(target_words[source]),
                              "userWordsAllHistory": sum(user_words[source].values()),
                              "userDistinctCasefoldedWords": len(user_words[source]),
                              "lastTargetDistinctBpeIds": len(target_token_counts[source])},
        }
    human_sources = [name for name in counts if name.startswith("OpenAssistant/")]
    human_rows = sum(counts[name] for name in human_sources)
    human_targets = sum(sum(metrics[name]["lastAssistantSupervisedTokensIncludingEos"]) for name in human_sources)
    result = {
        "schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "corpusPath": str(args.corpus), "corpusSha256": corpus_hash,
        "tokenizerPath": str(args.tokenizer), "tokenizerSha256": tokenizer_hash,
        "tokenizerVocabSize": tokenizer.get_vocab_size(),
        "helperSha256": sha256(Path(__file__)), "nativeSourceSha256": source_hashes,
        "scope": {"splitAudited": "train", "records": len(ids), "nonTrainHeadersSkipped": dict(skipped_headers),
                  "validationMessagesDecoded": False, "testMessagesDecoded": False,
                  "reservedSuitesRead": False, "ownerConversationIncluded": False,
                  "modelsLoaded": False, "torchImported": False, "trainingRun": False,
                  "tokenizerRefitted": False, "messageTextExported": False},
        "objective": "assistant_last_reply", "contextLength": 512,
        "quantileMethod": "nearest rank ceil(p*n)-1; integer token/word lengths",
        "wordMethod": "Unicode regex word units, casefolded; numbers and code identifiers count as words",
        "maskAccounting": dict(mask_checks), "windows": {"trainWindows": len(ids), "windowsPerRecord": 1,
                                                          "contextOverflowRows": 0, "croppedRows": 0},
        "systemConditioning": {"rowsUsingDefaultSystem": default_system_count,
                               "defaultSystemPayloadTokens": len(tokenizer.encode(DEFAULT_SYSTEM_PROMPT, add_special_tokens=False).ids)},
        "humanMixture": {"records": human_rows, "recordShare": human_rows / len(ids),
                         "supervisedTokensIncludingEos": human_targets,
                         "supervisedTokenShareIncludingEos": human_targets / target_total},
        "allTrainLengths": {name: quantiles(values) for name, values in metrics["ALL_TRAIN"].items()},
        "datasets": dataset_report,
        "repetition": {"uniqueCompleteDialoguesExact": len(full_dialogues),
                       "exactLastTargets": repetition(targets["ALL_TRAIN"]),
                       "normalizedLastTargets": repetition(normalized_targets["ALL_TRAIN"]),
                       "exactCompletePrompts": repetition(prompts["ALL_TRAIN"]),
                       "exactLastUserMessages": repetition(last_users["ALL_TRAIN"]),
                       "exactFirstUserMessagesAcrossPrefixes": repetition(first_users["ALL_TRAIN"]),
                       "completePromptClassesWithDifferentTargets": sum(len(values) > 1 for values in prompt_targets.values()),
                       "messagesOccurrencesAcrossPrefixes": dict(message_rehearsal),
                       "uniqueFamilyTurnMessages": dict(Counter(item[2] for item in unique_family_turns)),
                       "assistantHistoryOnlyMessageOccurrences": message_rehearsal["assistant"] - len(ids)},
        "composition": {"templateCount": len(template_counts), "uniqueDeclaredSemanticWorlds": len(worlds),
                        "recordsPerTemplate": dict(sorted(template_counts.items())),
                        "uniqueFamiliesPerTemplate": {key: len(value) for key, value in sorted(template_families.items())},
                        "changeKindRecords": dict(change_kinds),
                        "declaredDistinctWorldValues": {key: len(value) for key, value in world_values.items()}},
        "limits": {"cpuAffinityCount": len(os.sched_getaffinity(0)), "tokenizersParallelism": False,
                   "wallBudgetSeconds": 120, "maxAddressSpaceBytes": 1024 ** 3,
                   "maxOutputBytes": 32 * 1024 ** 2},
        "interpretationLimits": ["Exact or normalized uniqueness measures corpus strings, not competence or independent semantic novelty.",
                                 "Four prefixes rehearse earlier messages as context; only their final reply contributes positive supervision.",
                                 "Finite procedural templates and vocabulary can be memorized; no model learning or generalization was measured.",
                                 "Human row share and token share differ; token-weighting versus sequence-weighting depends on the fixed trainer loss configuration.",
                                 "Human provenance is not exhaustive factual certification."],
    }
    if sha256(args.corpus) != corpus_hash or sha256(args.tokenizer) != tokenizer_hash:
        raise ValueError("Frozen inputs changed during audit")
    if any(sha256(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Native source changed during audit")
    result["resources"] = {"wallSeconds": time.monotonic() - start,
                           "cpuUserSeconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                           "cpuSystemSeconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                           "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if len(raw) > 32 * 1024 ** 2:
        raise ValueError("Aggregate output exceeds disk budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    signal.alarm(0)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "records": len(ids), "humanMixture": result["humanMixture"],
                      "resources": result["resources"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
