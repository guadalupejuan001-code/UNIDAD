"""Author a separate recovery curriculum from clean variable-flow records.

Only one historical assistant response is replaced. User messages and the final
correct assistant response remain byte-exact. This is offline data authoring;
model inference never imports it. No model responses or evaluations are read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer


PARENT_DATASET = "unidad-variable-flow-v1"
DATASET = "unidad-conversation-recovery-v1"
SPLITS = ("train", "validation", "test")
EVENT_KIND = {
    "ask_own_name": "assistant-name",
    "ask_real_name": "user-name", "introduce_real_name": "user-name", "correct_real_name": "user-name",
    "ask_preferred_alias": "preferred-alias", "set_preferred_alias": "preferred-alias",
    "ask_city": "city", "set_city": "city", "correct_city": "city",
    "ask_topic": "topic", "set_topic": "topic", "correct_topic": "topic",
    "ask_intent": "intent", "set_intent_listen": "intent", "set_intent_act": "intent",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..4294967295")


def _fictional_label(record_id, kind, seed, attempt):
    """Create a short fictional error value without reading any other dataset."""
    number = int(sha(canonical([seed, record_id, kind, attempt]))[:16], 16)
    first = ("B", "C", "D", "F", "G", "J", "K", "L", "M", "N", "P", "R", "S", "T", "V", "W", "X", "Y", "Z")
    middle = ("a", "e", "i", "o", "u", "ia")
    last = ("lori", "nari", "vani", "mela", "rano", "sela", "tori", "rena", "mina", "lano")
    return first[number % len(first)] + middle[(number // len(first)) % len(middle)] + last[(number // (len(first)*len(middle))) % len(last)]


def _false_reply(record, kind, seed):
    event = record["source"]["events"][-1]
    if kind == "intent":
        target = record["messages"][-1]["content"]
        if event == "set_intent_listen" or "que te escuche" in target:
            return "Quieres que proponga un plan práctico."
        if event == "set_intent_act" or "que proponga" in target:
            return "Quieres que sólo te escuche, sin propuestas."
        # Clean variable-flow records currently ask intention only after an
        # explicit request. Refuse a different source instead of guessing it.
        raise ValueError("Intent recovery needs an explicit reviewed user request")
    texts = [message["content"] for message in record["messages"]]
    for attempt in range(100):
        label = _fictional_label(record["id"], kind, seed, attempt)
        value = "Villa " + label if kind == "city" else "Plan " + label if kind == "topic" else label
        if any(value in text for text in texts):
            continue
        return {
            "assistant-name": f"Me llamo {value}.",
            "user-name": f"Te llamas {value}.",
            "preferred-alias": f"Quieres que te llame {value}.",
            "city": f"Tu ciudad es {value}.",
            "topic": f"El tema actual es {value}.",
        }[kind]
    raise ValueError("Could not author a distinct fictional error value")


def corrupt_record(record, seed=20261015):
    """Return one explicitly labelled historical-error variant, or None.

    Only records with at least two exchanges and a reviewed fact/identity/intent
    target qualify. The immediate previous assistant message is changed; the
    latest user message and final correct target are always preserved.
    """
    _seed(seed)
    validate_record(record)
    source = record["source"]
    if source.get("dataset") != PARENT_DATASET or source.get("kind") != "authored-synthetic":
        raise ValueError("Recovery authoring accepts only clean variable-flow-v1 provenance")
    if source.get("privateConversationUsed") is not False or source.get("modelResponsesUsed") is not False:
        raise ValueError("Parent must explicitly exclude private conversations and model outputs")
    prefix = source.get("prefixPairs")
    events = source.get("events")
    messages = record["messages"]
    if type(prefix) is not int or prefix * 2 != len(messages) or messages[0]["role"] != "user":
        raise ValueError("Parent prefix metadata must match complete user/assistant exchanges")
    if not isinstance(events, list) or len(events) != prefix or not all(isinstance(event, str) for event in events):
        raise ValueError("Parent must provide its explicit clean authoring event sequence")
    if source.get("artificialPreviousAssistantError") or source.get("contextNoise"):
        raise ValueError("Parent history must be clean")
    if prefix < 2 or events[-1] not in EVENT_KIND:
        return None
    kind = EVENT_KIND[events[-1]]
    index = len(messages) - 3
    replacement = _false_reply(record, kind, seed)
    if replacement == messages[index]["content"]:
        raise ValueError("Historical error must differ from the original response")
    variant = copy.deepcopy(record)
    variant["id"] = record["id"] + "-recovery-v1"
    variant["messages"][index]["content"] = replacement
    variant["source"].update({
        "dataset": DATASET,
        "contextNoise": "authored-counterfactual-previous-assistant-error",
        "artificialPreviousAssistantError": True,
        "recovery": {
            "parentDataset": PARENT_DATASET, "parentRecordId": record["id"],
            "parentRecordSha256": sha(canonical(record)),
            "parentMessagesSha256": sha(canonical(messages)),
            "changedAssistantMessageIndex": index, "changedAssistantExchange": prefix - 1,
            "errorKind": kind, "seed": seed,
            "originalResponseSha256": sha(messages[index]["content"].encode("utf-8")),
            "artificialResponseSha256": sha(replacement.encode("utf-8")),
            "finalResponseSha256": sha(messages[-1]["content"].encode("utf-8")),
            "supervision": "last assistant response only; artificial history is never a target",
        },
    })
    validate_record(variant)
    return variant


def exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".recovery-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Another writer created output; it was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(parent_path, tokenizer_path, output_path, manifest_path, seed=20261015, fraction=.25, context_length=256):
    _seed(seed)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not .20 <= fraction <= .25:
        raise ValueError("fraction must be finite and in 0.20..0.25")
    if type(context_length) is not int or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer in 32..4096")
    parent, tokenizer_file, output, manifest = map(Path, (parent_path, tokenizer_path, output_path, manifest_path))
    if len({path.resolve() for path in (parent, tokenizer_file, output, manifest)}) != 4:
        raise ValueError("Parent, tokenizer and outputs must have distinct paths")
    if any(path.exists() or path.is_symlink() for path in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    raw_tokenizer = tokenizer_file.read_bytes()
    if len(raw_tokenizer) > 16_777_216:
        raise ValueError("Tokenizer exceeds its local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(raw_tokenizer.decode("utf-8")), strict=True)
    parent_records = load_corpus(parent)
    parent_sha = sha(parent.read_bytes())
    eligible_counts, candidates, reasons = Counter(), {split: [] for split in SPLITS}, Counter()
    for record in parent_records:
        if type(record["source"].get("prefixPairs")) is int and record["source"]["prefixPairs"] >= 2:
            eligible_counts[record["split"]] += 1
        variant = corrupt_record(record, seed)
        if variant is None:
            reasons["prefix1_or_nonrecovery_target_omitted"] += 1
            continue
        tokens, _ = dialogue_tokens(variant["messages"], tokenizer)
        if len(tokens) > context_length + 1:
            reasons["over_context_variant_omitted_without_truncation"] += 1
            continue
        variant["source"]["recovery"]["parentCorpusSha256"] = parent_sha
        candidates[variant["split"]].append((sha(canonical([seed, record["id"], "selection"])), variant, len(tokens)))
    selected, lengths, selection = [], [], {}
    for split in SPLITS:
        count = eligible_counts[split]
        lower, upper = math.ceil(.20 * count), math.floor(.25 * count)
        if lower > upper or upper < 1:
            raise ValueError("Each parent fold needs enough prefix>=2 records for a 20–25% selection")
        requested = min(upper, max(lower, round(fraction * count)))
        ranked = sorted(candidates[split], key=lambda item: item[0])
        if len(ranked) < requested:
            raise ValueError("Not enough full-fitting recovery variants for the requested fraction")
        selection[split] = {"parentPrefixesAtLeast2": count, "fullfitRecoveryCandidates": len(ranked),
                            "selected": requested, "fraction": requested / count}
        for _, variant, length in ranked[:requested]:
            selected.append(variant)
            lengths.append(length)
    prompts = {}
    for variant in selected:
        key = canonical(variant["messages"][:-1])
        value = (variant["split"], variant["messages"][-1]["content"])
        if key in prompts and prompts[key] != value:
            raise ValueError("A recovery full prompt crosses folds or has conflicting targets")
        prompts[key] = value
    raw = b"".join(canonical(record) + b"\n" for record in selected)
    report = {
        "schemaVersion": 1, "dataset": DATASET, "purpose": "separate-auxiliary-recovery-training-curriculum",
        "sha256": sha(raw), "seed": seed, "records": len(selected), "requestedFraction": fraction,
        "parent": {"dataset": PARENT_DATASET, "sha256": parent_sha, "records": len(parent_records)},
        "selection": selection, "splits": dict(Counter(record["split"] for record in selected)),
        "groupSplits": {split: len({record["groupId"] for record in selected if record["split"] == split}) for split in SPLITS},
        "errorKinds": dict(Counter(record["source"]["recovery"]["errorKind"] for record in selected)),
        "prefixPairs": dict(Counter(str(record["source"]["prefixPairs"]) for record in selected)),
        "fullDialogueTokens": {"min": min(lengths), "max": max(lengths)},
        "tokenizer": {"sha256": sha(raw_tokenizer), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
        "authoring": {
            "exactlyOnePreviousAssistantResponseChanged": True,
            "userMessagesByteExact": True, "lastCorrectResponseByteExact": True,
            "groupIdAndSplitPreserved": True, "prefix1Variants": False,
            "contextLength": context_length, "fullDialogueTokenLimit": context_length + 1,
            "dialoguesTruncated": False, "assistantTargets": "last_reply_only",
            "earlierAssistantHistoryMustNeverBeSupervised": True,
            "selectionPolicy": "stable SHA rank within unchanged parent folds; 20–25% of prefix>=2 records",
            "syntheticErrorPolicy": "fictional conflicting name/alias/city/topic or opposite explicitly requested intent",
            "inferenceImportsAuthoringHelper": False, "modelResponsesUsed": False,
            "evaluationRead": False, "privateConversationUsed": False,
        },
        "reasons": dict(reasons), "generatorSha256": sha(Path(__file__).read_bytes()),
        "limitations": [
            "Artificial prior errors are authored training history, not actual model failures or proof of recovery.",
            "Only the final correct reply may be supervised; training every assistant turn would teach fabricated history.",
            "The corpus is separate and does not modify the clean parent or its folds; no training was launched.",
            "Old-corpus collision checks remain required before a later merge.",
        ],
    }
    exclusive_write(output, raw)
    load_corpus(output)
    exclusive_write(manifest, canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--fraction", type=float, default=.25)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = build(args.parent, args.tokenizer, args.output, args.manifest, args.seed, args.fraction, args.context_length)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
