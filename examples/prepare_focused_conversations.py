"""Select brief contextual dialogues without changing existing evaluation folds.

Use an explicit tokenizer from Unidad's own parent model; no weights, model
responses or evaluation scores are read. All authored synthetic records are
retained, with over-context records reported. Reviewed human dialogues must fit
completely in one training window, including the final response and system turn.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, integer, validate_tokenizer


SPLITS = ("train", "validation", "test")
AUTHORIZED_KINDS = {"authored-synthetic", "public-human-reviewed"}


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1_048_576):
            result.update(block)
    return result.hexdigest()


def _load(path):
    before = _digest(path)
    records = load_corpus(path)
    if _digest(path) != before:
        raise ValueError("An input corpus changed while loading")
    return records, {"name": Path(path).name, "sha256": before, "records": len(records)}


def _text_key(record):
    return json.dumps(record["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _check_union(records):
    """Check every original record before filters or duplicate removal."""
    families, ids, conversations = {}, {}, {}
    for record in records:
        group, split = record["groupId"], record["split"]
        if group in families and families[group] != split:
            raise ValueError("A conversation family occurs in multiple input splits")
        families[group] = split
        if record["id"] in ids and ids[record["id"]] != record:
            raise ValueError("A record ID has conflicting content or provenance across inputs")
        ids[record["id"]] = record
        key = _text_key(record)
        if key in conversations and conversations[key] != split:
            raise ValueError("An exact dialogue occurs across different input splits")
        conversations[key] = split
        if record["source"]["kind"] not in AUTHORIZED_KINDS:
            raise ValueError("Focused training accepts authored-synthetic or reviewed human data only")


def _distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "min": values[0], "median": values[len(values) // 2],
            "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}


def _nearest_suffix(record, tokenizer, token_limit):
    """Diagnostic only: output always retains the original complete dialogue."""
    messages = record["messages"]
    system = messages[:1] if messages[0]["role"] == "system" else []
    turns = messages[len(system):]
    for start in range(2, len(turns), 2):
        tokens, _ = dialogue_tokens(system + turns[start:], tokenizer)
        if len(tokens) <= token_limit:
            return start
    return None


def prepare(input_path, output_path, *, curriculum_path, tokenizer_path,
            context_length=256, min_human_turns=4):
    """Preserve original IDs, messages, family IDs and splits in a new corpus.

    Selection depends only on explicit provenance, turn count and token length.
    The token limit is context_length + 1 because training shifts the sequence
    into next-token inputs/targets. This function never fits a new tokenizer.
    """
    integer(context_length, "context_length", 2, 4096)
    integer(min_human_turns, "min_human_turns", 2, 64)
    if min_human_turns % 2:
        raise ValueError("min_human_turns must be an even number of user/assistant turns")
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise ValueError("Choose a new output; existing corpora are immutable")
    input_paths = [Path(input_path), Path(curriculum_path), Path(tokenizer_path)]
    if any(target.resolve() == path.resolve() for path in input_paths):
        raise ValueError("Output must be distinct from every input")
    original, input_info = _load(input_path)
    curriculum, curriculum_info = _load(curriculum_path)
    union = original + curriculum
    _check_union(union)

    with Path(tokenizer_path).open("rb") as stream:
        tokenizer_raw = stream.read(16_777_217)
    if len(tokenizer_raw) > 16_777_216:
        raise ValueError("Tokenizer JSON exceeds the local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(tokenizer_raw.decode("utf-8")), strict=True)
    tokenizer_hash = hashlib.sha256(tokenizer_raw).hexdigest()
    token_limit = context_length + 1
    # Authored records come first so identical human text cannot displace them.
    ordered = [record for record in union if record["source"]["kind"] == "authored-synthetic"]
    ordered.extend(record for record in union if record["source"]["kind"] == "public-human-reviewed")
    selected, seen, reasons = [], set(), Counter()
    lengths, lengths_by_split, final_lengths = [], {split: [] for split in SPLITS}, []
    fit_splits, multi_splits = Counter(), Counter()
    authored_long, suffix_fit, suffix_cannot_fit = Counter(), Counter(), Counter()
    for record in ordered:
        messages = record["messages"]
        turns = len(messages) - (messages[0]["role"] == "system")
        authored = record["source"]["kind"] == "authored-synthetic"
        if not authored and turns < min_human_turns:
            reasons["human_too_few_turns"] += 1
            continue
        tokens, _ = dialogue_tokens(messages, tokenizer)
        length = len(tokens)
        if not authored and length > token_limit:
            reasons["human_full_dialogue_exceeds_context"] += 1
            continue
        key = _text_key(record)
        if key in seen:
            reasons["exact_duplicates_removed_within_split"] += 1
            continue
        seen.add(key)
        selected.append(record)
        lengths.append(length)
        lengths_by_split[record["split"]].append(length)
        final_lengths.append(len(tokenizer.encode(messages[-1]["content"], add_special_tokens=False).ids) + 1)
        if turns >= 4:
            multi_splits[record["split"]] += 1
        if length <= token_limit:
            fit_splits[record["split"]] += 1
        else:
            authored_long[record["split"]] += 1
            if _nearest_suffix(record, tokenizer, token_limit) is not None:
                suffix_fit[record["split"]] += 1
            else:
                suffix_cannot_fit[record["split"]] += 1
    splits = Counter(record["split"] for record in selected)
    if any(not splits[split] for split in SPLITS):
        raise ValueError("Focused corpus requires examples in train, validation and test")
    groups = {split: len({record["groupId"] for record in selected if record["split"] == split}) for split in SPLITS}
    raw = b"".join((json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for record in selected)

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".focused-", suffix=".jsonl", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        load_corpus(temporary)
        # Atomic publication with exclusive creation, including concurrent writers.
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ValueError("Another writer created the output; existing evidence was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"schemaVersion": 1, "records": len(selected), "groups": sum(groups.values()),
            "splits": {split: splits[split] for split in SPLITS}, "groupSplits": groups,
            "sources": dict(Counter(record["source"]["kind"] for record in selected)),
            "sha256": hashlib.sha256(raw).hexdigest(), "inputs": [input_info, curriculum_info],
            "tokenizer": {"name": Path(tokenizer_path).name, "sha256": tokenizer_hash,
                          "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
            "selection": {"contextLength": context_length, "fullDialogueTokenLimit": token_limit,
                          "minHumanTurns": min_human_turns, "authoredOverContextRetained": True,
                          "dialoguesTruncated": False, "keywordFiltersUsed": False,
                          "modelResponsesOrEvaluationMetricsUsed": False},
            "reasons": dict(reasons), "fullDialogueTokens": _distribution(lengths),
            "fullDialogueTokensBySplit": {split: _distribution(values) for split, values in lengths_by_split.items()},
            "finalAssistantTokensIncludingEos": _distribution(final_lengths),
            "fullContextFitBySplit": {split: fit_splits[split] for split in SPLITS},
            "multiTurnBySplit": {split: multi_splits[split] for split in SPLITS},
            "authoredOverContextBySplit": {split: authored_long[split] for split in SPLITS},
            "authoredNearestSuffixFitsBySplit": {split: suffix_fit[split] for split in SPLITS},
            "authoredFinalPairCannotFitBySplit": {split: suffix_cannot_fit[split] for split in SPLITS},
            "splitPolicy": "Validate all input folds and cross-split exact dialogue collisions before selecting; preserve existing family splits"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--min-human-turns", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.input, args.output, curriculum_path=args.curriculum,
        tokenizer_path=args.tokenizer, context_length=args.context_length,
        min_human_turns=args.min_human_turns), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
