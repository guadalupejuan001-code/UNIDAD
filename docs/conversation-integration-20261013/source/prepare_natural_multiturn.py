"""Prepare authored four-exchange families plus small reviewed-human anchors.

Authoring assistants are project AI assistants, not independent human authors.
This preparation reads no weights or evaluation results and never fits BPE.
All original family folds, user messages and gold replies remain unchanged.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from ia_propia.conversation.contract import DEFAULT_SYSTEM_PROMPT
from ia_propia.conversation.data import assign_split, load_corpus, validate_record


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1_048_576):
            digest.update(block)
    return digest.hexdigest()


def expand_authored(rows):
    expanded = []
    for row in rows:
        source, messages = row["source"], row["messages"]
        if (source.get("kind") != "authored-synthetic"
                or source.get("privateConversationUsed") is True
                or source.get("modelResponsesUsed") is True):
            raise ValueError("Inputs must identify project-authored fictional examples")
        offset = int(messages[0]["role"] == "system")
        if len(messages) - offset != 8:
            raise ValueError("Each authored record must contain exactly four complete exchanges")
        if row["split"] != assign_split(row["groupId"]):
            raise ValueError("Authored family does not retain its declared stable SHA fold")
        for prefix in range(1, 5):
            value = {**row, "id": f"{row['id']}:prefix-{prefix}",
                "messages": [dict(message) for message in messages[:offset + 2 * prefix]],
                "source": {**source, "prefixPairs": prefix, "parentRecordId": row["id"],
                           "parentRecordSha256": hashlib.sha256(canonical(row)).hexdigest()}}
            validate_record(value)
            expanded.append(value)
    return expanded


def check_union(rows):
    ids, families = set(), {}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Record IDs collide across input sources")
        ids.add(row["id"])
        if row["groupId"] in families and families[row["groupId"]] != row["split"]:
            raise ValueError("A family crosses input folds")
        families[row["groupId"]] = row["split"]


def crossfold_families(rows):
    """Exact complete dialogue-prefix collisions remove every involved family."""
    dialogues = defaultdict(list)
    for row in rows:
        messages = row["messages"]
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *messages]
        dialogues[hashlib.sha256(canonical(messages)).hexdigest()].append(row)
    excluded = set()
    for duplicates in dialogues.values():
        if len({row["split"] for row in duplicates}) > 1:
            excluded.update(row["groupId"] for row in duplicates)
    return excluded


def human_anchors(rows, train_limit=100, validation_limit=50):
    families = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            raise ValueError("Human anchor input must contain TRAIN/validation only, not TEST")
        if row["source"].get("kind") != "public-human-reviewed":
            raise ValueError("Human anchors require reviewed public-human provenance")
        families[row["groupId"]].append(row)
    selected, counts = set(), Counter()
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        groups = [group for group, members in families.items() if members[0]["split"] == split]
        rank = lambda group: hashlib.sha256(
            ("unidad-natural-multiturn-human-anchor-v1\0" + split + "\0" + group).encode()).hexdigest()
        for group in sorted(groups, key=rank):
            if counts[split] + len(families[group]) <= limit:
                selected.add(group)
                counts[split] += len(families[group])
    return [row for row in rows if row["groupId"] in selected]


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def write_corpus(path, rows):
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    if rows:
        load_corpus(path)


def prepare(input_paths, tokenizer_path, output, *, human_corpus=None,
            expected_human_corpus_sha256=None, context_length=512, max_reply_tokens=95):
    if (type(context_length) is not int or not 2 <= context_length <= 4096
            or type(max_reply_tokens) is not int or not 1 <= max_reply_tokens <= 1024):
        raise ValueError("Invalid context or final-reply payload limit")
    inputs = [Path(path).resolve() for path in input_paths]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Provide distinct authored input paths")
    tokenizer_path, output = Path(tokenizer_path).resolve(), Path(output).resolve()
    human = Path(human_corpus).resolve() if human_corpus is not None else None
    if output.exists():
        raise FileExistsError("Prepared corpus directories are immutable")
    frozen = {str(path): sha(path) for path in [*inputs, tokenizer_path, *([human] if human else [])]}
    if expected_human_corpus_sha256 is not None:
        if (human is None or re.fullmatch(r"[0-9a-fA-F]{64}", expected_human_corpus_sha256) is None
                or frozen[str(human)] != expected_human_corpus_sha256.lower()):
            raise ValueError("Human corpus differs from its declared SHA-256")
    authored = []
    for path in inputs:
        authored.extend(load_corpus(path))
    check_union(authored)
    expanded = expand_authored(authored)
    human_rows = load_corpus(human) if human is not None else []
    # Reject TEST before any length or lexical measurement on human input.
    if any(row["split"] == "test" for row in human_rows):
        raise ValueError("Human input must exclude TEST")
    combined = [*expanded, *human_rows]
    check_union(combined)
    collision_groups = crossfold_families(combined)
    from tokenizers import Tokenizer
    from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer
    tokenizer = validate_tokenizer(Tokenizer.from_file(str(tokenizer_path)), strict=True)
    length_groups, lengths = defaultdict(set), {}
    for row in combined:
        tokens, _ = dialogue_tokens(row["messages"], tokenizer)
        reply = len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False).ids)
        lengths[row["id"]] = {"inputTokens": len(tokens) - 1, "replyPayloadTokens": reply}
        if len(tokens) - 1 > context_length:
            length_groups[row["groupId"]].add("whole_context_exceeds_limit")
        if reply > max_reply_tokens:
            length_groups[row["groupId"]].add("assistant_payload_exceeds_limit")
    excluded = collision_groups | set(length_groups)
    retained_authored = [row for row in expanded if row["groupId"] not in excluded]
    anchors = human_anchors([row for row in human_rows if row["groupId"] not in excluded])
    retained = [*retained_authored, *anchors]
    check_union(retained)
    if crossfold_families(retained):
        raise AssertionError("Crossfold exact-prefix guard failed")
    training_rows = [row for row in retained if row["split"] != "test"]
    test_rows = [row for row in retained_authored if row["split"] == "test"]
    if not {"train", "validation"} <= {row["split"] for row in training_rows}:
        raise ValueError("Retained corpus requires TRAIN and validation families")
    lexical = defaultdict(set)
    for row in retained:
        for message in row["messages"]:
            lexical[row["split"]].update(re.findall(r"[^\W\d_]+", message["content"].casefold()))
    counts = Counter(row["split"] for row in retained)
    report = {"schemaVersion": 1, "purpose": "diverse_authored_multiturn_training_data_not_acceptance",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(), "inputsSha256": frozen,
        "builderSha256": sha(__file__), "tokenizerRefitted": False,
        "tokenizerVocabSize": tokenizer.get_vocab_size(), "contextLength": context_length,
        "maxReplyPayloadTokens": max_reply_tokens, "authoredCompleteFamilies": len(authored),
        "authoring": "Project AI assistants authored fictional dialogues; not independent human authorship.",
        "familyFoldsPreserved": True, "goldRepliesAndUserMessagesPreserved": True,
        "supervision": "Each 1/2/3/4-exchange prefix supervises only its final assistant reply and EOS.",
        "humanAnchors": {"records": len(anchors), "splits": dict(Counter(row["split"] for row in anchors)),
            "familyCount": len({row["groupId"] for row in anchors}),
            "selection": "SHA rank, whole families, at most 100 TRAIN and 50 validation records"},
        "excludedCrossfoldFamilyIds": sorted(collision_groups),
        "excludedLengthFamilies": {group: sorted(reasons) for group, reasons in sorted(length_groups.items())},
        "splits": dict(counts), "familiesBySplit": {split: len({row["groupId"] for row in retained
            if row["split"] == split}) for split in ("train", "validation", "test")},
        "sourceKinds": dict(Counter(row["source"]["kind"] for row in retained)),
        "lexicalWordTypesBySplit": {split: len(values) for split, values in sorted(lexical.items())},
        "lengthsBySplit": {split: {"records": len(subset),
            "minimumInputTokens": min(lengths[row["id"]]["inputTokens"] for row in subset),
            "maximumInputTokens": max(lengths[row["id"]]["inputTokens"] for row in subset),
            "maximumReplyPayloadTokens": max(lengths[row["id"]]["replyPayloadTokens"] for row in subset)}
            for split in ("train", "validation", "test")
            if (subset := [row for row in retained if row["split"] == split])},
        "testEvaluated": False, "modelsLoaded": False, "evaluationResultsRead": False,
        "licenseDeclarations": sorted({str(row["source"].get("license")) for row in anchors
                                       if row["source"].get("license")}),
        "limitations": ["Small authored dialogue families do not establish conversational competence.",
                        "Exact-prefix guards do not detect semantic paraphrases.",
                        "Word-type counts are descriptive corpus statistics, not linguistic capability."]}
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    os.chmod(stage, 0o700)
    try:
        write_corpus(stage / "train-validation.jsonl", training_rows)
        write_corpus(stage / "authored-test.jsonl", test_rows)
        if any(sha(path) != fingerprint for path, fingerprint in frozen.items()):
            raise ValueError("Source corpus or tokenizer changed while preparing")
        if sha(__file__) != report["builderSha256"]:
            raise ValueError("Preparation code changed while preparing")
        report["files"] = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                           for path in stage.iterdir() if path.is_file()}
        write_json(stage / "manifest.json", report)
        import fcntl
        descriptor = os.open(output.parent / f".{output.name}.publish.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if output.exists():
                raise FileExistsError("Another writer published this output")
            os.rename(stage, output)
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.close(descriptor)
    except BaseException as exc:
        write_json(stage / "failure.json", {"status": "failed", "error": str(exc),
                                           "inputsSha256": frozen, "testEvaluated": False})
        raise RuntimeError(f"Preparation failed; staged files preserved at {stage}: {exc}") from exc
    return {"output": str(output), "manifestSha256": sha(output / "manifest.json"), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("models/conversation-language-v7/tokenizer.json"))
    parser.add_argument("--human-corpus", type=Path)
    parser.add_argument("--expected-human-corpus-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.tokenizer, args.output, human_corpus=args.human_corpus,
        expected_human_corpus_sha256=args.expected_human_corpus_sha256),
        ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
