"""Expand an existing curriculum without moving its held-out conversations.

Keep every BASIC record first, then append canonical OASST records with new IDs
and previously unseen exact message arrays. Both inputs must already be local;
no data downloads, model scores or evaluation results are consulted. This checks
whole-dialogue duplicates, not paraphrases or copied dialogue prefixes.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ia_propia.conversation.data import CorpusError, SPLITS, load_corpus


def _messages_key(record):
    return json.dumps(record["messages"], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _snapshot(path):
    before = _sha(path)
    records = load_corpus(path)
    if _sha(path) != before:
        raise CorpusError("Input corpus changed while loading")
    return records, before


def _stats(records):
    groups = {row["groupId"]: row["split"] for row in records}
    splits = Counter(row["split"] for row in records)
    group_splits = Counter(groups.values())
    return {
        "records": len(records), "groups": len(groups),
        "splits": {split: splits[split] for split in SPLITS},
        "groupSplits": {split: group_splits[split] for split in SPLITS},
        "sources": dict(sorted(Counter(row["source"]["kind"] for row in records).items())),
        "duplicateIds": len(records) - len({row["id"] for row in records}),
        "exactDuplicateMessages": len(records) - len({_messages_key(row) for row in records}),
    }


def _write_new(destination, records):
    """Validate a complete temporary file, then publish without replacing a path."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".expanded-conversations-", dir=destination.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as stream:
            for record in records:
                raw = (json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode("utf-8")
                stream.write(raw)
                digest.update(raw)
            stream.flush()
            os.fsync(stream.fileno())
        verified = load_corpus(temporary)
        if verified != records:
            raise CorpusError("Prepared corpus changed during serialization")
        # link() is an atomic exclusive publication on the same filesystem.
        # A concurrently created output is never truncated or replaced.
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("Choose a new output; an existing corpus is immutable") from error
        return digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def prepare(basic_path, oasst_path, output_path):
    """Return a SHA/count report for a new BASIC-first expanded JSONL corpus.

    Duplicate IDs across inputs are omitted only when their entire record is
    identical. Conflicting identities or family splits fail before writing.
    Existing BASIC records, including any duplicate messages, are preserved.
    """
    destination = Path(output_path)
    if os.path.lexists(destination):
        raise ValueError("Choose a new output; an existing corpus is immutable")
    basic, basic_sha = _snapshot(basic_path)
    public, public_sha = _snapshot(oasst_path)
    ids = {row["id"]: row for row in basic}
    groups = {row["groupId"]: row["split"] for row in basic}

    # Validate the union before deduplication can hide a contradictory fold or ID.
    for record in public:
        group, split = record["groupId"], record["split"]
        if group in groups and groups[group] != split:
            raise CorpusError("Conversation family appears in multiple splits across corpora")
        groups[group] = split
        previous = ids.get(record["id"])
        if previous is not None and previous != record:
            raise CorpusError("Shared record ID conflicts with the BASIC record")

    merged = list(basic)
    messages_seen = {_messages_key(row) for row in basic}
    duplicate_ids = exact_duplicates = 0
    duplicates_against_basic = 0
    basic_messages = set(messages_seen)
    for record in public:
        if record["id"] in ids:
            duplicate_ids += 1
            continue
        key = _messages_key(record)
        if key in messages_seen:
            exact_duplicates += 1
            duplicates_against_basic += key in basic_messages
            continue
        messages_seen.add(key)
        merged.append(record)

    stats = _stats(merged)
    if any(not stats["splits"][split] for split in SPLITS):
        raise CorpusError("The expanded curriculum needs all three evaluation splits")
    raw_stats = {"basic": _stats(basic), "oasst": _stats(public),
                 "combined": _stats(basic + public)}
    # Verify that both hashes still describe the inputs when publishing the output.
    if _sha(basic_path) != basic_sha or _sha(oasst_path) != public_sha:
        raise CorpusError("Input corpus changed during preparation")
    output_sha = _write_new(destination, merged)
    return {
        "schemaVersion": 1, **stats, "sha256": output_sha,
        "inputSha256": {"basic": basic_sha, "oasst": public_sha},
        "preservedBasicRecords": len(basic), "appendedRecords": len(merged) - len(basic),
        "duplicateIdsSkipped": duplicate_ids, "exactDuplicatesRemoved": exact_duplicates,
        "exactDuplicatesAgainstBasicRemoved": duplicates_against_basic,
        "rawStats": raw_stats,
        "splitPolicy": "Preserve every BASIC record and every original family split",
        "selection": "BASIC first, then OASST file order; no score-based selection",
        "limitations": ["Exact full-dialogue deduplication does not detect paraphrases or copied prefixes",
                        "Existing duplicate messages within BASIC are preserved"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic-corpus", required=True, type=Path)
    parser.add_argument("--oasst-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.basic_corpus, args.oasst_corpus, args.output)
    except (CorpusError, ValueError, OSError) as error:
        parser.exit(2, f"prepare_expanded_conversations: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
