"""Bounded conversation data, with evaluation splits assigned to whole families.

Paths here are trusted application/CLI configuration. This module does not scan
personal memory or fetch data. Canonical JSONL records contain only ``id``,
``groupId``, ``split``, ``source`` and ``messages``. ``source`` is descriptive
provenance, never a model target; message content is the only conversational text.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator


SPLITS = ("train", "validation", "test")
MAX_FILE_BYTES = 1_073_741_824
MAX_LINE_BYTES = 8_388_608
MAX_RECORD_BYTES = 1_048_576
MAX_TEXT_BYTES = 16_384
MAX_MESSAGES = 64
MAX_DIALOGUE_BYTES = 65_536
# Keep ingestion independent of optional ML packages. These control strings and
# limits match tokenizer.py; import additionally reserves room for its system turn.
_CONTROL_TOKENS = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>")
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
MAX_RECORDS = 1_000_000
_RECORD_KEYS = {"id", "groupId", "split", "source", "messages"}


class CorpusError(ValueError):
    """A corpus violates the documented format or resource limits."""


def assign_split(group_id: str) -> str:
    """A stable 80/10/10 assignment, independent of text and branch count."""
    _identifier(group_id, "groupId")
    bucket = int.from_bytes(
        hashlib.sha256(("unidad-conversation-v1\0" + group_id).encode("utf-8")).digest()[:8],
        "big",
    ) % 10_000
    return "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise CorpusError(f"{name} must be a nonempty string of at most 256 characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CorpusError(f"{name} must be valid UTF-8") from exc
    if any(ord(char) < 32 for char in value):
        raise CorpusError(f"{name} cannot contain control characters")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CorpusError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_constant(_: str) -> Any:
    raise CorpusError("nonfinite JSON number")


def _check_json(value: Any) -> None:
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_NODES:
            raise CorpusError("JSON nesting or node limit exceeded")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise CorpusError("JSON string must be valid UTF-8") from exc
        elif isinstance(item, float) and not math.isfinite(item):
            raise CorpusError("nonfinite JSON number")


def _json_lines(path: str | Path, *, max_line: int = MAX_LINE_BYTES) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise CorpusError("source file byte limit exceeded")
    opener = gzip.open if file_path.suffix == ".gz" else open
    total = 0
    try:
        with opener(file_path, "rb") as stream:
            number = 0
            while raw := stream.readline(max_line + 1):
                number += 1
                total += len(raw)
                if len(raw) > max_line or total > MAX_FILE_BYTES:
                    raise CorpusError("decompressed file or JSONL line byte limit exceeded")
                if not raw.strip():
                    raise CorpusError(f"blank JSONL line {number}")
                try:
                    obj = json.loads(
                        raw.decode("utf-8", errors="strict"),
                        object_pairs_hook=_pairs,
                        parse_constant=_reject_constant,
                    )
                except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                    raise CorpusError(f"invalid UTF-8 JSON on line {number}") from exc
                _check_json(obj)
                if not isinstance(obj, dict):
                    raise CorpusError(f"JSONL line {number} must be an object")
                yield obj
    except (gzip.BadGzipFile, EOFError) as exc:
        raise CorpusError("invalid gzip source") from exc


def validate_record(record: dict[str, Any]) -> None:
    """Validate a canonical record without changing its content or provenance."""
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise CorpusError("record keys must be id, groupId, split, source, messages")
    _identifier(record["id"], "id")
    _identifier(record["groupId"], "groupId")
    if record["split"] not in SPLITS:
        raise CorpusError("split must be train, validation or test")
    source = record["source"]
    if not isinstance(source, dict) or not source:
        raise CorpusError("source must be a nonempty provenance object")
    if not isinstance(source.get("dataset"), str) or not source["dataset"].strip():
        raise CorpusError("source.dataset must identify provenance")
    if not isinstance(source.get("kind"), str) or not source["kind"].strip():
        raise CorpusError("source.kind must identify human or authored-synthetic provenance")
    messages = record["messages"]
    if not isinstance(messages, list) or not 2 <= len(messages) <= MAX_MESSAGES:
        raise CorpusError("messages must contain 2 to 64 turns")
    start = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    if (len(messages) - start) < 2 or (len(messages) - start) % 2:
        raise CorpusError("conversation must begin with user and end with assistant")
    total_bytes = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise CorpusError("message keys must be role and content")
        role = "system" if start and index == 0 else "user" if (index - start) % 2 == 0 else "assistant"
        if message["role"] != role:
            raise CorpusError("conversation roles must alternate user and assistant after optional system")
        text = message["content"]
        if not isinstance(text, str) or not text.strip():
            raise CorpusError("message content must be nonempty text")
        try:
            size = len(text.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise CorpusError("message content must be valid UTF-8") from exc
        if size > MAX_TEXT_BYTES:
            raise CorpusError("message content byte limit exceeded")
        total_bytes += size
        if total_bytes > MAX_DIALOGUE_BYTES:
            raise CorpusError("dialogue content byte limit exceeded")
        if any(token in text for token in _CONTROL_TOKENS):
            raise CorpusError("message text cannot contain reserved dialogue control tokens")
    _check_json(record)
    if len(_encoded(record)) > MAX_RECORD_BYTES:
        raise CorpusError("record byte limit exceeded")


def _encoded(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical JSONL; reject repeated IDs and family split leakage.

    Returns records in file order. Callers fit tokenizers only to ``train``
    messages, and supervise assistant text rather than source metadata.
    Explicit split assignments are permitted for authored/previous corpora;
    the importer uses :func:`assign_split` for whole OASST trees.
    """
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    groups: dict[str, str] = {}
    for record in _json_lines(path, max_line=MAX_RECORD_BYTES):
        validate_record(record)
        if record["id"] in ids:
            raise CorpusError("duplicate record id")
        if record["groupId"] in groups and groups[record["groupId"]] != record["split"]:
            raise CorpusError("conversation family appears in multiple splits")
        ids.add(record["id"])
        groups[record["groupId"]] = record["split"]
        records.append(record)
        if len(records) > MAX_RECORDS:
            raise CorpusError("record count limit exceeded")
    if not records:
        raise CorpusError("corpus contains no usable conversations")
    return records


def _label_values(message: dict[str, Any]) -> dict[str, float]:
    labels = message.get("labels", {})
    if isinstance(labels, list):
        labels = {entry.get("name"): entry for entry in labels if isinstance(entry, dict)}
    if not isinstance(labels, dict):
        return {}
    values: dict[str, float] = {}
    for name, label in labels.items():
        if not isinstance(label, dict) or not label.get("count", 0):
            continue
        value = label.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[name] = float(value)
    return values


def _filter_reason(message: dict[str, Any], language: str) -> str | None:
    if message.get("lang") != language:
        return "other_language"
    if message.get("deleted") is not False:
        return "deleted_or_unspecified"
    if message.get("review_result") is not True:
        return "not_positively_reviewed"
    if message.get("synthetic") is True or message.get("model_name"):
        return "external_synthetic"
    if message.get("role") not in ("prompter", "assistant"):
        return "unknown_role"
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return "empty_content"
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        return "content_too_long"
    if any(token in text for token in _CONTROL_TOKENS):
        return "reserved_control_token"
    labels = _label_values(message)
    for name in ("spam", "lang_mismatch", "pii", "not_appropriate"):
        if labels.get(name, 0) >= 0.5:
            return f"label_{name}"
    if "quality" in labels and labels["quality"] < 0.25:
        return "low_quality"
    return None


def _trees(path: str | Path) -> Iterator[dict[str, Any]]:
    """Accept official nested trees or reconstruct an official flat message file."""
    objects = _json_lines(path)
    first = next(objects, None)
    if first is None:
        raise CorpusError("source contains no messages")
    if "prompt" in first:
        yield first
        for tree in objects:
            if "prompt" not in tree:
                raise CorpusError("source mixes tree and flat message formats")
            yield tree
        return
    by_id: dict[str, dict[str, Any]] = {}
    tree_ids: dict[str, str] = {}
    def add(message: dict[str, Any]) -> None:
        message_id = message.get("message_id")
        _identifier(message_id, "message_id")
        if message_id in by_id or "prompt" in message:
            raise CorpusError("duplicate message id or mixed source formats")
        group_id = message.get("message_tree_id")
        _identifier(group_id, "message_tree_id")
        copied = dict(message)
        copied["replies"] = []
        by_id[message_id] = copied
        tree_ids[message_id] = group_id
        if len(by_id) > MAX_RECORDS:
            raise CorpusError("source message count limit exceeded")
    add(first)
    for message in objects:
        add(message)
    roots: list[dict[str, Any]] = []
    for message_id, message in by_id.items():
        parent_id = message.get("parent_id")
        if parent_id is None:
            roots.append(message)
        else:
            if parent_id not in by_id or tree_ids[parent_id] != tree_ids[message_id]:
                raise CorpusError("flat message parent is missing or belongs to another tree")
            by_id[parent_id]["replies"].append(message)
    visited: set[str] = set()
    for root in roots:
        stack = [(root, 0)]
        while stack:
            message, depth = stack.pop()
            message_id = message["message_id"]
            if message_id in visited or depth >= MAX_MESSAGES:
                raise CorpusError("flat message cycle, shared node or depth limit")
            visited.add(message_id)
            stack.extend((child, depth + 1) for child in message["replies"])
        yield {"message_tree_id": tree_ids[root["message_id"]], "tree_state": root.get("tree_state"), "prompt": root}
    if len(visited) != len(by_id):
        raise CorpusError("flat messages contain a cycle without a root")


def _tree_records(tree: dict[str, Any], language: str, revision: str | None, counts: Counter[str]) -> list[dict[str, Any]]:
    tree_id = tree.get("message_tree_id")
    _identifier(tree_id, "message_tree_id")
    if tree.get("tree_state") != "ready_for_export":
        counts["tree_not_ready"] += 1
        return []
    prompt = tree.get("prompt")
    if not isinstance(prompt, dict):
        raise CorpusError("tree prompt must be a message object")
    group_id = f"oasst:{tree_id}"
    split = assign_split(group_id)
    stack = [(prompt, [], [], None)]
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    while stack:
        message, history, message_ids, parent_id = stack.pop()
        if not isinstance(message, dict):
            raise CorpusError("tree reply must be a message object")
        message_id = message.get("message_id")
        _identifier(message_id, "message_id")
        if message_id in seen:
            raise CorpusError("duplicate message within a tree")
        seen.add(message_id)
        counts["source_messages"] += 1
        if "parent_id" in message and message["parent_id"] != parent_id:
            raise CorpusError("message parent differs from tree structure")
        replies = message.get("replies", [])
        if not isinstance(replies, list):
            raise CorpusError("message replies must be a list")
        reason = _filter_reason(message, language)
        if reason:
            counts[reason] += 1
            next_history = None
            next_ids = []
        elif history is None:
            counts["ancestor_filtered"] += 1
            next_history = None
            next_ids = []
        else:
            expected = "prompter" if len(history) % 2 == 0 else "assistant"
            if message["role"] != expected:
                counts["nonalternating_role"] += 1
                next_history = None
                next_ids = []
            elif len(history) >= MAX_MESSAGES - 1:
                counts["history_too_long"] += 1
                next_history = None
                next_ids = []
            elif sum(len(turn["content"].encode("utf-8")) for turn in history) + len(message["text"].encode("utf-8")) > MAX_DIALOGUE_BYTES - 512:
                counts["dialogue_too_long"] += 1
                next_history = None
                next_ids = []
            else:
                next_history = history + [{"role": "user" if expected == "prompter" else "assistant", "content": message["text"]}]
                next_ids = message_ids + [message_id]
                counts["accepted_messages"] += 1
                if expected == "assistant":
                    record = {
                        "id": f"oasst:{message_id}", "groupId": group_id, "split": split,
                        "source": {"dataset": "OpenAssistant/oasst1", "license": "Apache-2.0", "revision": revision or "unspecified", "kind": "public-human-reviewed", "language": language, "messageIds": next_ids},
                        "messages": next_history,
                    }
                    validate_record(record)
                    records.append(record)
        stack.extend((child, next_history, next_ids, message_id) for child in reversed(replies))
    return records


def import_oasst(input_path: str | Path, output_path: str | Path, *, language: str = "es", revision: str | None = None, max_records: int | None = None) -> dict[str, Any]:
    """Import reviewed human OASST paths, preserving complete root families.

    Every accepted assistant node supplies a target and its complete history;
    alternatives are retained regardless of rank. ``max_records`` is a soft
    limit checked between trees, so the final complete family may exceed it.
    Low-quality/rejected ancestors block descendants instead of splicing turns.
    Output replacement is atomic; files receive mode 0600 on POSIX systems.
    """
    _identifier(language, "language")
    if revision is not None:
        _identifier(revision, "revision")
    if max_records is not None and (isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0):
        raise CorpusError("max_records must be a positive integer")
    source_path, target = Path(input_path), Path(output_path)
    if source_path.resolve() == target.resolve():
        raise CorpusError("source and output must be different files")
    if source_path.stat().st_size > MAX_FILE_BYTES:
        raise CorpusError("source file byte limit exceeded")
    source_hash = hashlib.sha256()
    with source_path.open("rb") as stream:
        while block := stream.read(1_048_576):
            source_hash.update(block)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter({split: 0 for split in SPLITS})
    group_counts: Counter[str] = Counter({split: 0 for split in SPLITS})
    ids: set[str] = set()
    trees_seen: set[str] = set()
    output_hash = hashlib.sha256()
    total = 0
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".conversation-", suffix=".jsonl", dir=target.parent, delete=False) as stream:
            temporary = stream.name
            for tree in _trees(source_path):
                tree_id = tree.get("message_tree_id")
                _identifier(tree_id, "message_tree_id")
                if tree_id in trees_seen:
                    raise CorpusError("duplicate conversation tree")
                trees_seen.add(tree_id)
                counts["source_trees"] += 1
                records = _tree_records(tree, language, revision, counts)
                if records:
                    group_counts[records[0]["split"]] += 1
                for record in records:
                    if record["id"] in ids:
                        raise CorpusError("duplicate assistant id between trees")
                    ids.add(record["id"])
                    raw = _encoded(record)
                    stream.write(raw)
                    output_hash.update(raw)
                    split_counts[record["split"]] += 1
                    total += 1
                    if total > MAX_RECORDS:
                        raise CorpusError("record count limit exceeded")
                if max_records is not None and total >= max_records:
                    break
            if total == 0:
                raise CorpusError("source contains no usable conversations after filtering")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return {
        "records": total, "groups": sum(group_counts.values()), "splits": dict(split_counts), "groupSplits": dict(group_counts),
        "sha256": output_hash.hexdigest(), "sourceSha256": source_hash.hexdigest(), "filters": dict(counts),
        "source": {"dataset": "OpenAssistant/oasst1", "license": "Apache-2.0", "revision": revision or "unspecified", "language": language, "externalSyntheticExcluded": True},
        "splitMethod": "sha256(unidad-conversation-v1 + groupId), 80/10/10 by family",
        "maxRecordsIsSoftFamilyLimit": max_records is not None,
    }
