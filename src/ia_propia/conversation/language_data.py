"""Bounded plain-text LM corpus, separate from conversational SFT records.

Text and UTF-8 newlines are preserved. Metadata is never a model target.
Whole document families keep one fold, and exact text duplicates are rejected
globally before tokenization, including duplicates within the same fold.
"""
import hashlib

from .contract import MAX_DIALOGUE_BYTES, SPECIAL_TOKENS
from .data import (CorpusError, MAX_RECORD_BYTES, MAX_RECORDS, SPLITS,
                   _check_json, _identifier, _json_lines)

MAX_TEXT_BYTES = MAX_DIALOGUE_BYTES
TEXT_RECORD_KEYS = {"id", "groupId", "split", "source", "text"}


def validate_text_record(record):
    if type(record) is not dict or set(record) != TEXT_RECORD_KEYS:
        raise CorpusError("Text record keys must be id, groupId, split, source, text")
    _identifier(record["id"], "id")
    _identifier(record["groupId"], "groupId")
    if record["split"] not in SPLITS:
        raise CorpusError("Text split must be train, validation or test")
    source = record["source"]
    if type(source) is not dict or not source:
        raise CorpusError("Text source must contain explicit provenance")
    _identifier(source.get("dataset"), "source.dataset")
    _identifier(source.get("kind"), "source.kind")
    text = record["text"]
    if type(text) is not str or not text.strip():
        raise CorpusError("Document text must be nonempty UTF-8 text")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise CorpusError("Document text must be valid UTF-8") from error
    if len(raw) > MAX_TEXT_BYTES:
        raise CorpusError("Document text exceeds the UTF-8 byte limit")
    if any(token in text for token in SPECIAL_TOKENS):
        raise CorpusError("Document text cannot contain reserved control tokens")
    _check_json(record)


def iter_text_corpus(path):
    """Yield validated TEXT JSONL without retaining document text.

    Article chunks may share a group within one fold. Exact duplicate text is
    rejected rather than silently choosing which document/fold to keep.
    IDs, family splits and text digests take O(number of records) metadata.
    Exhaust the iterator to validate the entire corpus.
    """
    ids, groups, texts = set(), {}, set()
    count = 0
    for record in _json_lines(path, max_line=MAX_RECORD_BYTES):
        validate_text_record(record)
        if record["id"] in ids:
            raise CorpusError("Duplicate text record id")
        group, split = record["groupId"], record["split"]
        if group in groups and groups[group] != split:
            raise CorpusError("Document family appears in multiple splits")
        digest = hashlib.sha256(record["text"].encode("utf-8")).digest()
        if digest in texts:
            raise CorpusError("Exact document text is duplicated")
        ids.add(record["id"]); groups[group] = split; texts.add(digest)
        count += 1
        if count > MAX_RECORDS:
            raise CorpusError("Text record count limit exceeded")
        yield record
    if not count:
        raise CorpusError("Text corpus has no usable documents")


def load_text_corpus(path):
    """Read canonical TEXT JSONL; preserve supplied folds and provenance."""
    return list(iter_text_corpus(path))
