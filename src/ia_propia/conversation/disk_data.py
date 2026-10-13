"""Validated corpus passes and tokenized windows backed by read-only mmap.

Corpus validation keeps O(number of records) IDs, family splits and (for plain
text) duplicate digests, never document text. Encoding holds one record and one
window at a time. The cache index and token data stay on disk; accessing a window
returns ordinary input/label lists with the same order as in-memory encoding.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from contextlib import ExitStack
import errno
import hashlib
import json
import mmap
import operator
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

from .contract import IGNORE_INDEX
from .data import CorpusError, SPLITS, _pairs, _reject_constant, iter_corpus
from .language_data import iter_text_corpus

_HASH_BLOCK_BYTES = 1_048_576
_INDEX_ENTRY = struct.Struct("<QQ")  # byte offset, token count
_MANIFEST_MAX_BYTES = 65_536
_SCHEMA_VERSION = 1
_ENCODING = "little-endian-int32-inputs-labels-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(path):
    from .data import MAX_FILE_BYTES
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise CorpusError("source file byte limit exceeded")
    return _sha256(path)


def _encoding_options(objective, partition_policy):
    if objective not in (None, "assistant_last_reply", "causal_lm"):
        raise ValueError("Invalid encoding objective")
    if partition_policy is not None and (
            type(partition_policy) is not str or partition_policy != "train_validation_only"):
        raise ValueError("Invalid encoding partition_policy")


class StreamingCorpus:
    """Re-iterable validated JSONL, with source hashes checked on every pass.

    Construction validates every record, including TEST, and records fold
    counts. Each later pass independently checks global validation and verifies
    the source before and after reading. No records or text remain in this object.
    """

    def __init__(self, path, config):
        self.path = Path(path)
        self.objective = config.get("objective")
        self.partition_policy = config.get("partition_policy")
        _encoding_options(self.objective, self.partition_policy)
        self.sha256 = _source_sha256(self.path)
        self._counts = dict.fromkeys(SPLITS, 0)
        for record in self:
            self._counts[record["split"]] += 1

    @property
    def counts(self):
        return dict(self._counts)

    def verify(self):
        if _source_sha256(self.path) != self.sha256:
            raise CorpusError("Corpus changed while loading or encoding")

    def __iter__(self):
        self.verify()
        iterator = iter_text_corpus if self.objective == "causal_lm" else iter_corpus
        try:
            for record in iterator(self.path):
                if self.partition_policy == "train_validation_only" and record["split"] == "test":
                    raise ValueError("train_validation_only corpus must not contain TEST records")
                yield record
        finally:
            self.verify()


class WindowCacheError(ValueError):
    """A tokenized-window cache has a different identity or invalid contents."""


class WindowSequence(Sequence):
    """Random-access windows whose data and fixed-size index are memory mapped."""

    def __init__(self, directory, metadata, context_length, vocab_size):
        self._count = metadata["count"]
        self._data = self._index = None
        self._closed = False
        try:
            # Validate through bounded reads before mapping. Walking the whole
            # mmap here would fault every cache page into this process's RSS.
            self._validate(directory, metadata, context_length, vocab_size)
            if metadata["data"]["bytes"]:
                with (directory / metadata["data"]["name"]).open("rb") as stream:
                    self._data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            if metadata["index"]["bytes"]:
                with (directory / metadata["index"]["name"]).open("rb") as stream:
                    self._index = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise

    def _validate(self, directory, metadata, context_length, vocab_size):
        offset = 0
        with ((directory / metadata["data"]["name"]).open("rb") as data,
              (directory / metadata["index"]["name"]).open("rb") as index):
            for _ in range(self._count):
                start, length = _INDEX_ENTRY.unpack(index.read(_INDEX_ENTRY.size))
                if start != offset or not 1 <= length <= context_length:
                    raise WindowCacheError("Invalid cached window index bounds")
                offset += length * 8  # two int32 vectors
                if offset > metadata["data"]["bytes"]:
                    raise WindowCacheError("Cached window index exceeds token data")
                tokens = struct.unpack(f"<{length * 2}i", data.read(length * 8))
                if any(not 0 <= token < vocab_size for token in tokens[:length]):
                    raise WindowCacheError("Cached input token is outside the vocabulary")
                labels = tokens[length:]
                if (not any(label != IGNORE_INDEX for label in labels)
                        or any(label != IGNORE_INDEX and not 0 <= label < vocab_size for label in labels)):
                    raise WindowCacheError("Cached labels are invalid or have no supervised target")
            if (offset != metadata["data"]["bytes"] or data.read(1) or index.read(1)
                    or metadata["index"]["bytes"] != self._count * _INDEX_ENTRY.size):
                raise WindowCacheError("Cached window data or index length mismatch")

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if self._closed:
            raise RuntimeError("Tokenized window cache is closed")
        if isinstance(index, slice):
            return [self[number] for number in range(*index.indices(self._count))]
        index = operator.index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError("Tokenized window index out of range")
        offset, length = _INDEX_ENTRY.unpack_from(self._index, index * _INDEX_ENTRY.size)
        tokens = struct.unpack_from(f"<{length * 2}i", self._data, offset)
        return list(tokens[:length]), list(tokens[length:])

    def __iter__(self):
        for index in range(self._count):
            yield self[index]

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None
        if self._index is not None:
            self._index.close()
            self._index = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class WindowCache(dict):
    """Split sequences plus the small, path-independent verified manifest."""

    def __init__(self, manifest):
        super().__init__()
        self.cache_metadata = manifest

    def close(self):
        close_windows(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def close_windows(examples):
    for windows in examples.values():
        close = getattr(windows, "close", None)
        if close is not None:
            close()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_metadata(path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _check_file(directory, metadata, expected_name):
    if (type(metadata) is not dict or set(metadata) != {"name", "bytes", "sha256"}
            or metadata["name"] != expected_name or type(metadata["bytes"]) is not int
            or metadata["bytes"] < 0 or type(metadata["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is None):
        raise WindowCacheError("Invalid cached file metadata")
    path = directory / expected_name
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != metadata["bytes"]
            or _sha256(path) != metadata["sha256"]):
        raise WindowCacheError("Cached window file size or SHA-256 mismatch")


def _open_cache(directory, identity):
    examples = None
    try:
        path = directory / "manifest.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MANIFEST_MAX_BYTES:
            raise WindowCacheError("Missing or invalid tokenized cache manifest")
        manifest = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=_pairs, parse_constant=_reject_constant)
        if (type(manifest) is not dict or set(manifest) != {"schemaVersion", "identity", "splits"}
                or type(manifest["schemaVersion"]) is not int
                or manifest["schemaVersion"] != _SCHEMA_VERSION):
            raise WindowCacheError("Invalid tokenized cache schema")
        if _canonical_json(manifest["identity"]) != _canonical_json(identity):
            raise WindowCacheError("Tokenized cache identity mismatch")
        splits = manifest["splits"]
        if type(splits) is not dict or set(splits) != set(SPLITS):
            raise WindowCacheError("Invalid tokenized cache splits")
        expected_files = {"manifest.json"}
        for split in SPLITS:
            metadata = splits[split]
            if (type(metadata) is not dict or set(metadata) != {"count", "data", "index"}
                    or type(metadata["count"]) is not int or metadata["count"] < 0):
                raise WindowCacheError("Invalid cached window count")
            _check_file(directory, metadata["data"], f"{split}.windows.bin")
            _check_file(directory, metadata["index"], f"{split}.index.bin")
            if metadata["index"]["bytes"] != metadata["count"] * _INDEX_ENTRY.size:
                raise WindowCacheError("Cached window index length mismatch")
            if split == "test" and not identity["includeTest"] and metadata["count"]:
                raise WindowCacheError("Training cache must not contain TEST windows")
            expected_files.update((f"{split}.windows.bin", f"{split}.index.bin"))
        if {entry.name for entry in directory.iterdir()} != expected_files:
            raise WindowCacheError("Tokenized cache contains unexpected files")
        _require_targets(splits, identity, counts=True)
        examples = WindowCache(manifest)
        for split in SPLITS:
            examples[split] = WindowSequence(directory, splits[split],
                                             identity["contextLength"], identity["vocabSize"])
        return examples
    except (OSError, UnicodeError, json.JSONDecodeError, CorpusError, TypeError, KeyError,
            struct.error, RecursionError) as error:
        if examples is not None:
            close_windows(examples)
        raise WindowCacheError("Invalid tokenized window cache") from error
    except BaseException:
        if examples is not None:
            close_windows(examples)
        raise


def _require_targets(splits, identity, *, counts=False):
    required = ("train", "validation")
    if identity["includeTest"] and identity["partitionPolicy"] != "train_validation_only":
        required = SPLITS
    if any(not (splits[split]["count"] if counts else splits[split]) for split in required):
        raise ValueError("Each split needs at least one encoded text target"
                         if identity["objective"] == "causal_lm"
                         else "Each split needs at least one encoded assistant target")


def _sync(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_cache(directory, records, tokenizer, identity):
    # Import lazily: ingestion and corpus validation do not require torch.
    from .training import _record_windows

    counts = dict.fromkeys(SPLITS, 0)
    record_counts = dict.fromkeys(SPLITS, 0)
    offsets = dict.fromkeys(SPLITS, 0)
    with ExitStack() as stack:
        data = {split: stack.enter_context((directory / f"{split}.windows.bin").open("wb"))
                for split in SPLITS}
        indexes = {split: stack.enter_context((directory / f"{split}.index.bin").open("wb"))
                   for split in SPLITS}
        for record in records:
            split = record["split"]
            record_counts[split] += 1
            if identity["partitionPolicy"] == "train_validation_only" and split == "test":
                raise ValueError("train_validation_only corpus must not contain TEST records")
            # TEST is validated by StreamingCorpus, but never tokenized during training.
            if split == "test" and not identity["includeTest"]:
                continue
            for inputs, labels in _record_windows(record, tokenizer, identity["contextLength"],
                                                   identity["windowPolicy"], objective=identity["objective"]):
                length = len(inputs)
                if (not 1 <= length <= identity["contextLength"] or len(labels) != length
                        or any(type(token) is not int or not 0 <= token < identity["vocabSize"]
                               for token in inputs)
                        or not any(label != IGNORE_INDEX for label in labels)
                        or any(type(label) is not int or (label != IGNORE_INDEX
                               and not 0 <= label < identity["vocabSize"]) for label in labels)):
                    raise WindowCacheError("Encoder returned invalid tokenized window")
                tokens = array("i", inputs)
                tokens.extend(labels)
                if tokens.itemsize != 4:
                    raise WindowCacheError("Platform int array must use 32-bit elements")
                if sys.byteorder != "little":
                    tokens.byteswap()
                data[split].write(tokens.tobytes())
                indexes[split].write(_INDEX_ENTRY.pack(offsets[split], length))
                offsets[split] += length * 8
                counts[split] += 1
        if identity["partitionPolicy"] is None and not record_counts["test"]:
            raise ValueError("Each split needs at least one corpus record, including TEST")
        _require_targets(counts, identity)
        for stream in (*data.values(), *indexes.values()):
            _sync(stream)
    manifest = {"schemaVersion": _SCHEMA_VERSION, "identity": identity, "splits": {
        split: {"count": counts[split],
                "data": _file_metadata(directory / f"{split}.windows.bin"),
                "index": _file_metadata(directory / f"{split}.index.bin")}
        for split in SPLITS}}
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        stream.write(_canonical_json(manifest) + "\n")
        _sync(stream)
    _sync_directory(directory)


def build_or_load_windows(cache_dir, records, tokenizer, context_length, window_policy, *,
                          objective=None, partition_policy=None, corpus_sha256, include_test=False):
    """Open an exact-identity cache or atomically build one when absent.

    A damaged or mismatched existing cache is rejected, never silently replaced.
    Include TEST only for explicit final evaluation, using a separate cache path.
    Call ``close_windows`` (or the returned cache's ``close``) after use.
    """
    from .tokenizer import MAX_CONTEXT_LENGTH, validate_tokenizer

    _encoding_options(objective, partition_policy)
    if type(context_length) is not int or not 2 <= context_length <= MAX_CONTEXT_LENGTH:
        raise ValueError("Invalid encoding context_length")
    if window_policy not in ("contiguous_nonoverlapping", "sliding_context"):
        raise ValueError("Invalid window_policy")
    if type(include_test) is not bool:
        raise ValueError("include_test must be a boolean")
    if type(corpus_sha256) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", corpus_sha256) is None:
        raise ValueError("corpus_sha256 must be a SHA-256 hexadecimal digest")
    validate_tokenizer(tokenizer)
    corpus_sha256 = corpus_sha256.lower()
    if isinstance(records, StreamingCorpus):
        if partition_policy == "train_validation_only" and records.counts["test"]:
            raise ValueError("train_validation_only corpus must not contain TEST records")
        if (records.sha256 != corpus_sha256
                or (records.objective or "assistant_last_reply") != (objective or "assistant_last_reply")
                or records.partition_policy != partition_policy):
            raise ValueError("Streaming corpus identity or encoding configuration mismatch")
        records.verify()
    identity = {"corpusSha256": corpus_sha256,
                "tokenizerSha256": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest(),
                "vocabSize": tokenizer.get_vocab_size(), "contextLength": context_length,
                "windowPolicy": window_policy, "objective": objective or "assistant_last_reply",
                "partitionPolicy": partition_policy, "includeTest": include_test,
                "windowEncoding": _ENCODING}
    if isinstance(records, StreamingCorpus):
        if partition_policy is None and not records.counts["test"]:
            raise ValueError("Each split needs at least one corpus record, including TEST")
        _require_targets(records.counts, identity)
    directory = Path(cache_dir)
    if directory.exists() or directory.is_symlink():
        if directory.is_symlink() or not directory.is_dir():
            raise WindowCacheError("Tokenized cache path must be a directory")
        examples = _open_cache(directory, identity)
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
        try:
            _write_cache(temporary, records, tokenizer, identity)
            if isinstance(records, StreamingCorpus):
                records.verify()
            # A complete directory becomes visible in one rename. If another writer
            # published first, its manifest must satisfy exactly the same checks.
            try:
                os.rename(temporary, directory)
            except OSError as error:
                if (error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                        or directory.is_symlink() or not directory.is_dir()):
                    raise
            else:
                _sync_directory(directory.parent)
            examples = _open_cache(directory, identity)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    if isinstance(records, StreamingCorpus):
        try:
            records.verify()
        except BaseException:
            close_windows(examples)
            raise
    return examples
