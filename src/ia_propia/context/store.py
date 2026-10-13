"""Local, incremental conversation sources; retrieval does not accept facts.

The host chooses the database, product and session. Assistant text is retained
only as dialogue history and is never returned by memory search. ``append`` is
an O(batch) delta write; ``append_history`` validates and hashes the complete
snapshot, so its input cost grows with the supplied history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
import unicodedata
from typing import Any, Mapping, Sequence


MAX_BATCH = 64
MAX_TEXT_BYTES = 16_384
MAX_QUERY_CHARS = 4_096
MAX_RESULTS = 64
MAX_SEARCH_CANDIDATES = 256
MAX_QUERY_TERMS = 16
_QUERY_FILLERS = frozenset(("que", "qué", "cual", "cuál", "cuando", "cuándo",
                            "donde", "dónde", "como", "cómo", "el", "la", "los", "las",
                            "un", "una", "de", "del", "en", "a", "al", "es", "y"))
_EMPTY_PREFIX = hashlib.sha256(b"unidad-context-v1").hexdigest()


class ContextStoreError(ValueError):
    """Invalid conversation sources, bounds or a divergent history snapshot."""


@dataclass(frozen=True)
class Source:
    id: str
    text: str
    role: str
    turn_index: int
    sha256: str
    source: str = "conversation"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "role": self.role,
                "turn_index": self.turn_index, "sha256": self.sha256,
                "source": self.source}


@dataclass(frozen=True)
class AppendResult:
    inserted: int
    total: int
    sources: tuple[Source, ...]


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ContextStoreError(f"{name} must be nonempty text of at most 256 characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ContextStoreError(f"{name} must be valid UTF-8") from exc
    if any(ord(char) < 32 for char in value):
        raise ContextStoreError(f"{name} cannot contain control characters")
    return value


def _bound(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ContextStoreError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _message(value: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(value, Mapping) or set(value) != {"role", "content"}:
        raise ContextStoreError("messages must contain exactly role and content")
    role, text = value["role"], value["content"]
    if role not in ("user", "system", "assistant"):
        raise ContextStoreError("message role must be user, system or assistant")
    if not isinstance(text, str) or not text.strip():
        raise ContextStoreError("message content must be nonempty text")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ContextStoreError("message content must be valid UTF-8") from exc
    if len(raw) > MAX_TEXT_BYTES:
        raise ContextStoreError("message content exceeds 16384 bytes")
    return role, text, hashlib.sha256(raw).hexdigest()


def _batch(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    if not isinstance(messages, (list, tuple)) or len(messages) > MAX_BATCH:
        raise ContextStoreError("message batch must be a list or tuple of at most 64 messages")
    return [_message(message) for message in messages]


def _prefix(previous: str, role: str, text_hash: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + role.encode("ascii") +
                          bytes.fromhex(text_hash)).hexdigest()


def _search_scope(product_id: str, session_id: str, *, assistant: bool = False) -> str:
    scope = hashlib.sha256((product_id + "\0" + session_id).encode("utf-8")).hexdigest()
    return "scope" + scope + ("assistant" if assistant else "memory")


def _terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return tuple(dict.fromkeys(re.findall(r"[^\W_]+", normalized)))


class ContextStore:
    """SQLite sources isolated by a host-selected product and conversation."""

    def __init__(self, path: str | Path, *, product_id: str, session_id: str,
                 fallback_scan_limit: int = 4_096, use_fts: bool = True):
        self.product_id = _identifier(product_id, "product_id")
        self.session_id = _identifier(session_id, "session_id")
        self._memory_scope = _search_scope(product_id, session_id)
        self._assistant_scope = _search_scope(product_id, session_id, assistant=True)
        self.fallback_scan_limit = _bound(fallback_scan_limit, "fallback_scan_limit", 65_536)
        if type(use_fts) is not bool:
            raise ContextStoreError("use_fts must be a boolean")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        with self._connection:
            self._connection.execute("""CREATE TABLE IF NOT EXISTS context_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL, session_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, role TEXT NOT NULL,
                text TEXT NOT NULL, sha256 TEXT NOT NULL, prefix_sha256 TEXT NOT NULL,
                search_scope TEXT NOT NULL,
                UNIQUE(product_id, session_id, ordinal)
            )""")
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(context_sources)")}
            if "search_scope" not in columns:
                # Upgrade the first storage schema without changing source IDs,
                # original text, prefix hashes or the conversation's ordinals.
                self._connection.execute("ALTER TABLE context_sources ADD COLUMN search_scope TEXT NOT NULL DEFAULT ''")
                scopes = self._connection.execute("SELECT DISTINCT product_id,session_id FROM context_sources")
                for product, session in scopes:
                    self._connection.execute("""UPDATE context_sources SET search_scope=
                        CASE WHEN role='assistant' THEN ? ELSE ? END
                        WHERE product_id=? AND session_id=?""",
                        (_search_scope(product, session, assistant=True),
                         _search_scope(product, session), product, session))
            self._connection.execute("""CREATE INDEX IF NOT EXISTS context_scope_role
                ON context_sources(product_id, session_id, role, ordinal DESC)""")
            self._connection.execute("""CREATE INDEX IF NOT EXISTS context_scope_order
                ON context_sources(product_id, session_id, ordinal DESC)""")
        self.fts_enabled = False
        if use_fts:
            self._enable_fts()

    def _enable_fts(self) -> None:
        # Only missing FTS5 support selects the fallback. Other database errors
        # propagate instead of silently masking a broken index or storage.
        exists = self._connection.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='context_sources_fts'""").fetchone()
        try:
            with self._connection:
                columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(context_sources_fts)")}
                if exists and "search_scope" not in columns:
                    self._connection.execute("DROP TRIGGER IF EXISTS context_sources_insert")
                    self._connection.execute("DROP TRIGGER IF EXISTS context_sources_delete")
                    self._connection.execute("DROP TABLE context_sources_fts")
                    exists = None
                self._connection.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS context_sources_fts
                    USING fts5(text, search_scope, content='context_sources', content_rowid='source_id',
                               tokenize='unicode61 remove_diacritics 0')""")
                self._connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS context_sources_insert AFTER INSERT ON context_sources BEGIN
                        INSERT INTO context_sources_fts(rowid,text,search_scope)
                        VALUES(new.source_id,new.text,new.search_scope);
                    END
                """)
                self._connection.execute("""
                    CREATE TRIGGER IF NOT EXISTS context_sources_delete AFTER DELETE ON context_sources BEGIN
                        INSERT INTO context_sources_fts(context_sources_fts,rowid,text,search_scope)
                        VALUES('delete',old.source_id,old.text,old.search_scope);
                    END
                """)
                if not exists:
                    self._connection.execute("""INSERT INTO context_sources_fts(context_sources_fts)
                        VALUES('rebuild')""")
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" not in str(exc).lower():
                raise
        else:
            self.fts_enabled = True

    def _check_open(self) -> None:
        if self._closed:
            raise ContextStoreError("context store is closed")

    @property
    def _scope(self) -> tuple[str, str]:
        return self.product_id, self.session_id

    @staticmethod
    def _source(row: sqlite3.Row) -> Source:
        return Source(f"context:{row['source_id']}", row["text"], row["role"],
                      row["ordinal"], row["sha256"])

    def _last(self) -> sqlite3.Row | None:
        return self._connection.execute("""SELECT ordinal,prefix_sha256 FROM context_sources
            WHERE product_id=? AND session_id=? ORDER BY ordinal DESC LIMIT 1""",
            self._scope).fetchone()

    def _insert(self, values: list[tuple[str, str, str]], start: int,
                previous: str) -> AppendResult:
        sources = []
        for ordinal, (role, text, text_hash) in enumerate(values, start):
            previous = _prefix(previous, role, text_hash)
            cursor = self._connection.execute("""INSERT INTO context_sources
                (product_id,session_id,ordinal,role,text,sha256,prefix_sha256,search_scope)
                VALUES(?,?,?,?,?,?,?,?)""", (*self._scope, ordinal, role, text, text_hash, previous,
                    self._assistant_scope if role == "assistant" else self._memory_scope))
            sources.append(Source(f"context:{cursor.lastrowid}", text, role, ordinal, text_hash))
        return AppendResult(len(values), start + len(values), tuple(sources))

    def append(self, new_messages: Sequence[Mapping[str, Any]]) -> AppendResult:
        """Append a delta atomically. Identical messages at new turns stay distinct."""
        values = _batch(new_messages)
        with self._lock:
            self._check_open()
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                last = self._last()
                start = last["ordinal"] + 1 if last else 0
                previous = last["prefix_sha256"] if last else _EMPTY_PREFIX
                return self._insert(values, start, previous)

    def append_history(self, messages: Sequence[Mapping[str, Any]]) -> AppendResult:
        """Verify an entire <=64-turn snapshot and append only its new suffix.

        Matching shorter snapshots are no-ops. A changed earlier turn raises;
        callers must explicitly reset or choose another session for a new branch.
        """
        values = _batch(messages)
        prefixes = []
        previous = _EMPTY_PREFIX
        for role, _, text_hash in values:
            previous = _prefix(previous, role, text_hash)
            prefixes.append(previous)
        with self._lock:
            self._check_open()
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                last = self._last()
                total = last["ordinal"] + 1 if last else 0
                overlap = min(total, len(values))
                if overlap:
                    row = self._connection.execute("""SELECT prefix_sha256 FROM context_sources
                        WHERE product_id=? AND session_id=? AND ordinal=?""",
                        (*self._scope, overlap - 1)).fetchone()
                    if row is None or row["prefix_sha256"] != prefixes[overlap - 1]:
                        raise ContextStoreError("history prefix changed; reset explicitly or use another session")
                if len(values) <= total:
                    return AppendResult(0, total, ())
                previous = last["prefix_sha256"] if last else _EMPTY_PREFIX
                return self._insert(values[total:], total, previous)

    def count(self) -> int:
        with self._lock:
            self._check_open()
            last = self._last()
            return last["ordinal"] + 1 if last else 0

    def stats(self) -> dict[str, Any]:
        # Host paths are intentionally omitted from response-safe diagnostics.
        return {"count": self.count(), "fts_enabled": self.fts_enabled}

    def recent(self, *, limit: int = 8, include_assistant: bool = False) -> list[Source]:
        limit = _bound(limit, "limit", MAX_RESULTS)
        if type(include_assistant) is not bool:
            raise ContextStoreError("include_assistant must be a boolean")
        roles = "" if include_assistant else " AND role IN ('user','system')"
        with self._lock:
            self._check_open()
            rows = self._connection.execute(f"""SELECT * FROM context_sources INDEXED BY context_scope_order
                WHERE product_id=? AND session_id=?{roles}
                ORDER BY ordinal DESC LIMIT ?""", (*self._scope, limit)).fetchall()
            return [self._source(row) for row in reversed(rows)]

    def search(self, query: str, *, top_k: int = 5, max_chars: int = 4_096) -> list[Source]:
        """Retrieve complete user/system sources within a total text budget.

        Fallback scans at most the most recent ``fallback_scan_limit`` sources;
        FTS first prefers recent sources matching every significant query term,
        across the complete session. It fills a bounded candidate pool with
        recent matches for each separate term, then ranks by weighted term
        overlap and recency. Weights estimate rarity from bounded posting
        samples, not global BM25. This finds an old rare entity beside common words;
        older partial matches can fall outside the per-term candidate bound.
        No excerpt changes original text or its SHA-256.
        """
        top_k = _bound(top_k, "top_k", MAX_RESULTS)
        max_chars = _bound(max_chars, "max_chars", MAX_TEXT_BYTES * MAX_RESULTS)
        if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS:
            raise ContextStoreError("query must be text of at most 4096 characters")
        try:
            query.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ContextStoreError("query must be valid UTF-8") from exc
        terms = tuple(term for term in _terms(query) if term not in _QUERY_FILLERS)[:MAX_QUERY_TERMS]
        candidate_limit = min(MAX_SEARCH_CANDIDATES, max(64, top_k * 4))
        with self._lock:
            self._check_open()
            if not terms:
                return []
            if self.fts_enabled:
                scope_match = 'search_scope:"' + self._memory_scope + '" AND '
                full_match = scope_match + " AND ".join('text:"' + term + '"' for term in terms)
                rows = self._connection.execute("""SELECT s.* FROM context_sources_fts
                    JOIN context_sources s ON s.source_id=context_sources_fts.rowid
                    WHERE context_sources_fts MATCH ? AND s.product_id=? AND s.session_id=?
                    AND s.role IN ('user','system')
                    ORDER BY context_sources_fts.rowid DESC LIMIT ?""",
                    (full_match, *self._scope, candidate_limit)).fetchall()
                if len(self._budget(rows, top_k, max_chars)) < top_k:
                    seen = {row["source_id"] for row in rows}
                    weights = {term: 1.0 for term in terms}
                    remaining = candidate_limit - len(rows)
                    per_term = min(32, remaining // len(terms))
                    if per_term:
                        for term in terms:
                            match = scope_match + 'text:"' + term + '"'
                            partials = self._connection.execute("""SELECT s.* FROM context_sources_fts
                                JOIN context_sources s ON s.source_id=context_sources_fts.rowid
                                WHERE context_sources_fts MATCH ? AND s.product_id=? AND s.session_id=?
                                AND s.role IN ('user','system')
                                ORDER BY context_sources_fts.rowid DESC LIMIT ?""",
                                (match, *self._scope, per_term)).fetchall()
                            weights[term] = 1.0 / (1 + len(partials))
                            for row in partials:
                                if row["source_id"] not in seen:
                                    rows.append(row)
                                    seen.add(row["source_id"])
                    wanted = set(terms)
                    rows.sort(key=lambda row: (sum(weights[term] for term in wanted.intersection(_terms(row["text"]))),
                                               row["ordinal"]), reverse=True)
            else:
                rows = self._fallback(terms, candidate_limit)
            return self._budget(rows, top_k, max_chars)

    def _budget(self, rows: list[sqlite3.Row], top_k: int, max_chars: int) -> list[Source]:
        result, used = [], 0
        for row in rows:
            size = len(row["text"])
            if used + size <= max_chars:
                result.append(self._source(row))
                used += size
                if len(result) == top_k:
                    break
        return result

    def _fallback(self, terms: tuple[str, ...], candidate_limit: int) -> list[sqlite3.Row]:
        wanted = set(terms)
        best: list[tuple[int, int, sqlite3.Row]] = []
        rows = self._connection.execute("""SELECT * FROM context_sources INDEXED BY context_scope_order
            WHERE product_id=? AND session_id=? AND role IN ('user','system')
            ORDER BY ordinal DESC LIMIT ?""", (*self._scope, self.fallback_scan_limit))
        for row in rows:
            matches = len(wanted.intersection(_terms(row["text"])))
            if matches:
                item = matches, row["ordinal"], row
                if len(best) < candidate_limit:
                    heapq.heappush(best, item)
                elif item[:2] > best[0][:2]:
                    heapq.heapreplace(best, item)
        return [item[2] for item in sorted(best, key=lambda item: item[:2], reverse=True)]

    def reset(self) -> None:
        """Delete only this product/session, including its FTS entries."""
        with self._lock:
            self._check_open()
            with self._connection:
                self._connection.execute("""DELETE FROM context_sources
                    WHERE product_id=? AND session_id=?""", self._scope)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> ContextStore:
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
