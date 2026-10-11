"""Private, local records of explicit tasks, attempts and human feedback.

These records never execute tasks or update model parameters. Accepted examples
can be exported for a separate, explicitly requested language-model training run.
"""

from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

try:
    import fcntl
except ImportError:  # Windows: thread safety only, documented in workspace.md.
    fcntl = None


class WorkspaceError(ValueError):
    """Invalid input or damaged history; existing evidence is never reset."""


SCHEMA_VERSION = 1
LOG_NAME = "events.jsonl"
MAX_TEXT_BYTES = 65_536
MAX_EVIDENCE_BYTES = 8_192
MAX_EVIDENCE_REFERENCES = 16
MAX_RECORD_BYTES = 262_144
MAX_TIMESTAMP = 9_223_372_036_854_775_807
_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()
# The log remains authoritative. Cached replay is reused only after hashing all
# current bytes, including on same-size rewrites; no timestamp shortcut is used.
_REPLAY_CACHE = OrderedDict()
_REPLAY_CACHE_GUARD = threading.Lock()
_MAX_CACHED_WORKSPACES = 4
_MAX_CACHED_LOG_BYTES = 2 * 1024 * 1024


def _root(path):
    return Path(path).expanduser().absolute()


def _chmod(path, mode):
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # Filesystems without Unix modes still retain the local-only policy.


def _mkdir_private(path, *, tighten_existing=False):
    missing = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise WorkspaceError("Cannot use a broken directory symlink")
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    for directory in reversed(missing):
        _chmod(directory, 0o700)
    if tighten_existing:
        _chmod(path, 0o700)


def _id(value, prefix):
    if not isinstance(value, str) or re.fullmatch(prefix + r"_[0-9a-f]{32}", value) is None:
        raise WorkspaceError(f"Invalid {prefix} identifier")
    return value


def _text(value, name, *, allow_empty=False, limit=MAX_TEXT_BYTES):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise WorkspaceError(f"{name} must be explicit text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise WorkspaceError(f"{name} must be valid UTF-8 text") from error
    if size > limit:
        raise WorkspaceError(f"{name} exceeds {limit} UTF-8 bytes")
    return value


def _timestamp(value):
    if type(value) is not int or not 0 <= value <= MAX_TIMESTAMP:
        raise WorkspaceError("Event time must be a nonnegative integer in milliseconds")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise WorkspaceError(f"Invalid {name} schema")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkspaceError("Duplicate JSON keys in workspace history")
        result[key] = value
    return result


def _bad_constant(value):
    raise WorkspaceError(f"Nonfinite JSON number {value} in workspace history")


def _new_id(prefix):
    return prefix + "_" + uuid.uuid4().hex


def _event(kind, payload, last_at=0):
    return {"schemaVersion": SCHEMA_VERSION, "id": _new_id("event"), "at": max(time.time_ns() // 1_000_000, last_at),
            "kind": kind, "payload": payload}


@contextmanager
def _locked(root):
    if not root.is_dir() or root.is_symlink():
        raise WorkspaceError("Initialize a regular local workspace directory first")
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(str(root.resolve()), threading.RLock())
    with thread_lock:
        lock_path = root / ".lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WorkspaceError("Workspace lock must be a regular file")
            _chmod(lock_path, 0o600)
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _apply(state, event):
    _keys(event, {"schemaVersion", "id", "at", "kind", "payload"}, "event")
    if type(event["schemaVersion"]) is not int or event["schemaVersion"] != SCHEMA_VERSION:
        raise WorkspaceError("Unsupported workspace schema")
    event_id = _id(event["id"], "event")
    at = _timestamp(event["at"])
    if event_id in state["eventIds"] or at < state["lastAt"]:
        raise WorkspaceError("Duplicate event identity or decreasing event time")
    payload = event["payload"]
    kind = event["kind"]
    if kind == "workspace_initialized":
        _keys(payload, {"workspaceId", "privacy"}, "initialization")
        if state["eventIds"] or payload["privacy"] != "PRIVATE":
            raise WorkspaceError("Workspace initialization must be first and PRIVATE")
        state["workspaceId"] = _id(payload["workspaceId"], "workspace")
    elif not state["workspaceId"]:
        raise WorkspaceError("Workspace history lacks initialization")
    elif kind == "task_added":
        _keys(payload, {"taskId", "goal", "context"}, "task")
        task_id = _id(payload["taskId"], "task")
        _text(payload["goal"], "goal")
        _text(payload["context"], "context", allow_empty=True)
        if task_id in state["tasks"]:
            raise WorkspaceError("Duplicate task identity")
        state["tasks"][task_id] = {"id": task_id, "goal": payload["goal"], "context": payload["context"],
                                   "createdAt": at, "eventId": event_id, "attempts": []}
    elif kind == "attempt_recorded":
        _keys(payload, {"taskId", "attemptId", "text", "evidence"}, "attempt")
        task_id = _id(payload["taskId"], "task")
        attempt_id = _id(payload["attemptId"], "attempt")
        _text(payload["text"], "attempt text")
        evidence = payload["evidence"]
        if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_REFERENCES:
            raise WorkspaceError("Evidence must be a limited list of explicit references")
        for reference in evidence:
            _text(reference, "evidence reference", limit=MAX_EVIDENCE_BYTES)
        if task_id not in state["tasks"] or attempt_id in state["attempts"]:
            raise WorkspaceError("Attempt requires an existing task and a new identity")
        attempt = {"id": attempt_id, "taskId": task_id, "text": payload["text"], "evidence": list(evidence),
                   "recordedAt": at, "eventId": event_id, "accepted": None, "feedback": []}
        state["attempts"][attempt_id] = attempt
        state["tasks"][task_id]["attempts"].append(attempt)
    elif kind == "feedback_recorded":
        _keys(payload, {"taskId", "attemptId", "text", "accepted"}, "feedback")
        task_id = _id(payload["taskId"], "task")
        attempt_id = _id(payload["attemptId"], "attempt")
        _text(payload["text"], "feedback text")
        if type(payload["accepted"]) is not bool:
            raise WorkspaceError("Feedback needs an explicit accepted boolean")
        attempt = state["attempts"].get(attempt_id)
        if attempt is None or attempt["taskId"] != task_id:
            raise WorkspaceError("Feedback task and attempt association is invalid")
        attempt["feedback"].append({"eventId": event_id, "at": at, "taskId": task_id, "attemptId": attempt_id,
                                    "text": payload["text"], "accepted": payload["accepted"]})
        attempt["accepted"] = payload["accepted"]
    else:
        raise WorkspaceError("Unknown workspace event kind")
    state["eventIds"].add(event_id)
    state["lastAt"] = at


def _cached_replay(root, *, remove=False):
    key = str(root.resolve())
    with _REPLAY_CACHE_GUARD:
        if remove:
            return _REPLAY_CACHE.pop(key, None)
        entry = _REPLAY_CACHE.get(key)
        if entry is not None:
            _REPLAY_CACHE.move_to_end(key)
        return entry


def _cache_replay(root, state, digest, size):
    # All callers hold the workspace lock. A separate guard protects the bounded
    # process-wide LRU across independently locked product workspaces.
    key = str(root.resolve())
    with _REPLAY_CACHE_GUARD:
        _REPLAY_CACHE.pop(key, None)
        if size > _MAX_CACHED_LOG_BYTES:
            return
        _REPLAY_CACHE[key] = (state, digest.copy(), size)
        while len(_REPLAY_CACHE) > _MAX_CACHED_WORKSPACES:
            _REPLAY_CACHE.popitem(last=False)


def _load(root):
    log = root / LOG_NAME
    if log.is_symlink() or not log.is_file():
        raise WorkspaceError("Initialize a regular workspace event log first")
    with log.open("rb") as stream:
        cached = _cached_replay(root)
        if cached is not None:
            if hashlib.file_digest(stream, "sha256").digest() == cached[1].digest():
                return cached[0]  # Internal state only; public snapshots are detached.
            _cached_replay(root, remove=True)
            stream.seek(0)
        state = {"workspaceId": None, "tasks": {}, "attempts": {}, "eventIds": set(), "lastAt": 0}
        digest = hashlib.sha256()
        while True:
            line = stream.readline(MAX_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n") or not line.strip():
                raise WorkspaceError("Damaged or oversized workspace event; history preserved")
            try:
                event = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant)
                _apply(state, event)
            except (ValueError, UnicodeError, KeyError, TypeError, RecursionError) as error:
                raise WorkspaceError(f"Cannot reconstruct workspace history; original log preserved: {error}") from error
            digest.update(line)
        size = stream.tell()
    if not state["workspaceId"]:
        raise WorkspaceError("Empty workspace history; original log preserved")
    state["logSha256"] = digest.hexdigest()
    _cache_replay(root, state, digest, size)
    return state


def _append(root, state, event):
    # Validation may mutate state, and any write/fsync can fail. Remove cached
    # ownership before either, so a failed operation cannot publish phantom data.
    cached = _cached_replay(root, remove=True)
    _apply(state, event)  # Validate before writing any new bytes.
    record = (json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(record) > MAX_RECORD_BYTES:
        raise WorkspaceError("Workspace event is too large")
    descriptor = os.open(root / LOG_NAME, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    try:
        remaining = memoryview(record)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Cannot append workspace event")
            remaining = remaining[written:]
        os.fsync(descriptor)
        _chmod(root / LOG_NAME, 0o600)
    finally:
        os.close(descriptor)
    if cached is not None and cached[0] is state:
        digest = cached[1].copy()
        digest.update(record)
        state["logSha256"] = digest.hexdigest()
        _cache_replay(root, state, digest, cached[2] + len(record))


def _status(root, state):
    return {"path": str(root), "workspaceId": state["workspaceId"], "privacy": "PRIVATE",
            "eventCount": len(state["eventIds"]), "taskCount": len(state["tasks"]),
            "attemptCount": len(state["attempts"]),
            "acceptedAttemptCount": sum(attempt["accepted"] is True for attempt in state["attempts"].values())}


def init_workspace(path):
    root = _root(path)
    if root.is_symlink():
        raise WorkspaceError("Workspace directory cannot be a symlink")
    _mkdir_private(root, tighten_existing=True)
    with _locked(root):
        log = root / LOG_NAME
        if log.exists() or log.is_symlink():
            return _status(root, _load(root))
        descriptor = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(descriptor)
        state = {"workspaceId": None, "tasks": {}, "attempts": {}, "eventIds": set(), "lastAt": 0}
        _append(root, state, _event("workspace_initialized", {"workspaceId": _new_id("workspace"), "privacy": "PRIVATE"}))
        return _status(root, state)


def add_task(path, goal, context=""):
    root = _root(path)
    _text(goal, "goal")
    _text(context, "context", allow_empty=True)
    with _locked(root):
        state = _load(root)
        task_id = _new_id("task")
        _append(root, state, _event("task_added", {"taskId": task_id, "goal": goal, "context": context}, state["lastAt"]))
        return deepcopy(state["tasks"][task_id])


def record_attempt(path, task_id, text, evidence=None):
    root = _root(path)
    _id(task_id, "task")
    with _locked(root):
        state = _load(root)
        attempt_id = _new_id("attempt")
        event = _event("attempt_recorded", {"taskId": task_id, "attemptId": attempt_id, "text": text,
                                             "evidence": [] if evidence is None else evidence}, state["lastAt"])
        _append(root, state, event)
        return deepcopy(state["attempts"][attempt_id])


def feedback(path, task_id, attempt_id, text, accepted):
    root = _root(path)
    _id(task_id, "task")
    _id(attempt_id, "attempt")
    with _locked(root):
        state = _load(root)
        _append(root, state, _event("feedback_recorded", {"taskId": task_id, "attemptId": attempt_id,
                                                         "text": text, "accepted": accepted}, state["lastAt"]))
        return deepcopy(state["attempts"][attempt_id]["feedback"][-1])


def show(path, task_id=None):
    root = _root(path)
    with _locked(root):
        state = _load(root)
        if task_id is not None:
            _id(task_id, "task")
            if task_id not in state["tasks"]:
                raise WorkspaceError("Unknown task identifier")
            return deepcopy(state["tasks"][task_id])
        return {"workspaceId": state["workspaceId"], "privacy": "PRIVATE", "eventCount": len(state["eventIds"]),
                "tasks": deepcopy(list(state["tasks"].values()))}


def list_tasks(path):
    """Return task summaries without copying contexts, attempts or feedback."""
    root = _root(path)
    with _locked(root):
        state = _load(root)
        tasks = [{"id": task["id"], "goal": task["goal"], "createdAt": task["createdAt"],
                  "attemptCount": len(task["attempts"]),
                  "acceptedAttemptCount": sum(attempt["accepted"] is True for attempt in task["attempts"])}
                 for task in state["tasks"].values()]
        return {"workspaceId": state["workspaceId"], "privacy": "PRIVATE", "eventCount": len(state["eventIds"]),
                "tasks": tasks}


def status(path):
    root = _root(path)
    with _locked(root):
        return _status(root, _load(root))


def search_tasks(path, query, limit=10):
    """Search explicit local memory once, copying only bounded matching excerpts."""
    from .memory_search import search_snapshot, validate_search

    try:
        validate_search(query, limit)
    except ValueError as error:
        raise WorkspaceError(str(error)) from error
    root = _root(path)
    with _locked(root):
        return search_snapshot(_load(root), query, limit)


def _write_new(path, content):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _chmod(path, 0o600)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def export_training(path, destination):
    root = _root(path)
    output = _root(destination)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.resolve() in {(root / LOG_NAME).resolve(), (root / ".lock").resolve()}:
        raise WorkspaceError("Training export cannot replace workspace history or its lock")
    if any(candidate.exists() or candidate.is_symlink() for candidate in (output, manifest_path)):
        raise WorkspaceError("Training export requires new output and manifest paths; existing evidence is preserved")
    with _locked(root):
        state = _load(root)
        accepted = [attempt for attempt in state["attempts"].values() if attempt["accepted"] is True]
        if not accepted:
            raise WorkspaceError("No explicitly accepted attempts; no training examples were invented or written")
        chunks, samples = [], []
        for attempt in accepted:
            task = state["tasks"][attempt["taskId"]]
            last_feedback = attempt["feedback"][-1]
            parts = ["Encargo:\n" + task["goal"]]
            if task["context"]:
                parts.append("Contexto:\n" + task["context"])
            parts.extend(["Intento aceptado:\n" + attempt["text"], "Comentario explícito:\n" + last_feedback["text"]])
            chunk = "\n\n".join(parts)
            # The character model treats a standalone --- line as a document
            # boundary. Preserve one whole accepted example per document.
            chunk = re.sub(r"(?m)^(\s*)---(\s*)$", lambda match: match[1] + "\\---" + match[2], chunk)
            chunks.append(chunk)
            samples.append({"taskId": task["id"], "attemptId": attempt["id"],
                            "sourceEventIds": [task["eventId"], attempt["eventId"], last_feedback["eventId"]],
                            "evidenceReferences": attempt["evidence"], "accepted": True})
        corpus = ("\n\n---\n\n".join(chunks) + "\n").encode("utf-8")
        corpus_hash = hashlib.sha256(corpus).hexdigest()
        manifest = {"schemaVersion": SCHEMA_VERSION, "privacy": "PRIVATE", "format": "utf8-text",
                    "exportedAt": max(time.time_ns() // 1_000_000, state["lastAt"]),
                    "workspaceId": state["workspaceId"], "sourceEventLog": str(root / LOG_NAME),
                    "sourceEventLogSha256": state["logSha256"], "sourceEventCount": len(state["eventIds"]),
                    "corpusSha256": corpus_hash, "sampleCount": len(samples), "samples": samples,
                    "parametersTrained": False, "evidenceReferencesVerified": False,
                    "documentSeparator": "standalone --- line", "literalSeparatorEscape": "\\---; originals remain in event log"}
        _mkdir_private(output.parent)
        _write_new(output, corpus)
        try:
            _write_new(manifest_path, (json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
        except Exception:
            output.unlink(missing_ok=True)
            raise
    return {"destination": str(output), "manifest": str(manifest_path), "sampleCount": len(samples),
            "corpusSha256": corpus_hash, "parametersTrained": False, "privacy": "PRIVATE"}
