"""Bounded lexical retrieval from a validated workspace snapshot.

This module does not load models, edit records or interpret a query as an order.
The workspace owns locking and validation of the event log.
"""

import hashlib
import re
import unicodedata

MAX_QUERY_BYTES = 512
MAX_QUERY_TERMS = 16
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 20
MAX_EXCERPT_CHARS = 160
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
SEARCH_METHOD = "lexical_all_words_nfc_casefold"


def _words(text):
    """Yield whole words with combining marks and untouched original offsets."""
    consumed_end = 0
    for match in _WORDS.finditer(text):
        if match.start() < consumed_end:
            continue
        start, end = match.span()
        while end < len(text) and unicodedata.category(text[end]).startswith("M"):
            end += 1
            continuation = _WORDS.match(text, end)
            if continuation is not None:
                end = continuation.end()
        consumed_end = end
        yield unicodedata.normalize("NFC", text[start:end]).casefold(), start, end


def validate_search(query, limit=DEFAULT_SEARCH_LIMIT):
    """Return distinct NFC/casefolded words; reject input before log access."""
    if type(query) is not str or not query.strip():
        raise ValueError("Search query must be explicit text")
    try:
        size = len(query.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("Search query must be valid UTF-8") from error
    if size > MAX_QUERY_BYTES:
        raise ValueError("Search query exceeds 512 UTF-8 bytes")
    if type(limit) is not int or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ValueError("Search limit must be an integer from 1 to 20")
    terms = tuple(dict.fromkeys(term for term, _start, _end in _words(query)))
    if not terms or len(terms) > MAX_QUERY_TERMS:
        raise ValueError("Search query needs from 1 to 16 distinct words")
    return terms


def _fields(task):
    for field in ("goal", "context"):
        yield task[field], {"taskId": task["id"], "eventId": task["eventId"],
                            "field": field, "accepted": None}
    for attempt in task["attempts"]:
        if attempt["feedback"]:
            latest = attempt["feedback"][-1]
            yield latest["text"], {"taskId": task["id"], "attemptId": attempt["id"],
                                   "eventId": latest["eventId"], "field": "feedback.text",
                                   "accepted": latest["accepted"]}
        if attempt["accepted"] is True:
            yield attempt["text"], {"taskId": task["id"], "attemptId": attempt["id"],
                                    "eventId": attempt["eventId"], "field": "attempt.text",
                                    "accepted": True}


def _matches(task, terms):
    remaining = set(terms)
    found = {}
    for text, source in _fields(task):
        for term, start, end in _words(text):
            if term in remaining:
                found[term] = (text, source, start, end)
                remaining.remove(term)
                if not remaining:
                    return found
    return None


def _reference(term, match, hashes):
    text, source, start, end = match
    # Key by event and field, rather than retaining copies of long source text.
    key = (source["eventId"], source["field"])
    if key not in hashes:
        hashes[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    excerpt_start = max(0, start - 32)
    excerpt_end = min(len(text), excerpt_start + MAX_EXCERPT_CHARS)
    return {"term": term, "excerpt": text[excerpt_start:excerpt_end],
            "excerptStart": excerpt_start, "excerptEnd": excerpt_end,
            "source": {**source, "textSha256": hashes[key], "start": start, "end": end}}


def search_snapshot(state, query, limit=DEFAULT_SEARCH_LIMIT):
    """Read a validated state without mutation; copy only bounded result text."""
    terms = validate_search(query, limit)
    tasks, count, hashes = [], 0, {}
    ordered = sorted(state["tasks"].values(), key=lambda task: (-task["createdAt"], task["id"]))
    for task in ordered:
        matches = _matches(task, terms)
        if matches is None:
            continue
        count += 1
        if len(tasks) < limit:
            tasks.append({"id": task["id"], "createdAt": task["createdAt"],
                          "goalExcerpt": task["goal"][:MAX_EXCERPT_CHARS],
                          "goalTruncated": len(task["goal"]) > MAX_EXCERPT_CHARS,
                          "matches": [_reference(term, matches[term], hashes) for term in terms]})
    return {"workspaceId": state["workspaceId"], "privacy": "PRIVATE",
            "eventCount": len(state["eventIds"]), "queryTerms": list(terms),
            "method": SEARCH_METHOD, "limit": limit,
            "matchCount": count, "truncated": count > limit, "tasks": tasks}
