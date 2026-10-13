#!/usr/bin/env python3
"""Bounded retry transport for the frozen V2 collector; no training or new source.

The original collector, its lock, cache validation, partitions and publication
transaction are reused. This opt-in helper does not change a running supervisor.
Every attempted HTTP request has a receipt, including transient failures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_learning_wiki_v2 as collector

COLLECTOR_SHA256 = "0ba9290196f5689710a1988a0fa8e18b481962a1e1324816243df31c5eecbf95"
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def page_spec(order):
    count = collector.math.ceil(collector.public.WIKI_ROWS / 100)
    if not isinstance(order, int) or isinstance(order, bool) or not 0 <= order < count:
        raise ValueError("logical page order outside pinned source")
    if collector.math.gcd(7919, count) != 1:
        raise ValueError("pinned page stride is not coprime")
    offset = ((order * 7919 + 37) % count) * 100
    length = min(100, collector.public.WIKI_ROWS - offset)
    query = collector.urlencode({"dataset": collector.PIN["dataset"], "config": collector.PIN["configuration"],
                                 "split": "train", "offset": offset, "length": length,
                                 "revision": collector.PIN["revision"]})
    return offset, length, "https://datasets-server.huggingface.co/rows?" + query


def validate_coverage(result, order):
    offset, length, url = page_spec(order)
    actual_offset, raw, headers, value, actual_url = result
    rows = value["rows"]
    if (actual_offset != offset or actual_url != url or headers.get("status") != 200
            or headers.get("xRevision") != collector.PIN["revision"]
            or len(rows) != length or [row.get("row_idx") for row in rows] != list(range(offset, offset + length))):
        raise ValueError("pinned logical page coverage, URL, status or revision differs")
    return result


class RetryTransport:
    def __init__(self, receipt_root, *, max_attempts=4, max_requests=320,
                 max_network_bytes=512 * collector.MIB, disk_roots=(), max_disk_bytes=2048 * collector.MIB):
        if not 1 <= max_attempts <= 4 or not 1 <= max_requests <= 320:
            raise ValueError("retry/request limits outside finite authorized bounds")
        if not 1 <= max_network_bytes <= 512 * collector.MIB or not 1 <= max_disk_bytes <= 2048 * collector.MIB:
            raise ValueError("transport resource limits outside authorized bounds")
        self.receipt_root = Path(receipt_root)
        self.max_attempts, self.max_requests = max_attempts, max_requests
        self.max_network_bytes, self.max_disk_bytes = max_network_bytes, max_disk_bytes
        self.disk_roots = tuple(Path(root) for root in disk_roots)
        self.network_bytes = self.requests = 0
        self.original_fetch = collector.fetch_page
        self.original_cached = collector.cached_page

    def check(self, deadline, maximum_raw=0):
        if time.monotonic() >= deadline:
            raise TimeoutError("recovery transport deadline exhausted")
        if maximum_raw and self.requests >= self.max_requests:
            raise ValueError("recovery transport request cap exceeded")
        if self.network_bytes + maximum_raw + 1 > self.max_network_bytes:
            raise ValueError("recovery transport network byte cap exceeded")
        if sum(collector.causal.directory_bytes(root) for root in self.disk_roots) > self.max_disk_bytes:
            raise ValueError("recovery transport disk cap exceeded")

    def save_receipt(self, value):
        self.receipt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = "request-" + str(self.requests).zfill(6) + ".json"
        receipt_path = self.receipt_root / name
        if receipt_path.exists():
            raise ValueError("request receipt already exists; preserve previous recovery evidence")
        collector.atomic_json(receipt_path, value)
        return {"path": collector.portable(receipt_path), "sha256": collector.public.file_sha(receipt_path), **value}

    def fetch_page(self, order, maximum_raw, deadline):
        offset, _length, url = page_spec(order)
        receipts = []
        for attempt in range(1, self.max_attempts + 1):
            self.check(deadline, maximum_raw)
            self.requests += 1
            started = time.monotonic()
            try:
                with collector.urlopen(collector.Request(url, headers={
                        "User-Agent": "Unidad-public-corpus-preparation/2.0"}),
                        timeout=min(15, deadline - time.monotonic())) as response:
                    raw = response.read(maximum_raw + 1)
                    headers = {"status": response.status, "contentType": response.headers.get("Content-Type"),
                               "xRevision": response.headers.get("x-revision"), "etag": response.headers.get("ETag")}
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                headers = getattr(error, "headers", None)
                status = getattr(error, "code", None)
                raw = b""
                if isinstance(error, HTTPError):
                    raw = error.read(min(65536, self.max_network_bytes - self.network_bytes))
                    error.close()
                self.network_bytes += len(raw)
                transient = status in TRANSIENT_STATUS if status is not None else True
                receipts.append(self.save_receipt({"source": collector.PIN, "order": order, "attempt": attempt,
                    "url": url, "status": status, "xRevision": headers.get("x-revision") if headers else None,
                    "responseBodySha256": collector.public.sha(raw), "responseBodyBytesRead": len(raw),
                    "bodyReadLimitBytes": 65536, "errorType": type(error).__name__,
                    "retryable": transient, "elapsedSeconds": time.monotonic() - started}))
                if not transient or attempt == self.max_attempts:
                    raise
                delay = .25 * 2 ** (attempt - 1)
                if time.monotonic() + delay >= deadline:
                    raise TimeoutError("recovery transport deadline before retry") from error
                time.sleep(delay)
                continue
            self.network_bytes += len(raw)
            try:
                if len(raw) > maximum_raw or time.monotonic() >= deadline:
                    raise ValueError("bounded source response or time budget exceeded")
                value, spans = collector.metadata_only_document(raw)
                rows = value.get("rows")
                if (value.get("partial") or value.get("num_rows_total") != collector.public.WIKI_ROWS
                        or not isinstance(rows, list) or len(spans) != len(rows)):
                    raise ValueError("partial, changed or unrecognized pinned row schema")
                result = validate_coverage((offset, raw, headers, value, url), order)
                # TEST text is still encoded bytes; only TRAIN/VAL text is decoded.
                for wrapped, (start, end) in zip(rows, spans):
                    if collector.public.assign_split("wikipedia:es:" + str(wrapped["row"]["id"])) != "test":
                        wrapped["row"]["text"] = json.loads(raw[start:end])
            except (ValueError, KeyError, TypeError) as error:
                self.save_receipt({"source": collector.PIN, "order": order, "attempt": attempt,
                    "url": url, "status": headers["status"], "xRevision": headers["xRevision"],
                    "responseBodySha256": collector.public.sha(raw), "responseBodyBytesRead": len(raw),
                    "response": headers.copy(), "errorType": type(error).__name__,
                    "retryable": False, "validationError": str(error),
                    "elapsedSeconds": time.monotonic() - started})
                raise
            receipts.append(self.save_receipt({"source": collector.PIN, "order": order, "attempt": attempt,
                "url": result[4], "status": headers["status"], "xRevision": headers["xRevision"],
                "responseBodySha256": collector.public.sha(raw), "responseBodyBytesRead": len(raw),
                "response": headers.copy(), "errorType": None, "retryable": False,
                "elapsedSeconds": time.monotonic() - started}))
            headers["transportRecovery"] = {"kind": "bounded-retry-of-same-pinned-HTTP-page",
                "derivedAggregate": False,
                "helperSha256": collector.public.file_sha(Path(__file__)),
                "frozenCollectorSha256": COLLECTOR_SHA256, "requests": receipts,
                "requestBytesRead": sum(item["responseBodyBytesRead"] for item in receipts),
                "globalNetworkBytesReadSoFar": self.network_bytes,
                "globalRequestsSoFar": self.requests}
            self.check(deadline)
            return result
        raise RuntimeError("unreachable bounded retry state")

    def cached_page(self, cache_root, order, maximum_raw, deadline, *, offline=False):
        # Cache hits need no request budget. Frozen raw bytes are never rewritten.
        if time.monotonic() >= deadline:
            raise TimeoutError("recovery deadline before cache validation")
        result = self.original_cached(cache_root, order, maximum_raw, deadline, offline=offline)
        return validate_coverage(result, order)

    @contextmanager
    def installed(self):
        collector.fetch_page, collector.cached_page = self.fetch_page, self.cached_page
        try:
            yield self
        finally:
            collector.fetch_page, collector.cached_page = self.original_fetch, self.original_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--inbox-dir", type=Path, required=True)
    parser.add_argument("--supervisor-state", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True,
                        help="New private directory for this recovery's request receipts")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=320)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if collector.public.file_sha(Path(collector.__file__)) != COLLECTOR_SHA256:
        parser.error("frozen collector V2 hash differs")
    if args.receipt_dir.exists():
        parser.error("receipt directory must be new; existing evidence is preserved")
    recipe = collector.read(args.recipe)
    recipe["recipeSha256"] = collector.public.file_sha(args.recipe)
    limits = recipe["limits"]
    roots = (collector.path(recipe["outputRoot"]), collector.path(recipe["statePath"]).parent,
             args.inbox_dir, args.receipt_dir)
    transport = RetryTransport(args.receipt_dir, max_attempts=args.max_attempts, max_requests=args.max_requests,
        max_network_bytes=limits["maxNetworkMiB"] * collector.MIB,
        max_disk_bytes=limits["maxDiskMiB"] * collector.MIB, disk_roots=roots)
    with transport.installed():
        result = collector.collect(recipe, args.inbox_dir, args.supervisor_state, prepare_only=args.prepare_only)
    result.update(transportRequests=transport.requests, transportNetworkBytesRead=transport.network_bytes,
                  recoveryHelperSha256=collector.public.file_sha(Path(__file__)))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
