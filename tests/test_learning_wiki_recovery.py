"""Declared HTTP fixtures; these tests do not claim a live download."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collect_learning_wiki_recovery", ROOT / "examples/collect_learning_wiki_recovery.py")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)
COLLECTOR = RECOVERY.collector


class Response:
    status = 200

    def __init__(self, raw, revision=COLLECTOR.PIN["revision"]):
        self.raw = raw
        self.headers = {"x-revision": revision, "Content-Type": "application/json", "ETag": "fixture"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, maximum):
        return self.raw[:maximum]


def raw_page(order=0, *, missing_last=False):
    offset, length, _url = RECOVERY.page_spec(order)
    rows = [{"row_idx": index, "row": {"id": str(index), "text": "Texto español de fixture " + str(index)}}
            for index in range(offset, offset + length - int(missing_last))]
    return json.dumps({"partial": False, "num_rows_total": COLLECTOR.public.WIKI_ROWS, "rows": rows}).encode()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.transport = RECOVERY.RetryTransport(self.root / "receipts", max_attempts=2)
        self.cache = self.root / "cache"

    def test_retry_retains_actual_failure_and_success_receipts_and_test_text_is_not_decoded(self):
        _offset, _length, url = RECOVERY.page_spec(0)
        error = HTTPError(url, 502, "declared fixture", {"x-revision": "proxy"}, io.BytesIO(b"Bad Gateway"))
        raw = raw_page()
        with patch.object(COLLECTOR, "urlopen", side_effect=[error, Response(raw)]), patch.object(RECOVERY.time, "sleep"):
            with self.transport.installed():
                result = COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30)
        self.assertEqual(self.transport.requests, 2)
        self.assertEqual(self.transport.network_bytes, len(raw) + len(b"Bad Gateway"))
        receipts = [COLLECTOR.read(file) for file in sorted((self.root / "receipts").glob("*.json"))]
        self.assertEqual([r["status"] for r in receipts], [502, 200])
        self.assertEqual(receipts[0]["responseBodySha256"], COLLECTOR.public.sha(b"Bad Gateway"))
        self.assertEqual(receipts[1]["responseBodySha256"], COLLECTOR.public.sha(raw))
        self.assertFalse(result[2]["transportRecovery"]["derivedAggregate"])
        for wrapped in result[3]["rows"]:
            split = COLLECTOR.public.assign_split("wikipedia:es:" + wrapped["row"]["id"])
            self.assertEqual(wrapped["row"]["text"], "" if split == "test" else "Texto español de fixture " + wrapped["row"]["id"])
        self.assertEqual((self.cache / "page-000000.json").read_bytes(), raw)
        with patch.object(COLLECTOR, "urlopen") as opened, self.transport.installed():
            restored = COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30, offline=True)
            opened.assert_not_called()
        self.assertTrue(restored[2]["_cachedLocally"])
        self.assertEqual(self.transport.requests, 2)

    def test_exhausted_transient_failure_never_creates_success_cache_or_cursor(self):
        _offset, _length, url = RECOVERY.page_spec(0)
        errors = [HTTPError(url, 502, "fixture", {}, io.BytesIO(b"fail")) for _ in range(2)]
        with patch.object(COLLECTOR, "urlopen", side_effect=errors), patch.object(RECOVERY.time, "sleep"), self.transport.installed():
            with self.assertRaises(HTTPError):
                COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30)
        self.assertFalse((self.cache / "page-000000.json").exists())
        self.assertFalse((self.cache / "page-000000.receipt.json").exists())
        self.assertFalse((self.root / "state.json").exists())
        self.assertEqual(len(list((self.root / "receipts").glob("*.json"))), 2)

    def test_wrong_revision_is_logged_as_real_200_and_never_cached_or_retried(self):
        with patch.object(COLLECTOR, "urlopen", return_value=Response(raw_page(), revision="wrong")) as opened, self.transport.installed():
            with self.assertRaisesRegex(ValueError, "coverage, URL, status or revision"):
                COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30)
            self.assertEqual(opened.call_count, 1)
        receipt = COLLECTOR.read(next((self.root / "receipts").glob("*.json")))
        self.assertEqual(receipt["status"], 200)
        self.assertEqual(receipt["xRevision"], "wrong")
        self.assertFalse(receipt["retryable"])
        self.assertFalse((self.cache / "page-000000.json").exists())

    def test_missing_row_rejects_page_and_preserves_raw_body_hash(self):
        raw = raw_page(missing_last=True)
        with patch.object(COLLECTOR, "urlopen", return_value=Response(raw)), self.transport.installed():
            with self.assertRaises(ValueError):
                COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30)
        receipt = COLLECTOR.read(next((self.root / "receipts").glob("*.json")))
        self.assertEqual(receipt["responseBodySha256"], COLLECTOR.public.sha(raw))
        self.assertFalse((self.cache / "page-000000.json").exists())

    def test_exact_final_request_allowed_and_next_request_and_small_byte_budget_denied_before_io(self):
        transport = RECOVERY.RetryTransport(self.root / "one", max_requests=1)
        with patch.object(COLLECTOR, "urlopen", return_value=Response(raw_page())) as opened, transport.installed():
            COLLECTOR.cached_page(self.cache, 0, 100000, time.monotonic() + 30)
            with self.assertRaisesRegex(ValueError, "request cap"):
                COLLECTOR.cached_page(self.cache, 1, 100000, time.monotonic() + 30)
            self.assertEqual(opened.call_count, 1)
        low = RECOVERY.RetryTransport(self.root / "low", max_network_bytes=100)
        with patch.object(COLLECTOR, "urlopen") as opened, low.installed():
            with self.assertRaisesRegex(ValueError, "network byte cap"):
                COLLECTOR.cached_page(self.root / "cache-low", 0, 100000, time.monotonic() + 30)
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
