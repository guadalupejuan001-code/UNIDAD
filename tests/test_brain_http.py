"""Authenticated localhost transport contracts and actual shared-brain calls."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr
import http.client
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from ia_propia.brain.http import create_server
from ia_propia.brain.protocol import MAX_JSON_DEPTH, MAX_REQUEST_BYTES


TOKEN = "synthetic-local-test-token-00000000000000"


class TransportSession:
    body_id = "synthetic-product"

    def __init__(self):
        self.calls = []
        self.capability_calls = 0
        self.lock = threading.Lock()

    def capabilities(self):
        self.capability_calls += 1
        return {"schemaVersion": 1, "bodyId": self.body_id, "operations": ["fixture.echo"]}

    def request(self, payload):
        with self.lock:
            self.calls.append(payload)
        if payload.get("operation") == "fixture.explode":
            raise RuntimeError("PRIVATE_EXCEPTION_MARKER")
        if payload.get("operation") == "fixture.nonfinite":
            return {"status": "ok", "result": float("nan")}
        if payload.get("operation") == "fixture.failed":
            return {"schemaVersion": 1, "bodyId": self.body_id, "status": "error",
                    "error": {"code": "operation_failed", "message": "Synthetic operation failure"}}
        return {"schemaVersion": 1, "bodyId": self.body_id, "status": "ok", "result": payload.get("input", {})}


@contextmanager
def running_server(session, token=TOKEN):
    server = create_server(session, token, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise AssertionError("Local HTTP server did not stop")


def exchange(server, method="POST", path="/v1/requests", payload=None, headers=None, *, timeout=3):
    body = b"" if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    request_headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    if headers is not None:
        request_headers.update(headers)
        request_headers = {name: value for name, value in request_headers.items() if value is not None}
    connection = http.client.HTTPConnection(*server.server_address, timeout=timeout)
    try:
        connection.request(method, path, body=body if method == "POST" else None, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
        return response.status, dict(response.getheaders()), json.loads(raw) if raw else None
    finally:
        connection.close()


def raw_exchange(server, request):
    with socket.create_connection(server.server_address, timeout=3) as connection:
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        response = bytearray()
        while chunk := connection.recv(65536):
            response.extend(chunk)
    headers, body = bytes(response).split(b"\r\n\r\n", 1)
    return int(headers.split(b" ", 2)[1]), json.loads(body) if body else None


class BrainHttpTests(unittest.TestCase):
    def test_every_route_requires_auth_without_dispatch_or_capability_reads(self):
        session = TransportSession()
        with running_server(session) as server:
            for method, path in (("GET", "/health"), ("GET", "/v1/capabilities"), ("POST", "/v1/requests"),
                                 ("GET", "/unknown"), ("OPTIONS", "/v1/requests")):
                for authorization in (None, "Bearer incorrect", "Basic synthetic", "Bearer "):
                    with self.subTest(method=method, authorization=authorization):
                        status, headers, result = exchange(server, method, path, headers={"Authorization": authorization})
                        self.assertEqual(status, 401)
                        self.assertEqual(result["error"]["code"], "unauthorized")
                        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
            self.assertEqual(session.calls, [])
            self.assertEqual(session.capability_calls, 0)

    def test_authenticated_health_capabilities_and_fixed_session(self):
        session = TransportSession()
        with running_server(session) as server:
            for path in ("/health", "/v1/capabilities"):
                status, headers, result = exchange(server, "GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(result["bodyId"], session.body_id)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn("Access-Control-Allow-Origin", headers)
            status, _, result = exchange(server, payload={"schemaVersion": 1, "operation": "fixture.echo",
                                                          "bodyId": "another-product", "input": {"goal": "Synthetic work"}})
            self.assertEqual(status, 200)
            self.assertEqual(result["bodyId"], session.body_id)
            self.assertEqual(len(session.calls), 1)

    def test_independent_servers_never_share_session_or_token(self):
        first, second = TransportSession(), TransportSession()
        second.body_id = "other-product"
        other_token = "another-synthetic-local-token-0000000000"
        with running_server(first) as a, running_server(second, token=other_token) as b:
            self.assertEqual(exchange(a, "GET", "/health")[2]["bodyId"], first.body_id)
            self.assertEqual(exchange(b, "GET", "/health")[0], 401)
            status, _, result = exchange(b, "GET", "/health", headers={"Authorization": f"Bearer {other_token}"})
            self.assertEqual(status, 200)
            self.assertEqual(result["bodyId"], second.body_id)

    def test_invalid_json_is_rejected_before_any_side_effect(self):
        session = TransportSession()
        variants = [b'{"input":{},"input":{"secret":"PRIVATE_MARKER"}}', b'{"x":NaN}', b'{"x":Infinity}',
                    b'{"x":1e999}', b'\xff', b'{"unclosed":', b'[]', b'null', b'"string"',
                    json.dumps({"nested": [[[]]]}).encode()]
        nested = "[" * (MAX_JSON_DEPTH + 2) + "0" + "]" * (MAX_JSON_DEPTH + 2)
        variants[-1] = ('{"input":' + nested + '}').encode()
        with running_server(session) as server:
            for body in variants:
                with self.subTest(body=body[:50]):
                    status, _, result = exchange(server, payload=body)
                    self.assertEqual(status, 400)
                    self.assertIn(result["error"]["code"], {"invalid_json", "invalid_request"})
                    self.assertNotIn("PRIVATE_MARKER", json.dumps(result))
            self.assertEqual(session.calls, [])

    def test_content_length_and_transfer_encoding_are_bounded_and_unambiguous(self):
        session = TransportSession()
        common = f"POST /v1/requests HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n"
        cases = [("", b"", 411, "length_required"),
                 ("Content-Length: -1\r\n", b"", 400, "invalid_content_length"),
                 ("Content-Length: 9999999999999999999\r\n", b"", 400, "invalid_content_length"),
                 ("Content-Length: 2\r\nContent-Length: 2\r\n", b"{}", 400, "invalid_content_length"),
                 ("Content-Length: 2\r\nTransfer-Encoding: chunked\r\n", b"{}", 400, "unsupported_transfer_encoding"),
                 (f"Content-Length: {MAX_REQUEST_BYTES + 1}\r\n", b"", 413, "payload_too_large"),
                 ("Content-Length: 20\r\n", b"{}", 400, "incomplete_body")]
        with running_server(session) as server:
            for fields, body, expected_status, code in cases:
                status, result = raw_exchange(server, (common + fields + "\r\n").encode() + body)
                self.assertEqual(status, expected_status)
                self.assertEqual(result["error"]["code"], code)
            self.assertEqual(session.calls, [])

    def test_duplicate_credentials_unsupported_media_and_routes_never_dispatch(self):
        session = TransportSession()
        with running_server(session) as server:
            duplicate = f"GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
            self.assertEqual(raw_exchange(server, duplicate.encode())[0], 401)
            for headers in ({"Content-Type": None}, {"Content-Type": "text/plain"},
                            {"Content-Type": "application/json; charset=utf-16"}, {"Content-Encoding": "gzip"}):
                self.assertEqual(exchange(server, payload=b"{}", headers=headers)[0], 415)
            for path in ("/v1/requests?body=other", "/v1/requests/../../private", "/v1/tools/register"):
                self.assertEqual(exchange(server, path=path, payload=b"{}")[0], 404)
            self.assertEqual(exchange(server, "DELETE", "/v1/requests")[0], 405)
            self.assertEqual(session.calls, [])

    def test_dispatch_failure_and_nonfinite_response_are_sanitized_without_logs(self):
        session = TransportSession()
        logs = io.StringIO()
        with redirect_stderr(logs), running_server(session) as server:
            for operation, expected in (("fixture.explode", 500), ("fixture.nonfinite", 500), ("fixture.failed", 422)):
                status, _, result = exchange(server, payload={"operation": operation, "input": {"secret": "PRIVATE_PAYLOAD_MARKER"}})
                self.assertEqual(status, expected)
                self.assertEqual(result["status"], "error")
                self.assertNotIn("PRIVATE_EXCEPTION_MARKER", json.dumps(result))
        for marker in (TOKEN, "PRIVATE_EXCEPTION_MARKER", "PRIVATE_PAYLOAD_MARKER"):
            self.assertNotIn(marker, logs.getvalue())

    def test_parallel_requests_keep_their_results_bound_to_the_same_body(self):
        session = TransportSession()
        with running_server(session) as server, ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda i: exchange(server, payload={"operation": "fixture.echo", "input": {"number": i}}), range(12)))
        self.assertEqual(len(session.calls), 12)
        self.assertEqual({result[2]["result"]["number"] for result in results}, set(range(12)))
        self.assertTrue(all(result[2]["bodyId"] == session.body_id for result in results))

    def test_short_or_invalid_tokens_fail_before_opening_a_listener(self):
        for token in (None, "", "short-secret", "x" * 31, "x" * 32 + "\n", "ñ" * 40):
            with self.assertRaises(ValueError):
                create_server(TransportSession(), token)

    def test_actual_brain_operations_and_cross_body_memory_isolation(self):
        from ia_propia.brain import Brain
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "brain")
            web = brain.connect("web-product")
            other = brain.connect("other-product")
            with running_server(web) as server:
                request = {"schemaVersion": 1, "requestId": "synthetic-create", "operation": "memory.task.add",
                           "input": {"goal": "Synthetic HTTP task", "context": "Explicit fixture context"}}
                status, _, added = exchange(server, payload=request)
                self.assertEqual(status, 200, added)
                task = added["result"]
                listed = other.request({"schemaVersion": 1, "operation": "memory.task.list", "input": {}})
                self.assertEqual(listed["result"]["tasks"], [])
                request.update(operation="memory.task.show", input={"task_id": task["id"]}, bodyId="other-product")
                status, _, rejected = exchange(server, payload=request)
                self.assertEqual(status, 400)
                self.assertEqual(rejected["bodyId"], "web-product")
                status, _, generated = exchange(server, payload={"schemaVersion": 1, "operation": "language.generate",
                                                                "input": {"prompt": "Unidad", "length": 12, "seed": 4}})
                self.assertEqual(status, 200, generated)
                self.assertEqual(generated["bodyId"], "web-product")
                self.assertFalse(generated["workerCapabilityValidated"])


if __name__ == "__main__":
    unittest.main()
