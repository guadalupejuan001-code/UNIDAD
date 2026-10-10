"""Small authenticated HTTP adapter bound to one trusted BodySession.

The adapter never selects a body, memory scope, model, or tool from request data.
It dispatches strict JSON to the session created by the hosting application.
"""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import socket

from .protocol import BrainError, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, decode_json, validate_json


MIN_TOKEN_LENGTH = 32
MAX_TOKEN_LENGTH = 4096
READ_TIMEOUT_SECONDS = 5


def validate_token(token):
    """Require a configured opaque ASCII credential without displaying it."""
    if not isinstance(token, str) or not MIN_TOKEN_LENGTH <= len(token) <= MAX_TOKEN_LENGTH:
        raise ValueError("Configure an API token with at least 32 ASCII characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in token):
        raise ValueError("API token must contain printable ASCII characters without spaces")
    return token.encode("ascii")


def error_envelope(body_id, code, message):
    return {"schemaVersion": 1, "bodyId": body_id, "requestId": None,
            "operation": None, "status": "error", "workerCapabilityValidated": False,
            "error": {"code": code, "message": message}}


def response_status(payload):
    """Map domain failures to HTTP codes without inventing a successful result."""
    if payload.get("status") == "ok":
        return 200
    if payload.get("status") != "error":
        return 500
    error = payload.get("error", {})
    code = error.get("code") if isinstance(error, dict) else None
    if code in {"invalid_json", "invalid_request", "invalid_input", "reference_invalid", "unsupported_operation"}:
        return 400
    if code == "tool_denied":
        return 403
    if code in {"model_unavailable", "capability_unavailable"}:
        return 503
    return 422


class _BoundServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # The standard handler prints exception details. A host may add metrics,
        # but private request bodies, URLs and credentials never enter these logs.
        pass


def create_server(session, token, host="127.0.0.1", port=0):
    """Create a server; the caller owns serve_forever/shutdown/server_close.

    All routes, including health, require Bearer authentication. The token and
    fixed session are held by this server's handler closure, not a global map.
    """
    credential = validate_token(token)
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Configure a nonempty HTTP bind host")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("HTTP port must be an integer from 0 to 65535")
    body_id = session.body_id
    if not isinstance(body_id, str) or not body_id:
        raise ValueError("HTTP server requires a configured body session")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "Unidad"
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(READ_TIMEOUT_SECONDS)

        def log_message(self, format, *args):
            pass

        def _reply(self, status, payload, *, authenticate=False):
            try:
                validate_json(payload, max_bytes=MAX_RESPONSE_BYTES)
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (ValueError, TypeError, UnicodeError, RecursionError):
                status = 500
                encoded = json.dumps(error_envelope(body_id, "invalid_response", "The configured session returned an invalid JSON response")).encode("utf-8")
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            if authenticate:
                self.send_header("WWW-Authenticate", "Bearer")
            if status == 405:
                self.send_header("Allow", "GET, POST")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    pass

        def _error(self, status, code, message, *, authenticate=False):
            self._reply(status, error_envelope(body_id, code, message), authenticate=authenticate)

        def _authorized(self):
            headers = self.headers.get_all("Authorization", [])
            candidate = b""
            if len(headers) == 1 and headers[0].startswith("Bearer "):
                try:
                    candidate = headers[0][7:].encode("ascii")
                except UnicodeEncodeError:
                    pass
            matched = hmac.compare_digest(candidate, credential)
            if not matched:
                self._error(401, "unauthorized", "A valid Bearer credential is required", authenticate=True)
            return matched

        def send_error(self, code, message=None, explain=None):
            # Sanitize errors raised by the base HTTP parser too: it otherwise
            # reflects fragments of the URL/request line in an HTML response.
            if code == 501 and hasattr(self, "headers"):
                self._unsupported()
            else:
                self._error(code, "http_error", "Invalid HTTP request")

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/health":
                self._reply(200, {"schemaVersion": 1, "bodyId": body_id, "status": "ok", "health": "ready"})
            elif self.path == "/v1/capabilities":
                try:
                    self._reply(200, session.capabilities())
                except Exception:
                    self._error(503, "capability_unavailable", "Configured session capabilities are unavailable")
            else:
                self._error(404, "not_found", "Unknown endpoint")

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/v1/requests":
                self._error(404, "not_found", "Unknown endpoint")
                return
            if self.headers.get_all("Transfer-Encoding", []):
                self._error(400, "unsupported_transfer_encoding", "Use an explicit Content-Length; transfer encoding is unsupported")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self._error(411, "length_required", "Content-Length is required")
                return
            if len(lengths) != 1 or len(lengths[0]) > 10 or re.fullmatch(r"[0-9]+", lengths[0]) is None:
                self._error(400, "invalid_content_length", "Use one bounded decimal Content-Length")
                return
            length = int(lengths[0])
            if length > MAX_REQUEST_BYTES:
                self._error(413, "payload_too_large", "Request exceeds the configured JSON byte limit")
                return
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or self.headers.get_content_type() != "application/json":
                self._error(415, "unsupported_media_type", "Use application/json with UTF-8 JSON")
                return
            charset = self.headers.get_content_charset()
            if charset is not None and charset.lower() not in {"utf-8", "utf8"}:
                self._error(415, "unsupported_media_type", "Request JSON must use UTF-8")
                return
            if any(value.lower() != "identity" for value in self.headers.get_all("Content-Encoding", [])):
                self._error(415, "unsupported_content_encoding", "Content encoding is unsupported")
                return
            try:
                raw = self.rfile.read(length)
            except (socket.timeout, TimeoutError):
                self._error(408, "request_timeout", "Request body was not received within the read timeout")
                return
            if len(raw) != length:
                self._error(400, "incomplete_body", "Request body does not match Content-Length")
                return
            try:
                payload = decode_json(raw, max_bytes=MAX_REQUEST_BYTES)
            except (BrainError, ValueError, UnicodeError, RecursionError):
                self._error(400, "invalid_json", "Request body must be strict bounded UTF-8 JSON")
                return
            if not isinstance(payload, dict):
                self._error(400, "invalid_request", "Request envelope must be a JSON object")
                return
            try:
                response = session.request(payload)
            except Exception:
                self._error(500, "operation_failed", "Request failed after dispatch; inspect task history before retrying")
                return
            if not isinstance(response, dict) or response.get("status") not in {"ok", "error"} or response.get("bodyId") != body_id:
                self._error(500, "invalid_response", "Configured session returned an invalid response envelope")
                return
            self._reply(response_status(response), response)

        def _unsupported(self):
            if self._authorized():
                self._error(405, "method_not_allowed", "Use a supported endpoint and method")

        do_HEAD = _unsupported
        do_PUT = _unsupported
        do_DELETE = _unsupported
        do_PATCH = _unsupported
        do_OPTIONS = _unsupported
        do_TRACE = _unsupported
        do_CONNECT = _unsupported

    return _BoundServer((host, port), Handler)
