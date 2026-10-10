"""Small, bounded JSON protocol shared by local and transport adapters."""

from copy import deepcopy
import json
import math
import re
import uuid

SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 262_144
MAX_RESPONSE_BYTES = 1_048_576
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 20_000
MAX_WORKFLOW_STEPS = 16


class BrainError(ValueError):
    def __init__(self, code, message, *, result=None):
        self.code = code if type(code) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "operation_failed"
        self.message = (message if type(message) is str else "Operation failed")[:512].encode("utf-8", errors="replace").decode("utf-8")
        super().__init__(self.message)
        self.result = result


def validate_json(value, *, max_bytes=MAX_REQUEST_BYTES, max_depth=MAX_JSON_DEPTH):
    """Accept native JSON types only, bounded before serializing the result."""
    nodes, approximate_bytes = 0, 0
    active = set()

    def walk(item, depth):
        nonlocal nodes, approximate_bytes
        nodes += 1
        approximate_bytes += 4
        if nodes > MAX_JSON_NODES or depth > max_depth:
            raise BrainError("invalid_json", "JSON exceeds depth or element limits")
        kind = type(item)
        if kind is str:
            try:
                approximate_bytes += len(item.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise BrainError("invalid_json", "JSON text must be valid UTF-8") from error
        elif kind is int:
            if not -(2**63) <= item <= 2**63 - 1:
                raise BrainError("invalid_json", "JSON integer exceeds supported range")
            approximate_bytes += len(str(item))
        elif kind is float:
            if not math.isfinite(item):
                raise BrainError("invalid_json", "JSON numbers must be finite")
            approximate_bytes += 24
        elif kind in (bool, type(None)):
            pass
        elif kind in (dict, list):
            identity = id(item)
            if identity in active:
                raise BrainError("invalid_json", "JSON cannot contain cycles")
            active.add(identity)
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise BrainError("invalid_json", "JSON object keys must be strings")
                    walk(key, depth + 1)
                    walk(child, depth + 1)
            else:
                for child in item:
                    walk(child, depth + 1)
            active.remove(identity)
        else:
            raise BrainError("invalid_json", "Value is not a native JSON type")
        if approximate_bytes > max_bytes:
            raise BrainError("invalid_json", "JSON exceeds byte limit")

    walk(value, 0)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise BrainError("invalid_json", "JSON exceeds byte limit")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BrainError("invalid_json", "Duplicate JSON object keys")
        result[key] = value
    return result


def _bad_constant(_value):
    raise BrainError("invalid_json", "JSON numbers must be finite")


def decode_json(raw, *, max_bytes=MAX_REQUEST_BYTES):
    if type(raw) not in (str, bytes):
        raise BrainError("invalid_json", "JSON input must be text or bytes")
    try:
        encoded = raw.encode("utf-8") if type(raw) is str else raw
        if len(encoded) > max_bytes:
            raise BrainError("invalid_json", "JSON exceeds byte limit")
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    except BrainError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise BrainError("invalid_json", "Cannot decode bounded UTF-8 JSON") from error
    return validate_json(value, max_bytes=max_bytes)


def identifier(value, name):
    if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value) is None:
        raise BrainError("invalid_input", f"{name} must be a bounded identifier")
    return value


def request_identity(payload):
    request_id = payload.get("requestId") if type(payload) is dict else None
    try:
        valid_id = type(request_id) is str and 0 < len(request_id) <= 128 and len(request_id.encode("utf-8")) <= 128
    except UnicodeEncodeError:
        valid_id = False
    if not valid_id:
        request_id = uuid.uuid4().hex
    operation = payload.get("operation") if type(payload) is dict else None
    try:
        valid_operation = type(operation) is str and len(operation) <= 64 and bool(operation.encode("utf-8"))
    except UnicodeEncodeError:
        valid_operation = False
    if not valid_operation:
        operation = "unknown"
    return request_id, operation


def validate_request(payload):
    validate_json(payload)
    if type(payload) is not dict or not {"schemaVersion", "operation", "input"} <= payload.keys() or not payload.keys() <= {
            "schemaVersion", "requestId", "operation", "input", "context"}:
        raise BrainError("invalid_request", "Request has unsupported or missing fields")
    if type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != SCHEMA_VERSION:
        raise BrainError("invalid_request", "Unsupported request schema")
    identifier(payload["operation"], "operation")
    if type(payload["input"]) is not dict or type(payload.get("context", {})) is not dict:
        raise BrainError("invalid_request", "Request input and context must be objects")
    if "requestId" in payload and (type(payload["requestId"]) is not str or not payload["requestId"] or
                                    len(payload["requestId"].encode("utf-8")) > 128):
        raise BrainError("invalid_request", "Invalid request identity")
    result = deepcopy(payload)
    result.setdefault("requestId", uuid.uuid4().hex)
    result.setdefault("context", {})
    return result


def envelope(request_id, body_id, operation, *, result=None, error=None):
    response = {"schemaVersion": SCHEMA_VERSION, "requestId": request_id, "bodyId": body_id,
                "operation": operation, "status": "error" if error else "ok", "workerCapabilityValidated": False}
    if result is not None:
        response["result"] = result
    if error is not None:
        response["error"] = {"code": error.code, "message": error.message}
    validate_json(response, max_bytes=MAX_RESPONSE_BYTES)
    return response
