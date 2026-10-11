"""Portable brain services with explicit operations and product-bound tools."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Callable

from .. import workspace
from .models import ModelRuntime, ModelUnavailableError
from .protocol import (BrainError, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_WORKFLOW_STEPS,
                       envelope, identifier, request_identity, validate_json, validate_request)
from .workflow import has_references, is_reference, run_plan, validate_plan

MODEL_OPERATIONS = {"language.generate", "attention.forecast", "conversation.reply"}
MEMORY_OPERATIONS = {"memory.task.add", "memory.task.list", "memory.task.show", "memory.task.attempt",
                     "memory.task.feedback", "memory.task.status"}
OPERATIONS = MODEL_OPERATIONS | MEMORY_OPERATIONS | {"tool.call", "workflow.run"}


def _label(value, name):
    if type(value) is not str or not value.strip():
        raise BrainError("invalid_input", f"{name} must be explicit text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise BrainError("invalid_input", f"{name} must be valid UTF-8") from error
    if size > 128:
        raise BrainError("invalid_input", f"{name} exceeds 128 UTF-8 bytes")
    return value


@dataclass(frozen=True)
class Tool:
    name: str
    handler: Callable[[dict], object]
    description: str = ""
    validator: Callable[[dict], object] | None = None

    def __post_init__(self):
        identifier(self.name, "tool name")
        if not callable(self.handler) or (self.validator is not None and not callable(self.validator)):
            raise BrainError("invalid_input", "Tool handlers and optional validators must be callable")
        if type(self.description) is not str or len(self.description.encode("utf-8")) > 4096:
            raise BrainError("invalid_input", "Tool description must be bounded text")


class Brain:
    def __init__(self, state_dir, language_model=None, attention_model=None, conversation_model=None):
        if not isinstance(state_dir, (str, Path)):
            raise BrainError("invalid_input", "Brain state directory must be configured by the host")
        self._state_dir = Path(state_dir).expanduser().absolute()
        # One model runtime for all sessions. No per-body weight loading or chat cache.
        self.models = ModelRuntime(language_model=language_model, attention_model=attention_model,
                                   conversation_model=conversation_model)

    def connect(self, body_id, tools=(), memory_scope=None):
        _label(body_id, "body id")
        if memory_scope is not None:
            _label(memory_scope, "memory scope")
        try:
            bound = list(tools)
        except TypeError as error:
            raise BrainError("invalid_input", "Tools must be an iterable of Tool bindings") from error
        if any(not isinstance(tool, Tool) for tool in bound) or len({tool.name for tool in bound}) != len(bound):
            raise BrainError("invalid_input", "Tool bindings must be Tool instances with unique names")
        namespace = "body\0" + body_id if memory_scope is None else "shared\0" + memory_scope
        directory = self._state_dir / "memory" / hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        workspace.init_workspace(directory)
        return BodySession(self, body_id, {tool.name: tool for tool in bound}, directory, memory_scope is not None)

    def capabilities(self):
        model_capabilities = self.models.capabilities()
        available = (OPERATIONS - MODEL_OPERATIONS) | set(model_capabilities["operations"])
        result = {"schemaVersion": 1, "operations": sorted(available), "models": model_capabilities,
                  "workerCapabilityValidated": False, "instructionFollowingValidated": False,
                  "naturalLanguagePlanner": False, "automaticLearning": False,
                  "memory": {"privacy": "PRIVATE", "isolatedByBody": True, "sharingRequiresHostConfiguration": True},
                  "tools": "explicit bindings on each body session",
                  "workflow": {"maxSteps": MAX_WORKFLOW_STEPS, "planning": "explicit application-provided steps",
                               "referenceTraversal": "JSON dictionaries and lists only", "rollback": False},
                  "requestContext": "request-local metadata; model prompts must be explicit input",
                  "limits": {"requestBytes": MAX_REQUEST_BYTES, "responseBytes": MAX_RESPONSE_BYTES}}
        return validate_json(result, max_bytes=MAX_RESPONSE_BYTES)


class BodySession:
    def __init__(self, brain, body_id, tools, memory_directory, shared_memory):
        self.brain = brain
        self.body_id = body_id
        self._tools = tools
        self._memory_directory = memory_directory
        self._shared_memory = shared_memory

    @property
    def models(self):
        return self.brain.models

    def capabilities(self):
        result = self.brain.capabilities()
        result["bodyId"] = self.body_id
        result["tools"] = [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]
        result["memory"]["sharedScopeConfigured"] = self._shared_memory
        result["memory"]["isolatedByBody"] = not self._shared_memory
        return result

    def request(self, payload):
        request_id, operation = request_identity(payload)
        try:
            request = validate_request(payload)
            request_id, operation = request["requestId"], request["operation"]
            if operation == "workflow.run":
                steps = validate_plan(request["input"], self._validate_operation)
                result = run_plan(steps, self._execute)
            else:
                if has_references(request["input"]):
                    raise BrainError("reference_invalid", "References are available only inside an explicit workflow")
                result = self._execute(operation, request["input"])
            return envelope(request_id, self.body_id, operation, result=result)
        except BrainError as error:
            try:
                return envelope(request_id, self.body_id, operation, result=error.result, error=error)
            except BrainError:
                if error.code == "workflow_failed" and type(error.result) is dict:
                    steps = error.result.get("steps", [])
                    summary = {"status": "failed", "steps": [{"id": step["id"], "operation": step["operation"],
                                                                "resultOmitted": True} for step in steps],
                               "completedCount": len(steps), "effectsRolledBack": False,
                               "failedStep": error.result.get("failedStep")}
                    return envelope(request_id, self.body_id, operation, result=summary, error=error)
                return envelope(request_id, self.body_id, operation,
                                error=BrainError("response_too_large", "Result exceeds the bounded response protocol"))
        except Exception:
            return envelope(request_id, self.body_id, operation,
                            error=BrainError("operation_failed", "Operation failed; no automatic rollback was performed"))

    def _tool_validator(self, tool, arguments):
        if tool.validator is not None:
            try:
                if tool.validator(arguments) is False:
                    raise BrainError("invalid_input", "Bound tool rejected its arguments")
            except BrainError:
                raise
            except Exception as error:
                raise BrainError("invalid_input", "Bound tool rejected its arguments") from error

    def _validate_operation(self, operation, input_data, *, allow_refs=False):
        if type(operation) is not str or operation not in OPERATIONS or operation == "workflow.run":
            raise BrainError("unsupported_operation", "Operation is not an available explicit brain service")
        if type(input_data) is not dict:
            raise BrainError("invalid_input", "Operation input must be an object")
        if operation in MODEL_OPERATIONS:
            try:
                self.models.validate_input(operation, input_data, allow_refs=allow_refs)
            except ModelUnavailableError as error:
                raise BrainError("capability_unavailable", "A local conversation model was not configured by the host") from error
            except BrainError:
                raise
            except (ValueError, TypeError) as error:
                raise BrainError("invalid_input", "Invalid or unavailable model operation input") from error
            return
        if operation == "tool.call":
            if set(input_data) != {"name", "arguments"} or type(input_data["name"]) is not str:
                raise BrainError("invalid_input", "Tool calls require a literal name and arguments object")
            tool = self._tools.get(input_data["name"])
            if tool is None:
                raise BrainError("tool_denied", "Tool is not bound to this body")
            if type(input_data["arguments"]) is not dict:
                raise BrainError("invalid_input", "Tool arguments must be an object")
            if not (allow_refs and has_references(input_data["arguments"])):
                self._tool_validator(tool, input_data["arguments"])
            return
        self._validate_memory(operation, input_data, allow_refs)

    def _validate_memory(self, operation, values, allow_refs):
        shapes = {
            "memory.task.add": ({"goal"}, {"context"}),
            "memory.task.list": (set(), set()), "memory.task.status": (set(), set()),
            "memory.task.show": ({"task_id"}, set()),
            "memory.task.attempt": ({"task_id", "text"}, {"evidence"}),
            "memory.task.feedback": ({"task_id", "attempt_id", "text", "accepted"}, set()),
        }
        required, optional = shapes[operation]
        if not required <= values.keys() or not values.keys() <= required | optional:
            raise BrainError("invalid_input", "Memory operation has unsupported or missing fields")
        for key, value in values.items():
            if allow_refs and is_reference(value):
                continue
            if key in ("task_id", "attempt_id"):
                prefix = "task" if key == "task_id" else "attempt"
                if type(value) is not str or re.fullmatch(prefix + r"_[0-9a-f]{32}", value) is None:
                    raise BrainError("invalid_input", "Invalid memory record identity")
            elif key == "accepted":
                if type(value) is not bool:
                    raise BrainError("invalid_input", "Memory feedback needs an explicit accepted boolean")
            elif key == "evidence":
                if type(value) is not list or len(value) > workspace.MAX_EVIDENCE_REFERENCES:
                    raise BrainError("invalid_input", "Evidence must be a bounded list of references")
                for reference in value:
                    if allow_refs and is_reference(reference):
                        continue
                    try:
                        workspace._text(reference, "evidence reference", limit=workspace.MAX_EVIDENCE_BYTES)
                    except ValueError as error:
                        raise BrainError("invalid_input", "Invalid memory evidence reference") from error
            else:
                try:
                    workspace._text(value, key, allow_empty=key == "context")
                except ValueError as error:
                    raise BrainError("invalid_input", "Memory text is invalid or too large") from error

    def _execute(self, operation, input_data):
        # Revalidate resolved references before each effect; their value/path types
        # are available only at runtime, unlike bindings and graph structure.
        self._validate_operation(operation, input_data)
        try:
            if operation in MODEL_OPERATIONS:
                result = self.models.execute(operation, input_data)
            elif operation == "tool.call":
                result = self._tools[input_data["name"]].handler(input_data["arguments"])
            elif operation == "memory.task.add":
                result = workspace.add_task(self._memory_directory, input_data["goal"], input_data.get("context", ""))
            elif operation == "memory.task.list":
                result = workspace.list_tasks(self._memory_directory)
            elif operation == "memory.task.show":
                result = workspace.show(self._memory_directory, input_data["task_id"])
            elif operation == "memory.task.attempt":
                result = workspace.record_attempt(self._memory_directory, input_data["task_id"], input_data["text"], input_data.get("evidence"))
            elif operation == "memory.task.feedback":
                result = workspace.feedback(self._memory_directory, input_data["task_id"], input_data["attempt_id"],
                                            input_data["text"], accepted=input_data["accepted"])
            else:
                result = workspace.status(self._memory_directory)
            return validate_json(result, max_bytes=MAX_RESPONSE_BYTES)
        except BrainError:
            raise
        except Exception as error:
            raise BrainError("operation_failed", "Explicit operation failed; completed effects were not rolled back") from error
