"""Portable brain services with explicit operations and product-bound tools."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import threading
from typing import Callable

from .. import workspace
from ..memory_search import (DEFAULT_SEARCH_LIMIT, MAX_EXCERPT_CHARS, MAX_QUERY_BYTES,
                             MAX_QUERY_TERMS, MAX_SEARCH_LIMIT, SEARCH_METHOD, validate_search)
from .models import ModelRuntime, ModelUnavailableError
from .protocol import (BrainError, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_WORKFLOW_STEPS,
                       envelope, identifier, request_identity, validate_json, validate_request)
from .workflow import has_references, is_reference, run_plan, validate_plan

MODEL_OPERATIONS = {"language.generate", "language.score", "attention.forecast", "conversation.reply", "conversation.forecast"}
MEMORY_OPERATIONS = {"memory.task.add", "memory.task.list", "memory.task.search", "memory.task.show", "memory.task.attempt",
                     "memory.task.feedback", "memory.task.status"}
AUTONOMY_OPERATIONS = {"autonomy.start", "autonomy.status", "autonomy.step", "autonomy.run",
                       "autonomy.cancel", "autonomy.approve", "autonomy.resume"}
OPERATIONS = MODEL_OPERATIONS | MEMORY_OPERATIONS | AUTONOMY_OPERATIONS | {"tool.call", "workflow.run"}


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
    def __init__(self, state_dir, language_model=None, attention_model=None, conversation_model=None, *, language_backend="numpy",
                 conversation_decoder="torch", conversation_cache=True, conversation_models=None,
                 ensemble_budget=None, ensemble_primary=None):
        if not isinstance(state_dir, (str, Path)):
            raise BrainError("invalid_input", "Brain state directory must be configured by the host")
        self._state_dir = Path(state_dir).expanduser().absolute()
        # One model runtime for all sessions. No per-body weight loading or chat cache.
        self.models = ModelRuntime(language_model=language_model, attention_model=attention_model,
                                   conversation_model=conversation_model, language_backend=language_backend,
                                   conversation_decoder=conversation_decoder, conversation_cache=conversation_cache,
                                   conversation_models=conversation_models, ensemble_budget=ensemble_budget,
                                   ensemble_primary=ensemble_primary)
        self._autonomy_bindings = {}
        self._connection_lock = threading.Lock()

    def connect(self, body_id, tools=(), memory_scope=None, *, autonomy_tools=None,
                autonomy_proposer=None, autonomy_limits=None, autonomy_completion_verifier=None,
                autonomy_input_validator=None, autonomy_operator_token=None):
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
        autonomy = self._bind_autonomy(body_id, autonomy_tools, autonomy_proposer, autonomy_limits,
                                      autonomy_completion_verifier, autonomy_input_validator,
                                      autonomy_operator_token)
        workspace.init_workspace(directory)
        return BodySession(self, body_id, {tool.name: tool for tool in bound}, directory,
                           memory_scope is not None, autonomy)

    def _bind_autonomy(self, body_id, tools, proposer, limits, verifier, input_validator, operator_token):
        # This method is called only by the hosting application. Request input
        # cannot provide a registry, roots, model, proposer, or resource policy.
        if tools is None:
            if any(value is not None for value in (proposer, limits, verifier, input_validator, operator_token)):
                raise BrainError("invalid_input", "Autonomy requires explicit host tool bindings")
            return None
        from .autonomy import AutonomyController, AutonomyLimits, AutonomyTool
        from .autonomy_model import ModelActionProposer
        try:
            tools = tuple(tools)
        except TypeError as error:
            raise BrainError("invalid_input", "Autonomy tools must be explicit host bindings") from error
        if any(not isinstance(tool, AutonomyTool) for tool in tools) or len({tool.name for tool in tools}) != len(tools):
            raise BrainError("invalid_input", "Autonomy bindings must be unique AutonomyTool instances")
        if input_validator is not None and not callable(input_validator):
            raise BrainError("invalid_input", "Autonomy task input validator must be host callable")
        credential = None
        if operator_token is not None:
            from .http import validate_token
            credential = validate_token(operator_token)
        configured_limits = AutonomyLimits() if limits is None else limits
        validator_identity = None
        if input_validator is not None:
            validator_policy = getattr(input_validator, "policy_id", "")
            if type(validator_policy) is not str or len(validator_policy.encode("utf-8")) > 4096:
                raise BrainError("invalid_input", "Task input validator policy identity must be bounded host text")
            validator_identity = {"module": getattr(input_validator, "__module__", type(input_validator).__module__),
                                  "qualname": getattr(input_validator, "__qualname__", type(input_validator).__qualname__),
                                  "policy_id": validator_policy}
        host_policy_id = hashlib.sha256(json.dumps(validator_identity, ensure_ascii=False,
                                                   sort_keys=True).encode("utf-8")).hexdigest()
        signature = (tuple(id(tool) for tool in tools), id(proposer) if proposer is not None else None,
                     configured_limits, verifier, input_validator, credential)
        with self._connection_lock:
            existing = self._autonomy_bindings.get(body_id)
            if existing is not None:
                if existing[0] != signature:
                    raise BrainError("invalid_input", "Autonomy policy is already bound for this body")
                return existing[1]
            try:
                selected_proposer = ModelActionProposer(self.models) if proposer is None else proposer
            except ModelUnavailableError as error:
                raise BrainError("capability_unavailable", "Autonomy requires a host-selected local conversation model") from error
            namespace = "autonomy-body\0" + body_id
            state = self._state_dir / "autonomy" / hashlib.sha256(namespace.encode("utf-8")).hexdigest()
            controller = AutonomyController(state, selected_proposer, tools, limits=configured_limits,
                                            completion_verifier=verifier, policy_id=host_policy_id)
            binding = {"controller": controller, "tools": tools, "proposer": selected_proposer,
                       "limits": configured_limits, "input_validator": input_validator,
                       "operator_credential": credential}
            self._autonomy_bindings[body_id] = (signature, binding)
            return binding

    def capabilities(self):
        model_capabilities = self.models.capabilities()
        available = (OPERATIONS - MODEL_OPERATIONS - AUTONOMY_OPERATIONS) | set(model_capabilities["operations"])
        result = {"schemaVersion": 1, "operations": sorted(available), "models": model_capabilities,
                  "workerCapabilityValidated": False, "instructionFollowingValidated": False,
                  "naturalLanguagePlanner": False, "automaticLearning": False,
                  "memory": {"privacy": "PRIVATE", "isolatedByBody": True, "sharingRequiresHostConfiguration": True,
                             "retrieval": {"operation": "memory.task.search", "method": SEARCH_METHOD,
                                           "automaticContextInjection": False, "queryBytes": MAX_QUERY_BYTES,
                                           "queryTerms": MAX_QUERY_TERMS, "defaultResults": DEFAULT_SEARCH_LIMIT,
                                           "maxResults": MAX_SEARCH_LIMIT, "excerptCharacters": MAX_EXCERPT_CHARS}},
                  "tools": "explicit bindings on each body session",
                  "autonomy": {"enabled": False, "experimental": True,
                               "modelProposalValidated": False, "naturalLanguagePlanner": False,
                               "isolatedByBody": True, "configuration": "explicit host bindings only"},
                  "workflow": {"maxSteps": MAX_WORKFLOW_STEPS, "planning": "explicit application-provided steps",
                               "referenceTraversal": "JSON dictionaries and lists only", "rollback": False},
                  "requestContext": "request-local metadata; model prompts must be explicit input",
                  "limits": {"requestBytes": MAX_REQUEST_BYTES, "responseBytes": MAX_RESPONSE_BYTES}}
        return validate_json(result, max_bytes=MAX_RESPONSE_BYTES)


class BodySession:
    def __init__(self, brain, body_id, tools, memory_directory, shared_memory, autonomy=None):
        self.brain = brain
        self.body_id = body_id
        self._tools = tools
        self._memory_directory = memory_directory
        self._shared_memory = shared_memory
        self._autonomy = autonomy

    @property
    def models(self):
        return self.brain.models

    def capabilities(self):
        result = self.brain.capabilities()
        result["bodyId"] = self.body_id
        result["tools"] = [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]
        result["memory"]["sharedScopeConfigured"] = self._shared_memory
        result["memory"]["isolatedByBody"] = not self._shared_memory
        if self._autonomy is not None:
            binding = self._autonomy
            result["operations"] = sorted(set(result["operations"]) | AUTONOMY_OPERATIONS)
            limits = binding["limits"]
            result["autonomy"].update({"enabled": True, "source": binding["proposer"].origin,
                                       "proposerId": binding["proposer"].proposer_id,
                                       "operatorApprovalConfigured": binding["operator_credential"] is not None,
                                       "limits": {"maxSteps": limits.max_steps,
                                                  "maxFailures": limits.max_failures,
                                                  "maxArgumentRepairs": limits.max_argument_repairs,
                                                  "maxSeconds": limits.max_seconds},
                                       "tools": [tool.metadata() for tool in binding["tools"]]})
        return validate_json(result, max_bytes=MAX_RESPONSE_BYTES)

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
        if operation in AUTONOMY_OPERATIONS:
            if allow_refs:
                raise BrainError("invalid_input", "Autonomy is dispatched directly, outside explicit workflows")
            self._validate_autonomy(operation, input_data)
            return
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

    def _validate_autonomy(self, operation, values):
        if self._autonomy is None:
            raise BrainError("capability_unavailable", "Autonomy was not configured by the host for this body")
        required = {"goal"} if operation == "autonomy.start" else {"task_id"}
        optional = {"task_inputs"} if operation == "autonomy.start" else set()
        if operation == "autonomy.approve":
            required |= {"approval_id", "approved", "operator_token"}
        if not required <= values.keys() or not values.keys() <= required | optional:
            raise BrainError("invalid_input", "Autonomy operation has unsupported or missing fields")
        if operation == "autonomy.start":
            if type(values["goal"]) is not str or not values["goal"].strip():
                raise BrainError("invalid_input", "Autonomy requires an explicit goal")
            task_inputs = values.get("task_inputs", {})
            if type(task_inputs) is not dict:
                raise BrainError("invalid_input", "Autonomy task inputs must be an object")
            validator = self._autonomy["input_validator"]
            if validator is None:
                if task_inputs:
                    raise BrainError("invalid_input", "This host accepts no additional autonomy task inputs")
            else:
                try:
                    if validator(task_inputs) is False:
                        raise ValueError("Rejected task input")
                except BrainError:
                    raise
                except Exception as error:
                    raise BrainError("invalid_input", "Host rejected autonomy task inputs") from error
        else:
            identifier(values["task_id"], "autonomy task id")
        if operation == "autonomy.approve":
            if type(values["approval_id"]) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", values["approval_id"]) is None:
                raise BrainError("invalid_input", "Approval id must be the bounded nonce for the pending action")
            if type(values["approved"]) is not bool:
                raise BrainError("invalid_input", "Approval requires an explicit boolean")
            token = values["operator_token"]
            credential = self._autonomy["operator_credential"]
            try:
                candidate = token.encode("ascii") if type(token) is str else b""
            except UnicodeEncodeError:
                candidate = b""
            if credential is None or not hmac.compare_digest(candidate, credential):
                raise BrainError("tool_denied", "A separately configured operator credential is required")

    def _execute_autonomy(self, operation, values):
        controller = self._autonomy["controller"]
        if operation == "autonomy.start":
            result = controller.start(values["goal"], task_inputs=values.get("task_inputs", {}))
        elif operation == "autonomy.approve":
            result = controller.approve(values["task_id"], values["approval_id"], values["approved"],
                                        operator_authorized=True)
        else:
            result = getattr(controller, operation.split(".", 1)[1])(values["task_id"])
        try:
            return validate_json(result, max_bytes=MAX_RESPONSE_BYTES - 8192, max_depth=14)
        except BrainError:
            # Host persistence limits may exceed transport limits. Preserve a
            # truthful task receipt while retaining the full journal locally.
            receipt = {key: deepcopy(result[key]) for key in
                       ("task_id", "status", "steps", "failures", "argument_repairs", "source", "proposer_id", "limits",
                        "created_at", "updated_at", "cancel_requested") if key in result}
            receipt.update(snapshotOmitted=True, journalEntries=len(result.get("journal", [])))
            pending = result.get("pending_approval")
            try:
                receipt["pending_approval"] = deepcopy(validate_json(pending, max_bytes=262144, max_depth=12))
            except BrainError:
                receipt["pendingApprovalOmitted"] = True
            return validate_json(receipt, max_bytes=MAX_RESPONSE_BYTES - 8192, max_depth=14)

    def _validate_memory(self, operation, values, allow_refs):
        shapes = {
            "memory.task.add": ({"goal"}, {"context"}),
            "memory.task.list": (set(), set()), "memory.task.status": (set(), set()),
            "memory.task.search": ({"query"}, {"limit"}),
            "memory.task.show": ({"task_id"}, set()),
            "memory.task.attempt": ({"task_id", "text"}, {"evidence"}),
            "memory.task.feedback": ({"task_id", "attempt_id", "text", "accepted"}, set()),
        }
        required, optional = shapes[operation]
        if not required <= values.keys() or not values.keys() <= required | optional:
            raise BrainError("invalid_input", "Memory operation has unsupported or missing fields")
        if operation == "memory.task.search":
            query, limit = values["query"], values.get("limit", DEFAULT_SEARCH_LIMIT)
            deferred_query = allow_refs and is_reference(query)
            deferred_limit = allow_refs and is_reference(limit)
            try:
                validate_search("reference" if deferred_query else query,
                                DEFAULT_SEARCH_LIMIT if deferred_limit else limit)
            except ValueError as error:
                raise BrainError("invalid_input", "Memory search needs bounded words and a result limit from 1 to 20") from error
            return
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
            elif operation in AUTONOMY_OPERATIONS:
                result = self._execute_autonomy(operation, input_data)
            elif operation == "tool.call":
                result = self._tools[input_data["name"]].handler(input_data["arguments"])
            elif operation == "memory.task.add":
                result = workspace.add_task(self._memory_directory, input_data["goal"], input_data.get("context", ""))
            elif operation == "memory.task.list":
                result = workspace.list_tasks(self._memory_directory)
            elif operation == "memory.task.search":
                result = workspace.search_tasks(self._memory_directory, input_data["query"],
                                                input_data.get("limit", DEFAULT_SEARCH_LIMIT))
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
            # Coordination failures preserve completed model-work receipts while
            # keeping private exception text out of the transport protocol.
            from ..conversation.ensemble import EnsembleError
            if isinstance(error, EnsembleError):
                raise BrainError("ensemble_incomplete", "La coordinación no produjo los candidatos mínimos requeridos.",
                                 result={"ensemble": error.details}) from error
            raise BrainError("operation_failed", "Explicit operation failed; completed effects were not rolled back") from error
