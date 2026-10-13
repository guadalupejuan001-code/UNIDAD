"""Host-bound, persistent control of proposals; this does not teach a planner.

Callbacks run outside SQLite transactions. Cancellation stops future effects;
an already running external handler cannot be killed or rolled back here. Hosts
must give network/process handlers their own finite timeouts. The controller
checks its wall-clock budget before and after every callback.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Callable
import uuid

from .protocol import BrainError, MAX_JSON_NODES, decode_json, identifier, validate_json

EFFECTS = {"read", "write", "destructive", "publication", "expense"}
APPROVAL_EFFECTS = {"destructive", "publication", "expense"}
CONTROL_NAMES = {"approve", "autonomy.approve", "autonomy.start", "autonomy.cancel", "autonomy.resume"}
TERMINAL = {"completed", "cancelled", "stopped", "blocked_model", "blocked_context", "blocked_policy", "limit_reached"}


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value, name, limit):
    if type(value) is not str or not value.strip():
        raise BrainError("invalid_input", f"{name} must be explicit text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise BrainError("invalid_input", f"{name} must be valid UTF-8") from error
    if size > limit:
        raise BrainError("invalid_input", f"{name} exceeds its byte limit")
    return value


@dataclass(frozen=True)
class AutonomyLimits:
    max_steps: int = 16
    max_failures: int = 3
    max_seconds: float = 120
    max_goal_bytes: int = 8192
    max_proposal_bytes: int = 65536
    max_result_bytes: int = 131072
    max_state_bytes: int = 1048576

    def __post_init__(self):
        ceilings = {"max_steps": 1000, "max_failures": 1000, "max_goal_bytes": 262144,
                    "max_proposal_bytes": 1048576, "max_result_bytes": 1048576,
                    "max_state_bytes": 67108864}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise BrainError("invalid_input", f"{name} must be a finite bounded integer")
        if type(self.max_seconds) not in (int, float) or not 0 < self.max_seconds <= 86400:
            raise BrainError("invalid_input", "max_seconds must be finite and between zero and 86400")
        if self.max_state_bytes < 16384:
            raise BrainError("invalid_input", "Task state requires at least 16384 bytes")


@dataclass(frozen=True)
class AutonomyTool:
    """A trusted binding; effect classification must be explicit.

    ``policy_id`` identifies host resources/configuration hidden from the model.
    Hosts must change it when the callback keeps its name but changes authority.
    """
    name: str
    handler: Callable[[dict], object]
    validator: Callable[[dict], object]
    description: str = ""
    schema: dict = field(default_factory=dict)
    effect: str = "unclassified"
    retry_safe: bool = False
    verifier: Callable[[dict, object], object] | None = None
    policy_id: str = ""

    def __post_init__(self):
        identifier(self.name, "autonomy tool name")
        if self.name in CONTROL_NAMES:
            raise BrainError("invalid_input", "Operator controls cannot be model tools")
        if not callable(self.handler) or not callable(self.validator):
            raise BrainError("invalid_input", "Autonomy tools require host handlers and validators")
        if self.verifier is not None and not callable(self.verifier):
            raise BrainError("invalid_input", "Tool verifier must be callable")
        if self.effect not in EFFECTS or type(self.retry_safe) is not bool:
            raise BrainError("invalid_input", "Invalid host effect policy")
        if type(self.policy_id) is not str or len(self.policy_id.encode("utf-8")) > 4096:
            raise BrainError("invalid_input", "Host policy identity must be bounded text")
        if type(self.description) is not str or len(self.description.encode("utf-8")) > 4096:
            raise BrainError("invalid_input", "Tool description must be bounded text")
        if type(self.schema) is not dict:
            raise BrainError("invalid_input", "Tool schema must be a JSON object")
        validate_json(self.schema, max_bytes=16384)
        object.__setattr__(self, "schema", deepcopy(self.schema))

    def metadata(self):
        return {"name": self.name, "description": self.description, "schema": deepcopy(self.schema),
                "effect": self.effect, "retry_safe": self.retry_safe,
                "requires_operator_approval": self.effect in APPROVAL_EFFECTS}


class AutonomyController:
    """Control a host proposer and registry, isolated by the host state directory.

    Proposer(context)->raw JSON text must identify ``origin`` as ``own_model``
    or ``test_fixture`` and provide ``proposer_id``. Optional ``last_metadata``
    is journaled. Tool verifiers(args, result_or_None) return a bool or an object
    containing an explicit ``verified`` bool. Completion verifiers(snapshot)
    use the same contract and must check actual host artifacts/state.
    ``policy_id`` binds additional host policies (for example input validation)
    across restarts. Completion callbacks may also declare a ``policy_id``.
    A bound completion adapter may provide ``restore_task_observations(snapshot)``;
    it runs before each ready step/resume and must restore only trusted evidence.
    """

    def __init__(self, state_dir, proposer, tools, limits=None, completion_verifier=None, *, policy_id=""):
        if not isinstance(state_dir, (str, Path)) or not callable(proposer):
            raise BrainError("invalid_input", "State directory and proposer are host configuration")
        self.limits = limits if limits is not None else AutonomyLimits()
        if not isinstance(self.limits, AutonomyLimits):
            raise BrainError("invalid_input", "Limits must be configured by the host")
        if completion_verifier is not None and not callable(completion_verifier):
            raise BrainError("invalid_input", "Completion verifier must be callable")
        completion_policy_id = getattr(completion_verifier, "policy_id", "")
        for value in (policy_id, completion_policy_id):
            if type(value) is not str or len(value.encode("utf-8")) > 4096:
                raise BrainError("invalid_input", "Controller host policy identity must be bounded text")
        self.source = getattr(proposer, "origin", None)
        if self.source not in {"own_model", "test_fixture"}:
            raise BrainError("invalid_input", "Proposer origin must explicitly identify own_model or test_fixture")
        self.proposer_id = _text(getattr(proposer, "proposer_id", None), "proposer_id", 256)
        self.proposer = proposer
        self.completion_verifier = completion_verifier
        owner = getattr(completion_verifier, "__self__", None)
        self._task_restorer = getattr(owner, "restore_task_observations", None)
        if self._task_restorer is not None and not callable(self._task_restorer):
            raise BrainError("invalid_input", "Host task observation restorer must be callable")
        bound = list(tools)
        if any(not isinstance(tool, AutonomyTool) for tool in bound) or len({t.name for t in bound}) != len(bound):
            raise BrainError("invalid_input", "Autonomy tools require unique host bindings")
        # Snapshot metadata to prevent later caller mutation from changing policy.
        self._tools = {tool.name: tool for tool in bound}
        self._metadata = {tool.name: tool.metadata() for tool in bound}
        def callable_identity(value):
            return None if value is None else [getattr(value, "__module__", type(value).__module__),
                                               getattr(value, "__qualname__", type(value).__qualname__)]
        self._policy = _digest({"tools": self._metadata, "callbacks": {
            tool.name: {"handler": callable_identity(tool.handler), "validator": callable_identity(tool.validator),
                        "verifier": callable_identity(tool.verifier), "policy_id": tool.policy_id} for tool in bound},
            "completion_verifier": callable_identity(completion_verifier),
            "completion_policy_id": completion_policy_id, "controller_policy_id": policy_id,
            "task_restorer": callable_identity(self._task_restorer)})
        self._directory = Path(state_dir).expanduser().absolute()
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._database = self._directory / "autonomy.sqlite3"
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self._database.chmod(0o600)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._database, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
        finally:
            connection.close()

    def _task_id(self, value):
        if type(value) is not str or len(value) != 37 or not value.startswith("task_"):
            raise BrainError("invalid_input", "Invalid autonomy task identity")
        try:
            int(value[5:], 16)
        except ValueError as error:
            raise BrainError("invalid_input", "Invalid autonomy task identity") from error
        return value

    def _load(self, connection, task_id):
        row = connection.execute("SELECT state FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise BrainError("task_not_found", "Autonomy task is not in this host scope")
        return json.loads(row[0])

    def _mutate(self, task_id, change):
        self._task_id(task_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._load(connection, task_id)
                original = deepcopy(state)
                change(state)
                state["updated_at"] = time.time()
                try:
                    validate_json(state, max_bytes=state["limits"]["max_state_bytes"])
                except BrainError:
                    # If a real handler returned a very complex JSON object,
                    # retain its receipt instead of losing an already-run effect
                    # to a failed database transaction. The pre-effect record
                    # and exact arguments remain available for safe recovery.
                    receipts = []
                    for event in state["journal"][len(original["journal"]):][:4]:
                        receipt = {"kind": event["kind"], "sha256": _digest(event), "bytes": len(_encoded(event))}
                        if event["kind"] == "proposal":
                            receipt["raw"] = self._evidence(event.get("raw"), 1024)
                        receipts.append(receipt)
                    original["status"] = "cancelled" if state["cancel_requested"] else "limit_reached"
                    original["cancel_requested"] = state["cancel_requested"]
                    original["steps"] = state["steps"]
                    original["failures"] = state["failures"]
                    original["updated_at"] = state["updated_at"]
                    self._event(original, "state_limit", omitted_events=receipts,
                                effects_rolled_back=False, outcome_may_exist=bool(
                                    original["current"] and original["current"].get("effect_started")))
                    state = original
                    validate_json(state, max_bytes=state["limits"]["max_state_bytes"])
                connection.execute("UPDATE tasks SET state=? WHERE id=?", (_encoded(state).decode("utf-8"), task_id))
                connection.commit()
                return deepcopy(state)
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def _step_lock(self, task_id):
        self._task_id(task_id)
        with (self._directory / (task_id + ".lock")).open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BrainError("task_busy", "Another step or recovery is running for this task") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _event(self, state, kind, **values):
        state["journal"].append({"index": len(state["journal"]), "time": time.time(), "kind": kind, **values})

    def _failure(self, state, code, message, *, status="ready"):
        state["failures"] += 1
        self._event(state, "failure", code=code, message=str(message)[:512])
        state["status"] = status
        if state["failures"] >= state["limits"]["max_failures"] and status == "ready":
            state["status"] = "limit_reached"
        if state["cancel_requested"]:
            state["status"] = "cancelled"

    def _budget(self, state, *, next_step=False, reserve=0):
        reason = None
        if state["cancel_requested"]:
            state["status"] = "cancelled"
            return False
        limits = state["limits"]
        if time.time() - state["created_at"] >= limits["max_seconds"]:
            reason = "seconds"
        elif state["failures"] >= limits["max_failures"]:
            reason = "failures"
        elif next_step and state["steps"] >= limits["max_steps"]:
            reason = "steps"
        elif len(_encoded(state)) + reserve + 4096 > limits["max_state_bytes"]:
            reason = "state_bytes"
        else:
            def nodes(value):
                if type(value) is dict:
                    return 1 + len(value) + sum(nodes(child) for child in value.values())
                if type(value) is list:
                    return 1 + sum(nodes(child) for child in value)
                return 1
            if nodes(state) > MAX_JSON_NODES - 512:
                reason = "state_elements"
        if reason:
            state["status"] = "limit_reached"
            self._event(state, "limit", resource=reason)
            return False
        return True

    def _binding_valid(self, state):
        if (state["policy"] != self._policy or state["source"] != self.source or
                state["proposer_id"] != self.proposer_id):
            self._failure(state, "host_binding_changed", "Task policy or proposer identity changed",
                          status="blocked_policy")
            return False
        return True

    def start(self, goal, *, task_inputs=None):
        _text(goal, "goal", self.limits.max_goal_bytes)
        values = {} if task_inputs is None else task_inputs
        if type(values) is not dict:
            raise BrainError("invalid_input", "Task inputs must be a host-validated JSON object")
        validate_json(values, max_bytes=self.limits.max_goal_bytes)
        task_id = "task_" + uuid.uuid4().hex
        now = time.time()
        state = {"task_id": task_id, "goal": goal, "task_inputs": deepcopy(values), "status": "ready",
                 "steps": 0, "failures": 0, "source": self.source, "proposer_id": self.proposer_id,
                 "policy": self._policy,
                 "limits": asdict(self.limits), "created_at": now, "updated_at": now,
                 "cancel_requested": False, "pending_approval": None, "current": None, "journal": []}
        self._event(state, "started", source=self.source, proposer_id=self.proposer_id, policy=self._policy)
        validate_json(state, max_bytes=self.limits.max_state_bytes - 4096)
        with self._connect() as connection:
            connection.execute("INSERT INTO tasks VALUES (?,?)", (task_id, _encoded(state).decode("utf-8")))
        return deepcopy(state)

    def status(self, task_id):
        self._task_id(task_id)
        with self._connect() as connection:
            return self._load(connection, task_id)

    def cancel(self, task_id):
        def change(state):
            if state["status"] == "completed" or state["cancel_requested"]:
                return
            state["cancel_requested"] = True
            state["status"] = "cancelled"
            self._event(state, "cancelled", effects_rolled_back=False)
        return self._mutate(task_id, change)

    def _evidence(self, value, limit):
        try:
            validate_json(value, max_bytes=limit)
            return deepcopy(value)
        except (BrainError, ValueError, TypeError):
            if type(value) is str:
                data = value.encode("utf-8", errors="replace")
                return {"omitted": True, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "preview": data[: min(limit // 4, 2048)].decode("utf-8", errors="replace")}
            return {"omitted": True, "reason": "non_json_or_over_limit", "type": type(value).__name__}

    def _proposal(self, raw, limit):
        if type(raw) is not str:
            raise BrainError("invalid_proposal", "Model proposal must be raw JSON text")
        proposal = decode_json(raw, max_bytes=limit)
        if type(proposal) is not dict or proposal.get("action") not in {"tool", "finish", "stop"}:
            raise BrainError("invalid_proposal", "Unsupported proposal action")
        expected = {"action", "tool", "arguments", "reason"} if proposal["action"] == "tool" else {"action", "reason"}
        if set(proposal) != expected:
            raise BrainError("invalid_proposal", "Proposal contains missing or unsupported fields")
        _text(proposal["reason"], "reason", 4096)
        if proposal["action"] == "tool":
            identifier(proposal["tool"], "proposal tool")
            if type(proposal["arguments"]) is not dict:
                raise BrainError("invalid_proposal", "Tool arguments must be a JSON object")
        return proposal

    def _verification(self, value, limit):
        if type(value) is bool:
            return {"verified": value}
        validate_json(value, max_bytes=limit)
        if type(value) is not dict or type(value.get("verified")) is not bool:
            raise BrainError("verification_failed", "Host verifier must return an explicit verified boolean")
        return deepcopy(value)

    def _check_completion(self, task_id):
        state = self._mutate(task_id, lambda s: self._budget(s, reserve=s["limits"]["max_result_bytes"]))
        if state["cancel_requested"] or state["status"] == "limit_reached":
            return state
        try:
            observation = ({"verified": False, "reason": "no_host_completion_verifier"}
                           if self.completion_verifier is None else
                           self._verification(self.completion_verifier(deepcopy(state)), state["limits"]["max_result_bytes"]))
        except Exception as error:
            return self._mutate(task_id, lambda s: self._failure(s, "completion_verifier_failed", error,
                                                               status="blocked_verification"))
        def change(current):
            self._event(current, "completion_verification", observation=observation)
            if self._budget(current) and observation["verified"]:
                current["status"] = "completed"
                current["current"] = None
                current["pending_approval"] = None
        return self._mutate(task_id, change)

    def _restore_task_observations(self, task_id):
        """Replay trusted task evidence outside transactions, never model args."""
        if self._task_restorer is None:
            return self.status(task_id)
        state = self._mutate(task_id, lambda s: self._budget(s, reserve=s["limits"]["max_result_bytes"]))
        retrying = state["status"] == "blocked_verification" and any(
            event["kind"] == "failure" and event.get("code") == "task_restore_failed"
            for event in state["journal"][-2:])
        if state["status"] != "ready" and not retrying:
            return state
        try:
            result = self._task_restorer(deepcopy(state))
            validate_json(result, max_bytes=state["limits"]["max_result_bytes"])
        except Exception as error:
            return self._mutate(task_id, lambda s: self._failure(s, "task_restore_failed", error,
                                                               status="blocked_verification"))
        def restored(current):
            self._event(current, "observations_restored", result=deepcopy(result))
            if retrying:
                current["status"] = "ready"
            self._budget(current)
        return self._mutate(task_id, restored)

    def _approval_binding(self, state, action):
        if action["tool"] not in self._metadata:
            raise BrainError("approval_denied", "Approved tool is no longer bound to this host")
        return _digest({"task_id": state["task_id"], "proposal": action, "policy": self._policy,
                        "effect": self._metadata[action["tool"]]["effect"]})

    def approve(self, task_id, approval_id, approved, *, operator_authorized=False):
        if operator_authorized is not True or type(approved) is not bool:
            raise BrainError("approval_denied", "Approval requires a separate authorized operator channel")
        def change(state):
            if not self._binding_valid(state):
                raise BrainError("approval_denied", "Task host policy or proposer identity changed")
            pending = state["pending_approval"]
            if state["status"] != "needs_approval" or pending is None or pending["approval_id"] != approval_id:
                raise BrainError("approval_denied", "Approval nonce does not match a pending action")
            if pending["binding"] != self._approval_binding(state, state["current"]["proposal"]):
                raise BrainError("approval_denied", "Approval policy or exact arguments changed")
            if not self._budget(state):
                return
            pending["approved"] = approved
            self._event(state, "operator_approval", approval_id=approval_id, binding=pending["binding"], approved=approved)
            state["status"] = "ready" if approved else "stopped"
        return self._mutate(task_id, change)

    def step(self, task_id):
        with self._step_lock(task_id):
            state = self.status(task_id)
            if state["status"] in TERMINAL:
                return state
            def ready(current):
                if self._binding_valid(current):
                    self._budget(current)
            state = self._mutate(task_id, ready)
            if state["status"] != "ready":
                return state
            state = self._restore_task_observations(task_id)
            if state["status"] != "ready":
                return state
            pending = state["pending_approval"]
            if pending is not None and pending.get("approved") is True:
                return self._execute_tool(task_id, state["current"]["proposal"])
            def begin(current):
                if self._binding_valid(current) and self._budget(current, next_step=True,
                                reserve=current["limits"]["max_proposal_bytes"] * 3):
                    current["steps"] += 1
                    current["status"] = "proposing"
                    current["current"] = None
                    self._event(current, "proposal_requested", step=current["steps"])
            state = self._mutate(task_id, begin)
            if state["status"] != "proposing":
                return state
            context = {"task": deepcopy(state), "tools": deepcopy(list(self._metadata.values())),
                       "proposal_schema": {"tool": {"action": "tool", "tool": "registered name", "arguments": {}, "reason": "text"},
                                           "finish": {"action": "finish", "reason": "text"},
                                           "stop": {"action": "stop", "reason": "text"}}}
            try:
                raw = self.proposer(context)
            except Exception as error:
                context_blocked = isinstance(error, BrainError) and error.code == "context_too_large"
                def failed(current):
                    self._event(current, "proposer_evidence", source=self.source, proposer_id=self.proposer_id,
                                metadata=self._evidence(getattr(self.proposer, "last_metadata", None),
                                                        current["limits"]["max_proposal_bytes"]))
                    self._failure(current, "context_too_large" if context_blocked else "proposer_failed", error,
                                  status="blocked_context" if context_blocked else "ready")
                    if not context_blocked:
                        self._budget(current)
                return self._mutate(task_id, failed)
            limit = state["limits"]["max_proposal_bytes"]
            evidence = self._evidence(raw, limit)
            metadata = self._evidence(getattr(self.proposer, "last_metadata", None), limit)
            try:
                action = self._proposal(raw, limit)
            except Exception as error:
                def invalid(current):
                    self._event(current, "proposal", raw=evidence, metadata=metadata, source=self.source,
                                proposer_id=self.proposer_id, valid=False)
                    self._failure(current, "invalid_proposal", error, status="blocked_model")
                return self._mutate(task_id, invalid)
            def record(current):
                self._event(current, "proposal", raw=evidence, metadata=metadata, source=self.source,
                            proposer_id=self.proposer_id, valid=True)
                current["current"] = {"proposal": action, "fingerprint": _digest(action), "effect_started": False}
                if self._budget(current):
                    current["status"] = "ready"
            state = self._mutate(task_id, record)
            if state["status"] != "ready":
                return state
            if action["action"] == "tool":
                return self._execute_tool(task_id, action)
            if action["action"] == "stop":
                return self._mutate(task_id, lambda s: s.update(status="stopped"))
            state = self._check_completion(task_id)
            if state["status"] == "ready":
                return self._mutate(task_id, lambda s: self._failure(s, "finish_unverified",
                                  "The model requested finish without verified host completion", status="blocked_verification"))
            return state

    def _execute_tool(self, task_id, action):
        def before_validator(current):
            if self._binding_valid(current):
                self._budget(current)
        state = self._mutate(task_id, before_validator)
        if state["status"] != "ready":
            return state
        tool = self._tools.get(action["tool"])
        if tool is None:
            return self._mutate(task_id, lambda s: self._failure(s, "tool_denied", "Tool is not bound to this host",
                                                               status="blocked_policy"))
        try:
            if tool.validator(deepcopy(action["arguments"])) is False:
                raise BrainError("tool_denied", "Host validator rejected arguments")
        except Exception as error:
            return self._mutate(task_id, lambda s: self._failure(s, "arguments_denied", error, status="blocked_policy"))
        def prepare(current):
            if not self._binding_valid(current) or not self._budget(current, reserve=current["limits"]["max_result_bytes"] * 3):
                return
            if tool.effect != "read" and not tool.retry_safe:
                fingerprint = _digest({"tool": tool.name, "arguments": action["arguments"]})
                if any(event["kind"] == "effect_started" and event.get("effect_fingerprint") == fingerprint
                       for event in current["journal"]):
                    self._failure(current, "duplicate_effect_denied", "A non-idempotent effect was already attempted",
                                  status="blocked_policy")
                    return
            if tool.effect in APPROVAL_EFFECTS:
                binding = self._approval_binding(current, action)
                pending = current["pending_approval"]
                if pending is None:
                    current["pending_approval"] = {"approval_id": secrets.token_urlsafe(24), "binding": binding,
                                                   "tool": tool.name, "arguments": deepcopy(action["arguments"]),
                                                   "effect": tool.effect, "approved": None}
                    current["status"] = "needs_approval"
                    self._event(current, "approval_requested", binding=binding, effect=tool.effect)
                    return
                if pending.get("approved") is not True or pending["binding"] != binding:
                    self._failure(current, "approval_denied", "No approval matches the exact effect and policy",
                                  status="blocked_policy")
                    return
                self._event(current, "approval_consumed", approval_id=pending["approval_id"], binding=binding)
                current["pending_approval"] = None
            current["status"] = "executing"
            current["current"]["effect_started"] = True
            current["current"]["effect"] = tool.effect
            self._event(current, "effect_started", tool=tool.name, arguments=deepcopy(action["arguments"]), effect=tool.effect,
                        effect_fingerprint=_digest({"tool": tool.name, "arguments": action["arguments"]}))
        state = self._mutate(task_id, prepare)
        if state["status"] != "executing":
            return state
        # Cancellation while a proposer or validator was blocked is observed
        # before the handler. Once this invocation begins its effects may exist.
        state = self._mutate(task_id, self._budget)
        if state["status"] != "executing":
            return state
        result = None
        returned = False
        try:
            result = tool.handler(deepcopy(action["arguments"]))
            returned = True
            validate_json(result, max_bytes=state["limits"]["max_result_bytes"])
        except Exception as error:
            def failed(current):
                if returned:
                    self._event(current, "tool_result", tool=tool.name, arguments=deepcopy(action["arguments"]),
                                result=self._evidence(result, current["limits"]["max_result_bytes"]), bounded=False)
                self._failure(current, "tool_failed", error,
                              status="ready" if tool.effect == "read" or tool.retry_safe else "blocked_uncertain")
                if current["status"] == "ready":
                    self._budget(current)
            state = self._mutate(task_id, failed)
            if tool.verifier is not None and not state["cancel_requested"]:
                return self._recover_effect(task_id, tool, action, None)
            return state
        def returned(current):
            self._event(current, "handler_returned", tool=tool.name, arguments=deepcopy(action["arguments"]),
                        result=deepcopy(result))
            self._budget(current)
        state = self._mutate(task_id, returned)
        if state["status"] != "executing":
            return state
        observation = {"verified": False, "reason": "no_host_tool_verifier"}
        if tool.verifier is not None:
            try:
                observation = self._verification(tool.verifier(deepcopy(action["arguments"]), deepcopy(result)),
                                                 state["limits"]["max_result_bytes"])
            except Exception as error:
                def failed(current):
                    self._event(current, "tool_result", tool=tool.name, arguments=deepcopy(action["arguments"]),
                                result=deepcopy(result))
                    current["current"]["result"] = deepcopy(result)
                    self._failure(current, "tool_verifier_failed", error, status="blocked_verification")
                return self._mutate(task_id, failed)
        def finish(current):
            self._event(current, "tool_result", tool=tool.name, arguments=deepcopy(action["arguments"]),
                        result=deepcopy(result), verification=observation)
            if tool.verifier is not None and not observation["verified"]:
                current["current"]["result"] = deepcopy(result)
                self._failure(current, "tool_not_verified", "Host verifier did not confirm the tool result",
                              status="blocked_verification")
                self._budget(current)
            else:
                current["current"] = None
                current["status"] = "ready"
                self._budget(current)
        state = self._mutate(task_id, finish)
        if state["status"] == "ready":
            state = self._check_completion(task_id)
        return state

    def _recover_effect(self, task_id, tool, action, result):
        state = self._mutate(task_id, lambda s: self._budget(s, reserve=s["limits"]["max_result_bytes"]))
        if state["cancel_requested"] or state["status"] == "limit_reached":
            return state
        try:
            observation = self._verification(tool.verifier(deepcopy(action["arguments"]), deepcopy(result)),
                                             self.status(task_id)["limits"]["max_result_bytes"])
        except Exception as error:
            return self._mutate(task_id, lambda s: self._failure(s, "recovery_verifier_failed", error,
                                                               status="blocked_uncertain"))
        def recovered(current):
            self._event(current, "recovery_verification", tool=tool.name, observation=observation)
            if current["cancel_requested"]:
                current["status"] = "cancelled"
            elif observation["verified"]:
                current["current"] = None
                current["status"] = "ready"
                self._budget(current)
            elif tool.effect == "read" or tool.retry_safe:
                current["current"] = None
                self._failure(current, "recovery_not_verified", "Host verifier did not confirm the outcome")
                self._budget(current)
            else:
                self._failure(current, "recovery_not_verified", "Host verifier did not confirm the outcome",
                              status="blocked_uncertain")
                self._budget(current)
        state = self._mutate(task_id, recovered)
        return self._check_completion(task_id) if state["status"] == "ready" else state

    def resume(self, task_id):
        with self._step_lock(task_id):
            state = self.status(task_id)
            if state["status"] in TERMINAL:
                return state
            def can_resume(current_state):
                if self._binding_valid(current_state):
                    self._budget(current_state, reserve=current_state["limits"]["max_result_bytes"])
            state = self._mutate(task_id, can_resume)
            if state["status"] == "ready":
                return self._restore_task_observations(task_id)
            if state["status"] in TERMINAL or state["status"] == "needs_approval":
                return state
            if state["status"] == "blocked_verification" and any(
                    event["kind"] == "failure" and event.get("code") == "task_restore_failed"
                    for event in state["journal"][-2:]):
                return self._restore_task_observations(task_id)
            if state["status"] == "proposing":
                return self._mutate(task_id, lambda s: self._failure(s, "interrupted_proposer",
                                             "Previous proposal call ended without a recorded response"))
            current = state["current"]
            if current is not None and current.get("effect_started"):
                action = current["proposal"]
                tool = self._tools.get(action["tool"])
                if tool is None:
                    return self._mutate(task_id, lambda s: self._failure(s, "recovery_tool_denied",
                                               "Original tool is unavailable", status="blocked_uncertain"))
                if tool.verifier is not None:
                    recorded = current.get("result")
                    if recorded is None:
                        for event in reversed(state["journal"]):
                            if (event["kind"] in {"handler_returned", "tool_result"} and
                                    event["tool"] == tool.name and event.get("arguments") == action["arguments"]):
                                recorded = event.get("result")
                                break
                    return self._recover_effect(task_id, tool, action, recorded)
                if tool.effect == "read" or tool.retry_safe:
                    def retry(current_state):
                        current_state["current"] = None
                        self._failure(current_state, "interrupted_tool", "Safe tool may be proposed again")
                        self._budget(current_state)
                    return self._mutate(task_id, retry)
                return self._mutate(task_id, lambda s: self._failure(s, "interrupted_effect",
                                            "Effect outcome is unknown; no blind retry", status="blocked_uncertain"))
            return self._check_completion(task_id)

    def run(self, task_id):
        while True:
            state = self.step(task_id)
            if state["status"] != "ready":
                return state
