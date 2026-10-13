"""Bounded coordination of verified own Transformers through the common engine.

Models exchange text, not incompatible token IDs or hidden states. Selection is
an explicit rule: within-critic likelihood ranks, lexical source support, and an
optional auxiliary diagnostic. Neither likelihood nor overlap proves truth or
conversation quality. The returned text is exactly one model's original output.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from time import perf_counter
import unicodedata

from .contract import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, DEFAULT_TEMPERATURE
from .runtime import ConversationEngine, validate_reply_request


class EnsembleError(ValueError):
    """Coordination could not produce the declared minimum; trace has no text."""

    def __init__(self, reason, trace=None):
        super().__init__(reason)
        self.reason = reason
        self.trace = deepcopy(trace) if trace is not None else None
        self.details = self.trace


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_dir: str | Path
    expected_model_id: str
    decoder_backend: str = "torch"
    use_cache: bool = True
    role: str = "candidate"

    def __post_init__(self):
        if type(self.name) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.name) is None:
            raise ValueError("Model name must be a bounded host-configured identifier")
        if not isinstance(self.model_dir, (str, Path)) or not str(self.model_dir):
            raise ValueError("Model directory must be a host-configured local path")
        if type(self.expected_model_id) is not str or not self.expected_model_id or len(self.expected_model_id) > 128:
            raise ValueError("Model identity must be pinned by the host")
        if self.decoder_backend not in ("torch", "nerva", "nerva-fast") or type(self.use_cache) is not bool:
            raise ValueError("Model execution requires a known decoder and boolean cache")
        if type(self.role) is not str or not self.role or len(self.role) > 64:
            raise ValueError("Declared role must be bounded text, not a capability claim")


@dataclass(frozen=True)
class EnsembleBudget:
    max_candidates: int = 4
    min_candidates: int = 2
    max_resident: int = 4
    max_seconds: float = 60
    total_new_tokens: int = 256
    max_score_calls: int = 8
    critic_count: int = 2
    max_scoring_tokens: int = 256
    max_sliding_score_tokens: int = 0

    def __post_init__(self):
        for name, low, high in (("max_candidates", 2, 8), ("min_candidates", 2, 8),
                ("max_resident", 1, 64), ("total_new_tokens", 2, 8192),
                ("max_score_calls", 0, 64), ("critic_count", 1, 8),
                ("max_scoring_tokens", 1, 1025), ("max_sliding_score_tokens", 0, 1025)):
            _integer(getattr(self, name), name, low, high)
        if self.min_candidates > self.max_candidates or self.total_new_tokens < self.min_candidates:
            raise ValueError("Budget must admit the minimum number of candidates")
        if type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 3600:
            raise ValueError("max_seconds must be finite, positive and at most 3600")


def _read_json(path, limit=4_194_304):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise EnsembleError("artifact_unavailable")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise EnsembleError("artifact_manifest_invalid")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EnsembleError("artifact_manifest_invalid") from exc
    if type(result) is not dict:
        raise EnsembleError("artifact_manifest_invalid")
    return result


def _preflight(spec):
    """Admit identity and bounded header before importing/loading neural weights.

    Full file hashes, architecture and tensors are still verified by
    ConversationEngine. This light check is not advertised as weight validation.
    """
    root = Path(spec.model_dir)
    if root.is_symlink():
        raise EnsembleError("artifact_unavailable")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("modelId") != spec.expected_model_id:
        raise EnsembleError("model_identity_mismatch")
    if manifest.get("modelType") != "unidad-conversation-transformer" or type(manifest.get("config")) is not dict:
        raise EnsembleError("artifact_manifest_invalid")
    path = root / "weights.safetensors"
    descriptor = manifest.get("files", {}).get("weights.safetensors", {})
    if path.is_symlink() or not path.is_file():
        raise EnsembleError("artifact_unavailable")
    if type(descriptor.get("bytes")) is not int or path.stat().st_size != descriptor["bytes"]:
        raise EnsembleError("weights_size_mismatch")
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= 1_048_576 or length + 8 > descriptor["bytes"]:
            raise EnsembleError("weights_header_invalid")
        try:
            header = json.loads(stream.read(length))
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise EnsembleError("weights_header_invalid") from exc
        if type(header) is not dict or not header:
            raise EnsembleError("weights_header_invalid")
        tensors = [description for key, description in header.items() if key != "__metadata__"]
        payload_bytes = descriptor["bytes"] - length - 8
        for tensor in tensors:
            if type(tensor) is not dict or set(tensor) != {"dtype", "shape", "data_offsets"} or tensor["dtype"] != "F32":
                raise EnsembleError("weights_header_invalid")
            shape, offsets = tensor["shape"], tensor["data_offsets"]
            if type(shape) is not list or any(type(size) is not int or size < 0 for size in shape):
                raise EnsembleError("weights_header_invalid")
            if type(offsets) is not list or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
                raise EnsembleError("weights_header_invalid")
            if not 0 <= offsets[0] <= offsets[1] <= payload_bytes or offsets[1] - offsets[0] != math.prod(shape) * 4:
                raise EnsembleError("weights_header_invalid")
        if not tensors:
            raise EnsembleError("weights_header_invalid")
    return manifest


_FILLERS = frozenset(("el", "la", "los", "las", "un", "una", "de", "del", "en", "a", "al", "y",
                     "es", "son", "que", "qué", "se", "lo", "por", "para", "con", "como", "cómo"))
_FAILURE_REASONS = frozenset(("artifact_unavailable", "artifact_manifest_invalid", "model_identity_mismatch",
    "weights_size_mismatch", "weights_header_invalid", "minimum_context_exceeds_capacity", "request_budget",
    "generation_result_invalid", "generation_token_budget_invalid", "scoring_target_token_budget",
    "scoring_result_invalid", "scoring_sliding_budget"))


def _terms(text):
    return set(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())) - _FILLERS


def _source_terms(messages, context_sources):
    sources = [message["content"] for message in messages if message["role"] in ("user", "system")]
    if context_sources is not None:
        if type(context_sources) not in (list, tuple) or len(context_sources) > 64:
            raise ValueError("Context sources must be a host-provided bounded list")
        size = 0
        for source in context_sources:
            if type(source) is not dict:
                raise ValueError("Context source must be an object")
            if source.get("role") not in ("user", "system"):
                continue
            text = source.get("text", source.get("content"))
            if type(text) is not str:
                raise ValueError("Context source must preserve original text")
            size += len(text.encode("utf-8"))
            if size > 65_536:
                raise ValueError("Context source text exceeds the byte budget")
            sources.append(text)
    result = set()
    for text in sources:
        result.update(_terms(text))
    return result


def _auxiliary_metadata(extra):
    """Keep only bounded diagnostic literals, never arbitrary callback output."""
    if type(extra) is not dict:
        raise EnsembleError("scoring_result_invalid")
    result = {}
    for key in ("meanNll", "unknownRatio"):
        value = extra.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            if key != "unknownRatio" or value <= 1:
                result[key] = float(value)
        elif key == "meanNll" and value is None:
            result[key] = None
    for key in ("scoreable", "truncated", "teacherForced", "includesEos"):
        if type(extra.get(key)) is bool:
            result[key] = extra[key]
    for key in ("targetChars", "scoredCharacters", "charLength"):
        if type(extra.get(key)) is int and 0 <= extra[key] <= 16_384:
            result[key] = extra[key]
    model_id = extra.get("modelId")
    if type(model_id) is str and re.fullmatch(r"(?:language|unidad-conversation)-[0-9a-f]{64}", model_id):
        result["modelId"] = model_id
    digest = extra.get("artifactSha256")
    if type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest):
        result["artifactSha256"] = digest
    for key in ("source", "task"):
        if type(extra.get(key)) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", extra[key]):
            result[key] = extra[key]
    if type(extra.get("executionBackend")) is str and extra["executionBackend"] in ("numpy", "torch", "nerva", "nerva-0.2"):
        result["executionBackend"] = extra["executionBackend"]
    if type(extra.get("modelType")) is str and extra["modelType"] == "ia-propia-char-rnn":
        result["modelType"] = extra["modelType"]
    result["usedForTieBreak"] = result.get("scoreable") is not False and type(result.get("meanNll")) is float
    return result


def _auxiliary_key(candidate):
    diagnostic = candidate["metadata"].get("auxiliary", {})
    if not diagnostic.get("usedForTieBreak", False):
        return 1.0, float("inf")
    return diagnostic.get("unknownRatio", 1.0), diagnostic["meanNll"]


class ConversationEnsemble:
    """Lazy LRU engines and a serialized, cooperatively bounded request pipeline."""

    def __init__(self, specs, budget=None, primary_name=None, *,
                 engine_factory=ConversationEngine, auxiliary_scorer=None):
        if type(specs) not in (list, tuple) or not 2 <= len(specs) <= 64 or any(not isinstance(spec, ModelSpec) for spec in specs):
            raise ValueError("Coordination needs 2 to 64 host-configured own model specifications")
        if len({spec.name for spec in specs}) != len(specs) or len({spec.expected_model_id for spec in specs}) != len(specs):
            raise ValueError("Members must have distinct names and pinned model identities")
        self.budget = budget if budget is not None else EnsembleBudget()
        if not isinstance(self.budget, EnsembleBudget) or self.budget.min_candidates > len(specs):
            raise ValueError("The configured members must admit the minimum candidates")
        self._specs = OrderedDict((spec.name, spec) for spec in specs)
        self.primary_name = primary_name if primary_name is not None else specs[0].name
        if self.primary_name not in self._specs:
            raise ValueError("Primary must identify one configured member")
        if not callable(engine_factory) or (auxiliary_scorer is not None and not callable(auxiliary_scorer)):
            raise ValueError("Engine and optional auxiliary scorer must be host-provided callables")
        self._factory, self._auxiliary = engine_factory, auxiliary_scorer
        self._resident = OrderedDict()
        self._states = {name: "not_loaded" for name in self._specs}
        self._loads, self._evictions, self._peak_resident, self._rotation = 0, 0, 0, 0
        self._lock = threading.RLock()

    def _engine(self, name, trace=None):
        if name in self._resident:
            self._resident.move_to_end(name)
            return self._resident[name]
        spec = self._specs[name]
        started = perf_counter()
        try:
            if self._factory is ConversationEngine:
                _preflight(spec)
            if len(self._resident) >= self.budget.max_resident:
                evicted_name, evicted = self._resident.popitem(last=False)
                self._states[evicted_name] = "evicted"
                self._evictions += 1
                if callable(getattr(evicted, "close", None)):
                    evicted.close()
                del evicted
            engine = self._factory(spec.model_dir, decoder_backend=spec.decoder_backend, use_cache=spec.use_cache)
            if engine.model_id != spec.expected_model_id:
                raise EnsembleError("model_identity_mismatch")
            self._resident[name] = engine
            self._states[name] = "resident"
            self._loads += 1
            self._peak_resident = max(self._peak_resident, len(self._resident))
            return engine
        except Exception:
            self._states[name] = "failed"
            raise
        finally:
            if trace is not None:
                trace["loads"].append({"name": name, "seconds": perf_counter() - started,
                                       "state": self._states[name]})

    @property
    def primary_engine(self):
        with self._lock:
            return self._engine(self.primary_name)

    @property
    def _tokenizer(self):
        return self.primary_engine._tokenizer

    @property
    def model_id(self):
        return self._specs[self.primary_name].expected_model_id

    @property
    def manifest(self):
        with self._lock:
            if self._factory is ConversationEngine:
                return deepcopy(_preflight(self._specs[self.primary_name]))
            return deepcopy(self._engine(self.primary_name).manifest)

    @property
    def parameter_count(self):
        from .configuration import count_config_parameters
        return count_config_parameters(self.manifest["config"])

    @property
    def execution(self):
        spec = self._specs[self.primary_name]
        return {"neuralBackend": "torch", "decoderBackend": spec.decoder_backend,
                "kvCache": spec.use_cache, "coordination": "own_transformer_ensemble",
                "hostRuntimeRequired": "Node.js 18+" if spec.decoder_backend in ("nerva", "nerva-fast") else None}

    def cache_info(self):
        with self._lock:
            return {"residentNames": list(self._resident), "residentCount": len(self._resident),
                    "maximumResident": self.budget.max_resident, "peakResident": self._peak_resident,
                    "loads": self._loads, "evictions": self._evictions,
                    "states": dict(self._states)}

    def describe(self):
        """Report host configuration and observed cache, without loading weights."""
        with self._lock:
            availability = {member["name"]: member for member in self._availability()}
            members = self.catalog_state()
            for member in members:
                path = Path(self._specs[member["name"]].model_dir)
                member["artifactFilesPresent"] = all((path / filename).is_file() for filename in
                                                     ("manifest.json", "config.json", "tokenizer.json", "weights.safetensors"))
                member.update({key: value for key, value in availability[member["name"]].items()
                               if key not in ("name", "modelId")})
            return {"method": "own_transformer_ensemble", "primaryName": self.primary_name,
                    "primaryModelId": self.model_id, "memberCount": len(members),
                    "members": members, "budget": asdict(self.budget), "cache": self.cache_info(),
                    "cohortPolicy": {"method": "two_fixed_available_anchors_then_round_robin",
                                     "anchors": self._anchor_names(availability.values()),
                                     "rotatingSlots": max(0, self.budget.max_candidates - 2)},
                    "learnedSelector": False, "conversationValidated": False}

    def catalog_state(self):
        with self._lock:
            return [{"name": name, "modelId": spec.expected_model_id,
                     "declaredRole": spec.role, "state": self._states[name]} for name, spec in self._specs.items()]

    def prompt_usage(self, messages):
        with self._lock:
            return self._engine(self.primary_name).prompt_usage(messages)

    def score_reply(self, messages, text):
        # Compatibility diagnostic keeps its original single-primary semantics.
        with self._lock:
            return self._engine(self.primary_name).score_reply(messages, text)

    @staticmethod
    def _prepared(engine, messages, reserve):
        prepared = deepcopy(messages)
        minimum = [prepared[0], prepared[-1]]
        minimum_usage = engine.prompt_usage(minimum)
        if minimum_usage["tokens"] + reserve > minimum_usage["capacity"]:
            raise EnsembleError("minimum_context_exceeds_capacity")
        omitted = 0
        usage = engine.prompt_usage(prepared)
        while usage["tokens"] + reserve > usage["capacity"] and len(prepared) > 2:
            remove = 2 if len(prepared) > 3 and prepared[1]["role"] == "user" and prepared[2]["role"] == "assistant" else 1
            del prepared[1:1 + remove]
            omitted += remove
            usage = engine.prompt_usage(prepared)
        return prepared, {"promptTokens": usage["tokens"], "capacity": usage["capacity"],
                          "omittedMessages": omitted, "reservedTokens": reserve}

    def _availability(self):
        """Light host-side admission; absent catalog entries remain observable."""
        result = []
        for name, spec in self._specs.items():
            member = {"name": name, "modelId": spec.expected_model_id, "available": True,
                      "verification": "injected_host_factory"}
            if name in self._resident:
                member["verification"] = "resident_snapshot"
            elif self._factory is ConversationEngine:
                member["verification"] = "identity_and_header_only"
                try:
                    _preflight(spec)
                except Exception as exc:
                    member["available"] = False
                    member["reason"] = self._failure(name, "availability", exc)["reason"]
                    self._states[name] = "unavailable"
                else:
                    if self._states[name] == "unavailable":
                        self._states[name] = "not_loaded"
            result.append(member)
        return result

    def _scheduled(self, trace):
        availability = self._availability()
        trace["registryAvailability"] = availability
        for member in availability:
            if not member["available"]:
                trace["skips"].append({"name": member["name"], "stage": "availability",
                                       "reason": member["reason"]})
        available = {member["name"] for member in availability if member["available"]}
        anchors = self._anchor_names(availability)
        others = [name for name in self._specs if name not in anchors and name in available]
        policy = {"method": "two_fixed_available_anchors_then_round_robin",
                  "anchors": anchors, "rotating": []}
        trace["cohortPolicy"] = policy
        if not others:
            return anchors
        count = min(self.budget.max_candidates - len(anchors), len(others))
        selected = [others[(self._rotation + index) % len(others)] for index in range(count)]
        self._rotation = (self._rotation + count) % len(others)
        policy["rotating"] = selected
        return [*anchors, *selected]

    def _anchor_names(self, availability):
        available = {member["name"] for member in availability if member["available"]}
        ordered = ([self.primary_name] if self.primary_name in available else []) + [
            name for name in self._specs if name != self.primary_name and name in available]
        return ordered[:2]

    def _generate(self, name, values, quota, trace, deadline):
        engine = self._engine(name, trace)
        prepared, context = self._prepared(engine, values["messages"], quota)
        if perf_counter() >= deadline:
            raise EnsembleError("request_budget")
        started = perf_counter()
        if engine.model_id not in trace["modelIdsExecuted"]:
            trace["modelIdsExecuted"].append(engine.model_id)
        # A failed neural call may already have spent its entire allowance.
        # Reserve before calling and refund only a verified successful result.
        trace["budget"]["chargedNewTokens"] += quota
        reply = engine.reply(**{**values, "messages": prepared, "max_new_tokens": quota})
        if type(reply) is not dict or type(reply.get("text")) is not str or reply.get("modelId") != engine.model_id:
            raise EnsembleError("generation_result_invalid")
        tokens = reply.get("generatedTokens")
        if type(tokens) is not int or not 0 <= tokens <= quota:
            raise EnsembleError("generation_token_budget_invalid")
        trace["budget"]["chargedNewTokens"] -= quota - tokens
        metadata = {"name": name, "modelId": engine.model_id, "declaredRole": self._specs[name].role,
            "textSha256": hashlib.sha256(reply["text"].encode("utf-8")).hexdigest(),
            "textCharacters": len(reply["text"]), "generatedTokens": tokens,
            "stopReason": reply.get("stopReason"), "seconds": perf_counter() - started,
            "context": context, "empty": not bool(reply["text"].strip())}
        trace["candidates"].append(metadata)
        return {"name": name, "reply": reply, "metadata": metadata}

    def _score_critic(self, name, candidates, values, deadline, remaining, trace):
        engine = self._engine(name, trace)
        scorer = {"name": name, "modelId": engine.model_id, "complete": False, "scores": []}
        trace["critics"].append(scorer)
        # The actual critic tokenizer controls the work estimate, including EOS.
        counts = [len(engine._tokenizer.encode(candidate["reply"]["text"], add_special_tokens=False).ids) + 1
                  for candidate in candidates]
        if max(counts) > self.budget.max_scoring_tokens:
            raise EnsembleError("scoring_target_token_budget")
        reserve = max(counts) - self.budget.max_sliding_score_tokens
        prepared, context = self._prepared(engine, values["messages"], max(1, reserve))
        scorer["context"] = context
        used = 0
        for candidate, target_count in zip(candidates, counts):
            if perf_counter() >= deadline or used >= remaining:
                break
            started = perf_counter()
            used += 1
            trace["budget"]["scoreCalls"] += 1
            try:
                if engine.model_id not in trace["modelIdsExecuted"]:
                    trace["modelIdsExecuted"].append(engine.model_id)
                result = engine.score_reply(prepared, candidate["reply"]["text"])
                if type(result) is not dict:
                    raise EnsembleError("scoring_result_invalid")
                nll = result.get("meanNll")
                if type(nll) not in (int, float) or not math.isfinite(nll) or nll < 0:
                    raise EnsembleError("scoring_result_invalid")
                sliding = result.get("targetsWithSlidingPrompt", 0)
                target_tokens = result.get("targetTokens", target_count)
                if type(sliding) is not int or sliding < 0 or type(target_tokens) is not int or target_tokens != target_count:
                    raise EnsembleError("scoring_result_invalid")
                if sliding > self.budget.max_sliding_score_tokens:
                    raise EnsembleError("scoring_sliding_budget")
                total = -result.get("totalLogProbability", -nll * target_count)
                if type(total) not in (int, float) or not math.isfinite(total) or total < 0:
                    raise EnsembleError("scoring_result_invalid")
                row = {"candidate": candidate["name"], "meanNll": float(nll),
                    "characterNormalizedNll": total / max(1, len(candidate["reply"]["text"])),
                    "targetTokens": target_tokens,
                    "includesEos": True, "seconds": perf_counter() - started,
                    "discardedPromptTokens": result.get("discardedPromptTokens", 0),
                    "targetsWithSlidingPrompt": result.get("targetsWithSlidingPrompt", 0)}
                scorer["scores"].append(row)
                trace["budget"]["scoredTargetTokens"] += row["targetTokens"]
            except Exception as exc:
                trace["failed"].append(self._failure(name, "scoring", exc, candidate["name"]))
        scorer["complete"] = len(scorer["scores"]) == len(candidates)
        if scorer["complete"]:
            ordered = sorted(scorer["scores"], key=lambda row: row["characterNormalizedNll"])
            for row in ordered:
                before = sum(other["characterNormalizedNll"] < row["characterNormalizedNll"] for other in ordered)
                equal = sum(other["characterNormalizedNll"] == row["characterNormalizedNll"] for other in ordered)
                row["rankCost"] = (before + (equal - 1) / 2) / max(1, len(ordered) - 1)
        return used

    @staticmethod
    def _failure(name, stage, exc, candidate=None):
        reason = getattr(exc, "reason", None)
        result = {"name": name, "stage": stage, "reason": reason if type(reason) is str and reason in _FAILURE_REASONS else "operation_failed",
                  "errorType": type(exc).__name__}
        if candidate is not None:
            result["candidate"] = candidate
        return result

    def reply(self, messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
              temperature=DEFAULT_TEMPERATURE, seed=DEFAULT_SEED, *,
              top_k=0, top_p=1.0, repetition_penalty=1.0, context_sources=None):
        values = validate_reply_request({"messages": messages, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "seed": seed, "top_k": top_k, "top_p": top_p,
            "repetition_penalty": repetition_penalty})
        from .tokenizer import prepare_messages
        values["messages"] = prepare_messages(values["messages"])
        source_terms = _source_terms(values["messages"], context_sources)
        with self._lock:
            started = perf_counter()
            deadline = started + self.budget.max_seconds
            trace = {"method": "explicit_cross_model_rank_and_lexical_support",
                "learnedSelector": False, "conversationValidated": False,
                "candidateTextsIncluded": False, "modelIdsExecuted": [], "selectedModelId": None,
                "candidates": [], "critics": [], "loads": [], "skips": [], "failed": [],
                "budget": {"maxSeconds": self.budget.max_seconds, "timeLimit": "cooperative_between_operations",
                    "totalNewTokens": self.budget.total_new_tokens, "generatedTokens": 0,
                    "chargedNewTokens": 0, "failedGenerationCost": "full_reserved_allowance_when_unknown",
                    "maxScoreCalls": self.budget.max_score_calls, "scoreCalls": 0, "scoredTargetTokens": 0}}
            scheduled = self._scheduled(trace)
            candidates = []
            for index, name in enumerate(scheduled):
                remaining = self.budget.total_new_tokens - trace["budget"]["chargedNewTokens"]
                if perf_counter() >= deadline or remaining < 1:
                    trace["skips"].append({"name": name, "stage": "generation", "reason": "request_budget"})
                    continue
                quota = min(values["max_new_tokens"], max(1, remaining // (len(scheduled) - index)))
                try:
                    candidate = self._generate(name, values, quota, trace, deadline)
                    candidates.append(candidate)
                    trace["budget"]["generatedTokens"] += candidate["reply"]["generatedTokens"]
                except Exception as exc:
                    if getattr(exc, "reason", None) == "request_budget":
                        trace["skips"].append({"name": name, "stage": "generation", "reason": "request_budget"})
                    else:
                        trace["failed"].append(self._failure(name, "generation", exc))
            trace["minimumCandidatesSatisfied"] = len(candidates) >= self.budget.min_candidates
            if not trace["minimumCandidatesSatisfied"]:
                self._finish_trace(trace, started)
                raise EnsembleError("minimum_successful_candidates_not_met", trace)
            usable = [candidate for candidate in candidates if not candidate["metadata"]["empty"]]
            if not usable:
                self._finish_trace(trace, started)
                raise EnsembleError("no_nonempty_generated_candidate", trace)
            # Score resident successful generators first, one whole critic block
            # at a time, to avoid repeatedly loading the same pair of weights.
            names = [candidate["name"] for candidate in usable]
            fixed_critics = names[:self.budget.critic_count]
            trace["plannedCriticModelIds"] = [self._specs[name].expected_model_id for name in fixed_critics]
            admitted = fixed_critics[:self.budget.max_score_calls // len(usable)]
            for name in fixed_critics[len(admitted):]:
                trace["skips"].append({"name": name, "stage": "scoring", "reason": "request_budget"})
            resident = list(reversed(self._resident))
            critics = [name for name in resident if name in admitted] + [name for name in admitted if name not in resident]
            for name in critics:
                remaining = self.budget.max_score_calls - trace["budget"]["scoreCalls"]
                if perf_counter() >= deadline or remaining < len(usable):
                    trace["skips"].append({"name": name, "stage": "scoring", "reason": "request_budget"})
                    continue
                try:
                    self._score_critic(name, usable, values, deadline, remaining, trace)
                except Exception as exc:
                    trace["failed"].append(self._failure(name, "scoring", exc))
            shared = [critic for critic in trace["critics"] if critic["complete"]]
            trace["sharedCritics"] = [critic["modelId"] for critic in shared]
            for candidate in usable:
                ranks = [next(row["rankCost"] for row in critic["scores"] if row["candidate"] == candidate["name"])
                         for critic in shared]
                terms = _terms(candidate["reply"]["text"])
                support = len(terms & source_terms) / max(1, len(terms))
                cost = math.fsum(ranks) / len(ranks) if ranks else 0.0
                candidate["metadata"].update(criticRankCost=cost if ranks else None,
                    lexicalSourceSupport=support, selectionCost=cost - 0.25 * support)
                if self._auxiliary is not None and perf_counter() < deadline:
                    try:
                        auxiliary_started = perf_counter()
                        extra = self._auxiliary(candidate["reply"]["text"])
                        candidate["metadata"]["auxiliary"] = _auxiliary_metadata(extra)
                        candidate["metadata"]["auxiliary"]["seconds"] = perf_counter() - auxiliary_started
                    except Exception as exc:
                        trace["failed"].append(self._failure(candidate["name"], "auxiliary", exc))
                elif self._auxiliary is not None:
                    trace["skips"].append({"name": candidate["name"], "stage": "auxiliary", "reason": "request_budget"})
            winner = min(usable, key=lambda candidate: (
                candidate["metadata"]["selectionCost"],
                *_auxiliary_key(candidate),
                names.index(candidate["name"])))
            trace["selectionRule"] = {"criticAggregation": "mean_within_critic_character_nll_rank",
                "sourceSupportWeight": 0.25, "sourceSupportIsLexical": True,
                "auxiliary": "diagnostic_and_tie_break_only", "completeSharedCritics": len(shared),
                "unscoredSelection": not bool(shared)}
            trace["selectedModelId"] = winner["reply"]["modelId"]
            self._finish_trace(trace, started)
            return {**deepcopy(winner["reply"]), "ensemble": trace}

    def _finish_trace(self, trace, started):
        elapsed = perf_counter() - started
        trace["budget"].update(elapsedSeconds=elapsed,
            timeBudgetExceeded=elapsed > self.budget.max_seconds,
            secondsOverBudget=max(0.0, elapsed - self.budget.max_seconds))
        trace["cache"] = self.cache_info()

    def close(self):
        with self._lock:
            while self._resident:
                name, engine = self._resident.popitem()
                if callable(getattr(engine, "close", None)):
                    engine.close()
                self._states[name] = "evicted"
