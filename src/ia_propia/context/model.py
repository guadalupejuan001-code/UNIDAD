"""Small learned relevance ranker; it selects source text and never generates facts.

Features are explicitly lexical and temporal. Learned weights are independent
of the conversational Transformer and do not grant tools or filesystem access.
"""

from copy import deepcopy
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re
import unicodedata

MODEL_TYPE = "context-linear-ranker-v1"
FEATURE_NAMES = (
    "query_overlap", "query_coverage", "thread_overlap", "thread_coverage",
    "recency", "user_source", "correction", "goal", "constraint", "question",
    "length", "short_query_thread", "query_correction", "query_goal",
    "query_constraint", "thread_correction", "thread_goal", "thread_constraint",
    "query_complete", "thread_complete",
)
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset("a al algo como con cual cuando de del el ella en es esa ese eso esta este esto hay la las lo los me mi mis no o para pero por que se si sin su sus te tu tus un una unas unos y yo".split())
_KINDS = {
    "correction": ("corrijo", "correccion", "rectifico", "en realidad", "cambio de", "no era", "ahora es"),
    "goal": ("quiero", "necesito", "objetivo", "vamos a", "estamos", "trabajamos", "mi plan"),
    "constraint": ("no publiques", "no modifiques", "no compartas", "no guardes", "no uses", "evita", "mantener", "manten", "solo ", "unicamente", "sin permiso"),
}


def _normalize(text):
    return "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(char) != "Mn")


def terms(text):
    return frozenset(word for word in _WORDS.findall(_normalize(text))
                     if len(word) > 1 and word not in _STOP)


def _text(value, name, limit):
    if type(value) is not str or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} must be bounded UTF-8 text")
    return value


def _candidates(candidates):
    if type(candidates) is not list or len(candidates) > 512:
        raise ValueError("Context candidates must be a list of at most 512 sources")
    for candidate in candidates:
        if type(candidate) is not dict or "text" not in candidate:
            raise ValueError("Each context candidate requires source text")
        if not _text(candidate["text"], "candidate text", 16384).strip():
            raise ValueError("Context source text must be nonempty")
        if candidate.get("role", "user") not in ("user", "assistant", "system"):
            raise ValueError("Context source role must be explicit")
        index = candidate.get("turn_index", 0)
        if type(index) is not int or not 0 <= index <= 10**12:
            raise ValueError("Context turn_index must be a nonnegative integer")
        _text(candidate.get("kind", ""), "candidate kind", 64)
    return candidates


def features(query, candidate, *, thread="", max_turn=0):
    """Expose the fixed, bounded features used during both training and inference."""
    return _features_from_terms(terms(query), terms(thread), candidate, max_turn)


def _features_from_terms(query_terms, thread_terms, candidate, max_turn):
    source_terms = terms(candidate["text"])
    query_hits = len(query_terms & source_terms)
    thread_hits = len(thread_terms & source_terms)
    q = query_hits / max(1, len(query_terms))
    t = thread_hits / max(1, len(thread_terms))
    normalized = _normalize(candidate["text"])
    flags = {kind: float(candidate.get("kind") == kind or any(marker in normalized for marker in markers))
             for kind, markers in _KINDS.items()}
    values = (
        query_hits / max(1, len(query_terms | source_terms)), q,
        thread_hits / max(1, len(thread_terms | source_terms)), t,
        candidate.get("turn_index", 0) / max(1, max_turn),
        float(candidate.get("role", "user") == "user"),
        flags["correction"], flags["goal"], flags["constraint"],
        float("?" in candidate["text"] or "¿" in candidate["text"]),
        min(len(source_terms), 64) / 64,
        t * float(len(query_terms) <= 2),
        q * flags["correction"], q * flags["goal"], q * flags["constraint"],
        t * flags["correction"], t * flags["goal"], t * flags["constraint"],
        float(bool(query_terms) and query_hits == len(query_terms)),
        float(bool(thread_terms) and thread_hits == len(thread_terms)),
    )
    return values


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def artifact_identity(artifact):
    payload = {key: value for key, value in artifact.items() if key not in ("artifactSha256", "modelId")}
    return hashlib.sha256(_canonical(payload)).hexdigest()


class ContextRanker:
    """A local linear classifier with its own trained parameters and provenance."""

    def __init__(self, weights, bias, *, metadata=None):
        if (type(weights) not in (tuple, list) or len(weights) != len(FEATURE_NAMES)
                or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1000000 for value in weights)
                or type(bias) not in (int, float) or not math.isfinite(bias) or abs(bias) > 1000000):
            raise ValueError("Context ranker requires one finite weight per feature and a finite bias")
        self._weights = tuple(float(value) for value in weights)
        self._bias = float(bias)
        if metadata is not None and type(metadata) is not dict:
            raise ValueError("Context ranker metadata must be a provenance object")
        self._metadata = deepcopy(metadata or {})

    @property
    def parameter_count(self):
        return len(self._weights) + 1

    @property
    def model_id(self):
        return self.artifact()["modelId"]

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def score_candidates(self, query, candidates, *, thread=""):
        _text(query, "query", 16384)
        _text(thread, "thread", 32768)
        _candidates(candidates)
        max_turn = max((candidate.get("turn_index", 0) for candidate in candidates), default=0)
        query_terms, thread_terms = terms(query), terms(thread)
        result = []
        for candidate in candidates:
            values = _features_from_terms(query_terms, thread_terms, candidate, max_turn)
            score = sigmoid(self._bias + sum(weight * value for weight, value in zip(self._weights, values)))
            result.append({**deepcopy(candidate), "score": score,
                           "featureValues": dict(zip(FEATURE_NAMES, values))})
        return result

    def score(self, query, candidate, *, candidates=None, thread=""):
        sources = [candidate] if candidates is None else candidates
        _text(query, "query", 16384)
        _text(thread, "thread", 32768)
        _candidates(sources)
        _candidates([candidate])
        maximum = max((source.get("turn_index", 0) for source in sources), default=0)
        values = features(query, candidate, thread=thread, max_turn=maximum)
        return sigmoid(self._bias + sum(weight * value for weight, value in zip(self._weights, values)))

    def artifact(self):
        result = {"schemaVersion": 1, "modelType": MODEL_TYPE, "featureNames": list(FEATURE_NAMES),
                  "weights": list(self._weights), "bias": self._bias, "parameterCount": self.parameter_count,
                  "metadata": deepcopy(self._metadata), "experimental": True,
                  "generalUnderstandingValidated": False, "generatesAnswers": False}
        digest = artifact_identity(result)
        return {**result, "artifactSha256": digest, "modelId": "unidad-context-" + digest}

    @classmethod
    def from_artifact(cls, artifact):
        keys = {"schemaVersion", "modelType", "featureNames", "weights", "bias", "parameterCount", "metadata",
                "experimental", "generalUnderstandingValidated", "generatesAnswers", "artifactSha256", "modelId"}
        if (type(artifact) is not dict or set(artifact) != keys
                or type(artifact["schemaVersion"]) is not int or artifact["schemaVersion"] != 1
                or artifact["modelType"] != MODEL_TYPE or artifact["featureNames"] != list(FEATURE_NAMES)
                or artifact["experimental"] is not True or artifact["generalUnderstandingValidated"] is not False
                or artifact["generatesAnswers"] is not False or type(artifact["metadata"]) is not dict):
            raise ValueError("Invalid independent context-ranker artifact")
        digest = artifact_identity(artifact)
        if artifact["artifactSha256"] != digest or artifact["modelId"] != "unidad-context-" + digest:
            raise ValueError("Context-ranker artifact hash or identity mismatch")
        model = cls(artifact["weights"], artifact["bias"], metadata=artifact["metadata"])
        if type(artifact["parameterCount"]) is not int or artifact["parameterCount"] != model.parameter_count:
            raise ValueError("Context-ranker parameter count mismatch")
        return model


def load_ranker(path=None):
    """Load explicitly configured weights, or the packaged trained reference."""
    if path is None:
        raw = resources.files("ia_propia.assets").joinpath("context-ranker-v1.json").read_bytes()
    else:
        with Path(path).open("rb") as stream:
            raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Context-ranker artifact exceeds its byte limit")
    def distinct(pairs):
        value = {}
        for key, entry in pairs:
            if key in value:
                raise ValueError("Duplicate context-ranker artifact key")
            value[key] = entry
        return value
    artifact = json.loads(raw.decode("utf-8"), object_pairs_hook=distinct,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite artifact number")))
    return ContextRanker.from_artifact(artifact)
