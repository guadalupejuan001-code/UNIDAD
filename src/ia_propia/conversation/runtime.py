"""One local conversation engine shared by product adapters and assessment.

Importing this facade does not import optional ML dependencies. Artifact loading,
validation and inference stay explicit; messages never choose files or tools.
"""

from copy import deepcopy
from pathlib import Path

from .contract import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, DEFAULT_TEMPERATURE,
                       MAX_MESSAGES, MAX_NEW_TOKENS, ROLE_IDS)

REPLY_KEYS = frozenset({"messages", "max_new_tokens", "temperature", "seed", "top_k", "top_p", "repetition_penalty"})
REPLY_INPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["messages"], "properties": {
    "messages": {"type": "array", "minItems": 1, "maxItems": MAX_MESSAGES, "items": {
        "type": "object", "additionalProperties": False, "required": ["role", "content"], "properties": {
            "role": {"enum": list(ROLE_IDS)}, "content": {"type": "string"}}}},
    "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": MAX_NEW_TOKENS, "default": DEFAULT_MAX_NEW_TOKENS},
    "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": DEFAULT_TEMPERATURE},
    "top_k": {"type": "integer", "minimum": 0, "maximum": 65536, "default": 0},
    "top_p": {"type": "number", "minimum": 0.000001, "maximum": 1, "default": 1},
    "repetition_penalty": {"type": "number", "minimum": 1, "maximum": 2, "default": 1},
    "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": DEFAULT_SEED}}}


def validate_reply_request(values, *, partial=False):
    """Validate and detach reply values; partial mode checks present literals.

    Partial input is used only for workflow preflight: the brain retains and
    validates reference expressions, then validates all resolved values again.
    """
    if type(partial) is not bool:
        raise ValueError("partial must be boolean")
    if type(values) is not dict or set(values) - REPLY_KEYS or (not partial and "messages" not in values):
        raise ValueError("Conversation reply requires messages and only known input fields")
    from .tokenizer import integer, number, validate_messages
    result = {}
    for key, value in values.items():
        if key == "messages":
            result[key] = validate_messages(value)
            if result[key][-1]["role"] != "user":
                raise ValueError("Conversation messages must end with an explicit user turn")
        elif key == "max_new_tokens":
            result[key] = integer(value, key, 1, MAX_NEW_TOKENS)
        elif key == "seed":
            result[key] = integer(value, key)
        elif key in ("top_k", "top_p", "repetition_penalty"):
            from .decoding import validate_controls
            result[key] = validate_controls(**{key: value})[key]
        else:
            result[key] = number(value, key, 0, 2)
    if not partial:
        result = {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "temperature": DEFAULT_TEMPERATURE,
                  "seed": DEFAULT_SEED, **result}
    return result


class ConversationEngine:
    """Load one verified local artifact once and use the same weights for every call."""

    def __init__(self, model_dir, *, decoder_backend="torch", use_cache=True):
        if not isinstance(model_dir, (str, Path)):
            raise ValueError("Conversation model must be a host-configured local artifact directory")
        if type(decoder_backend) is not str or decoder_backend not in ("torch", "nerva") or type(use_cache) is not bool:
            raise ValueError("Host must select a known conversation decoder and boolean use_cache")
        self._decoder_backend = decoder_backend
        self._use_cache = use_cache
        from .artifacts import load_artifact
        self._model, self._tokenizer, self._manifest = load_artifact(Path(model_dir))

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @property
    def model_id(self):
        return self._manifest["modelId"]

    @property
    def parameter_count(self):
        return self._model.parameter_count

    @property
    def execution(self):
        return {"neuralBackend": "torch", "decoderBackend": self._decoder_backend,
                "kvCache": self._use_cache, "hostRuntimeRequired": "Node.js 18+" if self._decoder_backend == "nerva" else None}

    def reply(self, messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
              temperature=DEFAULT_TEMPERATURE, seed=DEFAULT_SEED, *,
              top_k=0, top_p=1.0, repetition_penalty=1.0):
        values = validate_reply_request({"messages": messages, "max_new_tokens": max_new_tokens,
                                         "temperature": temperature, "seed": seed,
                                         "top_k": top_k, "top_p": top_p, "repetition_penalty": repetition_penalty})
        from .model import chat
        result = chat(self._model, self._tokenizer, decoder_backend=self._decoder_backend,
                      use_cache=self._use_cache, **values)
        return {**result, "modelId": self.model_id, "workerCapabilityValidated": False}

    def score_reply(self, messages, text):
        from .scoring import score_reply
        return score_reply(self._model, self._tokenizer, messages, text)
