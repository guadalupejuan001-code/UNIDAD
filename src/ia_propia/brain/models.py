"""Local adapters shared by every Unidad body, with explicitly selected weights.

Only trusted host construction may choose model files. Requests contain values,
never paths. Optional conversation dependencies are loaded only by host selection.
"""
from copy import deepcopy
import hashlib
from importlib import resources
import json
from pathlib import Path

import numpy as np

from ..language import model as language
from ..trend import data as attention_data
from ..trend import model as attention


class ModelRuntimeError(ValueError):
    """Invalid model configuration or model-operation request."""


class ModelUnavailableError(ModelRuntimeError):
    """A known optional model was not configured by the trusted host."""


LANGUAGE_KEYS = {"prompt", "length", "temperature", "seed"}
ATTENTION_KEYS = {"observations", "signal_id", "as_of", "historical", "source", "metric"}
INPUT_SCHEMAS = {
    "language.generate": {"type": "object", "additionalProperties": False, "properties": {
        "prompt": {"type": "string", "maxLength": 4096, "default": ""},
        "length": {"type": "integer", "minimum": 0, "maximum": 4096, "default": 200},
        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.8},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1, "default": 42}}},
    "attention.forecast": {"type": "object", "additionalProperties": False, "required": ["observations"], "properties": {
        "observations": {"oneOf": [{"type": "array"}, {"type": "object"}]},
        "signal_id": {"type": "string", "minLength": 1},
        "as_of": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1},
        "historical": {"type": "boolean", "default": False},
        "source": {"type": "string", "minLength": 1}, "metric": {"type": "string", "minLength": 1}}}}


def _strict_json(raw):
    def distinct(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ModelRuntimeError(f"Duplicate asset key: {key}")
            value[key] = item
        return value
    def invalid(value):
        raise ModelRuntimeError(f"Nonfinite asset constant: {value}")
    return json.loads(raw, object_pairs_hook=distinct, parse_constant=invalid)


def _packaged_models(selected):
    root = resources.files("ia_propia.assets")
    manifest = _strict_json(root.joinpath("manifest.json").read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or type(manifest["schemaVersion"]) is not int or not isinstance(manifest.get("models"), dict):
        raise ModelRuntimeError("Invalid packaged-model manifest")
    result = {}
    for name, filename, module in (("language", "language-demo.json", language), ("attention", "attention-v1.json", attention)):
        if name not in selected:
            continue
        descriptor = manifest["models"].get(name)
        if not isinstance(descriptor, dict) or descriptor.get("file") != filename:
            raise ModelRuntimeError(f"Missing packaged {name} descriptor")
        raw = root.joinpath(filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor.get("fileSha256"):
            raise ModelRuntimeError(f"Packaged {name} file hash differs from its manifest")
        artifact = module.validate_artifact(_strict_json(raw))
        if artifact["modelId"] != descriptor.get("modelId") or artifact["modelType"] != descriptor.get("modelType"):
            raise ModelRuntimeError(f"Packaged {name} model identity differs from its manifest")
        if type(descriptor.get("parameterCount")) is not int or _parameter_count(artifact, name) != descriptor["parameterCount"]:
            raise ModelRuntimeError(f"Packaged {name} parameter count differs from its manifest")
        result[name] = artifact
    return result


def _parameter_count(artifact, kind):
    if kind == "language":
        return sum(np.asarray(values).size for values in artifact["weights"].values())
    return sum(np.asarray(artifact[key]).size for key in ("weightsInput", "biasHidden", "weightsOutput")) + 1


def _is_reference(value, allow_refs):
    if not allow_refs or not isinstance(value, dict) or "$ref" not in value:
        return False
    if set(value) != {"$ref"} or not isinstance(value["$ref"], str) or not value["$ref"].strip():
        raise ModelRuntimeError("A reference must be exactly {'$ref': nonempty string}")
    return True


class ModelRuntime:
    """Loads host-selected models once; operations cannot replace their weights."""

    def __init__(self, language_model=None, attention_model=None, conversation_model=None):
        defaults = [name for name, selected in (("language", language_model), ("attention", attention_model)) if selected is None]
        packaged = _packaged_models(defaults) if defaults else {}
        self._language = packaged["language"] if language_model is None else language.load_artifact(Path(language_model))
        self._attention = packaged["attention"] if attention_model is None else attention.load_artifact(Path(attention_model))
        self._conversation = None
        self._capabilities = {
            "schemaVersion": 1, "execution": "local_cpu", "networkRequired": False,
            "experimental": True, "productionAccepted": False, "workerCapabilityValidated": False,
            "operations": list(INPUT_SCHEMAS), "inputSchemas": INPUT_SCHEMAS,
            "models": {
                "language": {"available": True, "modelId": self._language["modelId"],
                    "parameterCount": _parameter_count(self._language, "language"), "task": "next_character_generation",
                    "instructionFollowingValidated": False, "generalLanguageUnderstandingValidated": False},
                "attention": {"available": True, "modelId": self._attention["modelId"],
                    "parameterCount": _parameter_count(self._attention, "attention"), "task": "scoped_metric_point_forecast",
                    "source": self._attention["source"], "metric": self._attention["metric"],
                    "privacy": "PUBLIC", "liveForecastValidated": False}}}
        if conversation_model is not None:
            if not isinstance(conversation_model, (str, Path)):
                raise ModelRuntimeError("The conversation artifact must be a host-configured local path")
            # Optional dependencies and scratch-trained weights are loaded only
            # when explicitly selected. Legacy bodies never import torch here.
            from ..conversation.runtime import ConversationEngine, REPLY_INPUT_SCHEMA
            self._conversation = ConversationEngine(conversation_model)
            self._capabilities["operations"].append("conversation.reply")
            self._capabilities["inputSchemas"] = deepcopy(INPUT_SCHEMAS)
            self._capabilities["inputSchemas"]["conversation.reply"] = deepcopy(REPLY_INPUT_SCHEMA)
            self._capabilities["models"]["conversation"] = {
                "available": True, "modelId": self._conversation.model_id,
                "parameterCount": self._conversation.parameter_count,
                "task": "conversation_token_generation", "instructionFollowingValidated": False,
                "generalLanguageUnderstandingValidated": False, "workerCapabilityValidated": False}

    def capabilities(self):
        """A detached JSON description, never mutable model parameters."""
        return deepcopy(self._capabilities)

    def validate_input(self, operation, inputdict, *, allow_refs=False):
        """Validate literals before work, optionally allowing whole-field graph refs.

        Reference expressions and graph dependencies belong to the workflow
        resolver. After resolution execute validates the actual values again.
        """
        self._validate_input(operation, inputdict, allow_refs=allow_refs)

    def _validate_input(self, operation, inputdict, *, allow_refs=False):
        if operation == "conversation.reply":
            return self._validate_conversation(inputdict, allow_refs=allow_refs)
        if not isinstance(operation, str) or operation not in INPUT_SCHEMAS:
            raise ModelRuntimeError(f"Unsupported model operation: {operation}")
        if not isinstance(inputdict, dict) or any(not isinstance(key, str) for key in inputdict):
            raise ModelRuntimeError("Model input must be an object with string keys")
        if operation == "language.generate":
            if set(inputdict) - LANGUAGE_KEYS:
                raise ModelRuntimeError("Unknown language input field")
            for key, value in inputdict.items():
                if _is_reference(value, allow_refs):
                    continue
                if key == "prompt":
                    if not isinstance(value, str) or len(value) > 4096:
                        raise ModelRuntimeError("prompt must be a string of at most 4096 characters")
                elif key == "length":
                    language.integer(value, "length", 0, 4096)
                elif key == "seed":
                    language.integer(value, "seed")
                else:
                    if not 0 <= language.number(value, "temperature") <= 2:
                        raise ModelRuntimeError("temperature must be in [0, 2]")
        else:
            if set(inputdict) - ATTENTION_KEYS or "observations" not in inputdict:
                raise ModelRuntimeError("attention requires observations and only known input fields")
            for key, value in inputdict.items():
                if _is_reference(value, allow_refs):
                    continue
                if key in ("signal_id", "source", "metric"):
                    attention_data.nonempty(value, key)
                    if key in ("source", "metric") and value != self._attention[key]:
                        raise ModelRuntimeError(f"Requested {key} does not match the loaded attention model")
                elif key == "as_of":
                    attention_data.timestamp(value, "as_of")
                elif key == "historical":
                    if type(value) is not bool:
                        raise ModelRuntimeError("historical must be an explicit boolean")
                elif not isinstance(value, (dict, list)):
                    raise ModelRuntimeError("observations must contain raw Observation JSON, not a path")
            if not _is_reference(inputdict["observations"], allow_refs):
                observations, _ = attention_data.observations_from_payload(inputdict["observations"],
                    source=self._attention["source"], metric=self._attention["metric"])
                return observations
        return None

    def _validate_conversation(self, values, *, allow_refs=False):
        if self._conversation is None:
            raise ModelUnavailableError("Configure a local conversation model to enable conversation.reply")
        from ..conversation.runtime import REPLY_KEYS, validate_reply_request
        if type(values) is not dict or set(values) - REPLY_KEYS or "messages" not in values:
            raise ModelRuntimeError("conversation.reply requires messages and only known input fields")
        literals = {key: value for key, value in values.items() if not _is_reference(value, allow_refs)}
        validate_reply_request(literals, partial=allow_refs)

    def execute(self, operation, inputdict):
        """Dispatch pure local inference using fixed, previously loaded artifacts."""
        if not isinstance(inputdict, dict):
            raise ModelRuntimeError("Model input must be an object")
        request = deepcopy(inputdict)
        observations = self._validate_input(operation, request)
        if operation == "conversation.reply":
            return self._conversation.reply(**request)
        if operation == "language.generate":
            return language.generate(self._language, **request)
        signal_id = request.get("signal_id")
        if signal_id is not None:
            observations = [row for row in observations if row.signal_id == signal_id]
        return {"schemaVersion": 1, "forecasts": attention.predict(self._attention, observations,
            as_of=request.get("as_of"), historical=request.get("historical", False))}
