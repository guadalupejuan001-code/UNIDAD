"""Hash-verified portable Transformer directories; inference never reads pickle."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors.torch import load as load_tensors, save as save_tensors
from tokenizers import Tokenizer

from .model import ConversationModel, MODEL_TYPE, validate_config
from .tokenizer import ConversationError, validate_tokenizer


SCHEMA_VERSION = 1
ARTIFACT_FILES = ("config.json", "tokenizer.json", "weights.safetensors")
MAX_JSON_BYTES = 4_194_304
MAX_WEIGHT_BYTES = 268_435_456
MANIFEST_KEYS = {"schemaVersion", "modelType", "modelId", "config", "files", "training", "evaluation"}


def _json_bytes(value):
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ConversationError("Artifact metadata must be finite UTF-8 JSON") from error
    if len(raw) > MAX_JSON_BYTES:
        raise ConversationError("Artifact JSON exceeds the byte limit")
    return raw


def _parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConversationError("Artifact JSON contains duplicate object keys")
            result[key] = value
        return result

    def invalid(_value):
        raise ConversationError("Artifact JSON must contain finite numbers")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
        _json_bytes(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ConversationError("Cannot decode bounded artifact JSON") from error
    return value


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def artifact_model_id(files):
    identity = {"schemaVersion": SCHEMA_VERSION, "modelType": MODEL_TYPE,
                "files": {name: files[name]["sha256"] for name in ARTIFACT_FILES}}
    return "unidad-conversation-" + _digest(_json_bytes(identity))


def _bounded_read(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ConversationError("Artifact members must be regular files, not symlinks")
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise ConversationError("Artifact member exceeds its byte limit or is empty")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != size or len(raw) > limit:
        raise ConversationError("Artifact member changed while being read or exceeds its limit")
    return raw


def _write_atomic(path, raw):
    descriptor, temporary = tempfile.mkstemp(prefix=".unidad-artifact-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_artifact(directory, model, tokenizer, *, training=None, evaluation=None):
    """Export fixed filenames and a final manifest; no executable serialization.

    An existing experimental checkpoint may be replaced. The manifest is written
    last, so a concurrent reader rejects intermediate hash mismatches rather than
    accepting a mixed checkpoint. Hosts should load a completed export once.
    """
    if not isinstance(model, ConversationModel):
        raise ConversationError("Expected a ConversationModel for artifact export")
    validate_tokenizer(tokenizer, strict=True)
    config = validate_config(model.config)
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        raise ConversationError("Tokenizer vocabulary differs from exported model")
    if training is not None and type(training) is not dict:
        raise ConversationError("training metadata must be an object")
    if evaluation is not None and type(evaluation) is not dict:
        raise ConversationError("evaluation metadata must be an object")
    tensors = {}
    for name, value in model.state_dict().items():
        if value.dtype != torch.float32 or not torch.all(torch.isfinite(value)):
            raise ConversationError("Exported parameters must be finite float32 tensors")
        tensors[name] = value.detach().cpu().contiguous().clone()
    payloads = {"config.json": _json_bytes(config), "tokenizer.json": tokenizer.to_str().encode("utf-8"),
                "weights.safetensors": save_tensors(tensors)}
    if len(payloads["tokenizer.json"]) > MAX_JSON_BYTES or len(payloads["weights.safetensors"]) > MAX_WEIGHT_BYTES:
        raise ConversationError("Exported tokenizer or weights exceed artifact byte limits")
    files = {name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in payloads.items()}
    manifest = {"schemaVersion": SCHEMA_VERSION, "modelType": MODEL_TYPE, "modelId": artifact_model_id(files),
                "config": config, "files": files,
                "training": {**(training or {}), "parameterCount": model.parameter_count},
                "evaluation": {**(evaluation or {}), "experimental": True, "productionAccepted": False}}
    encoded_manifest = _json_bytes(manifest)
    target = Path(directory)
    if target.is_symlink():
        raise ConversationError("Artifact directory cannot be a symlink")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.is_dir():
        raise ConversationError("Artifact output must be a directory")
    for name, raw in payloads.items():
        if (target / name).is_symlink():
            raise ConversationError("Artifact members cannot replace symlinks")
        _write_atomic(target / name, raw)
    if (target / "manifest.json").is_symlink():
        raise ConversationError("Artifact manifest cannot replace a symlink")
    _write_atomic(target / "manifest.json", encoded_manifest)
    return manifest


def load_artifact(directory):
    """Return (CPU eval model, local tokenizer, manifest), verifying all bytes first."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ConversationError("Expected a regular portable-artifact directory")
    try:
        manifest = _parse_json(_bounded_read(root / "manifest.json", MAX_JSON_BYTES))
        if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
            raise ConversationError("Unsupported conversation manifest fields")
        if type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] != SCHEMA_VERSION or manifest["modelType"] != MODEL_TYPE:
            raise ConversationError("Unsupported conversation artifact schema or type")
        files = manifest["files"]
        if type(files) is not dict or set(files) != set(ARTIFACT_FILES):
            raise ConversationError("Manifest must identify only the three fixed artifact files")
        payloads = {}
        for name in ARTIFACT_FILES:
            descriptor = files[name]
            if type(descriptor) is not dict or set(descriptor) != {"sha256", "bytes"}:
                raise ConversationError("Invalid artifact file descriptor")
            if type(descriptor["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is None:
                raise ConversationError("Invalid artifact SHA-256")
            limit = MAX_WEIGHT_BYTES if name.endswith(".safetensors") else MAX_JSON_BYTES
            if type(descriptor["bytes"]) is not int or not 0 < descriptor["bytes"] <= limit:
                raise ConversationError("Invalid artifact member byte count")
            raw = _bounded_read(root / name, limit)
            if len(raw) != descriptor["bytes"] or _digest(raw) != descriptor["sha256"]:
                raise ConversationError("Artifact member hash or byte count differs from manifest")
            payloads[name] = raw
        if manifest["modelId"] != artifact_model_id(files):
            raise ConversationError("Artifact model identity differs from its hashed files")
        config = validate_config(_parse_json(payloads["config.json"]))
        if validate_config(manifest["config"]) != config or _json_bytes(manifest["config"]) != _json_bytes(config):
            raise ConversationError("Manifest configuration differs from config.json")
        if type(manifest["training"]) is not dict or type(manifest["evaluation"]) is not dict:
            raise ConversationError("Artifact training/evaluation metadata must be objects")
        if manifest["evaluation"].get("experimental") is not True or manifest["evaluation"].get("productionAccepted") is not False:
            raise ConversationError("Artifact does not preserve the experimental acceptance boundary")
        try:
            tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode("utf-8"))
        except Exception as error:
            # The Rust parser raises a plain Exception rather than ValueError.
            raise ConversationError("Invalid serialized local tokenizer") from error
        validate_tokenizer(tokenizer, strict=True)
        if tokenizer.get_vocab_size() != config["vocab_size"]:
            raise ConversationError("Tokenizer vocabulary differs from artifact configuration")
        try:
            tensors = load_tensors(payloads["weights.safetensors"])
        except Exception as error:
            # SafetensorError is not a ValueError/RuntimeError. Never let a
            # malformed inference artifact bypass the public validation error.
            raise ConversationError("Invalid safetensors parameter file") from error
        model = ConversationModel(config)
        if type(manifest["training"].get("parameterCount")) is not int or manifest["training"]["parameterCount"] != model.parameter_count:
            raise ConversationError("Manifest parameter count differs from the architecture")
        expected = model.state_dict()
        if set(tensors) != set(expected):
            raise ConversationError("Artifact tensor names differ from the model architecture")
        for name, tensor in tensors.items():
            if tensor.dtype != torch.float32 or tensor.shape != expected[name].shape or not torch.all(torch.isfinite(tensor)):
                raise ConversationError("Artifact tensor shape, dtype or finite-value check failed")
        model.load_state_dict(tensors, strict=True)
        model.eval()
        return model, tokenizer, manifest
    except ConversationError:
        raise
    except (OSError, ValueError, UnicodeError, RuntimeError, KeyError, TypeError) as error:
        raise ConversationError("Cannot load verified conversation artifact") from error
