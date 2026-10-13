"""Hash-verified portable Transformer directories; inference never reads pickle."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from .model import ConversationModel, MODEL_TYPE, count_config_parameters, validate_config
from .tokenizer import ConversationError, validate_tokenizer


SCHEMA_VERSION = 1
ARTIFACT_FILES = ("config.json", "tokenizer.json", "weights.safetensors")
MAX_JSON_BYTES = 4_194_304
MAX_WEIGHT_BYTES = 268_435_456
MAX_SAFETENSORS_HEADER_BYTES = 1_048_576
IO_CHUNK_BYTES = 1_048_576
FINITE_CHUNK_ELEMENTS = 262_144
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


def _member_identity(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ConversationError("Artifact members must be regular files, not symlinks")
    info = path.stat()
    if not 0 < info.st_size <= limit:
        raise ConversationError("Artifact member exceeds its byte limit or is empty")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_descriptor(path, limit):
    """Hash bounded chunks instead of retaining a serialized weight file."""
    identity = _member_identity(path, limit)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(IO_CHUNK_BYTES):
            size += len(block)
            if size > limit:
                raise ConversationError("Artifact member changed while being read or exceeds its limit")
            digest.update(block)
    if size != identity[2] or _member_identity(path, limit) != identity:
        raise ConversationError("Artifact member changed while being read or exceeds its limit")
    return {"sha256": digest.hexdigest(), "bytes": size}, identity


def _weight_limit(config):
    if "parameter_budget" not in config:
        return MAX_WEIGHT_BYTES
    # No allowance for the whole budget: bound this exact architecture and a
    # small header. F32 storage remains part of the portable artifact contract.
    return 4 * count_config_parameters(config) + 8 + MAX_SAFETENSORS_HEADER_BYTES


def _finite_float32(tensor):
    if tensor.dtype != torch.float32:
        return False
    flat = tensor.detach().reshape(-1)
    return all(bool(torch.isfinite(flat[start:start + FINITE_CHUNK_ELEMENTS]).all())
               for start in range(0, flat.numel(), FINITE_CHUNK_ELEMENTS))


def _validate_weight_header(path, expected, size):
    """Check tensor sizes on meta before mapping any payload into Torch."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ConversationError("Invalid safetensors parameter file")
        header_size = int.from_bytes(prefix, "little")
        if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
            raise ConversationError("Safetensors header exceeds its bounded byte limit")
        raw = stream.read(header_size)
    if len(raw) != header_size:
        raise ConversationError("Invalid safetensors parameter file")
    header = _parse_json(raw)
    if type(header) is not dict or set(header) - {"__metadata__"} != set(expected):
        raise ConversationError("Artifact tensor names differ from the model architecture")
    metadata = header.get("__metadata__", {})
    if type(metadata) is not dict or any(type(key) is not str or type(value) is not str
                                         for key, value in metadata.items()):
        raise ConversationError("Invalid safetensors text metadata")
    elements = 0
    for name, tensor in expected.items():
        value = header[name]
        if (type(value) is not dict or set(value) != {"dtype", "shape", "data_offsets"}
                or value["dtype"] != "F32" or type(value["shape"]) is not list
                or any(type(dimension) is not int for dimension in value["shape"])
                or value["shape"] != list(tensor.shape)):
            raise ConversationError("Artifact tensor shape, dtype or finite-value check failed")
        offsets = value["data_offsets"]
        if (type(offsets) is not list or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
                or offsets[1] - offsets[0] != 4 * tensor.numel()):
            raise ConversationError("Invalid safetensors tensor byte offsets")
        elements += tensor.numel()
    if size != 8 + header_size + 4 * elements:
        raise ConversationError("Safetensors payload size differs from the exact architecture")


def _staged_weights(target, tensors, limit):
    descriptor, name = tempfile.mkstemp(prefix=".unidad-weights-", dir=target)
    os.close(descriptor)
    temporary = Path(name)
    try:
        save_file(tensors, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        file_description, _ = _file_descriptor(temporary, limit)
        return temporary, file_description
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
        if not _finite_float32(value):
            raise ConversationError("Exported parameters must be finite float32 tensors")
        tensors[name] = value.detach().cpu().contiguous()
    payloads = {"config.json": _json_bytes(config), "tokenizer.json": tokenizer.to_str().encode("utf-8")}
    if len(payloads["tokenizer.json"]) > MAX_JSON_BYTES:
        raise ConversationError("Exported tokenizer or weights exceed artifact byte limits")
    target = Path(directory)
    if target.is_symlink():
        raise ConversationError("Artifact directory cannot be a symlink")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.is_dir():
        raise ConversationError("Artifact output must be a directory")
    for name in (*ARTIFACT_FILES, "manifest.json"):
        if (target / name).is_symlink():
            raise ConversationError("Artifact members cannot replace symlinks")
    temporary = None
    try:
        temporary, weight_description = _staged_weights(target, tensors, _weight_limit(config))
        files = {name: {"sha256": _digest(raw), "bytes": len(raw)} for name, raw in payloads.items()}
        files["weights.safetensors"] = weight_description
        manifest = {"schemaVersion": SCHEMA_VERSION, "modelType": MODEL_TYPE, "modelId": artifact_model_id(files),
                    "config": config, "files": files,
                    "training": {**(training or {}), "parameterCount": model.parameter_count},
                    "evaluation": {**(evaluation or {}), "experimental": True, "productionAccepted": False}}
        encoded_manifest = _json_bytes(manifest)
        for name, raw in payloads.items():
            _write_atomic(target / name, raw)
        os.replace(temporary, target / "weights.safetensors")
        _write_atomic(target / "manifest.json", encoded_manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_artifact(directory, *, expected_model_config=None):
    """Verify and load a CPU snapshot, optionally pinning admitted architecture.

    The host-selected pin is checked against hash-verified configuration before
    any weight payload is mapped or a model skeleton is constructed. A prior
    resource plan alone cannot authorize a different concurrently replaced model.
    """
    expected_config = (validate_config(expected_model_config)
                       if expected_model_config is not None else None)
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
            if type(descriptor["bytes"]) is not int or descriptor["bytes"] <= 0:
                raise ConversationError("Invalid artifact member byte count")
            if name == "weights.safetensors":
                continue
            if descriptor["bytes"] > MAX_JSON_BYTES:
                raise ConversationError("Invalid artifact member byte count")
            raw = _bounded_read(root / name, MAX_JSON_BYTES)
            if len(raw) != descriptor["bytes"] or _digest(raw) != descriptor["sha256"]:
                raise ConversationError("Artifact member hash or byte count differs from manifest")
            payloads[name] = raw
        if manifest["modelId"] != artifact_model_id(files):
            raise ConversationError("Artifact model identity differs from its hashed files")
        config = validate_config(_parse_json(payloads["config.json"]))
        if validate_config(manifest["config"]) != config or _json_bytes(manifest["config"]) != _json_bytes(config):
            raise ConversationError("Manifest configuration differs from config.json")
        if expected_config is not None and config != expected_config:
            raise ConversationError("Artifact architecture differs from admitted model configuration")
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
        if (type(manifest["training"].get("parameterCount")) is not int
                or manifest["training"]["parameterCount"] != count_config_parameters(config)):
            raise ConversationError("Manifest parameter count differs from the architecture")
        limit = _weight_limit(config)
        if files["weights.safetensors"]["bytes"] > limit:
            raise ConversationError("Invalid artifact member byte count")
        weights_path = root / "weights.safetensors"
        description, identity = _file_descriptor(weights_path, limit)
        if description != files["weights.safetensors"]:
            raise ConversationError("Artifact member hash or byte count differs from manifest")
        model = ConversationModel._loading_skeleton(config)
        expected = model.state_dict()
        _validate_weight_header(weights_path, expected, description["bytes"])
        try:
            tensors = load_file(weights_path, device="cpu")
        except Exception as error:
            # SafetensorError does not derive from ValueError/RuntimeError.
            raise ConversationError("Invalid safetensors parameter file") from error
        if set(tensors) != set(expected):
            raise ConversationError("Artifact tensor names differ from the model architecture")
        for name in list(tensors):
            tensor = tensors[name]
            if tensor.shape != expected[name].shape or not _finite_float32(tensor):
                raise ConversationError("Artifact tensor shape, dtype or finite-value check failed")
            # A single CPU snapshot keeps the running model independent of later
            # in-place file edits. Meta construction avoids a second random
            # model; no serialized payload or full temporary clone is retained.
            tensors[name] = tensor.clone()
        if _member_identity(weights_path, limit) != identity:
            raise ConversationError("Artifact member changed while being loaded")
        model.load_state_dict(tensors, strict=True, assign=True)
        model.eval()
        return model, tokenizer, manifest
    except ConversationError:
        raise
    except (OSError, ValueError, UnicodeError, RuntimeError, KeyError, TypeError) as error:
        raise ConversationError("Cannot load verified conversation artifact") from error
