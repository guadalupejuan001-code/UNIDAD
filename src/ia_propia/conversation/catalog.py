"""Host-configured catalog of own Transformers, inspected without loading tensors.

``load_catalog(path, project_root)`` returns detached entries plus explicit
availability fields. Missing artifacts remain listed. Paths and membership are
host configuration, never inferred from a conversation or downloaded on a reply.
Weight bytes are verified by the existing artifact loader when a member loads;
catalog inspection reads only bounded manifests, configuration and tokenizers.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .configuration import count_config_parameters, validate_config

_FIELDS = {"name", "path", "modelId", "weightsSha256", "tokenizerSha256", "config",
           "parameterCount", "artifactKind", "rootOrigin", "productionAccepted"}
_NAMES = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {"complete", "delta_source", "complete_reconstructed"}
_FILES = ("config.json", "tokenizer.json", "weights.safetensors")
MAX_CATALOG_BYTES = 2097152
MAX_CATALOG_MODELS = 128
MAX_HEADER_BYTES = 4194304


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _read_json(path, *, limit=MAX_HEADER_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Catalog metadata must be a bounded regular JSON file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Catalog metadata exceeds its byte limit")
    def distinct(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate catalog metadata key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Catalog metadata requires finite JSON")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=distinct, parse_constant=invalid)


def _digest(value, name):
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"Catalog {name} must be a lowercase SHA-256")
    return value


def _artifact_directory(relative, root):
    if type(relative) is not str or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Catalog artifact path must be a project-relative directory")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError("Catalog artifact path cannot escape the project")
    directory = root.joinpath(*path.parts)
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ValueError("Catalog artifact directory cannot escape the project through symlinks")
    return directory


def _validate_entry(entry):
    if type(entry) is not dict or set(entry) != _FIELDS:
        raise ValueError("Catalog entries require the documented Transformer identity fields")
    if type(entry["name"]) is not str or not _NAMES.fullmatch(entry["name"]):
        raise ValueError("Catalog member name must be a bounded stable identifier")
    if type(entry["modelId"]) is not str or not entry["modelId"].startswith("unidad-conversation-"):
        raise ValueError("Catalog modelId must identify an own conversational Transformer")
    _digest(entry["modelId"][len("unidad-conversation-"):], "modelId")
    _digest(entry["weightsSha256"], "weightsSha256")
    _digest(entry["tokenizerSha256"], "tokenizerSha256")
    if type(entry["artifactKind"]) is not str or entry["artifactKind"] not in _KINDS:
        raise ValueError("Catalog artifactKind is unsupported")
    if entry["rootOrigin"] != "random_weights" or type(entry["rootOrigin"]) is not str:
        raise ValueError("The catalog requires declared own random-weight lineage")
    if type(entry["productionAccepted"]) is not bool:
        raise ValueError("Catalog productionAccepted must be explicit")
    if type(entry["parameterCount"]) is not int or entry["parameterCount"] != count_config_parameters(entry["config"]):
        raise ValueError("Catalog parameter count differs from its Transformer architecture")
    if validate_config(entry["config"]) != entry["config"]:
        raise ValueError("Catalog architecture must preserve its complete serialized configuration")


def _inspect_header(entry, directory):
    members = ("manifest.json", *_FILES)
    missing = [name for name in members if not (directory / name).is_file()]
    for name in members:
        if (directory / name).is_symlink():
            raise ValueError("Catalog artifact members cannot be symlinks")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if (type(manifest) is not dict or type(manifest.get("schemaVersion")) is not int or manifest.get("schemaVersion") != 1
                or manifest.get("modelType") != "unidad-conversation-transformer"
                or manifest.get("modelId") != entry["modelId"] or manifest.get("config") != entry["config"]):
            raise ValueError(f"Catalog identity or architecture differs from {entry['name']} manifest")
        descriptors = manifest.get("files")
        if type(descriptors) is not dict or set(descriptors) != set(_FILES):
            raise ValueError("Catalog manifest requires complete inference file descriptors")
        for name, descriptor in descriptors.items():
            if (type(descriptor) is not dict or set(descriptor) != {"bytes", "sha256"}
                    or type(descriptor["bytes"]) is not int or not 0 < descriptor["bytes"] <= 5000000000):
                raise ValueError("Invalid catalog artifact descriptor")
            _digest(descriptor["sha256"], name)
        if (descriptors["weights.safetensors"]["sha256"] != entry["weightsSha256"]
                or descriptors["tokenizer.json"]["sha256"] != entry["tokenizerSha256"]):
            raise ValueError("Catalog file identities differ from its manifest")
        identity = {"schemaVersion": 1, "modelType": "unidad-conversation-transformer",
                    "files": {name: descriptors[name]["sha256"] for name in _FILES}}
        if "unidad-conversation-" + hashlib.sha256(_canonical(identity)).hexdigest() != entry["modelId"]:
            raise ValueError("Catalog Transformer identity does not match manifest hashes")
        training, evaluation = manifest.get("training"), manifest.get("evaluation")
        if (type(training) is not dict or training.get("parameterCount") != entry["parameterCount"]
                or type(evaluation) is not dict or evaluation.get("productionAccepted") is not entry["productionAccepted"]):
            raise ValueError("Catalog provenance or acceptance differs from its manifest")
        lineage = training.get("lineage", {})
        if type(lineage) is not dict:
            raise ValueError("Catalog manifest lineage must be explicit metadata")
        origin = "random_weights" if training.get("trainedFrom") == "random_weights" else lineage.get("rootOrigin")
        if origin != entry["rootOrigin"]:
            raise ValueError("Catalog own-weight lineage differs from its manifest")
        for name in _FILES:
            member = directory / name
            if not member.is_file():
                continue
            if member.stat().st_size != descriptors[name]["bytes"]:
                raise ValueError("Catalog artifact member byte count differs from manifest")
            if name != "weights.safetensors":
                if member.stat().st_size > MAX_HEADER_BYTES:
                    raise ValueError("Catalog metadata member exceeds its byte limit")
                raw = member.read_bytes()
                if hashlib.sha256(raw).hexdigest() != descriptors[name]["sha256"]:
                    raise ValueError("Catalog metadata hash differs from manifest")
                if name == "config.json" and _read_json(member) != entry["config"]:
                    raise ValueError("Catalog architecture differs from config.json")
    ready = not missing and entry["artifactKind"] != "delta_source"
    status = "ready" if ready else "pending_reconstruction" if entry["artifactKind"] == "delta_source" else "missing_files"
    return {"available": ready, "loadable": ready, "status": status, "missingFiles": missing,
            "artifactDirectory": str(directory), "weightsVerifiedDuringCatalogLoad": False}


def load_catalog(path, project_root):
    """Return validated entries and availability without Torch, tensors or network.

    JSON format is ``{"schemaVersion": 1, "entries": [...]}``. Every entry has
    the identity fields listed in ``_FIELDS``. A missing artifact is returned
    with ``available=False`` and ``missingFiles``; corrupt present metadata
    raises a clear error. Loading a catalog never reconstructs or downloads.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("Catalog project_root must be an existing directory")
    catalog_path = Path(path)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path
    value = _read_json(catalog_path, limit=MAX_CATALOG_BYTES)
    if (type(value) is not dict or set(value) != {"schemaVersion", "entries"}
            or type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1
            or type(value["entries"]) is not list or not 1 <= len(value["entries"]) <= MAX_CATALOG_MODELS):
        raise ValueError("Unsupported own-Transformer catalog format")
    names, ids, result = set(), set(), []
    for entry in value["entries"]:
        _validate_entry(entry)
        if entry["name"] in names or entry["modelId"] in ids:
            raise ValueError("Catalog names and Transformer identities must be distinct; keep aliases separately")
        names.add(entry["name"])
        ids.add(entry["modelId"])
        directory = _artifact_directory(entry["path"], root)
        result.append({**deepcopy(entry), **_inspect_header(entry, directory)})
    return result
