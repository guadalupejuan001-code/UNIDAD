"""Restore ACTION BEST from its exact metadata and shared LAST transport bytes.

Stdlib-only; no Torch, pickle interpretation, model loading or training. The
reference records each complete native file and every component size/SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ("weights.safetensors", "resume.pt", "tokenizer.json", "state.json", "manifest.json")
BLOCK = 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir")
    parser.add_argument("destination")
    args = parser.parse_args()
    os.nice(10)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    source = Path(args.reference_dir).absolute()
    destination = Path(args.destination).absolute()
    record = json.loads((source / "native-reference.json").read_bytes())
    if (record.get("schemaVersion") != 1 or record.get("format") != "unidad-native-files-by-reference-v1"
            or destination.name != record["checkpointName"] or set(record["files"]) != set(NATIVE)):
        raise ValueError("Reference schema, native file set or original checkpoint basename mismatch")
    spec = importlib.util.spec_from_file_location("native_transport", ROOT / "examples/package_native_checkpoint.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport._basename(record["checkpointName"])
    components = {}
    for name in NATIVE:
        item = record["files"][name]
        if (type(item["bytes"]) is not int or item["bytes"] <= 0 or not transport._digest(item["sha256"])
                or not isinstance(item["sources"], list) or not item["sources"]):
            raise ValueError("Invalid complete native file identity")
        entries = []
        for part in item["sources"]:
            relative = Path(part["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Component must have a repository-relative physical path")
            path = ROOT / relative
            transport._physical_directory(path.parent)
            with transport._directory(path.parent) as fd:
                info = transport._hash_file(fd, path.name, part["bytes"])
            if info["sha256"] != part["sha256"] or not transport._digest(part["sha256"]):
                raise ValueError("Native reference component size/hash mismatch")
            entries.append((path, part, info["signature"]))
        if sum(part["bytes"] for _, part, _ in entries) != item["bytes"]:
            raise ValueError("Component byte counts do not equal the native file size")
        components[name] = entries
    with transport._temporary_output(source, destination) as output_fd:
        for name in NATIVE:
            whole = hashlib.sha256()
            count = 0
            with transport._new_file(output_fd, name) as target:
                for path, part, signature in components[name]:
                    digest = hashlib.sha256()
                    component_count = 0
                    with transport._directory(path.parent) as source_fd:
                        with transport._regular(source_fd, path.name) as stream:
                            if transport._signature(os.fstat(stream.fileno())) != signature:
                                raise ValueError("Native component changed before restoration")
                            while block := stream.read(BLOCK):
                                component_count += len(block)
                                if component_count > part["bytes"]:
                                    raise ValueError("Component exceeded its declared size")
                                target.write(block)
                                digest.update(block)
                                whole.update(block)
                            if transport._signature(os.fstat(stream.fileno())) != signature:
                                raise ValueError("Native component changed during restoration")
                    if component_count != part["bytes"] or digest.hexdigest() != part["sha256"]:
                        raise ValueError("Component size/hash mismatch during restoration")
                    count += component_count
            if count != record["files"][name]["bytes"] or whole.hexdigest() != record["files"][name]["sha256"]:
                raise ValueError("Reconstructed complete native file mismatch")
        manifest, _ = transport._read_json(output_fd, "manifest.json")
        transport._native_manifest(manifest, {name: item["sha256"] for name, item in record["files"].items()})
    print(json.dumps({"restoredCheckpoint": record["checkpointName"], "destination": str(destination),
                      "allFiveNativeFileHashesVerified": True, "torchImported": False, "trainingExecuted": False}))


if __name__ == "__main__":
    main()
