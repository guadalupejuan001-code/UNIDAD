"""Stream a frozen native checkpoint into bounded parts, or restore its bytes.

This transport helper uses only Python's standard library. It never imports
PyTorch, interprets pickle, trains, or changes the original checkpoint. Linux
renameat2(RENAME_NOREPLACE) publishes a complete new directory atomically.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid


NATIVE_FILES = ("weights.safetensors", "resume.pt", "tokenizer.json", "state.json", "manifest.json")
MANIFEST_NAME = "native-checkpoint-package.json"
POINTER_NAME = "native-checkpoint-pointer.json"
PACKAGE_FORMAT = "unidad-native-checkpoint-parts-v1"
POINTER_FORMAT = "unidad-native-checkpoint-pointer-v1"
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
IO_BLOCK_BYTES = 1024 * 1024


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _basename(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", value) is None:
        raise ValueError("Unsafe checkpoint basename")
    return value


def _absolute(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal is not permitted")
    return path


def _physical_directory(path):
    """Reject symlink ancestors instead of hiding them through resolve()."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Directory is missing, symlinked or not physical: {current}")


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _check_directory(path, directory_fd):
    info = path.lstat()
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
        raise ValueError("Directory changed during operation")


@contextlib.contextmanager
def _directory(path):
    _physical_directory(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def _regular(directory_fd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Expected a regular file without hard links: {name}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(fd)


def _hash_file(directory_fd, name, expected_bytes=None):
    digest = hashlib.sha256()
    with _regular(directory_fd, name) as stream:
        before = _signature(os.fstat(stream.fileno()))
        if expected_bytes is not None and before[2] != expected_bytes:
            raise ValueError(f"File size differs from package metadata: {name}")
        count = 0
        while count < before[2]:
            block = stream.read(min(IO_BLOCK_BYTES, before[2] - count))
            if not block:
                raise ValueError(f"Source file was truncated while hashing: {name}")
            count += len(block)
            digest.update(block)
        if stream.read(1) or _signature(os.fstat(stream.fileno())) != before:
            raise ValueError(f"Source file changed while hashing: {name}")
    return {"bytes": count, "sha256": digest.hexdigest(), "signature": before}


def _snapshot(directory_fd, names, expected_sizes=None):
    if set(os.listdir(directory_fd)) != set(names):
        raise ValueError("Directory contains missing, extra or shadow files")
    return {name: _hash_file(directory_fd, name,
                             None if expected_sizes is None else expected_sizes[name])
            for name in names}


def _read_json(directory_fd, name):
    with _regular(directory_fd, name) as stream:
        before = _signature(os.fstat(stream.fileno()))
        if before[2] > MAX_METADATA_BYTES:
            raise ValueError("JSON metadata exceeds the byte limit")
        raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES or len(raw) != before[2] or _signature(os.fstat(stream.fileno())) != before:
            raise ValueError("JSON metadata changed or exceeds the byte limit")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid UTF-8 JSON metadata") from exc
    return value, raw


def _native_manifest(value, native_hashes):
    expected = set(NATIVE_FILES) - {"manifest.json"}
    if (type(value) is not dict or set(value) != {"schemaVersion", "files"}
            or type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1
            or type(value["files"]) is not dict or set(value["files"]) != expected
            or any(not _digest(digest) or digest != native_hashes[name]
                   for name, digest in value["files"].items())):
        raise ValueError("Native checkpoint manifest/hash mismatch")


def _part_name(name, index):
    return f"{name}.part-{index:06d}"


def _metadata_bytes(value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Package metadata exceeds the byte limit")
    return raw


@contextlib.contextmanager
def _new_file(directory_fd, name):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                 dir_fd=directory_fd)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            yield stream
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(directory_fd, name, value):
    raw = _metadata_bytes(value)
    with _new_file(directory_fd, name) as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _rename_new(parent_fd, source_name, destination_name):
    """Rename within the pinned parent; an existing destination is never replaced."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise RuntimeError("Linux renameat2 is required for atomic no-replace publication") from exc
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(destination_name), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination_name)


@contextlib.contextmanager
def _temporary_output(source, destination):
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Output must be outside the source directory")
    _basename(destination.name)
    parent = destination.parent
    with _directory(parent) as parent_fd:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Output directory must be new")
        temporary_name = ".native-checkpoint-" + uuid.uuid4().hex
        os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
        temporary_fd = os.open(temporary_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=parent_fd)
        published = False
        try:
            yield temporary_fd
            _check_directory(parent, parent_fd)
            os.fsync(temporary_fd)
            _rename_new(parent_fd, temporary_name, destination.name)
            published = True
            os.fsync(parent_fd)
        finally:
            if not published:
                # All helper-created children are files in this private directory.
                for name in os.listdir(temporary_fd):
                    try:
                        os.unlink(name, dir_fd=temporary_fd)
                    except OSError:
                        pass
                try:
                    existing = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
                    opened = os.fstat(temporary_fd)
                    if (existing.st_dev, existing.st_ino) == (opened.st_dev, opened.st_ino):
                        os.rmdir(temporary_name, dir_fd=parent_fd)
                except OSError:
                    pass
            os.close(temporary_fd)


def pack(checkpoint_dir, output_dir, *, part_bytes=MAX_PART_BYTES):
    """Package a closed five-file native checkpoint; source bytes must remain stable."""
    if sys.platform != "linux":
        raise RuntimeError("This helper requires Linux atomic directory publication")
    if type(part_bytes) is not int or not 1 <= part_bytes <= MAX_PART_BYTES:
        raise ValueError("part_bytes must be an integer between 1 and 32 MiB")
    source, destination = _absolute(checkpoint_dir), _absolute(output_dir)
    name = _basename(source.name)
    with _directory(source) as source_fd:
        before = _snapshot(source_fd, NATIVE_FILES)
        hashes = {filename: info["sha256"] for filename, info in before.items()}
        native, _ = _read_json(source_fd, "manifest.json")
        _native_manifest(native, hashes)
        package = {"schemaVersion": 1, "format": PACKAGE_FORMAT, "checkpointName": name,
                   "partBytes": part_bytes, "totalNativeBytes": sum(info["bytes"] for info in before.values()),
                   "nativeFileHashes": hashes, "files": []}
        with _temporary_output(source, destination) as output_fd:
            for filename in NATIVE_FILES:
                info = before[filename]
                if info["bytes"] <= 0:
                    raise ValueError("Native checkpoint files must be nonempty")
                record = {"name": filename, "bytes": info["bytes"], "sha256": info["sha256"], "parts": []}
                whole = hashlib.sha256()
                with _regular(source_fd, filename) as stream:
                    if _signature(os.fstat(stream.fileno())) != info["signature"]:
                        raise ValueError("Checkpoint changed before partitioning")
                    remaining = info["bytes"]
                    index = 0
                    while remaining:
                        count = min(part_bytes, remaining)
                        part_name = _part_name(filename, index)
                        digest = hashlib.sha256()
                        with _new_file(output_fd, part_name) as target:
                            left = count
                            while left:
                                block = stream.read(min(left, IO_BLOCK_BYTES))
                                if not block:
                                    raise ValueError("Checkpoint was truncated during partitioning")
                                target.write(block)
                                digest.update(block)
                                whole.update(block)
                                left -= len(block)
                        record["parts"].append({"name": part_name, "bytes": count, "sha256": digest.hexdigest()})
                        remaining -= count
                        index += 1
                    if stream.read(1) or _signature(os.fstat(stream.fileno())) != info["signature"]:
                        raise ValueError("Checkpoint changed during partitioning")
                if whole.hexdigest() != info["sha256"]:
                    raise ValueError("Partitioned file does not match the frozen checkpoint")
                package["files"].append(record)
            digest = _write_json(output_fd, MANIFEST_NAME, package)
            _write_json(output_fd, POINTER_NAME, {"schemaVersion": 1, "format": POINTER_FORMAT,
                                                "manifest": MANIFEST_NAME, "manifestSha256": digest,
                                                "checkpointName": name})
            if _snapshot(source_fd, NATIVE_FILES) != before:
                raise ValueError("Checkpoint changed before package publication")
            _check_directory(source, source_fd)
    return {"operation": "pack", "directory": str(destination), "checkpointName": name,
            "manifestSha256": digest, "nativeFileHashes": hashes,
            "parts": sum(len(record["parts"]) for record in package["files"])}


def _package_metadata(pointer, package, raw_manifest):
    if (type(pointer) is not dict
            or set(pointer) != {"schemaVersion", "format", "manifest", "manifestSha256", "checkpointName"}
            or type(pointer["schemaVersion"]) is not int or pointer["schemaVersion"] != 1
            or pointer["format"] != POINTER_FORMAT or pointer["manifest"] != MANIFEST_NAME
            or not _digest(pointer["manifestSha256"])
            or hashlib.sha256(raw_manifest).hexdigest() != pointer["manifestSha256"]):
        raise ValueError("Invalid package pointer or manifest digest")
    _basename(pointer["checkpointName"])
    if (type(package) is not dict
            or set(package) != {"schemaVersion", "format", "checkpointName", "partBytes", "totalNativeBytes", "nativeFileHashes", "files"}
            or type(package["schemaVersion"]) is not int or package["schemaVersion"] != 1
            or package["format"] != PACKAGE_FORMAT or package["checkpointName"] != pointer["checkpointName"]
            or type(package["partBytes"]) is not int or not 1 <= package["partBytes"] <= MAX_PART_BYTES
            or type(package["totalNativeBytes"]) is not int or package["totalNativeBytes"] <= 0
            or type(package["nativeFileHashes"]) is not dict or set(package["nativeFileHashes"]) != set(NATIVE_FILES)
            or any(not _digest(value) for value in package["nativeFileHashes"].values())
            or type(package["files"]) is not list or len(package["files"]) != len(NATIVE_FILES)):
        raise ValueError("Invalid package manifest fields")
    names = {MANIFEST_NAME, POINTER_NAME}
    total = 0
    for filename, record in zip(NATIVE_FILES, package["files"]):
        if (type(record) is not dict or set(record) != {"name", "bytes", "sha256", "parts"}
                or record["name"] != filename or type(record["bytes"]) is not int or record["bytes"] <= 0
                or not _digest(record["sha256"]) or record["sha256"] != package["nativeFileHashes"][filename]
                or type(record["parts"]) is not list
                or len(record["parts"]) != (record["bytes"] + package["partBytes"] - 1) // package["partBytes"]):
            raise ValueError("Invalid native file/parts metadata")
        remaining = record["bytes"]
        for index, part in enumerate(record["parts"]):
            expected_name = _part_name(filename, index)
            expected_bytes = min(remaining, package["partBytes"])
            if (type(part) is not dict or set(part) != {"name", "bytes", "sha256"}
                    or part["name"] != expected_name or expected_name in names
                    or type(part["bytes"]) is not int or part["bytes"] != expected_bytes
                    or not _digest(part["sha256"])):
                raise ValueError("Invalid part name, size or digest")
            names.add(expected_name)
            remaining -= expected_bytes
        total += record["bytes"]
    if total != package["totalNativeBytes"]:
        raise ValueError("Package total byte count mismatch")
    return names


def restore(package_dir, output_dir):
    """Restore exact native bytes into a new directory with the original basename."""
    if sys.platform != "linux":
        raise RuntimeError("This helper requires Linux atomic directory publication")
    source, destination = _absolute(package_dir), _absolute(output_dir)
    with _directory(source) as source_fd:
        pointer, raw_pointer = _read_json(source_fd, POINTER_NAME)
        package, raw = _read_json(source_fd, MANIFEST_NAME)
        names = _package_metadata(pointer, package, raw)
        if destination.name != pointer["checkpointName"]:
            raise ValueError("Restore directory must preserve the original checkpoint basename")
        expected_sizes = {POINTER_NAME: len(raw_pointer), MANIFEST_NAME: len(raw)}
        expected_hashes = {POINTER_NAME: hashlib.sha256(raw_pointer).hexdigest(),
                           MANIFEST_NAME: hashlib.sha256(raw).hexdigest()}
        for record in package["files"]:
            for part in record["parts"]:
                expected_sizes[part["name"]] = part["bytes"]
                expected_hashes[part["name"]] = part["sha256"]
        before = _snapshot(source_fd, sorted(names), expected_sizes)
        if any(before[name]["sha256"] != expected_hashes[name] for name in names):
            raise ValueError("Package metadata/part digest mismatch")
        with _temporary_output(source, destination) as output_fd:
            for record in package["files"]:
                whole = hashlib.sha256()
                count = 0
                with _new_file(output_fd, record["name"]) as target:
                    for part in record["parts"]:
                        digest = hashlib.sha256()
                        part_count = 0
                        with _regular(source_fd, part["name"]) as stream:
                            if _signature(os.fstat(stream.fileno())) != before[part["name"]]["signature"]:
                                raise ValueError("Package changed before restoration")
                            while block := stream.read(IO_BLOCK_BYTES):
                                part_count += len(block)
                                if part_count > part["bytes"]:
                                    raise ValueError("Part exceeds its declared size")
                                target.write(block)
                                digest.update(block)
                                whole.update(block)
                            if _signature(os.fstat(stream.fileno())) != before[part["name"]]["signature"]:
                                raise ValueError("Package changed during restoration")
                        if part_count != part["bytes"] or digest.hexdigest() != part["sha256"]:
                            raise ValueError("Part size/SHA256 mismatch")
                        count += part_count
                if count != record["bytes"] or whole.hexdigest() != record["sha256"]:
                    raise ValueError("Restored native file size/SHA256 mismatch")
            native, _ = _read_json(output_fd, "manifest.json")
            _native_manifest(native, package["nativeFileHashes"])
            if _snapshot(source_fd, sorted(names), expected_sizes) != before:
                raise ValueError("Package changed before restored checkpoint publication")
            _check_directory(source, source_fd)
    return {"operation": "restore", "directory": str(destination),
            "checkpointName": pointer["checkpointName"], "manifestSha256": pointer["manifestSha256"],
            "nativeFileHashes": package["nativeFileHashes"], "nativeBytesVerified": package["totalNativeBytes"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="Package only a closed, pinned native checkpoint")
    packing.add_argument("checkpoint_dir", type=Path)
    packing.add_argument("output_dir", type=Path)
    packing.add_argument("--part-bytes", type=int, default=MAX_PART_BYTES)
    restoring = commands.add_parser("restore", help="Verify and restore five original native files")
    restoring.add_argument("package_dir", type=Path)
    restoring.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (pack(args.checkpoint_dir, args.output_dir, part_bytes=args.part_bytes)
                  if args.command == "pack" else restore(args.package_dir, args.output_dir))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Checkpoint packaging failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
