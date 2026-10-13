"""Host-bound file observations and reports; model arguments never select roots.

Only files present when the host constructs this adapter can be read. Reports
contain measurements derived from those bytes, never text supplied by an actor.
The completion callback additionally requires evidence in the current task's
journal; an existing report or another task's observations cannot finish a task.
"""

import base64
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import threading

from .protocol import BrainError


MAX_TASK_FILES = 128
_OBSERVATION_FIELDS = {"filename", "size_bytes", "sha256", "statistics"}
_REPORT_FIELDS = {"filename", "size_bytes", "sha256", "files"}


def _basename(value, *, report=False):
    if (type(value) is not str or not value or value in {".", ".."}
            or "/" in value or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise BrainError("invalid_input", "File names must be safe basenames")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise BrainError("invalid_input", "File names must be valid UTF-8") from error
    if len(encoded) > 255 or (report and (not value.endswith(".json") or value.startswith("."))):
        raise BrainError("invalid_input", "Reports need a bounded .json basename")
    return value


def _open_root(path):
    """Walk every directory component without following symlinks, including ancestors."""
    normalized = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in normalized.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return normalized, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _identity(status):
    return status.st_dev, status.st_ino


def _revision(status):
    return (_identity(status), status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_mode)


def _report_bytes(observations):
    document = {"kind": "unidad.file-report", "version": 1, "files": observations}
    return (json.dumps(document, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _same_json(left, right):
    """JSON equality keeps booleans distinct from integers in recorded evidence."""
    return (json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            == json.dumps(right, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


class SafeFileTools:
    """Three narrow tools configured by the host with disjoint, existing roots.

    ``filenames`` may restrict the captured whitelist; omitted, all existing
    regular files with safe basenames are captured. The adapter pins directory
    and input-file revisions and rejects replacements or later changes. A host
    must explicitly reconfigure inputs after changing their bytes or metadata.
    """

    def __init__(self, input_root, report_root, filenames=None, max_file_bytes=65_536):
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise BrainError("invalid_file_configuration", "max_file_bytes must be a positive integer")
        if not hasattr(os, "O_NOFOLLOW"):
            raise BrainError("invalid_file_configuration", "This host cannot safely open configured files")
        self._lock = threading.RLock()
        self._input_fd = self._report_fd = None
        # A shared adapter can serve different task workers. Each controller
        # callback restores only its own journal on the invoking thread.
        self._task_observations = threading.local()
        self.max_file_bytes = max_file_bytes
        try:
            self.input_root, self._input_fd = _open_root(input_root)
            self.report_root, self._report_fd = _open_root(report_root)
            if (self.input_root == self.report_root or self.input_root in self.report_root.parents
                    or self.report_root in self.input_root.parents
                    or _identity(os.fstat(self._input_fd)) == _identity(os.fstat(self._report_fd))):
                raise BrainError("invalid_file_configuration", "Input and report roots must be disjoint")
            self._input_identity = _identity(os.fstat(self._input_fd))
            self._report_identity = _identity(os.fstat(self._report_fd))
            self._whitelist = self._capture_files(filenames)
            self.filenames = tuple(sorted(self._whitelist))
            # Stable across instances bound to the same resources, but changes
            # when a host redirects roots, replaces captured inputs or changes
            # limits. Only this digest enters the controller's private policy;
            # filesystem paths and identities never enter model tool metadata.
            binding = {"adapter": "unidad.safe-file-tools.v1",
                       "input_root": {"path": str(self.input_root), "identity": self._input_identity},
                       "report_root": {"path": str(self.report_root), "identity": self._report_identity},
                       "files": [[name, self._whitelist[name]] for name in self.filenames],
                       "max_file_bytes": self.max_file_bytes, "max_task_files": MAX_TASK_FILES}
            policy_id = hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":")).encode("utf-8")).hexdigest()
            # Imported here so merely importing the adapter does not create a
            # dependency from the controller back to filesystem implementations.
            from .autonomy import AutonomyTool
            filename_schema = {"type": "string", "minLength": 1, "maxLength": 255,
                               "description": "A host-configured basename from task_inputs.files"}
            read_schema = {"type": "object", "properties": {"filename": filename_schema},
                           "required": ["filename"], "additionalProperties": False}
            report_schema = {"type": "object", "properties": {
                "filename": {"type": "string", "description": "A new .json report basename"},
                "files": {"type": "array", "items": filename_schema, "minItems": 1,
                          "maxItems": MAX_TASK_FILES, "uniqueItems": True}},
                "required": ["filename", "files"], "additionalProperties": False}
            self.tools = (
                AutonomyTool("inspect_file", self.inspect_file, self._validate_file,
                             description="Measure a configured file's real bytes and SHA-256.",
                             schema=deepcopy(read_schema), effect="read", retry_safe=True,
                             verifier=self._verify_inspection, policy_id=policy_id),
                AutonomyTool("read_file", self.read_file, self._validate_file,
                             description="Read configured bytes as UTF-8 text or base64, with measurements.",
                             schema=deepcopy(read_schema), effect="read", retry_safe=True,
                             verifier=self._verify_read, policy_id=policy_id),
                AutonomyTool("save_report", self.save_report, self._validate_report,
                             description="Save a new report from prior real file observations; never supply report text.",
                             schema=report_schema, effect="write", retry_safe=False,
                             verifier=self.verify_report, policy_id=policy_id),
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            for attribute in ("_input_fd", "_report_fd"):
                descriptor = getattr(self, attribute, None)
                if descriptor is not None:
                    setattr(self, attribute, None)
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()

    def __del__(self):
        self.close()

    @property
    def _observations(self):
        values = getattr(self._task_observations, "values", None)
        if values is None:
            values = {}
            self._task_observations.values = values
        return values

    def _capture_files(self, filenames):
        explicit = filenames is not None
        if explicit and (type(filenames) not in (list, tuple) or not filenames
                         or len(filenames) > MAX_TASK_FILES):
            raise BrainError("invalid_file_configuration", "The file whitelist must be a bounded sequence")
        captured = {}
        def capture(name):
            _basename(name)
            if name in captured:
                raise BrainError("invalid_file_configuration", "Whitelist file names must be unique")
            status = os.stat(name, dir_fd=self._input_fd, follow_symlinks=False)
            if not stat.S_ISREG(status.st_mode):
                if explicit:
                    raise BrainError("invalid_file_configuration", "Whitelisted inputs must be existing regular files")
                return
            # Filesystems may recycle an inode immediately after unlinking it.
            # Bind the captured revision as well as its device/inode identity;
            # atime is intentionally excluded because reads may update it.
            captured[name] = _revision(status)

        if explicit:
            for name in filenames:
                try:
                    capture(name)
                except (BrainError, OSError):
                    raise BrainError("invalid_file_configuration", "Whitelist contains an unsafe or unavailable file") from None
        else:
            # Bound directory work as well as the resulting whitelist. Avoid
            # materializing arbitrary numbers of unconfigured directory entries.
            with os.scandir(self._input_fd) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > MAX_TASK_FILES * 8:
                        raise BrainError("invalid_file_configuration", "Configure an explicit whitelist for a large directory")
                    try:
                        capture(entry.name)
                    except (BrainError, OSError):
                        continue
                    if len(captured) > MAX_TASK_FILES:
                        raise BrainError("invalid_file_configuration", "Configure at most 128 input files")
        return captured

    def _validate_file(self, arguments):
        if type(arguments) is not dict or set(arguments) != {"filename"}:
            raise BrainError("invalid_input", "File tools accept only filename")
        name = _basename(arguments["filename"])
        if name not in self._whitelist:
            raise BrainError("file_denied", "File is outside the host's captured whitelist")

    def _validate_files(self, files):
        if type(files) is not list or not 1 <= len(files) <= MAX_TASK_FILES:
            raise BrainError("invalid_input", "Files must be a nonempty bounded list")
        seen = set()
        for name in files:
            self._validate_file({"filename": name})
            if name in seen:
                raise BrainError("invalid_input", "Files must be unique")
            seen.add(name)

    def _validate_report(self, arguments):
        if type(arguments) is not dict or set(arguments) != {"filename", "files"}:
            raise BrainError("invalid_input", "Reports accept only filename and observed files")
        _basename(arguments["filename"], report=True)
        self._validate_files(arguments["files"])

    def validate_task_inputs(self, inputs):
        if type(inputs) is not dict or set(inputs) != {"files", "report"}:
            raise BrainError("invalid_input", "File tasks require only files and report")
        self._validate_report({"filename": inputs["report"], "files": inputs["files"]})
        return deepcopy(inputs)

    def _check_roots(self):
        if self._input_fd is None or self._report_fd is None:
            raise BrainError("file_unavailable", "Configured file tools are closed")
        for path, identity in ((self.input_root, self._input_identity),
                               (self.report_root, self._report_identity)):
            descriptor = None
            try:
                _path, descriptor = _open_root(path)
                if _identity(os.fstat(descriptor)) != identity:
                    raise BrainError("file_unavailable", "Configured directory was replaced")
            except OSError:
                raise BrainError("file_unavailable", "Configured directory cannot be accessed safely") from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def _read_bytes(self, root_fd, filename, limit, expected_revision=None):
        descriptor = None
        try:
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                                 dir_fd=root_fd)
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode)
                    or (expected_revision is not None and _identity(before) != expected_revision[0])):
                raise BrainError("file_unavailable", "Configured file was replaced or is not regular")
            if expected_revision is not None and _revision(before) != expected_revision:
                raise BrainError("file_changed", "Configured file revision changed after host capture")
            if before.st_size > limit:
                raise BrainError("file_too_large", "Configured file exceeds the host's byte limit")
            chunks, size = [], 0
            while size <= limit:
                chunk = os.read(descriptor, min(65_536, limit + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            if size > limit:
                raise BrainError("file_too_large", "Configured file exceeds the host's byte limit")
            after = os.fstat(descriptor)
            current = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
            if (_revision(before) != _revision(after) or _revision(after) != _revision(current)
                    or not stat.S_ISREG(current.st_mode) or size != after.st_size):
                raise BrainError("file_changed", "File changed during observation")
            return b"".join(chunks)
        except OSError:
            raise BrainError("file_unavailable", "Configured file cannot be accessed safely") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _observe(self, filename, *, content=False):
        self._check_roots()
        raw = self._read_bytes(self._input_fd, filename, self.max_file_bytes, self._whitelist[filename])
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            text = None
        result = {"filename": filename, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                  "statistics": {"lines": raw.count(b"\n") + int(bool(raw) and not raw.endswith(b"\n")),
                                 "utf8": text is not None, "characters": len(text) if text is not None else None,
                                 "words": len(text.split()) if text is not None else None}}
        if content:
            # Control-heavy UTF-8 can expand sixfold in JSON. Base64 keeps
            # ordinary default-size reads within the controller's result budget.
            encoded_text_size = (len(json.dumps(text, ensure_ascii=False).encode("utf-8"))
                                 if text is not None else None)
            use_text = text is not None and encoded_text_size <= 2 + 4 * ((len(raw) + 2) // 3)
            result["encoding"] = "utf-8" if use_text else "base64"
            result["content"] = text if use_text else base64.b64encode(raw).decode("ascii")
        return result

    def inspect_file(self, arguments):
        self._validate_file(arguments)
        with self._lock:
            result = self._observe(arguments["filename"])
            self._observations[arguments["filename"]] = deepcopy(result)
            return result

    def read_file(self, arguments):
        self._validate_file(arguments)
        with self._lock:
            result = self._observe(arguments["filename"], content=True)
            self._observations[arguments["filename"]] = {key: deepcopy(result[key]) for key in _OBSERVATION_FIELDS}
            return result

    def _verify_observation(self, arguments, recorded_result, *, content=False):
        try:
            self._validate_file(arguments)
            with self._lock:
                current = self._observe(arguments["filename"], content=content)
                if not _same_json(current, recorded_result):
                    return {"verified": False}
                self._observations[arguments["filename"]] = {
                    key: deepcopy(current[key]) for key in _OBSERVATION_FIELDS}
                return {"verified": True, "result": current}
        except (BrainError, OSError, ValueError, TypeError):
            return {"verified": False}

    def _verify_inspection(self, arguments, recorded_result):
        return self._verify_observation(arguments, recorded_result)

    def _verify_read(self, arguments, recorded_result):
        return self._verify_observation(arguments, recorded_result, content=True)

    def _install_report(self, filename, raw):
        """Install a complete, synced inode with an exclusive hard link from its fd."""
        descriptor, temporary = None, None
        try:
            if hasattr(os, "O_TMPFILE"):
                try:
                    descriptor = os.open(".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
                                         0o600, dir_fd=self._report_fd)
                except OSError as error:
                    if error.errno not in {errno.EOPNOTSUPP, errno.EINVAL, errno.EISDIR, errno.ENOSYS}:
                        raise
            if descriptor is None:
                temporary = ".unidad-report-" + secrets.token_hex(16)
                descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     0o600, dir_fd=self._report_fd)
            remaining = memoryview(raw)
            while remaining:
                count = os.write(descriptor, remaining)
                if not count:
                    raise OSError("Cannot write report")
                remaining = remaining[count:]
            os.fsync(descriptor)
            # /proc resolves a kernel-owned descriptor, not a model-supplied
            # path or a temporary directory entry that an attacker can swap.
            os.link(f"/proc/self/fd/{descriptor}", filename,
                    dst_dir_fd=self._report_fd, follow_symlinks=True)
            os.fsync(self._report_fd)
        except FileExistsError:
            raise BrainError("report_exists", "Report already exists; no file was replaced") from None
        except OSError:
            raise BrainError("report_failed", "Report could not be installed safely") from None
        finally:
            if temporary is not None and descriptor is not None:
                try:
                    if _identity(os.stat(temporary, dir_fd=self._report_fd, follow_symlinks=False)) == _identity(os.fstat(descriptor)):
                        os.unlink(temporary, dir_fd=self._report_fd)
                except OSError:
                    pass
            if descriptor is not None:
                os.close(descriptor)

    def save_report(self, arguments):
        self._validate_report(arguments)
        with self._lock:
            self._check_roots()
            observations = []
            for name in arguments["files"]:
                previous = self._observations.get(name)
                if previous is None:
                    raise BrainError("observation_required", "Read or inspect every source before saving a report")
                current = self._observe(name)
                if current != previous:
                    raise BrainError("file_changed", "A source changed after its recorded observation")
                observations.append(current)
            raw = _report_bytes(observations)
            result = {"filename": arguments["filename"], "size_bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(), "files": observations}
            self._install_report(arguments["filename"], raw)
            verified = self.verify_report(arguments, result)
            if not verified["verified"]:
                raise BrainError("report_changed", "Saved report or a source changed during verification")
            return result

    def verify_report(self, arguments, recorded_result):
        """Verify an already recorded write without repeating it, including after restart."""
        try:
            self._validate_report(arguments)
            if type(recorded_result) is not dict or set(recorded_result) != _REPORT_FIELDS:
                return {"verified": False}
            if recorded_result["filename"] != arguments["filename"]:
                return {"verified": False}
            with self._lock:
                self._check_roots()
                observations = [self._observe(name) for name in arguments["files"]]
                expected = _report_bytes(observations)
                result = {"filename": arguments["filename"], "size_bytes": len(expected),
                          "sha256": hashlib.sha256(expected).hexdigest(), "files": observations}
                if not _same_json(recorded_result, result):
                    return {"verified": False}
                current = self._read_bytes(self._report_fd, arguments["filename"], len(expected))
                if current != expected:
                    return {"verified": False}
                return {"verified": True, "result": result}
        except (BrainError, OSError, ValueError, KeyError, TypeError):
            return {"verified": False}

    def _task_evidence(self, snapshot):
        """Replay committed receipts, rechecking every accepted read against its file."""
        observed, pending = {}, {}
        verification = {"verified": False}
        with self._lock:
            if (type(snapshot) is not dict or type(snapshot.get("task_id")) is not str
                    or not snapshot["task_id"] or type(snapshot.get("journal")) is not list):
                raise BrainError("invalid_input", "Restoration requires a host task snapshot")
            inputs = self.validate_task_inputs(snapshot["task_inputs"])
            self._check_roots()
            for event in snapshot["journal"]:
                if type(event) is not dict:
                    raise BrainError("invalid_input", "Task journal entries must be objects")
                name, kind = event.get("tool"), event.get("kind")
                if name not in {"inspect_file", "read_file", "save_report"}:
                    continue
                if kind == "effect_started":
                    pending[name] = {"arguments": event.get("arguments"), "recorded": False}
                    continue
                if kind == "tool_failed":
                    pending.pop(name, None)
                    continue
                if kind not in {"handler_returned", "tool_result", "recovery_verification"}:
                    continue
                started = pending.get(name)
                if started is None:
                    continue
                arguments = started["arguments"]
                if kind in {"handler_returned", "tool_result"}:
                    if not _same_json(arguments, event.get("arguments")):
                        continue
                    started["result"] = event.get("result")
                    started["recorded"] = True
                    if kind == "handler_returned":
                        # A receipt alone confirms neither a read nor a write;
                        # a later matching host verifier is required.
                        continue
                    confirmed = event.get("verification")
                else:
                    confirmed = event.get("observation")
                if (not started["recorded"] or type(confirmed) is not dict
                        or confirmed.get("verified") is not True):
                    continue
                result = started["result"]
                if ("result" in confirmed and not _same_json(confirmed["result"], result)):
                    continue
                pending.pop(name, None)
                if name in {"inspect_file", "read_file"}:
                    self._validate_file(arguments)
                    filename = arguments["filename"]
                    if filename not in inputs["files"]:
                        continue
                    try:
                        current = self._observe(filename, content=name == "read_file")
                    except BrainError as error:
                        if error.code not in {"file_changed", "file_unavailable", "file_too_large"}:
                            raise
                        observed.pop(filename, None)
                        continue
                    if _same_json(current, result):
                        observed[filename] = {key: current[key] for key in _OBSERVATION_FIELDS}
                    continue
                self._validate_report(arguments)
                if (arguments["filename"] != inputs["report"]
                        or set(arguments["files"]) != set(inputs["files"])
                        or set(observed) != set(inputs["files"])):
                    continue
                if (type(result) is not dict
                        or not _same_json(result.get("files"), [observed[file] for file in arguments["files"]])):
                    continue
                verified = self.verify_report(arguments, result)
                if verified["verified"]:
                    verification = verified
        return observed, verification

    def restore_task_observations(self, snapshot):
        """Host-only callback before a task step; no model-provided evidence is accepted.

        The controller checks its persisted policy before invoking this callback.
        Replacing this thread's cache, including for an empty journal, prevents
        observations from a previous task authorizing a new task's report.
        """
        self._task_observations.values = {}
        observed, _verification = self._task_evidence(snapshot)
        self._task_observations.values = deepcopy(observed)
        return {"restored_files": sorted(observed)}

    def completion_verifier(self, snapshot):
        """Check this task's committed reads, returned report and actual current bytes."""
        try:
            _observed, verification = self._task_evidence(snapshot)
            return verification
        except (BrainError, OSError, ValueError, TypeError, KeyError):
            return {"verified": False}
