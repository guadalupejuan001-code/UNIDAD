"""Filesystem evidence remains constrained by the host and the current task."""

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from ia_propia.brain.file_tools import SafeFileTools
from ia_propia.brain.autonomy import AutonomyController, AutonomyTool
from ia_propia.brain.protocol import BrainError


class FileTaskFixture:
    origin = "test_fixture"
    proposer_id = "file-task-recovery-contract"

    def __init__(self, actions):
        self.actions = iter(actions)

    def __call__(self, _context):
        tool, arguments = next(self.actions)
        return json.dumps({"action": "tool", "tool": tool, "arguments": arguments,
                           "reason": "Exercise actual file recovery"})


class SafeFileToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.reports = self.root / "reports"
        self.inputs.mkdir()
        self.reports.mkdir()
        self.source = self.inputs / "primero.txt"
        self.source.write_bytes("Hola Perú.\nOtra línea".encode("utf-8"))
        self.other = self.inputs / "segundo.bin"
        self.other.write_bytes(b"\xff\x00\n")
        self.adapter = self.make_adapter()

    def make_adapter(self, **kwargs):
        adapter = SafeFileTools(self.inputs, self.reports, **kwargs)
        self.addCleanup(adapter.close)
        return adapter

    def report(self, files=None):
        files = files or ["primero.txt"]
        for filename in files:
            self.adapter.inspect_file({"filename": filename})
        arguments = {"filename": "informe.json", "files": files}
        return arguments, self.adapter.save_report(arguments)

    def test_observations_measure_exact_utf8_and_binary_bytes(self):
        text = self.adapter.read_file({"filename": "primero.txt"})
        raw = self.source.read_bytes()
        self.assertEqual(text["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(text["size_bytes"], len(raw))
        self.assertEqual(text["content"], raw.decode("utf-8"))
        self.assertEqual(text["statistics"], {"lines": 2, "utf8": True, "characters": 21, "words": 4})
        binary = self.adapter.read_file({"filename": "segundo.bin"})
        self.assertEqual(binary["encoding"], "base64")
        self.assertEqual(base64.b64decode(binary["content"]), self.other.read_bytes())
        self.assertIsNone(binary["statistics"]["words"])

    def test_rejects_traversal_roots_text_and_files_added_after_configuration(self):
        (self.inputs / "nuevo.txt").write_text("not configured", encoding="utf-8")
        for filename in ("../primero.txt", str(self.source), "..", "a\\b", "nuevo.txt"):
            with self.subTest(filename=filename), self.assertRaises(BrainError):
                self.adapter.read_file({"filename": filename})
        with self.assertRaises(BrainError):
            self.adapter.read_file({"filename": "primero.txt", "root": str(self.root)})
        with self.assertRaises(BrainError):
            self.adapter.save_report({"filename": "informe.json", "files": ["primero.txt"], "text": "invented"})
        with self.assertRaises(BrainError):
            self.adapter.validate_task_inputs({"files": ["primero.txt"], "report": "informe.json", "root": str(self.root)})
        with self.assertRaises(BrainError):
            self.adapter.validate_task_inputs({"files": ["primero.txt", "primero.txt"], "report": "informe.json"})

    def test_symlink_roots_ancestors_files_and_replacements_are_rejected(self):
        link = self.root / "input-link"
        link.symlink_to(self.inputs, target_is_directory=True)
        with self.assertRaises((BrainError, OSError)):
            SafeFileTools(link, self.reports)
        with self.assertRaises((BrainError, OSError)):
            SafeFileTools(link / ".", self.reports)
        (self.inputs / "nested").mkdir()
        with self.assertRaises((BrainError, OSError)):
            SafeFileTools(link / "nested", self.reports)
        self.source.unlink()
        self.source.symlink_to(self.other)
        with self.assertRaises(BrainError):
            self.adapter.inspect_file({"filename": "primero.txt"})
        self.source.unlink()
        replacement = self.root / "replacement"
        replacement.write_text("different inode", encoding="utf-8")
        replacement.replace(self.source)
        with self.assertRaises(BrainError):
            self.adapter.inspect_file({"filename": "primero.txt"})

    def test_overlapping_roots_are_rejected(self):
        nested = self.inputs / "reports"
        nested.mkdir()
        with self.assertRaises(BrainError):
            SafeFileTools(self.inputs, nested)
        with self.assertRaises(BrainError):
            SafeFileTools(self.inputs, self.inputs)
        with self.assertRaises(BrainError):
            SafeFileTools(nested, self.inputs)

    def test_replacement_with_reused_inode_still_rejects_real_new_bytes(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        captured = self.source.stat()
        original_bytes = self.source.read_bytes()
        self.source.unlink()
        self.source.write_bytes(b"X" * len(original_bytes))
        os.utime(self.source, ns=(captured.st_atime_ns, captured.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(self.source.stat().st_size, captured.st_size)
        # Deterministically simulate inode recycling while retaining the new
        # file's real revision and bytes. No read or replace operation is mocked.
        replacement = self.source.stat()
        actual_fstat, actual_stat = os.fstat, os.stat
        def recycled_identity(status):
            if (status.st_dev, status.st_ino) == (replacement.st_dev, replacement.st_ino):
                return SimpleNamespace(st_dev=captured.st_dev, st_ino=captured.st_ino,
                                       st_size=status.st_size, st_mtime_ns=status.st_mtime_ns,
                                       st_ctime_ns=status.st_ctime_ns, st_mode=status.st_mode)
            return status
        with patch("ia_propia.brain.file_tools.os.fstat", side_effect=lambda fd: recycled_identity(actual_fstat(fd))), \
                patch("ia_propia.brain.file_tools.os.stat", side_effect=lambda *args, **kwargs: recycled_identity(actual_stat(*args, **kwargs))):
            with self.assertRaises(BrainError) as error:
                self.adapter.read_file({"filename": "primero.txt"})
        self.assertEqual(error.exception.code, "file_changed")

    def test_in_place_changes_require_explicit_host_reconfiguration(self):
        captured = self.source.stat()
        self.source.write_bytes(b"changed bytes")
        self.assertEqual((self.source.stat().st_dev, self.source.stat().st_ino),
                         (captured.st_dev, captured.st_ino))
        with self.assertRaises(BrainError):
            self.adapter.inspect_file({"filename": "primero.txt"})
        refreshed = self.make_adapter()
        result = refreshed.read_file({"filename": "primero.txt"})
        self.assertEqual(result["sha256"], hashlib.sha256(b"changed bytes").hexdigest())
        self.assertNotEqual(self.adapter.tools[0].policy_id, refreshed.tools[0].policy_id)

    def test_limit_prevents_read_and_no_report_exists_without_observation(self):
        limited = self.make_adapter(max_file_bytes=2)
        with self.assertRaises(BrainError) as error:
            limited.read_file({"filename": "primero.txt"})
        self.assertEqual(error.exception.code, "file_too_large")
        with self.assertRaises(BrainError) as error:
            self.adapter.save_report({"filename": "informe.json", "files": ["primero.txt"]})
        self.assertEqual(error.exception.code, "observation_required")
        self.assertFalse((self.reports / "informe.json").exists())

    def test_report_is_real_exclusive_and_verifiable_after_restart(self):
        arguments, result = self.report(["primero.txt", "segundo.bin"])
        raw = (self.reports / "informe.json").read_bytes()
        report = json.loads(raw)
        self.assertEqual(report["files"], result["files"])
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertNotIn("content", report["files"][0])
        restarted = self.make_adapter()
        self.assertTrue(restarted.verify_report(arguments, result)["verified"])
        with self.assertRaises(BrainError) as error:
            self.adapter.save_report(arguments)
        self.assertEqual(error.exception.code, "report_exists")
        self.assertEqual(raw, (self.reports / "informe.json").read_bytes())

    def test_verifiers_reject_changed_sources_reports_and_forged_evidence(self):
        observation = self.adapter.inspect_file({"filename": "primero.txt"})
        arguments, result = self.report()
        wrong = deepcopy(result)
        wrong["sha256"] = "0" * 64
        self.assertFalse(self.adapter.verify_report(arguments, wrong)["verified"])
        (self.reports / "informe.json").write_text("forged report", encoding="utf-8")
        self.assertFalse(self.adapter.verify_report(arguments, result)["verified"])
        self.source.write_text("source changed", encoding="utf-8")
        self.assertFalse(self.adapter._verify_inspection({"filename": "primero.txt"}, observation)["verified"])
        with self.assertRaises(BrainError) as error:
            self.adapter.save_report({"filename": "otro.json", "files": ["primero.txt"]})
        self.assertEqual(error.exception.code, "file_changed")

    def test_existing_report_symlink_cannot_overwrite_its_target(self):
        outside = self.root / "outside.txt"
        outside.write_text("preserve", encoding="utf-8")
        (self.reports / "informe.json").symlink_to(outside)
        self.adapter.inspect_file({"filename": "primero.txt"})
        with self.assertRaises(BrainError) as error:
            self.adapter.save_report({"filename": "informe.json", "files": ["primero.txt"]})
        self.assertEqual(error.exception.code, "report_exists")
        self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")

    def test_root_replacement_does_not_redirect_reads_or_writes(self):
        self.inputs.rename(self.root / "original")
        self.inputs.mkdir()
        (self.inputs / "primero.txt").write_text("redirected", encoding="utf-8")
        with self.assertRaises(BrainError):
            self.adapter.inspect_file({"filename": "primero.txt"})

    def journal_entry(self, tool, arguments, result, *, verification=True):
        return [{"kind": "effect_started", "tool": tool, "arguments": deepcopy(arguments)},
                {"kind": "tool_result", "tool": tool, "arguments": deepcopy(arguments),
                 "result": deepcopy(result), "verification": {"verified": verification}}]

    def snapshot(self, journal, files=None):
        return {"task_id": "task_current", "task_inputs": {
            "files": files or ["primero.txt"], "report": "informe.json"}, "journal": journal}

    def test_completion_needs_current_task_reads_and_recorded_report(self):
        arguments, result = self.report(["primero.txt", "segundo.bin"])
        write = self.journal_entry("save_report", arguments, result)
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(write, arguments["files"]))["verified"])
        one = self.adapter.inspect_file({"filename": "primero.txt"})
        reads = self.journal_entry("inspect_file", {"filename": "primero.txt"}, one)
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(reads + write, arguments["files"]))["verified"])
        two = self.adapter.read_file({"filename": "segundo.bin"})
        reads += self.journal_entry("read_file", {"filename": "segundo.bin"}, two)
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(reads, arguments["files"]))["verified"])
        self.assertTrue(self.adapter.completion_verifier(self.snapshot(reads + write, arguments["files"]))["verified"])
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(write + reads, arguments["files"]))["verified"])
        forged = deepcopy(reads + write)
        forged[-1]["result"]["sha256"] = "0" * 64
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(forged, arguments["files"]))["verified"])
        self.source.write_text("changed", encoding="utf-8")
        self.assertFalse(self.adapter.completion_verifier(self.snapshot(reads + write, arguments["files"]))["verified"])

    def test_recovery_needs_recorded_result_and_rechecks_real_files(self):
        read_args = {"filename": "primero.txt"}
        observation = self.adapter.inspect_file(read_args)
        arguments, result = self.report()
        reads = self.journal_entry("inspect_file", read_args, observation)
        write = self.journal_entry("save_report", arguments, result)
        del write[-1]["verification"]
        recovery = {"kind": "recovery_verification", "tool": "save_report",
                    "observation": {"verified": True, "result": result}}
        restarted = self.make_adapter()
        self.assertFalse(restarted.verify_report(arguments, None)["verified"])
        self.assertTrue(restarted.completion_verifier(self.snapshot(reads + write + [recovery]))["verified"])
        self.assertFalse(restarted.completion_verifier(self.snapshot(reads + write[:1] + [recovery]))["verified"])

    def test_returned_receipt_requires_matching_successful_recovery(self):
        read_args = {"filename": "primero.txt"}
        observation = self.adapter.inspect_file(read_args)
        arguments, result = self.report()
        reads = self.journal_entry("inspect_file", read_args, observation)
        write = self.journal_entry("save_report", arguments, result)
        write[-1]["kind"] = "handler_returned"
        del write[-1]["verification"]
        recovery = {"kind": "recovery_verification", "tool": "save_report",
                    "observation": {"verified": True, "result": result}}
        restarted = self.make_adapter()
        self.assertFalse(restarted.completion_verifier(self.snapshot(reads + write))["verified"])
        self.assertTrue(restarted.completion_verifier(self.snapshot(reads + write + [recovery]))["verified"])
        for altered in (dict(recovery, observation={"verified": False, "result": result}),
                        dict(recovery, observation={"verified": True, "result": {"sha256": "0" * 64}})):
            self.assertFalse(restarted.completion_verifier(self.snapshot(reads + write + [altered]))["verified"])
        forged = deepcopy(write)
        forged[-1]["arguments"]["filename"] = "otro.json"
        self.assertFalse(restarted.completion_verifier(self.snapshot(reads + forged + [recovery]))["verified"])

    def test_ordinary_restart_restores_verified_reads_before_step(self):
        originals = {path.name: path.read_bytes() for path in (self.source, self.other)}
        for resume_first in (False, True):
            with self.subTest(resume_first=resume_first):
                report = f"restart-{int(resume_first)}.json"
                inputs = {"files": ["primero.txt", "segundo.bin"], "report": report}
                state = self.root / f"restart-state-{int(resume_first)}"
                initial = self.make_adapter()
                controller = AutonomyController(state, FileTaskFixture([
                    ("inspect_file", {"filename": "primero.txt"}),
                    ("read_file", {"filename": "segundo.bin"})]), initial.tools,
                    completion_verifier=initial.completion_verifier)
                task = controller.start("Resume prior reads and save a report", task_inputs=inputs)
                controller.step(task["task_id"])
                self.assertEqual(controller.step(task["task_id"])["status"], "ready")
                initial.close()
                restarted = self.make_adapter()
                reopened = AutonomyController(state, FileTaskFixture([
                    ("save_report", {"filename": report, "files": inputs["files"]})]), restarted.tools,
                    completion_verifier=restarted.completion_verifier)
                if resume_first:
                    self.assertEqual(reopened.resume(task["task_id"])["status"], "ready")
                result = reopened.step(task["task_id"])
                self.assertEqual(result["status"], "completed", result)
                rows = json.loads((self.reports / report).read_bytes())["files"]
                for row in rows:
                    self.assertEqual(row["sha256"], hashlib.sha256(originals[row["filename"]]).hexdigest())
        for name, raw in originals.items():
            self.assertEqual((self.inputs / name).read_bytes(), raw)

    def test_real_report_recovers_after_interrupted_verifier_without_rewrite(self):
        calls = []
        def bound_tools(adapter, interrupt):
            original = adapter.tools[-1]
            def save(arguments):
                calls.append(arguments)
                return original.handler(arguments)
            def verify(arguments, result):
                if interrupt:
                    raise KeyboardInterrupt("Interrupted after the returned receipt was committed")
                return original.verifier(arguments, result)
            tool = AutonomyTool(original.name, save, original.validator, description=original.description,
                                schema=original.schema, effect=original.effect,
                                verifier=verify, policy_id=original.policy_id)
            return (*adapter.tools[:-1], tool)
        values = {"files": ["primero.txt"], "report": "recover-returned.json"}
        state = self.root / "returned-state"
        original = self.make_adapter()
        controller = AutonomyController(state, FileTaskFixture([
            ("inspect_file", {"filename": "primero.txt"}),
            ("save_report", {"filename": values["report"], "files": values["files"]})]),
            bound_tools(original, True), completion_verifier=original.completion_verifier)
        task = controller.start("Recover an actual returned report", task_inputs=values)
        controller.step(task["task_id"])
        with self.assertRaises(KeyboardInterrupt):
            controller.step(task["task_id"])
        interrupted = controller.status(task["task_id"])
        self.assertEqual(interrupted["status"], "executing")
        self.assertEqual(interrupted["journal"][-1]["kind"], "handler_returned")
        raw = (self.reports / values["report"]).read_bytes()
        original.close()
        restarted = self.make_adapter()
        reopened = AutonomyController(state, FileTaskFixture([]), bound_tools(restarted, False),
                                      completion_verifier=restarted.completion_verifier)
        result = reopened.resume(task["task_id"])
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(calls), 1)
        self.assertEqual((self.reports / values["report"]).read_bytes(), raw)
        self.assertTrue(any(event["kind"] == "recovery_verification" and event["observation"]["verified"]
                            for event in result["journal"]))

    def test_restoration_rejects_stale_forged_and_other_task_cache(self):
        arguments = {"filename": "primero.txt"}
        result = self.adapter.inspect_file(arguments)
        journal = self.journal_entry("inspect_file", arguments, result)
        restarted = self.make_adapter()
        self.assertEqual(restarted.restore_task_observations(self.snapshot(journal))["restored_files"], ["primero.txt"])
        self.assertEqual(restarted.restore_task_observations(self.snapshot([]))["restored_files"], [])
        with self.assertRaises(BrainError):
            restarted.save_report({"filename": "other-task.json", "files": ["primero.txt"]})
        self.assertFalse((self.reports / "other-task.json").exists())
        forged = deepcopy(journal)
        forged[-1]["result"]["sha256"] = "0" * 64
        self.assertEqual(restarted.restore_task_observations(self.snapshot(forged))["restored_files"], [])
        self.source.write_text("changed source", encoding="utf-8")
        self.assertEqual(restarted.restore_task_observations(self.snapshot(journal))["restored_files"], [])
        with self.assertRaises(BrainError):
            restarted.save_report({"filename": "stale.json", "files": ["primero.txt"]})
        self.assertFalse((self.reports / "stale.json").exists())

    def test_unreturned_report_effect_stays_uncertain_without_blind_retry(self):
        calls = []
        def bound_tools(adapter):
            original = adapter.tools[-1]
            def interrupted_save(arguments):
                calls.append(arguments)
                original.handler(arguments)
                raise KeyboardInterrupt("Interrupted before any returned result was committed")
            return (*adapter.tools[:-1], AutonomyTool(
                original.name, interrupted_save, original.validator, description=original.description,
                schema=original.schema, effect=original.effect,
                verifier=original.verifier, policy_id=original.policy_id))
        values = {"files": ["primero.txt"], "report": "unreturned.json"}
        state = self.root / "unreturned-state"
        controller = AutonomyController(state, FileTaskFixture([
            ("inspect_file", {"filename": "primero.txt"}),
            ("save_report", {"filename": values["report"], "files": values["files"]})]),
            bound_tools(self.adapter), completion_verifier=self.adapter.completion_verifier)
        task = controller.start("Keep an unknown write outcome explicit", task_inputs=values)
        controller.step(task["task_id"])
        with self.assertRaises(KeyboardInterrupt):
            controller.step(task["task_id"])
        raw = (self.reports / values["report"]).read_bytes()
        self.assertEqual(controller.status(task["task_id"])["journal"][-1]["kind"], "effect_started")
        restarted = self.make_adapter()
        reopened = AutonomyController(state, FileTaskFixture([]), bound_tools(restarted),
                                      completion_verifier=restarted.completion_verifier)
        result = reopened.resume(task["task_id"])
        self.assertEqual(result["status"], "blocked_uncertain")
        self.assertEqual(reopened.step(task["task_id"])["status"], "blocked_uncertain")
        self.assertFalse(restarted.completion_verifier(result)["verified"])
        self.assertEqual(len(calls), 1)
        self.assertEqual((self.reports / values["report"]).read_bytes(), raw)

    def test_parallel_task_restoration_keeps_observations_on_each_worker(self):
        result = self.adapter.inspect_file({"filename": "primero.txt"})
        journal = self.journal_entry("inspect_file", {"filename": "primero.txt"}, result)
        both_restored = threading.Barrier(2)
        results = {}
        def worker(name, entries):
            try:
                self.adapter.restore_task_observations(self.snapshot(entries))
                both_restored.wait(timeout=2)
                results[name] = self.adapter.save_report({"filename": name + ".json", "files": ["primero.txt"]})
            except BaseException as error:
                results[name] = error
        threads = [threading.Thread(target=worker, args=("own-read", journal)),
                   threading.Thread(target=worker, args=("other-task", []))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertIsInstance(results["other-task"], BrainError)
        self.assertEqual(results["own-read"]["sha256"], hashlib.sha256((self.reports / "own-read.json").read_bytes()).hexdigest())
        self.assertFalse((self.reports / "other-task.json").exists())

    def test_controller_completes_from_real_report_and_not_shared_cache(self):
        class Fixture:
            origin = "test_fixture"
            proposer_id = "file-tools-integration"
            def __init__(self, actions):
                self.actions = iter(actions)
            def __call__(self, _context):
                return json.dumps(next(self.actions))
        actions = [
            {"action": "tool", "tool": "inspect_file", "arguments": {"filename": "primero.txt"}, "reason": "measure"},
            {"action": "tool", "tool": "read_file", "arguments": {"filename": "segundo.bin"}, "reason": "read"},
            {"action": "tool", "tool": "save_report", "arguments": {
                "filename": "informe.json", "files": ["primero.txt", "segundo.bin"]}, "reason": "save"},
        ]
        controller = AutonomyController(self.root / "state", Fixture(actions), self.adapter.tools,
                                        completion_verifier=self.adapter.completion_verifier)
        values = self.adapter.validate_task_inputs({"files": ["primero.txt", "segundo.bin"], "report": "informe.json"})
        task = controller.start("Measure the configured files and save the report", task_inputs=values)
        for _ in actions:
            task = controller.step(task["task_id"])
        self.assertEqual(task["status"], "completed", task)
        other = controller.start("Another task", task_inputs=values)
        self.assertFalse(self.adapter.completion_verifier(other)["verified"])

    def test_fd_link_install_is_atomic_when_named_temp_entry_is_replaced(self):
        # Force the portable named-temp branch, then replace its directory
        # entry just before installation. The descriptor must supply the inode.
        from unittest.mock import patch
        original_open, original_link = os.open, os.link
        target = self.root / "unrelated.txt"
        target.write_text("unrelated", encoding="utf-8")
        def open_without_tmpfile(path, flags, *args, **kwargs):
            if flags & getattr(os, "O_TMPFILE", 0) == getattr(os, "O_TMPFILE", -1):
                import errno
                raise OSError(errno.EOPNOTSUPP, "test unnamed temp unsupported")
            return original_open(path, flags, *args, **kwargs)
        def replace_temporary_then_link(source, destination, *args, **kwargs):
            temporary = next(self.reports.glob(".unidad-report-*"))
            temporary.rename(self.reports / "moved-temporary")
            temporary.symlink_to(target)
            return original_link(source, destination, *args, **kwargs)
        self.adapter.inspect_file({"filename": "primero.txt"})
        with patch("ia_propia.brain.file_tools.os.open", side_effect=open_without_tmpfile), \
                patch("ia_propia.brain.file_tools.os.link", side_effect=replace_temporary_then_link):
            result = self.adapter.save_report({"filename": "informe.json", "files": ["primero.txt"]})
        raw = (self.reports / "informe.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["sha256"])
        self.assertEqual(target.read_text(encoding="utf-8"), "unrelated")
        self.assertFalse((self.reports / "informe.json").is_symlink())

    def test_control_heavy_read_and_long_whitelist_keep_metadata_bounded(self):
        self.source.write_bytes(b"\x00" * 65_536)
        configured_after_write = self.make_adapter()
        result = configured_after_write.read_file({"filename": "primero.txt"})
        self.assertEqual(result["encoding"], "base64")
        self.assertLess(len(json.dumps(result).encode("utf-8")), 131_072)
        names = [f"{index:03d}_" + "x" * 200 for index in range(128)]
        for name in names:
            (self.inputs / name).write_bytes(b"sample")
        wide = self.make_adapter(filenames=names)
        for tool in wide.tools:
            self.assertLess(len(json.dumps(tool.schema).encode("utf-8")), 16_384)

    def other_adapter(self):
        inputs, reports = self.root / "other-inputs", self.root / "other-reports"
        inputs.mkdir()
        reports.mkdir()
        for filename in self.adapter.filenames:
            (inputs / filename).write_bytes((self.inputs / filename).read_bytes())
        adapter = SafeFileTools(inputs, reports)
        self.addCleanup(adapter.close)
        return adapter

    def test_policy_identity_binds_resources_and_limits_without_exposing_paths(self):
        restarted = self.make_adapter()
        different = self.other_adapter()
        limited = self.make_adapter(max_file_bytes=64)
        restricted = self.make_adapter(filenames=["primero.txt"])
        for original, same, changed, smaller, fewer in zip(
                self.adapter.tools, restarted.tools, different.tools, limited.tools, restricted.tools):
            self.assertEqual(original.policy_id, same.policy_id)
            self.assertEqual(original.metadata(), changed.metadata())
            self.assertNotEqual(original.policy_id, changed.policy_id)
            self.assertNotEqual(original.policy_id, smaller.policy_id)
            self.assertNotEqual(original.policy_id, fewer.policy_id)
            self.assertNotIn("policy_id", original.metadata())
            self.assertNotIn(str(self.inputs), json.dumps(original.metadata()))

    def test_controller_rebinding_rejects_resume_and_step_for_same_named_files(self):
        class Fixture:
            origin = "test_fixture"
            proposer_id = "file-policy-rebinding"
            calls = 0
            def __call__(self, _context):
                self.calls += 1
                return json.dumps({"action": "stop", "reason": "fixture done"})
        state_root = self.root / "binding-state"
        original = AutonomyController(state_root, Fixture(), self.adapter.tools,
                                      completion_verifier=self.adapter.completion_verifier)
        inputs = {"files": ["primero.txt"], "report": "informe.json"}
        same_task = original.start("Restart with the same roots", task_inputs=inputs)
        resume_task = original.start("Reject different roots on resume", task_inputs=inputs)
        step_task = original.start("Reject different roots on step", task_inputs=inputs)
        restarted = self.make_adapter()
        same = AutonomyController(state_root, Fixture(), restarted.tools,
                                  completion_verifier=restarted.completion_verifier)
        self.assertEqual(same.resume(same_task["task_id"])["status"], "ready")
        different = self.other_adapter()
        fixture = Fixture()
        changed = AutonomyController(state_root, fixture, different.tools,
                                     completion_verifier=different.completion_verifier)
        for state in (changed.resume(resume_task["task_id"]), changed.step(step_task["task_id"])):
            self.assertEqual(state["status"], "blocked_policy")
            self.assertTrue(any(event.get("code") == "host_binding_changed" for event in state["journal"]))
        self.assertEqual(fixture.calls, 0)


if __name__ == "__main__":
    unittest.main()
