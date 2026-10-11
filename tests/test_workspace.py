"""Synthetic local records verify approval and history contracts, not model skill."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ia_propia import workspace as ws


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.root = self.folder / "private" / "default"
        self.initial = ws.init_workspace(self.root)

    def task_attempt(self, goal="Encargo sintético para pruebas", text="Intento sintético", evidence=None):
        task = ws.add_task(self.root, goal, "Contexto declarado por el usuario")
        attempt = ws.record_attempt(self.root, task["id"], text, evidence)
        return task, attempt

    def approve(self, task, attempt, text="Aceptado explícitamente para esta prueba"):
        return ws.feedback(self.root, task["id"], attempt["id"], text, accepted=True)

    def log_bytes(self):
        return (self.root / ws.LOG_NAME).read_bytes()

    def test_init_is_private_and_idempotent_without_replacing_history(self):
        before = self.log_bytes()
        repeated = ws.init_workspace(self.root)
        self.assertEqual(self.initial["workspaceId"], repeated["workspaceId"])
        self.assertEqual(repeated["privacy"], "PRIVATE")
        self.assertEqual(repeated["eventCount"], 1)
        self.assertEqual(before, self.log_bytes())
        self.assertEqual(ws.show(self.root)["tasks"], [])

    @unittest.skipUnless(os.name == "posix", "Unix permission bits")
    def test_private_directories_log_lock_and_exports_have_restricted_modes(self):
        task, attempt = self.task_attempt()
        self.approve(task, attempt)
        result = ws.export_training(self.root, self.folder / "exports" / "corpus.txt")
        for directory in (self.root, self.root.parent, self.folder / "exports"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for file in (self.root / ws.LOG_NAME, self.root / ".lock", Path(result["destination"]), Path(result["manifest"])):
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)

    def test_task_attempt_and_feedback_append_preserving_original_bytes(self):
        before = self.log_bytes()
        task, attempt = self.task_attempt(evidence=["local-reference.txt", "https://example.invalid/reference"])
        response = self.approve(task, attempt)
        shown = ws.show(self.root, task["id"])
        self.assertTrue(self.log_bytes().startswith(before))
        self.assertEqual(shown["goal"], task["goal"])
        self.assertEqual(shown["attempts"][0]["id"], attempt["id"])
        self.assertEqual(shown["attempts"][0]["feedback"][-1]["eventId"], response["eventId"])
        self.assertIs(shown["attempts"][0]["accepted"], True)
        counts = ws.status(self.root)
        self.assertEqual((counts["taskCount"], counts["attemptCount"], counts["acceptedAttemptCount"], counts["eventCount"]), (1, 1, 1, 4))
        shown["goal"] = "Caller changed returned snapshot"
        self.assertEqual(ws.show(self.root, task["id"])["goal"], task["goal"])

    def test_invalid_identifiers_and_cross_task_feedback_never_write_events(self):
        task, attempt = self.task_attempt()
        other = ws.add_task(self.root, "Otro encargo sintético")
        before = self.log_bytes()
        for task_id in ("../../events.jsonl", "task_" + "0" * 32):
            with self.assertRaises(ws.WorkspaceError):
                ws.record_attempt(self.root, task_id, "No guardar")
        with self.assertRaises(ws.WorkspaceError):
            ws.feedback(self.root, other["id"], attempt["id"], "Wrong association", accepted=True)
        with self.assertRaises(ws.WorkspaceError):
            ws.show(self.root, "invalid")
        self.assertEqual(before, self.log_bytes())

    def test_list_summarizes_current_approval_without_returning_full_history(self):
        context = "CONTEXT_MARKER " * 80
        text = "ATTEMPT_MARKER " * 100
        comment = "FEEDBACK_MARKER " * 90
        task = ws.add_task(self.root, "Synthetic compact listing", context)
        first = ws.record_attempt(self.root, task["id"], text, ["EVIDENCE_MARKER"])
        ws.record_attempt(self.root, task["id"], "Unreviewed synthetic attempt")
        self.approve(task, first, comment)
        listed = ws.list_tasks(self.root)
        self.assertEqual(listed["privacy"], "PRIVATE")
        self.assertEqual(listed["eventCount"], ws.status(self.root)["eventCount"])
        summary = listed["tasks"][0]
        self.assertEqual(set(summary), {"id", "goal", "createdAt", "attemptCount", "acceptedAttemptCount"})
        self.assertEqual((summary["attemptCount"], summary["acceptedAttemptCount"]), (2, 1))
        self.assertEqual(summary["id"], task["id"])
        self.assertEqual(summary["goal"], task["goal"])
        serialized = json.dumps(listed)
        for marker in ("CONTEXT_MARKER", "ATTEMPT_MARKER", "FEEDBACK_MARKER", "EVIDENCE_MARKER"):
            self.assertNotIn(marker, serialized)
        ws.feedback(self.root, task["id"], first["id"], "Latest explicit rejection", accepted=False)
        before = self.log_bytes()
        self.assertEqual(ws.list_tasks(self.root)["tasks"][0]["acceptedAttemptCount"], 0)
        shown = ws.show(self.root, task["id"])
        self.assertEqual(shown["context"], context)
        self.assertEqual(shown["attempts"][0]["text"], text)
        self.assertEqual(shown["attempts"][0]["feedback"][0]["text"], comment)
        self.assertEqual(len(shown["attempts"][0]["feedback"]), 2)
        self.assertEqual(self.log_bytes(), before)

    def test_approval_requires_actual_boolean_and_inputs_have_limits(self):
        task, attempt = self.task_attempt()
        before = self.log_bytes()
        for accepted in ("true", 1, None):
            with self.assertRaises(ws.WorkspaceError):
                ws.feedback(self.root, task["id"], attempt["id"], "Explicit feedback", accepted=accepted)
        with self.assertRaises(ws.WorkspaceError):
            ws.add_task(self.root, "x" * (ws.MAX_TEXT_BYTES + 1))
        with self.assertRaises(ws.WorkspaceError):
            ws.record_attempt(self.root, task["id"], "Some attempt", evidence="Not a list")
        with self.assertRaises(ws.WorkspaceError):
            ws.record_attempt(self.root, task["id"], "Some attempt", evidence=["reference"] * (ws.MAX_EVIDENCE_REFERENCES + 1))
        self.assertEqual(before, self.log_bytes())

    def test_export_uses_only_explicitly_accepted_attempts_with_source_manifest(self):
        task, approved = self.task_attempt(text="APPROVED_CONTENT", evidence=["Declared reference only"])
        ignored = ws.record_attempt(self.root, task["id"], "UNREVIEWED_CONTENT")
        rejected = ws.record_attempt(self.root, task["id"], "REJECTED_CONTENT")
        acceptance = self.approve(task, approved, "Only this attempt is accepted")
        ws.feedback(self.root, task["id"], rejected["id"], "Reject this", accepted=False)
        result = ws.export_training(self.root, self.folder / "corpus.txt")
        corpus = Path(result["destination"]).read_text(encoding="utf-8")
        self.assertIn("APPROVED_CONTENT", corpus)
        self.assertIn(task["goal"], corpus)
        self.assertIn(task["context"], corpus)
        self.assertIn(acceptance["text"], corpus)
        self.assertNotIn("UNREVIEWED_CONTENT", corpus)
        self.assertNotIn("REJECTED_CONTENT", corpus)
        self.assertIsNone(ignored["accepted"])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["sampleCount"], 1)
        self.assertEqual(manifest["samples"][0]["sourceEventIds"], [task["eventId"], approved["eventId"], acceptance["eventId"]])
        self.assertEqual(manifest["samples"][0]["evidenceReferences"], approved["evidence"])
        self.assertEqual(manifest["sourceEventLogSha256"], hashlib.sha256(self.log_bytes()).hexdigest())
        self.assertEqual(manifest["corpusSha256"], hashlib.sha256(Path(result["destination"]).read_bytes()).hexdigest())
        self.assertFalse(manifest["parametersTrained"])
        self.assertFalse(manifest["evidenceReferencesVerified"])

    def test_latest_feedback_wins_and_later_rejection_removes_export_eligibility(self):
        task, attempt = self.task_attempt()
        self.approve(task, attempt)
        ws.export_training(self.root, self.folder / "before.txt")
        ws.feedback(self.root, task["id"], attempt["id"], "Ahora rechazado", accepted=False)
        self.assertIs(ws.show(self.root, task["id"])["attempts"][0]["accepted"], False)
        destination = self.folder / "after.txt"
        with self.assertRaises(ws.WorkspaceError):
            ws.export_training(self.root, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(Path(str(destination) + ".manifest.json").exists())
        last = self.approve(task, attempt, "Aceptado otra vez explícitamente")
        result = ws.export_training(self.root, self.folder / "readmitted.txt")
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["samples"][0]["sourceEventIds"][-1], last["eventId"])
        self.assertEqual(len(ws.show(self.root, task["id"])["attempts"][0]["feedback"]), 3)

    def test_three_accepted_examples_remain_three_language_documents_even_with_literal_separators(self):
        from ia_propia.language.model import read_documents, split_documents

        for number in range(3):
            task = ws.add_task(self.root, f"Synthetic task {number}\n---\nKeep one example", "Context\n \t---  \nMore context")
            attempt = ws.record_attempt(self.root, task["id"], f"Synthetic attempt {number}\n---\nContinuation")
            self.approve(task, attempt, "Accepted\n---\nFeedback continuation")
        result = ws.export_training(self.root, self.folder / "language-corpus.txt")
        documents, _ = read_documents(result["destination"])
        self.assertEqual(len(documents), 3)
        splits, _ = split_documents(documents)
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [1, 1, 1])
        for document in documents:
            self.assertIn("\\---", document)
            self.assertIn("Feedback continuation", document)
        self.assertIn("\n---\n", ws.show(self.root)["tasks"][0]["goal"])

    def test_empty_or_unreviewed_workspace_never_invents_training_examples(self):
        for _ in range(2):
            destination = self.folder / "empty.txt"
            with self.assertRaises(ws.WorkspaceError):
                ws.export_training(self.root, destination)
            self.assertFalse(destination.exists())
            self.task_attempt()

    def test_export_cannot_replace_history_existing_evidence_or_manifest(self):
        evidence = self.folder / "evidence.txt"
        evidence.write_text("Original explicit evidence", encoding="utf-8")
        task, attempt = self.task_attempt(evidence=[str(evidence)])
        self.approve(task, attempt)
        before = self.log_bytes()
        for protected in (self.root / ws.LOG_NAME, self.root / ".lock", evidence):
            with self.assertRaises(ws.WorkspaceError):
                ws.export_training(self.root, protected)
        self.assertEqual(before, self.log_bytes())
        self.assertEqual(evidence.read_text(), "Original explicit evidence")
        output = self.folder / "new.txt"
        manifest = Path(str(output) + ".manifest.json")
        manifest.write_text("Keep existing manifest", encoding="utf-8")
        with self.assertRaises(ws.WorkspaceError):
            ws.export_training(self.root, output)
        self.assertFalse(output.exists())
        self.assertEqual(manifest.read_text(), "Keep existing manifest")

    def test_corruption_is_reported_without_reset_by_read_write_or_init(self):
        task, _ = self.task_attempt()
        log = self.root / ws.LOG_NAME
        with log.open("ab") as stream:
            stream.write(b'{"unfinished":')
        original = self.log_bytes()
        operations = [lambda: ws.show(self.root), lambda: ws.list_tasks(self.root), lambda: ws.init_workspace(self.root),
                      lambda: ws.record_attempt(self.root, task["id"], "Do not write")]
        for operation in operations:
            with self.assertRaises(ws.WorkspaceError):
                operation()
            self.assertEqual(original, self.log_bytes())

    def test_reconstruction_rejects_duplicate_keys_unknown_schema_and_wrong_numbers(self):
        original = self.log_bytes()
        event = json.loads(original)
        variants = []
        for field, value in (("at", "123"), ("at", True), ("at", -1), ("schemaVersion", True),
                             ("schemaVersion", 2), ("kind", "not-a-kind")):
            altered = dict(event)
            altered[field] = value
            variants.append(json.dumps(altered).encode() + b"\n")
        altered = dict(event, unknownField="unexpected")
        variants.append(json.dumps(altered).encode() + b"\n")
        variants.append(original.replace(b'"at":', b'"at":0,"at":', 1))
        variants.append(original.replace(str(event["at"]).encode(), b"NaN", 1))
        variants.append(original + original)  # Duplicate initialization/event identity.
        for variant in variants:
            with self.subTest(raw=variant[:100]):
                (self.root / ws.LOG_NAME).write_bytes(variant)
                with self.assertRaises(ws.WorkspaceError):
                    ws.status(self.root)
                self.assertEqual((self.root / ws.LOG_NAME).read_bytes(), variant)
        (self.root / ws.LOG_NAME).write_bytes(original)

    def test_concurrent_local_writers_keep_all_tasks_and_reconstructible_records(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = list(executor.map(lambda number: ws.add_task(self.root, f"Synthetic concurrent task {number}"), range(12)))
        self.assertEqual(len({task["id"] for task in tasks}), 12)
        self.assertEqual(ws.status(self.root)["taskCount"], 12)
        self.assertEqual(ws.status(self.root)["eventCount"], 13)

    def test_repeated_reads_reuse_replay_but_detect_same_size_rewrite_with_restored_mtime(self):
        task = ws.add_task(self.root, "alpha")
        ws.status(self.root)
        with patch.object(ws.json, "loads", wraps=ws.json.loads) as decoder:
            ws.status(self.root)
            ws.show(self.root, task["id"])
            ws.list_tasks(self.root)
            self.assertEqual(decoder.call_count, 0)
        log = self.root / ws.LOG_NAME
        before = log.read_bytes()
        timestamps = log.stat()
        changed = before.replace(b'"goal":"alpha"', b'"goal":"omega"')
        self.assertEqual(len(changed), len(before))
        self.assertNotEqual(changed, before)
        log.write_bytes(changed)
        os.utime(log, ns=(timestamps.st_atime_ns, timestamps.st_mtime_ns))
        self.assertEqual(ws.show(self.root, task["id"])["goal"], "omega")
        with ws._locked(self.root):
            self.assertEqual(ws._load(self.root)["logSha256"], hashlib.sha256(changed).hexdigest())
        # A cached valid history must not conceal in-place corruption either.
        damaged = changed.replace(b'"schemaVersion":1', b'"schemaVersion":2', 1)
        log.write_bytes(damaged)
        os.utime(log, ns=(timestamps.st_atime_ns, timestamps.st_mtime_ns))
        for operation in (lambda: ws.status(self.root), lambda: ws.list_tasks(self.root),
                          lambda: ws.init_workspace(self.root)):
            with self.assertRaises(ws.WorkspaceError):
                operation()
            self.assertEqual(log.read_bytes(), damaged)

    def test_cache_does_not_publish_failed_or_partially_written_events(self):
        before = self.log_bytes()
        ws.status(self.root)
        with patch.object(ws.os, "write", side_effect=OSError("Synthetic write failure")):
            with self.assertRaises(OSError):
                ws.add_task(self.root, "Must not become phantom cached memory")
        self.assertEqual(self.log_bytes(), before)
        self.assertEqual(ws.show(self.root)["tasks"], [])
        original_write = os.write
        calls = []

        def partial_write(descriptor, data):
            if calls:
                raise OSError("Synthetic failure after a partial write")
            calls.append(True)
            return original_write(descriptor, data[:4])

        with patch.object(ws.os, "write", side_effect=partial_write):
            with self.assertRaises(OSError):
                ws.add_task(self.root, "Interrupted event")
        damaged = self.log_bytes()
        self.assertEqual(damaged, before + b'{"sc')
        with self.assertRaises(ws.WorkspaceError):
            ws.status(self.root)
        self.assertEqual(self.log_bytes(), damaged)

    def test_fsync_error_reconstructs_actual_bytes_instead_of_claiming_rollback(self):
        ws.status(self.root)
        with patch.object(ws.os, "fsync", side_effect=OSError("Synthetic durability failure")):
            with self.assertRaises(OSError):
                ws.add_task(self.root, "Complete bytes were written before failed fsync")
        actual = self.log_bytes()
        shown = ws.show(self.root)
        self.assertEqual(len(shown["tasks"]), 1)
        self.assertEqual(shown["tasks"][0]["goal"], "Complete bytes were written before failed fsync")
        self.assertEqual(ws.status(self.root)["eventCount"], 2)
        self.assertEqual(self.log_bytes(), actual)

    def test_other_process_correction_invalidates_warm_replay_and_export_eligibility(self):
        task, attempt = self.task_attempt()
        self.approve(task, attempt)
        self.assertIs(ws.show(self.root, task["id"])["attempts"][0]["accepted"], True)
        subprocess.run([sys.executable, "-c",
                        "import sys; from ia_propia import workspace as ws; "
                        "ws.feedback(sys.argv[1], sys.argv[2], sys.argv[3], "
                        "'Synthetic correction from another process', accepted=False)",
                        str(self.root), task["id"], attempt["id"]], check=True, capture_output=True)
        shown = ws.show(self.root, task["id"])
        self.assertIs(shown["attempts"][0]["accepted"], False)
        self.assertEqual(len(shown["attempts"][0]["feedback"]), 2)
        self.assertEqual(ws.status(self.root)["acceptedAttemptCount"], 0)
        with self.assertRaises(ws.WorkspaceError):
            ws.export_training(self.root, self.folder / "revoked.txt")
        shown["attempts"][0]["accepted"] = True
        shown["attempts"][0]["feedback"].clear()
        self.assertIs(ws.show(self.root, task["id"])["attempts"][0]["accepted"], False)

    def test_replay_cache_is_bounded_and_eviction_does_not_mix_workspaces(self):
        roots = [self.folder / f"product-{index}" for index in range(4)]
        with patch.object(ws, "_MAX_CACHED_WORKSPACES", 2):
            with ws._REPLAY_CACHE_GUARD:
                ws._REPLAY_CACHE.clear()
            for index, root in enumerate(roots):
                ws.init_workspace(root)
                ws.add_task(root, f"Only product {index}")
                ws.status(root)
                with ws._REPLAY_CACHE_GUARD:
                    self.assertLessEqual(len(ws._REPLAY_CACHE), 2)
            for index, root in enumerate(roots):
                self.assertEqual([task["goal"] for task in ws.show(root)["tasks"]], [f"Only product {index}"])
        oversized_root = self.folder / "uncached"
        with patch.object(ws, "_MAX_CACHED_LOG_BYTES", 1):
            ws.init_workspace(oversized_root)
            ws.add_task(oversized_root, "Too large for the deliberately tiny cache")
            ws.status(oversized_root)
            with ws._REPLAY_CACHE_GUARD:
                self.assertNotIn(str(oversized_root.resolve()), ws._REPLAY_CACHE)

    def test_independent_workspaces_use_the_global_lru_safely_with_parallel_writers(self):
        roots = [self.folder / f"parallel-product-{index}" for index in range(6)]
        for root in roots:
            ws.init_workspace(root)

        def writes(index):
            for number in range(10):
                ws.add_task(roots[index], f"product {index} task {number}")
            return ws.status(roots[index])

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(writes, range(6)))
        for index, result in enumerate(results):
            self.assertEqual((result["taskCount"], result["eventCount"]), (10, 11))
            self.assertEqual([task["goal"] for task in ws.show(roots[index])["tasks"]],
                             [f"product {index} task {number}" for number in range(10)])
        with ws._REPLAY_CACHE_GUARD:
            self.assertLessEqual(len(ws._REPLAY_CACHE), ws._MAX_CACHED_WORKSPACES)

    def test_search_validates_before_loading_and_copies_only_detached_results(self):
        task, attempt = self.task_attempt(goal="Synthetic viento task", text="Synthetic approved clue")
        self.approve(task, attempt, "Latest explicit viento correction")
        with patch.object(ws, "_load", wraps=ws._load) as loader:
            result = ws.search_tasks(self.root, "VIENTO", limit=1)
            self.assertEqual(loader.call_count, 1)
        self.assertEqual(result["tasks"][0]["id"], task["id"])
        before = self.log_bytes()
        result["tasks"][0]["matches"][0]["source"]["eventId"] = "caller edit"
        result["tasks"].clear()
        self.assertEqual(ws.search_tasks(self.root, "VIENTO", limit=1)["tasks"][0]["id"], task["id"])
        self.assertEqual(self.log_bytes(), before)
        with patch.object(ws, "_load", wraps=ws._load) as loader:
            for query, limit in (("", 1), ("viento", True), ("viento", 21), ("x" * 513, 1)):
                with self.assertRaises(ws.WorkspaceError):
                    ws.search_tasks(self.folder / "not-initialized", query, limit)
            self.assertEqual(loader.call_count, 0)


if __name__ == "__main__":
    unittest.main()
