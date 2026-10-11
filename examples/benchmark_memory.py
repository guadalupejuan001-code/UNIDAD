"""Benchmark workspace replay and bounded search on fictional local records.

Only stdlib and the selected local workspace/search sources are imported. This
measures warmed CPU/filesystem operations, not model training or learning.
Use separate, new output files when comparing historical and current sources.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


def fixture(directory):
    directory.mkdir(mode=0o700)
    events = []

    def event(kind, payload):
        number = len(events) + 1
        events.append({"schemaVersion": 1, "id": f"event_{number:032x}",
                       "at": 1_600_000_000_000 + number, "kind": kind, "payload": payload})

    event("workspace_initialized", {"workspaceId": "workspace_" + "a" * 32, "privacy": "PRIVATE"})
    for number in range(200):
        task_id = f"task_{number + 1:032x}"
        event("task_added", {"taskId": task_id, "goal": f"Synthetic benchmark task {number}",
                             "context": "Explicit fictional context for performance only. " * 8})
        for branch in range(3):
            attempt_id = f"attempt_{number * 3 + branch + 1:032x}"
            event("attempt_recorded", {"taskId": task_id, "attemptId": attempt_id,
                                      "text": "Synthetic proposal; never a real creator record. " * 12,
                                      "evidence": ["fictional-reference"]})
            for revision in range(2):
                event("feedback_recorded", {"taskId": task_id, "attemptId": attempt_id,
                                           "text": f"Synthetic explicit correction {revision}; performance fixture only.",
                                           "accepted": revision == 1 and branch % 2 == 0})
    raw = b"".join((json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for item in events)
    (directory / "events.jsonl").write_bytes(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trusted local workspace.py implementation to measure")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path; use a separate new file for each measurement")
    parser.add_argument("--retrieval-helper", type=Path, help="Optional current memory_search.py; both sources use this same helper")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file; existing evidence is never overwritten")
    script_bytes = Path(__file__).read_bytes()
    source = args.source.read_bytes()
    helper_bytes = args.retrieval_helper.read_bytes() if args.retrieval_helper is not None else None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    spec = importlib.util.spec_from_file_location("ia_propia._bench_workspace", args.source)
    workspace = importlib.util.module_from_spec(spec)
    exec(compile(source, str(args.source), "exec"), workspace.__dict__)
    helper = None
    if args.retrieval_helper is not None:
        helper_spec = importlib.util.spec_from_file_location("ia_propia.memory_search", args.retrieval_helper)
        helper = importlib.util.module_from_spec(helper_spec)
        exec(compile(helper_bytes, str(args.retrieval_helper), "exec"), helper.__dict__)
        # search_tasks imports this exact name; both routes must use these bytes.
        sys.modules["ia_propia.memory_search"] = helper
    measurements = {}
    with tempfile.TemporaryDirectory(prefix="unidad-memory-benchmark-") as temp:
        directory = Path(temp) / "synthetic"
        log = fixture(directory)
        expected = {"taskCount": 200, "attemptCount": 600, "acceptedAttemptCount": 400, "eventCount": 2001}
        assert all(workspace.status(directory)[key] == value for key, value in expected.items())
        task_id = "task_" + f"{1:032x}"
        for name, operation in [("status", lambda: workspace.status(directory)),
                                ("list_tasks", lambda: workspace.list_tasks(directory)),
                                ("show_one_task", lambda: workspace.show(directory, task_id))]:
            for _ in range(3):
                operation()
            times = []
            for _ in range(31):
                started = time.perf_counter()
                operation()
                times.append(time.perf_counter() - started)
            measurements[name] = {"medianSeconds": statistics.median(times), "samples": 31,
                                  "minSeconds": min(times), "maxSeconds": max(times)}
        if helper is not None:
            query = "synthetic correction"

            def via_full_snapshot():
                snapshot = workspace.show(directory)
                state = {"workspaceId": snapshot["workspaceId"], "eventIds": range(snapshot["eventCount"]),
                         "tasks": {task["id"]: task for task in snapshot["tasks"]}}
                return helper.search_snapshot(state, query, limit=10)

            operation = (lambda: workspace.search_tasks(directory, query, limit=10)) if hasattr(workspace, "search_tasks") else via_full_snapshot
            assert operation() == via_full_snapshot()
            assert operation()["matchCount"] == 200
            for _ in range(3):
                operation()
            times = []
            for _ in range(31):
                started = time.perf_counter()
                operation()
                times.append(time.perf_counter() - started)
            measurements["lexical_retrieval"] = {"medianSeconds": statistics.median(times), "samples": 31,
                                                  "minSeconds": min(times), "maxSeconds": max(times),
                                                  "route": "bounded_under_lock" if hasattr(workspace, "search_tasks") else "full_detached_snapshot"}
        assert (directory / "events.jsonl").read_bytes() == log
        # A real append through the API, with the same fixed-size log at each trial.
        append_times = []
        append_root = Path(temp) / "append"
        for _ in range(11):
            if append_root.exists():
                (append_root / "events.jsonl").write_bytes(log)
            else:
                fixture(append_root)
            workspace.status(append_root)
            started = time.perf_counter()
            workspace.feedback(append_root, task_id, "attempt_" + f"{1:032x}",
                               "Synthetic measured later rejection", accepted=False)
            append_times.append(time.perf_counter() - started)
            current = workspace.status(append_root)
            assert current["acceptedAttemptCount"] == 399 and current["eventCount"] == 2002
        measurements["feedback_append"] = {"medianSeconds": statistics.median(append_times), "samples": 11,
                                            "minSeconds": min(append_times), "maxSeconds": max(append_times)}
    report = {"schemaVersion": 1, "sourceSha256": hashlib.sha256(source).hexdigest(),
              "fixtureSha256": hashlib.sha256(log).hexdigest(), "fixtureBytes": len(log), "fixtureCounts": expected,
              "measurements": measurements, "pythonVersion": sys.version, "torchImported": "torch" in sys.modules,
              "benchmarkScriptSha256": hashlib.sha256(script_bytes).hexdigest(),
              "retrievalHelperSha256": hashlib.sha256(helper_bytes).hexdigest() if helper is not None else None,
              "retrievalHelperModule": "ia_propia.memory_search" if helper is not None else None,
              "scope": "One process, warmed synthetic local log, existing public workspace operations; no models or creator memory.",
              "limits": ["Timing depends on CPU/filesystem/cache; not a model-learning or general latency guarantee.",
                         "Each feedback trial starts from identical bytes and warms the log before measuring the actual API append."]}
    assert report["torchImported"] is False
    if args.source.read_bytes() != source or Path(__file__).read_bytes() != script_bytes:
        raise RuntimeError("Benchmark or workspace source changed during measurement")
    if helper_bytes is not None and args.retrieval_helper.read_bytes() != helper_bytes:
        raise RuntimeError("Retrieval helper changed during measurement")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(args.output, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
