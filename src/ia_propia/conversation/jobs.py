"""Detached local jobs. Processes remain dependent on the workspace lifetime."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .training import _atomic_json, _birth, _configs, _json_read, _run_lock


def _alive(pid, birth):
    if type(pid) is not int or pid <= 0 or not isinstance(birth, str):
        return False
    try:
        os.kill(pid, 0)
        actual = _birth(pid)
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return actual == birth and stat[0] != "Z"
    except (OSError, IndexError):
        return False


def status(run_dir):
    """Read durable state and report live/stale separately without needing torch."""
    path = Path(run_dir) / "status.json"
    if not path.exists():
        return {"schemaVersion": 1, "status": "unplanned", "processAlive": False,
                "environmentLifetimeGuaranteed": False}
    value = _json_read(path)
    live = _alive(value.get("pid"), value.get("processBirth"))
    value["processAlive"] = live
    value["heartbeatAgeSeconds"] = max(0.0, time.time() - value.get("heartbeatAt", time.time()))
    if value.get("status") in ("planned", "running") and not live:
        value["recordedStatus"] = value["status"]
        value["status"] = "stale"
        value["reason"] = "worker_not_alive"
    elif value.get("status") == "running" and value["heartbeatAgeSeconds"] > 120:
        value["heartbeatStale"] = True  # A live process may still be evaluating or writing.
    return value


def start(corpus_path, run_dir, *, model_config=None, training_config=None,
          resume=False, budget_seconds=None, initial_model=None):
    """Validate a trusted local request, launch one independent worker, return its PID."""
    _configs(model_config, training_config)
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if resume and initial_model is not None:
        raise ValueError("initial_model is only allowed for a new stage, not resume")
    if initial_model is not None:
        initial_model = str(Path(initial_model).resolve())
        if not Path(initial_model).is_dir():
            raise ValueError("Initial model must be an existing own-artifact directory")
    import math
    if budget_seconds is not None and (type(budget_seconds) not in (int, float) or
            not math.isfinite(budget_seconds) or not 0 < budget_seconds <= 86400):
        raise ValueError("Invalid invocation time budget")
    corpus_path = Path(corpus_path).resolve()
    if not corpus_path.is_file():
        raise ValueError("Corpus path must be an existing file")
    run_dir = Path(run_dir).resolve()
    with _run_lock(run_dir):
        existing = status(run_dir)
        if existing.get("processAlive") and existing.get("status") in ("planned", "running"):
            raise RuntimeError("This run already has an active worker")
        if resume:
            if not existing.get("lastCheckpoint"):
                raise ValueError("Run has no resumable checkpoint")
            if (run_dir / "evaluation.json").exists():
                raise ValueError("Finalized runs cannot resume; start a new run")
        elif existing["status"] != "unplanned":
            raise ValueError("Run already exists; resume it or choose a new directory")
        request_path = run_dir / f"request-{uuid.uuid4().hex}.json"
        request = {"corpus_path": str(corpus_path), "run_dir": str(run_dir),
                   "model_config": model_config, "training_config": training_config,
                   "resume": resume, "budget_seconds": budget_seconds, "initial_model": initial_model}
        _atomic_json(request_path, request)
        # Launch and publish under the lock; the worker briefly retries this startup lock.
        log_path = run_dir / "worker.log"
        fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "ia_propia.conversation.training", "worker", "--request", str(request_path)],
                stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        finally:
            os.close(fd)
        state = {key: value for key, value in existing.items()
                 if key not in {"processAlive", "recordedStatus", "heartbeatAgeSeconds", "heartbeatStale"}}
        state.update({"schemaVersion": 1, "status": "planned", "reason": "worker_starting",
                      "pid": process.pid, "processBirth": _birth(process.pid), "heartbeatAt": time.time(),
                      "environmentLifetimeGuaranteed": False})
        _atomic_json(run_dir / "status.json", state)
    # Retain and reap the child in long-lived callers; detached CLI callers may exit.
    import threading
    threading.Thread(target=process.wait, daemon=True, name="unidad-job-reaper").start()
    return status(run_dir)


def stop(run_dir):
    """Request a cooperative checkpoint/stop, signaling only the recorded live process."""
    value = status(run_dir)
    if value.get("status") not in ("planned", "running") or not value.get("processAlive"):
        return {**value, "stopRequested": False}
    pid, birth = value["pid"], value["processBirth"]
    _atomic_json(Path(run_dir) / "stop-request.json", {"pid": pid, "processBirth": birth, "requestedAt": time.time()})
    if value["status"] == "planned" or not value.get("cooperativeSignalInstalled", False):
        return {**status(run_dir), "stopRequested": True}
    # Check identity immediately before signaling. A pidfd prevents PID reuse on supported Linux.
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            descriptor = os.pidfd_open(pid)
            try:
                if not _alive(pid, birth):
                    return {**status(run_dir), "stopRequested": False}
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            finally:
                os.close(descriptor)
        except ProcessLookupError:
            return {**status(run_dir), "stopRequested": False}
    else:
        if not _alive(pid, birth):
            return {**status(run_dir), "stopRequested": False}
        os.kill(pid, signal.SIGTERM)
    return {**status(run_dir), "stopRequested": True}
