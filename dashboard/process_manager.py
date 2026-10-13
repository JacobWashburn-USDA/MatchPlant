"""
Tracks subprocesses launched from the dashboard so their stdout/stderr can be
streamed to the browser (via SSE) and so they can be stopped from the UI.
"""

import subprocess
import threading
import time
import uuid
import os
import signal

_LOCK = threading.Lock()
_RUNS = {}  # run_id -> Run


class Run:
    def __init__(self, run_id, module_id, command, cwd):
        self.run_id = run_id
        self.module_id = module_id
        self.command = command
        self.cwd = str(cwd)
        self.started_at = time.time()
        self.done = False
        self.returncode = None
        self.popen = None
        self.lines = []
        self.condition = threading.Condition()
        self.stop_requested = False


def start(module_id, command, cwd):
    run_id = uuid.uuid4().hex[:12]
    run = Run(run_id, module_id, command, cwd)

    with _LOCK:
        if any(existing.module_id == module_id and not existing.done for existing in _RUNS.values()):
            raise ValueError("This module already has a running process. View its log or stop it before starting another.")
        _RUNS[run_id] = run

    def target():
        try:
            popen = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                start_new_session=os.name != "nt",
                bufsize=1,
            )
            run.popen = popen
            if run.stop_requested:
                stop(run_id)
            for line in popen.stdout:
                message = line.rstrip("\n")
                with run.condition:
                    run.lines.append(message)
                    run.condition.notify_all()
            popen.wait()
            run.returncode = popen.returncode
        except Exception as exc:  # surfaced to the UI log, not swallowed
            message = f"[dashboard] failed to launch: {exc}"
            with run.condition:
                run.lines.append(message)
            run.returncode = -1
        finally:
            with run.condition:
                run.done = True
                run.condition.notify_all()

    threading.Thread(target=target, daemon=True).start()
    return run_id


def get(run_id):
    with _LOCK:
        return _RUNS.get(run_id)


def output(run, after=0):
    """Yield numbered output without consuming other viewers' log lines."""
    cursor = max(0, after)
    while True:
        with run.condition:
            if cursor >= len(run.lines) and not run.done:
                run.condition.wait(timeout=15)
            batch = list(enumerate(run.lines[cursor:], cursor + 1))
            finished = run.done
        for number, line in batch:
            cursor = number
            yield number, line
        if finished:
            return
        if not batch:
            yield cursor, None  # SSE keep-alive


def stop(run_id):
    run = get(run_id)
    if run is None or run.done:
        return False
    run.stop_requested = True
    if run.popen is None:
        return True
    try:
        if os.name == 'nt':
            result = subprocess.run(['taskkill', '/PID', str(run.popen.pid), '/T', '/F'],
                                    capture_output=True, timeout=10)
            return result.returncode == 0
        os.killpg(run.popen.pid, signal.SIGTERM)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def list_runs():
    with _LOCK:
        return sorted(_RUNS.values(), key=lambda r: r.started_at, reverse=True)
