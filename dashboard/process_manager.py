"""
Tracks subprocesses launched from the dashboard so their stdout/stderr can be
streamed to the browser (via SSE) and so they can be stopped from the UI.
"""

import queue
import subprocess
import threading
import time
import uuid

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
        self.queue = queue.Queue()
        self.popen = None


def start(module_id, command, cwd):
    run_id = uuid.uuid4().hex[:12]
    run = Run(run_id, module_id, command, cwd)

    with _LOCK:
        _RUNS[run_id] = run

    def target():
        try:
            popen = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            run.popen = popen
            for line in popen.stdout:
                run.queue.put(line.rstrip("\n"))
            popen.wait()
            run.returncode = popen.returncode
        except Exception as exc:  # surfaced to the UI log, not swallowed
            run.queue.put(f"[dashboard] failed to launch: {exc}")
            run.returncode = -1
        finally:
            run.done = True
            run.queue.put(None)  # sentinel

    threading.Thread(target=target, daemon=True).start()
    return run_id


def get(run_id):
    with _LOCK:
        return _RUNS.get(run_id)


def stop(run_id):
    run = get(run_id)
    if run is None or run.popen is None:
        return False
    run.popen.terminate()
    return True


def list_runs():
    with _LOCK:
        return sorted(_RUNS.values(), key=lambda r: r.started_at, reverse=True)
