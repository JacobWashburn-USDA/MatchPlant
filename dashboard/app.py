"""
MatchPlant Dashboard: a local web front end for launching each pipeline
module (native GUIs and CLI/config-driven scripts) and watching their output.

This only ever talks to processes on the machine it runs on: GUI modules pop
open a native Tkinter window, and CLI modules run with a live log stream.
Run it with `python app.py` on your own workstation, not on a remote server.
"""

import platform
import subprocess
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

import dependency_check
import docker_check
import process_manager
import settings
import venv_manager
from modules import MODULES, MODULES_BY_ID, github_links, module_dir, requirements_path, requirements_text

app = Flask(__name__)


@app.context_processor
def inject_repo_status():
    return {"needs_repo_setup": not settings.is_valid_repo_root(settings.get_repo_root())}


def current_variant():
    system = platform.system()
    if system == "Darwin":
        return "mac", None
    if system == "Windows":
        return "win", None
    return "mac", (
        f"Detected OS '{system}'. This pipeline ships mac/win variants only; "
        "defaulting to the mac scripts. Override the interpreter/script in Settings if needed."
    )


def resolve_script(module, variant=None):
    scripts = module["script"]
    if "any" in scripts:
        return module_dir(module) / scripts["any"]
    variant = variant or current_variant()[0]
    rel = scripts.get(variant) or scripts.get("mac") or scripts.get("win")
    return module_dir(module) / rel


def build_command(module, python_exe, script_path, form):
    if module["kind"] == "shell":
        if script_path.suffix == ".ps1":
            return ["powershell", "-File", str(script_path)]
        return ["bash", str(script_path)]

    cmd = [python_exe, str(script_path)]
    for f in module["fields"]:
        if f["kind"] == "cwd":
            continue
        if f["kind"] == "checkbox":
            if form.get(f["name"]) == "on":
                cmd.append(f["flag"])
            continue
        val = (form.get(f["name"]) or "").strip()
        if val:
            cmd.append(f["flag"])
            cmd.append(val)
    return cmd


@app.route("/")
def index():
    stages = {}
    for m in MODULES:
        stages.setdefault(m["stage"], []).append(m)
    _, warning = current_variant()
    return render_template("index.html", stages=stages, os_warning=warning)


@app.route("/module/<module_id>")
def module_detail(module_id):
    module = MODULES_BY_ID.get(module_id)
    if module is None:
        return "Module not found", 404
    variant, warning = current_variant()
    req_name, req_text = requirements_text(module, variant)
    config_text = None
    if module["kind"] == "config":
        cfg = module_dir(module) / module["config_file"]
        config_text = cfg.read_text() if cfg.exists() else ""

    python_exe = settings.get_python(module_id)
    missing = dependency_check.missing_packages(python_exe, req_text) if req_text else []
    docker_status = docker_check.docker_status() if module["kind"] == "shell" else None

    return render_template(
        "module.html",
        module=module,
        req_name=req_name,
        req_text=req_text,
        has_requirements=req_name is not None,
        missing_packages=missing,
        docker_status=docker_status,
        source_links=github_links(module, req_name),
        variant=variant,
        os_warning=warning,
        python_exe=python_exe,
        config_text=config_text,
        runs=[r for r in process_manager.list_runs() if r.module_id == module_id][:5],
    )


@app.route("/module/<module_id>/config/save", methods=["POST"])
def save_config(module_id):
    module = MODULES_BY_ID.get(module_id)
    if module is None or module["kind"] != "config":
        return "Not found", 404
    cfg = module_dir(module) / module["config_file"]
    cfg.write_text(request.form.get("config_text", ""))
    return redirect(url_for("module_detail", module_id=module_id))


@app.route("/module/<module_id>/run", methods=["POST"])
def run_module(module_id):
    module = MODULES_BY_ID.get(module_id)
    if module is None:
        return jsonify({"error": "unknown module"}), 404

    variant = request.form.get("variant")
    script_path = resolve_script(module, variant)
    if not script_path.exists():
        return jsonify({"error": f"script not found: {script_path}"}), 400

    if module["kind"] == "shell":
        cwd = Path(request.form.get("project_folder", "")).expanduser()
        if not cwd.is_dir():
            return jsonify({"error": f"project folder does not exist: {cwd}"}), 400
    else:
        cwd = module_dir(module)

    python_exe = settings.get_python(module_id)
    command = build_command(module, python_exe, script_path, request.form)
    run_id = process_manager.start(module_id, command, cwd)
    return jsonify({"run_id": run_id, "command": " ".join(command)})


@app.route("/module/<module_id>/install", methods=["POST"])
def install_requirements(module_id):
    module = MODULES_BY_ID.get(module_id)
    if module is None:
        return jsonify({"error": "unknown module"}), 404

    variant, _ = current_variant()
    req_file = requirements_path(module, variant)
    if req_file is None:
        return jsonify({"error": "this module has no requirements file"}), 400

    try:
        python_path = venv_manager.ensure_venv(module_id)
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"failed to create virtual environment: {exc}"}), 500

    # Point this module at its dedicated venv now, so the next Run uses it
    # once the install below finishes (or picks up whatever partially
    # installed if it fails, which is still the honest current state).
    settings.set_python(module_id, str(python_path))

    command = [str(python_path), "-m", "pip", "install", "-r", str(req_file)]
    run_id = process_manager.start(module_id, command, module_dir(module))
    return jsonify({"run_id": run_id, "command": " ".join(command)})


@app.route("/api/stream/<run_id>")
def stream(run_id):
    run = process_manager.get(run_id)
    if run is None:
        return "Run not found", 404

    def generate():
        yield f"data: [dashboard] running: {' '.join(run.command)}\n\n"
        yield f"data: [dashboard] cwd: {run.cwd}\n\n"
        while True:
            line = run.queue.get()
            if line is None:
                code = run.returncode
                yield f"data: [dashboard] process exited with code {code}\n\n"
                yield "event: done\ndata: done\n\n"
                break
            safe = line.replace("\r", "")
            yield f"data: {safe}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stop/<run_id>", methods=["POST"])
def stop_run(run_id):
    ok = process_manager.stop(run_id)
    return jsonify({"stopped": ok})


@app.route("/api/browse-folder", methods=["POST"])
def browse_folder():
    """Opens a native folder picker on this machine, so the user can point
    at their download without typing a path by hand. Uses the OS's own
    picker (AppleScript / PowerShell) rather than Python's tkinter, since
    that isn't reliably available in every Python install (some pyenv
    builds skip it entirely)."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select the MatchPlant repo folder")'],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return jsonify({"path": ""})  # user cancelled
            return jsonify({"path": result.stdout.strip()})

        if system == "Windows":
            script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
                "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return jsonify({"path": result.stdout.strip()})
    except subprocess.SubprocessError as exc:
        return jsonify({"error": str(exc)}), 500

    # Other platforms: fall back to tkinter, if this interpreter has it.
    script = (
        "import tkinter, tkinter.filedialog\n"
        "root = tkinter.Tk()\n"
        "root.withdraw()\n"
        "print(tkinter.filedialog.askdirectory() or '')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.SubprocessError as exc:
        return jsonify({"error": str(exc)}), 500
    if result.returncode != 0:
        return jsonify({"error": "No folder picker available. Type the path in manually instead."}), 500
    return jsonify({"path": result.stdout.strip()})


@app.route("/guide")
def guide():
    return render_template("guide.html")


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    repo_error = None
    if request.method == "POST":
        if "repo_path" in request.form:
            path = request.form.get("repo_path", "").strip()
            if not path:
                repo_error = "Enter a folder path."
            elif not settings.is_valid_repo_root(path):
                repo_error = f"That doesn't look like the MatchPlant folder: {path}"
            else:
                settings.set_repo_root(path)
                return redirect(url_for("settings_page"))
        else:
            for module in MODULES:
                key = f"python_{module['id']}"
                if key in request.form and request.form[key].strip():
                    settings.set_python(module["id"], request.form[key].strip())
            return redirect(url_for("settings_page"))

    repo_root = settings.get_repo_root()
    return render_template(
        "settings.html",
        modules=MODULES,
        get_python=settings.get_python,
        current_path=str(repo_root),
        is_valid=settings.is_valid_repo_root(repo_root),
        repo_error=repo_error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)
