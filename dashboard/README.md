# MatchPlant Dashboard

![License](https://img.shields.io/badge/License-MIT-blue)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

A local web front end for the [MatchPlant](https://github.com/JacobWashburn-USDA/MatchPlant) pipeline,
included in this repository as the `dashboard/` folder.
Landing page: **[matchplant-dashboard.github.io](https://matchplant-dashboard.github.io)**.

- Lists all 11 pipeline modules across four stages (steps 0-9; the two step-6 training routes are alternatives), with a pipeline-flow card for jumping straight to any module
- Launch a module's native GUI, or run its CLI/script with real arguments
- Live log output streamed to the browser
- Downloaded together with the pipeline modules; the dashboard finds them automatically since they live in the same repository
- Runs entirely on your own machine. Desktop modules (GCP Finder, BBox Drawer, etc.) open a native GUI window on the machine running the dashboard, so don't deploy this to a remote server and expect the GUI to appear in your browser

## Setup

**Double-click the files for your operating system, in this order:**

| System | First-time setup | Start the dashboard |
| --- | --- | --- |
| Windows | **1_Setup_Windows.bat** | **2_Start_Windows.bat** |
| macOS | **1_Setup_macOS.command** | **2_Start_macOS.command** |

The platform names stay visible even when the file browser hides extensions. Run setup once, then use Start each time. Keep the MatchPlant Dashboard server window open while working.

The `_internal/` folder contains application code, assets, requirements, and helper scripts. Users do not need to open or run its files directly. Keep that folder alongside the four launchers. Existing `.venv/` and `instance/` folders stay at the dashboard root so installed environments and saved settings keep their paths.

**Prefer the terminal?** Run the setup script. It checks for Python, tells you what to install if it's missing, and does the rest for you:

```bash
bash _internal/setup.sh          # Windows: 1_Setup_Windows.bat
```

Then start the dashboard:

```bash
./.venv/bin/python _internal/app.py     # Windows: .venv\Scripts\python _internal/app.py
```

Then open http://127.0.0.1:5050

**Already comfortable with Python/venvs?** The manual equivalent:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r _internal/requirements.txt
python _internal/app.py
```

## Python setup and first-run help

The Windows launchers are included in this folder. Run `1_Setup_Windows.bat` once, then use `2_Start_Windows.bat` to start the dashboard.

`1_Setup_Windows.bat` checks a working dashboard environment, active virtual/Conda environments, common Python and Anaconda/Miniconda/Miniforge locations, and Python commands on PATH. An optional embedded PowerShell helper also searches Windows Python registrations, registered installation locations, Python launcher entries, Python install-manager runtimes, and known Conda environments, including custom locations recorded in `.conda/environments.txt`. No username or personal installation path is hardcoded. Each candidate must actually run Python 3.9+ with `venv`; inherited `PYTHONHOME` and `PYTHONPATH` overrides are cleared for setup. Portable or custom installations that are not registered or discoverable can be selected by pressing **E** and entering their executable path. Or press **P** for Python downloads, **A** for Anaconda downloads, or **C** to close. The download options open the official websites; finish installation and run setup again. Clickable terminal links depend on the terminal, so the keyboard menu is the reliable option. The discovery helper is embedded in `_internal/setup_windows.bat`, called by the Windows setup launcher.

On macOS, `1_Setup_macOS.command` calls `_internal/setup.sh`. Discovery checks a working dashboard environment, active virtual/Conda environments, Python framework versions, standard and custom Homebrew prefixes, pyenv versions, Conda environment records and common named-environment folders, and multiple Python versions across PATH. Candidates must run Python 3.9+ with `venv`; setup clears inherited `PYTHONHOME` and `PYTHONPATH` overrides and reports why a discovered executable fails validation. If it misses a portable or custom installation, press **E** and paste its executable path. You can find that path by running `command -v python3` in a separate Terminal window. Press **P** to open the official macOS Python downloads, **A** for Anaconda downloads, or **C** to close. Finish installing, then double-click `1_Setup_macOS.command` again. The setup script does not install Python without your action. On Linux, `_internal/setup.sh` prints the download links for manual installation.

If macOS shows “1_Setup_macOS.command” Not Opened before setup starts, follow the [macOS installation help](https://matchplant-dashboard.github.io/#get-started). The Python download choices appear only after the script is allowed to open.

The dashboard installs Flask, PyYAML, and packaging from `_internal/requirements.txt`; module dependencies are installed separately. If updating an existing dashboard, rerun setup to install added dependencies.

## Forms and guidance

Use **Browse…** to select local folders and files, or paste/type paths. Output folders can be typed before they exist. Help icons show field explanations on hover, click, or keyboard focus; Escape closes them. Defaults remain visible. Read each module's README before running. These pickers select locations on the computer running the dashboard; they do not upload data.

## Connecting to the MatchPlant pipeline

- Since `dashboard/` lives inside the MatchPlant repository, the dashboard finds the pipeline modules automatically, no setup needed
- If you ever move this folder somewhere else, or want to point it at a different MatchPlant checkout, the **Settings** page inside the dashboard has a **Browse...** button that opens a native folder picker to relocate it

## Per-module Python environments

- Modules have very different dependencies (Tkinter-only GUIs, PyTorch training, GDAL/rasterio geospatial code), so one environment usually can't satisfy all of them
- Each module page checks its required package versions automatically and offers an **Install requirements** button that creates a dedicated environment for that module
- The **Settings** page lists which interpreter each module is using. Only needed if you already have a working environment (e.g. an existing PyTorch/CUDA setup) you'd rather point a module at instead

## Notes per module

- **ODM Runner** is launched by the cross-platform Python runner `run_ODM_process.py`, which calls Docker directly without PowerShell or Bash. It needs Docker Desktop running (the dashboard checks this live on that module's page), and needs a project folder path with an `images/` subfolder. Enter that path in the form; the script runs with that folder as its working directory.
- **GPS Embedder** takes UTM zone, hemisphere, and the image/events folder in its form. Standalone terminal prompts remain available.
- **Transfer Learning** uses a settings form with file and folder pickers. Run saves the settings to `transfer_config.yaml` before launching; an advanced YAML editor is also available.
- Modules with mac/win variants (GCP Finder, Min Image Finder, BBox Drawer, Image Splitter, Spatial Stats Extractor) auto-select the script for your OS; a dropdown lets you override it.

## Form validation and run controls

Browse buttons accept existing input files/folders; output fields also accept typed new folder paths. Required paths and numeric values are checked before a process starts. Relative input paths use the module folder, except Module 8’s DSM/output paths, which use its ODM dataset folder. Transfer Learning includes Browse fields for checkpoint and validation output folders.

Only one process per module runs at a time. **View log** reopens previous output without consuming another viewer’s log. A dropped log connection reconnects; **Stop** stops the launched process tree. Installing requirements uses the selected operating-system variant. Installed package versions are checked against requirements, but this is not a full runtime/hardware test.

The dashboard binds to localhost and starts with Flask debug/reloader disabled, avoiding extra development processes. Update an existing dashboard environment with `python -m pip install -r _internal/requirements.txt` to install the added `packaging` dependency.

## How it works

```mermaid
flowchart LR
    Browser(["Browser"])
    App["app.py<br/>Flask routes"]
    Modules["modules.py<br/>module manifest"]
    Settings["settings.py<br/>instance/settings.json"]
    Docker["docker_check.py<br/>Docker install/running check"]
    ProcMgr["process_manager.py<br/>launches subprocess"]
    VenvMgr["venv_manager.py<br/>per-module venv"]
    Script["module script or native GUI"]

    Browser <-->|"HTTP + live log (SSE)"| App
    App -->|"module list, form fields"| Modules
    App -->|"pipeline location,<br/>per-module interpreter"| Settings
    App -->|"ODM Runner page"| Docker
    App -->|"Install requirements"| VenvMgr
    App -->|"run / stop"| ProcMgr
    VenvMgr -->|"pip install"| ProcMgr
    ProcMgr -->|"stdout / stderr"| Script
```

- `_internal/modules.py`: manifest of all pipeline modules, script location(s), kind (`gui`, `cli`, `config`, `shell`), and for CLI tools, the argparse-derived form fields. Resolves each module's folder against the MatchPlant pipeline location configured in Settings, not this dashboard's own folder.
- `_internal/process_manager.py`: launches each run as a subprocess and retains its stdout/stderr for live streaming and log replay.
- `_internal/transfer_form.py`: reads and validates transfer-learning form values, updating the existing YAML configuration while retaining advanced values.
- `_internal/app.py`: Flask routes for the dashboard, module detail/form, run (`POST`), live log (`GET`, Server-Sent Events), and stop.
- `_internal/settings.py`: persists the MatchPlant pipeline location and per-module interpreter paths to `instance/settings.json` (not committed).
- `_internal/docker_check.py`: live Docker install/running check, used on the ODM Runner module page.
- `_internal/venv_manager.py`: creates a dedicated virtual environment per module (used by the **Install requirements** button), so one module's dependencies never collide with another's.
- `_internal/static/logo.png`, `_internal/static/images/img.png`: the dashboard's own bundled branding and pipeline-diagram assets (independent of wherever the MatchPlant pipeline is located).

## Dashboard module badges

- **Desktop app:** GCP Finder, Minimum Image Finder, BBox Drawer, Image Splitter, and Spatial Stats Extractor open a separate desktop window.
- **Form:** GPS Embedder, Object Detection Tester, and Image-to-Ortho Projector use page inputs and a live log.
- **Training:** Both training routes use settings forms and live logs; Transfer Learning also has an advanced YAML editor.
- **Docker:** ODM Runner calls Docker through Python.

These user-facing badges are separate from the internal execution categories.

## Developer checks

From the dashboard folder, run `python -m unittest discover -s _internal/tests` using the dashboard environment. The Flask app is `_internal/app.py`; templates and static assets live beside it. Keep `.venv/`, `instance/`, caches, and generated data out of commits. When updating an existing checkout, record the file moves and remove the old root-level launchers and application files rather than leaving duplicate copies.

## Native file and folder pickers

Browse uses the repository-level `matchplant_ui/` package, also used by all five native GUI modules. Folder pickers use the standard system dialog and select folders only. Windows may hide individual files in this dialog. Desktop file fields retain their own CSV/raster requirements while showing all file types. Keep `matchplant_ui/` alongside the pipeline folders when distributing the dashboard.
