"""
Small persisted settings store: which Python interpreter to use per module,
and where the MatchPlant pipeline repo is located on this machine.

Modules in this pipeline have very different dependency stacks (Tkinter-only
GUIs, PyTorch training, GDAL/rasterio geospatial code). One interpreter
rarely satisfies all of them, so each module can point at its own venv.

The dashboard (this webapp/ folder) doesn't have to live inside the same
download as the pipeline modules, so the repo location is configurable
rather than assumed.
"""

import json
import sys
from pathlib import Path

INSTANCE_DIR = Path(__file__).resolve().parent / "instance"
SETTINGS_FILE = INSTANCE_DIR / "settings.json"

# If nobody has configured a repo location yet, assume the common case:
# webapp/ still lives inside the downloaded pipeline repo, one level up.
_AUTO_DETECTED_REPO_ROOT = Path(__file__).resolve().parent.parent

# A handful of files that only exist in a real MatchPlant checkout, used to
# tell "this is the right folder" from "this is some other folder".
_REPO_MARKERS = (
    "0_gps_embeder/gps_embed.py",
    "6-1_obj_det_trainer/train.py",
)


def _load():
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(data):
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def get_python(module_id):
    data = _load()
    return data.get("python", {}).get(module_id) or sys.executable


def set_python(module_id, python_path):
    data = _load()
    data.setdefault("python", {})[module_id] = python_path
    _save(data)


def is_valid_repo_root(path):
    path = Path(path)
    return path.is_dir() and all((path / marker).exists() for marker in _REPO_MARKERS)


def get_repo_root():
    """The configured MatchPlant repo folder, or the auto-detected default
    if none has been set yet. Always returns a Path; check
    is_valid_repo_root() separately before trusting it."""
    data = _load()
    configured = data.get("repo_root")
    return Path(configured) if configured else _AUTO_DETECTED_REPO_ROOT


def repo_root_is_configured():
    return "repo_root" in _load()


def set_repo_root(path):
    data = _load()
    data["repo_root"] = str(Path(path))
    _save(data)
