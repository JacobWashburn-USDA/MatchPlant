"""
Creates a dedicated virtual environment per module for the "Install
requirements" action, so a module's dependencies never have to share an
environment with another module's (PyTorch vs. GDAL vs. Tkinter-only).
"""

import subprocess
import sys
from pathlib import Path

VENVS_DIR = Path(__file__).resolve().parent / "instance" / "venvs"


def venv_python(module_id):
    root = VENVS_DIR / module_id
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def ensure_venv(module_id):
    """Return the venv's python executable, creating the venv first if needed."""
    python_path = venv_python(module_id)
    if not python_path.exists():
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENVS_DIR / module_id)],
            check=True,
        )
    return python_path
