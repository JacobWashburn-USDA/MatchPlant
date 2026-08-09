"""
Checks whether a module's currently-configured Python interpreter already
has its requirements installed, so the dashboard can tell a user "this is
ready" or "here's what's missing" without them ever needing to open Settings
or run anything first.

Compares against installed *distribution* names (importlib.metadata), not
import names, since requirements.txt lists distribution names (e.g.
"opencv-python"), which usually differ from their import name (e.g. "cv2"),
so this is the only reliable way to check without a hardcoded mapping table.
"""

import re
import subprocess

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name):
    """PEP 503 normalization, so e.g. 'PyYAML' matches 'pyyaml'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(requirements_text):
    names = []
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _NAME_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def missing_packages(python_exe, requirements_text, timeout=10):
    """Return the subset of requirement names not installed for python_exe.
    Returns all requirement names (i.e. "assume nothing is installed") if
    the interpreter can't be queried at all: a stale or nonexistent path
    should read as "needs setup", not silently pass."""
    names = _requirement_names(requirements_text)
    if not names:
        return []

    probe = (
        "import importlib.metadata as m;"
        "print('\\n'.join(d.metadata['Name'] for d in m.distributions() if d.metadata.get('Name')))"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return names

    installed = {_normalize(n) for n in result.stdout.splitlines() if n.strip()}
    return [n for n in names if _normalize(n) not in installed]
