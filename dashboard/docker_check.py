"""
Checks whether Docker is installed and running, for modules (like ODM
Runner) that shell out to it. Mirrors what dependency_check.py does for
Python packages, but for a system-level Docker install instead.
"""

import subprocess


def docker_status():
    """Returns one of "ready", "not_installed", "not_running"."""
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "not_installed"

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "not_running"

    return "ready" if result.returncode == 0 else "not_running"
