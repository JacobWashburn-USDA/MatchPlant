#!/usr/bin/env bash
# One-time setup for the MatchPlant Dashboard on macOS/Linux.
# Checks for Python, creates a venv, installs Flask, and tells you how to
# start the dashboard. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "MatchPlant Dashboard setup"
echo "---------------------------"

PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  cat <<'EOF'
No Python 3.9+ was found on this machine.

If you use Anaconda/Miniconda for other work, open an "Anaconda Prompt" /
activate your conda environment and re-run this script from there instead.

Otherwise, install Python from:
  https://www.python.org/downloads/macos/   (macOS)
  https://www.python.org/downloads/          (other)

During the installer, no special options are needed. Once installed, open a
new terminal window and re-run this script:
  ./setup.sh
EOF
  exit 1
fi

echo "Found Python: $("$PYTHON_BIN" --version) ($PYTHON_BIN)"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv"
fi

./.venv/bin/pip install --quiet --upgrade pip
echo "Installing dashboard dependencies (Flask) ..."
./.venv/bin/pip install --quiet -r requirements.txt

cat <<'EOF'

Setup complete.

To start the dashboard, double-click start_dashboard.command.

Note: this only sets up the dashboard itself. Individual pipeline modules
(training, testing, GUIs) may need their own packages installed the first
time you run them from the dashboard -- it will tell you what's missing.
EOF
