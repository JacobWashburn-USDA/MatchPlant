#!/usr/bin/env bash
# One-time setup for the MatchPlant Dashboard on macOS/Linux.
# Finds a usable Python, creates a venv, and installs dashboard dependencies.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

echo "MatchPlant Dashboard setup"
echo "---------------------------"

OS_NAME="$(uname -s)"
PYTHON_BIN=""

try_python() {
  local candidate="$1"
  [ -z "$PYTHON_BIN" ] || return 0
  command -v "$candidate" >/dev/null 2>&1 || return 0
  # Apple's developer-tools Python is incomplete for installing applications.
  if [ "$OS_NAME" = "Darwin" ] && [ "$(command -v "$candidate")" = "/usr/bin/python3" ]; then
    return 0
  fi
  if "$candidate" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
  fi
}

try_python "./.venv/bin/python"
try_python python3
try_python python
if [ -n "${CONDA_PREFIX:-}" ]; then
  try_python "$CONDA_PREFIX/bin/python"
fi
for candidate in \
  /opt/homebrew/bin/python3 \
  /usr/local/bin/python3 \
  /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
  "$HOME/anaconda3/bin/python" \
  "$HOME/miniconda3/bin/python" \
  "$HOME/miniforge3/bin/python" \
  /opt/anaconda3/bin/python \
  /opt/miniconda3/bin/python; do
  try_python "$candidate"
done

if [ -z "$PYTHON_BIN" ]; then
  echo
  echo "No usable Python 3.9+ installation was found."
  echo "Install Python or Anaconda, then run setup again."
  echo
  if [ "$OS_NAME" = "Darwin" ]; then
    echo "[P] Python:   https://www.python.org/downloads/macos/"
    echo "[A] Anaconda: https://www.anaconda.com/download"
    echo "[C] Close setup"
    echo
    if [ -t 0 ]; then
      while true; do
        printf 'Type P to open Python downloads, A for Anaconda, or C to close: '
        if ! IFS= read -r -n 1 choice; then
          echo
          exit 1
        fi
        echo
        case "$choice" in
          [Pp]) open "https://www.python.org/downloads/macos/" || true; break ;;
          [Aa]) open "https://www.anaconda.com/download" || true; break ;;
          [Cc]) exit 1 ;;
          *) echo "Choose P, A, or C." ;;
        esac
      done
      echo
      echo "Finish installing Python or Anaconda, then close this window"
      echo "and double-click setup.command again."
    fi
  else
    echo "Python:   https://www.python.org/downloads/"
    echo "Anaconda: https://www.anaconda.com/download"
    echo "Run ./setup.sh again after installation."
  fi
  exit 1
fi

echo "Found Python: $("$PYTHON_BIN" --version) ($PYTHON_BIN)"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv"
fi

./.venv/bin/python -m pip install --quiet --upgrade pip
echo "Installing dashboard dependencies ..."
./.venv/bin/python -m pip install --quiet -r requirements.txt

cat <<'EOF'

Setup complete.

To start the dashboard, double-click start_dashboard.command.

Note: this only sets up the dashboard itself. Individual pipeline modules
(training, testing, GUIs) may need their own packages installed the first
time you run them from the dashboard -- it will tell you what's missing.
EOF
