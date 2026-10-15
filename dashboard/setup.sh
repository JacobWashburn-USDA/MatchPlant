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
  if "$candidate" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
  fi
}

try_python "./.venv/bin/python"
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
try_python python3
try_python python

if [ -z "$PYTHON_BIN" ]; then
  echo
  echo "Setup could not find a usable Python 3.9+ installation automatically."
  if [ "$OS_NAME" = "Darwin" ]; then
    if [ -t 0 ]; then
      while [ -z "$PYTHON_BIN" ]; do
        echo
        echo "[E] I already have Python (enter its full path)"
        echo "[P] Python downloads: https://www.python.org/downloads/macos/"
        echo "[A] Anaconda downloads: https://www.anaconda.com/download"
        echo "[C] Close setup"
        printf 'Choose E, P, A, or C, then press Return: '
        IFS= read -r choice || exit 1
        case "$choice" in
          [Ee])
            printf 'Paste the full path to your Python executable: '
            IFS= read -r candidate || exit 1
            candidate="${candidate#\"}"
            candidate="${candidate%\"}"
            try_python "$candidate"
            if [ -z "$PYTHON_BIN" ]; then
              echo "That path does not run Python 3.9+ with venv. Try another path."
            fi
            ;;
          [Pp])
            open "https://www.python.org/downloads/macos/" || true
            echo "After installing Python, double-click setup.command again."
            exit 1
            ;;
          [Aa])
            open "https://www.anaconda.com/download" || true
            echo "After installing Anaconda, double-click setup.command again."
            exit 1
            ;;
          [Cc]) exit 1 ;;
          *) echo "Choose E, P, A, or C." ;;
        esac
      done
    else
      echo "Python:   https://www.python.org/downloads/macos/"
      echo "Anaconda: https://www.anaconda.com/download"
      exit 1
    fi
  else
    echo "Python:   https://www.python.org/downloads/"
    echo "Anaconda: https://www.anaconda.com/download"
    echo "Run ./setup.sh again after installation."
    exit 1
  fi
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
