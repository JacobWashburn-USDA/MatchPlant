#!/usr/bin/env bash
# One-time setup for the MatchPlant Dashboard on macOS/Linux.
# Finds a usable Python, creates a venv, and installs dashboard dependencies.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "MatchPlant Dashboard setup"
echo "---------------------------"
echo "Setup file: $PWD/_internal/setup.sh"

# Other Python applications may leave overrides that break an unrelated install.
unset PYTHONHOME PYTHONPATH

OS_NAME="$(uname -s)"
PYTHON_BIN=""

try_python() {
  local candidate="$1"
  local resolved
  [ -z "$PYTHON_BIN" ] || return 0
  resolved="$(command -v "$candidate" 2>/dev/null)" || return 0
  [ -f "$resolved" ] && [ -x "$resolved" ] || return 0
  echo "Checking Python: $resolved"
  if "$resolved" -I -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$resolved"
    echo "Python detected successfully."
  else
    echo "This Python could not pass the Python 3.9+ and venv check:"
    "$resolved" -I -c 'import sys, venv; print(sys.version); sys.exit(0 if sys.version_info >= (3, 9) else 1)' || true
  fi
}

try_python "./.venv/bin/python"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  try_python "$VIRTUAL_ENV/bin/python"
fi
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
  "$HOME/mambaforge/bin/python" \
  /opt/anaconda3/bin/python \
  /opt/miniconda3/bin/python; do
  try_python "$candidate"
done
# Conda records retain custom installation and named-environment locations.
if [ -r "$HOME/.conda/environments.txt" ]; then
  while IFS= read -r environment || [ -n "$environment" ]; do
    environment="${environment%$'\r'}"
    [ -z "$environment" ] || try_python "$environment/bin/python"
  done < "$HOME/.conda/environments.txt"
fi
if [ -n "${CONDA_EXE:-}" ]; then
  try_python "$(dirname "$(dirname "$CONDA_EXE")")/bin/python"
fi

# Standard framework installs, Conda environments, and both Homebrew layouts.
# Unmatched patterns are harmless because try_python requires an executable file.
for candidate in \
  /Library/Frameworks/Python.framework/Versions/*/bin/python3 \
  "$HOME"/Library/Python/*/bin/python3 \
  "$HOME"/.conda/envs/*/bin/python \
  "$HOME"/{anaconda3,miniconda3,miniforge3,mambaforge}/envs/*/bin/python \
  /opt/{anaconda3,miniconda3,miniforge3,mambaforge}/bin/python \
  /opt/{anaconda3,miniconda3,miniforge3,mambaforge}/envs/*/bin/python \
  /opt/homebrew/opt/python*/bin/python3 \
  /opt/homebrew/opt/python*/bin/python3.[0-9] \
  /opt/homebrew/opt/python*/bin/python3.[0-9][0-9] \
  /opt/homebrew/opt/python*/libexec/bin/python3 \
  /usr/local/opt/python*/bin/python3 \
  /usr/local/opt/python*/bin/python3.[0-9] \
  /usr/local/opt/python*/bin/python3.[0-9][0-9] \
  /usr/local/opt/python*/libexec/bin/python3; do
  try_python "$candidate"
done

# Respect custom Homebrew and pyenv roots without changing the user's shell.
brew_command="$(command -v brew 2>/dev/null || true)"
brew_prefix="${HOMEBREW_PREFIX:-}"
if [ -z "$brew_prefix" ] && [ -n "$brew_command" ]; then
  brew_prefix="$(dirname "$(dirname "$brew_command")")"
fi
if [ -n "$brew_prefix" ]; then
  try_python "$brew_prefix/bin/python3"
  for candidate in "$brew_prefix"/opt/python*/bin/python3 \
    "$brew_prefix"/opt/python*/bin/python3.[0-9] \
    "$brew_prefix"/opt/python*/bin/python3.[0-9][0-9] \
    "$brew_prefix"/opt/python*/libexec/bin/python3; do
    try_python "$candidate"
  done
fi
pyenv_root="${PYENV_ROOT:-$HOME/.pyenv}"
for candidate in "$pyenv_root"/versions/*/bin/python "$pyenv_root"/versions/*/envs/*/bin/python; do
  try_python "$candidate"
done

# Check every PATH directory so an old Python cannot hide a newer installation.
IFS=: read -r -a python_path_dirs <<< "${PATH:-}"
for python_dir in "${python_path_dirs[@]}"; do
  [ -n "$python_dir" ] || continue
  for candidate in "$python_dir/python3" "$python_dir/python" "$python_dir/pypy3" \
    "$python_dir"/python3.[0-9] "$python_dir"/python3.[0-9][0-9]; do
    try_python "$candidate"
  done
done
try_python python3
try_python python
try_python pypy3

if [ -z "$PYTHON_BIN" ]; then
  echo
  echo "Setup could not find a usable Python 3.9+ installation automatically."
  echo "Portable or custom installations can be selected with E below."
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
            echo "After installing Python, double-click 1_Setup_macOS.command again."
            exit 1
            ;;
          [Aa])
            open "https://www.anaconda.com/download" || true
            echo "After installing Anaconda, double-click 1_Setup_macOS.command again."
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
    echo "Run bash _internal/setup.sh again after installation."
    exit 1
  fi
fi

echo "Found Python: $("$PYTHON_BIN" --version) ($PYTHON_BIN)"

if [ -e .venv ] && [ "$PYTHON_BIN" != "./.venv/bin/python" ]; then
  echo "The existing dashboard .venv could not pass the Python check."
  echo "Rename that .venv folder to keep a backup, then run setup again."
  echo "Do not copy a Windows virtual environment to macOS; setup creates a local one."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv"
fi

./.venv/bin/python -m pip install --quiet --upgrade pip
echo "Installing dashboard dependencies ..."
./.venv/bin/python -m pip install --quiet -r _internal/requirements.txt

cat <<'EOF'

Setup complete.

To start the dashboard, double-click 2_Start_macOS.command.

Note: this only sets up the dashboard itself. Individual pipeline modules
(training, testing, GUIs) may need their own packages installed the first
time you run them from the dashboard -- it will tell you what's missing.
EOF
