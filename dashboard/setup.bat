@echo off
REM One-time setup for the MatchPlant Dashboard on Windows.
REM Checks for Python, creates a venv, installs Flask, and tells you how to
REM start the dashboard. Safe to re-run.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo MatchPlant Dashboard setup
echo ---------------------------

set PYTHON_BIN=
py -3 --version >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_BIN=py -3
) else (
    python --version >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON_BIN=python
    )
)

if "%PYTHON_BIN%"=="" (
    echo No Python 3.9+ was found on this machine.
    echo.
    echo If you use Anaconda/Miniconda for other work, open an "Anaconda Prompt"
    echo and re-run this script from there instead.
    echo.
    echo Otherwise, install Python from:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo IMPORTANT: on the first installer screen, check the box
    echo "Add python.exe to PATH" before clicking Install.
    echo.
    echo Once installed, double-click this file again.
    echo.
    pause
    exit /b 1
)

echo Found Python via: %PYTHON_BIN%
%PYTHON_BIN% --version

if not exist .venv (
    echo Creating virtual environment in .venv ...
    %PYTHON_BIN% -m venv .venv
) else (
    echo Reusing existing .venv
)

.venv\Scripts\python -m pip install --quiet --upgrade pip
echo Installing dashboard dependencies (Flask) ...
.venv\Scripts\python -m pip install --quiet -r requirements.txt

echo.
echo Setup complete.
echo.
echo To start the dashboard, double-click start_dashboard.bat.
echo.
echo Note: this only sets up the dashboard itself. Individual pipeline
echo modules (training, testing, GUIs) may need their own packages
echo installed the first time you run them from the dashboard -- it will
echo tell you what's missing.
echo.
pause
