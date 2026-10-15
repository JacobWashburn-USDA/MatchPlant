@echo off
setlocal
cd /d "%~dp0"
echo MatchPlant Dashboard setup
echo ---------------------------
set "PYTHON_BIN="
call :try_python "%~dp0.venv\Scripts\python.exe"
if defined PYTHON_BIN goto :install
call :try_python py -3
call :try_python python
call :try_python python3
if defined CONDA_PREFIX call :try_python "%CONDA_PREFIX%\python.exe"
call :try_python "%USERPROFILE%\anaconda3\python.exe"
call :try_python "%USERPROFILE%\miniconda3\python.exe"
call :try_python "%LOCALAPPDATA%\anaconda3\python.exe"
call :try_python "%LOCALAPPDATA%\miniconda3\python.exe"
call :try_python "%ProgramData%\anaconda3\python.exe"
call :try_python "%ProgramData%\miniconda3\python.exe"
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do call :try_python "%%~D\python.exe"
if not defined PYTHON_BIN goto :missing_python
:create_env
echo Using: "%PYTHON_BIN%"
"%PYTHON_BIN%" --version
if not exist ".venv\Scripts\python.exe" (
    echo Creating dashboard environment...
    "%PYTHON_BIN%" -m venv .venv
    if errorlevel 1 goto :failed
)
:install
echo Installing dashboard dependencies...
".venv\Scripts\python.exe" -m pip --disable-pip-version-check install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -c "import flask, yaml, packaging"
if errorlevel 1 goto :failed
echo.
echo Setup complete. Double-click start_dashboard.bat.
pause
exit /b 0

:try_python
if defined PYTHON_BIN exit /b 0
%* -c "import sys, venv; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "delims=" %%P in ('call %* -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_BIN=%%P"
exit /b 0

:missing_python
echo.
echo Setup could not find a usable Python 3.9+ installation automatically.
echo.
echo [E] I already have Python (enter its full path)
echo [P] Python:   https://www.python.org/downloads/windows/
echo [A] Anaconda: https://www.anaconda.com/download
echo [C] Close setup
echo.
choice /C EPAC /N /M "Type E, P, A, or C: "
if errorlevel 4 exit /b 1
if errorlevel 3 (
    start "" "https://www.anaconda.com/download"
    goto :install_instructions
)
if errorlevel 2 (
    start "" "https://www.python.org/downloads/windows/"
    goto :install_instructions
)
if errorlevel 1 goto :existing_python
exit /b 1
:existing_python
set "PYTHON_PATH="
set /p "PYTHON_PATH=Paste the full path to python.exe: "
if not defined PYTHON_PATH goto :missing_python
set "PYTHON_PATH=%PYTHON_PATH:"=%"
call :try_python "%PYTHON_PATH%"
if not defined PYTHON_BIN (
    echo That path does not run Python 3.9+ with venv. Try another path.
    goto :missing_python
)
goto :create_env
:install_instructions
echo.
echo Finish installing Python or Anaconda before continuing.
echo For a traditional Python installer, select Add python.exe to PATH.
echo If using the Python install manager, also install a Python runtime.
echo For Anaconda, the default installation location is detected automatically.
echo After installation, close this window and double-click setup.bat again.
pause
exit /b 1

:failed
echo.
echo Setup failed. Review the error above, then run setup.bat again.
pause
exit /b 1
