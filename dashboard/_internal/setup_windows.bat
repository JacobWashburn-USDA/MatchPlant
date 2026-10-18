@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0.."
echo MatchPlant Dashboard setup
echo ---------------------------
echo Setup file: "%~f0"
rem Ignore Python environment overrides inherited from other applications.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHON_BIN="
call :try_python "%CD%\.venv\Scripts\python.exe"
if defined PYTHON_BIN goto :install
rem Prefer actual installations before PATH commands and Windows app aliases.
if defined VIRTUAL_ENV call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"
if defined CONDA_PREFIX call :try_python "%CONDA_PREFIX%\python.exe"
call :try_python "%USERPROFILE%\anaconda3\python.exe"
call :try_python "%USERPROFILE%\miniconda3\python.exe"
call :try_python "%LOCALAPPDATA%\anaconda3\python.exe"
call :try_python "%LOCALAPPDATA%\miniconda3\python.exe"
call :try_python "%ProgramData%\anaconda3\python.exe"
call :try_python "%ProgramData%\miniconda3\python.exe"
call :try_python "%USERPROFILE%\miniforge3\python.exe"
call :try_python "%USERPROFILE%\mambaforge\python.exe"
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do call :try_python "%%~D\python.exe"
if not defined PYTHON_BIN call :discover_python
call :try_python py -3
call :try_python python
call :try_python python3
call :try_python pypy3
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
".venv\Scripts\python.exe" -m pip --disable-pip-version-check install -r "_internal\requirements.txt"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -c "import flask, yaml, packaging"
if errorlevel 1 goto :failed
echo.
echo Setup complete. Double-click 2_Start_Windows.bat.
pause
exit /b 0

:try_python
if defined PYTHON_BIN exit /b 0
if exist "%~1" goto :try_python_path
rem PATH/launcher probes report a tagged result, ignoring unrelated shell output.
%* -I -c "import sys, venv; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "tokens=1,* delims=|" %%P in ('%* -I -c "import sys; print('MATCHPLANT_PYTHON|' + sys.executable)" 2^>nul') do if "%%P"=="MATCHPLANT_PYTHON" if exist "%%Q" set "PYTHON_BIN=%%Q"
exit /b 0

:discover_python
echo Searching registered installations, PATH, and known environments...
rem The discovery helper is embedded below, so setup remains a single file.
rem PowerShell is optional: basic probes and manual entry still work without it.
set "MATCHPLANT_SETUP=%~f0"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -NonInteractive -Command "$lines=Get-Content -LiteralPath $env:MATCHPLANT_SETUP; $code=($lines -match '^::MPPS ' -replace '^::MPPS ', '') -join [Environment]::NewLine; & ([scriptblock]::Create($code))" 2^>nul`) do call :try_python "%%P"
exit /b 0

:try_python_path
echo Checking Python: "%~1"
"%~1" -I -c "import sys, venv; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 (
    echo This Python could not pass the Python 3.9+ and venv check:
    "%~1" -I -c "import sys, venv; print(sys.version); sys.exit(0 if sys.version_info >= (3,9) else 1)"
    exit /b 0
)
rem Use the validated executable directly; no second shell is needed.
set "PYTHON_BIN=%~f1"
echo Python detected successfully.
exit /b 0

:missing_python
echo.
echo Setup could not find a usable Python 3.9+ installation automatically.
echo Portable or custom installations can be selected with E below.
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

rem Embedded optional discovery helper. Only existing executable paths are output.
::MPPS $ErrorActionPreference = 'SilentlyContinue'
::MPPS $found = New-Object 'System.Collections.Generic.List[string]'
::MPPS function Add-Python([string]$path) {
::MPPS   if ([string]::IsNullOrWhiteSpace($path)) { return }
::MPPS   $path = [Environment]::ExpandEnvironmentVariables($path.Trim().Trim('"'))
::MPPS   if (Test-Path -LiteralPath $path -PathType Container) {
::MPPS     Add-Python (Join-Path $path 'python.exe')
::MPPS     Add-Python (Join-Path $path 'Scripts\python.exe')
::MPPS   } elseif ((Test-Path -LiteralPath $path -PathType Leaf) -and $path -match '(?i)(python[^\\]*|pypy[^\\]*)\.exe$') {
::MPPS     if ((Get-Item -LiteralPath $path).Length -eq 0) { return }
::MPPS     if ($found -notcontains $path) { [void]$found.Add($path) }
::MPPS   }
::MPPS }
::MPPS foreach ($name in @('python','python3','pypy3')) {
::MPPS   Get-Command $name -All -CommandType Application | ForEach-Object { Add-Python $_.Source }
::MPPS }
::MPPS if (Get-Command py -CommandType Application) {
::MPPS   & py -0p 2>$null | ForEach-Object { if ($_ -match '([A-Za-z]:\\.*\\(?:python|pypy)[^\\]*\.exe)\s*$') { Add-Python $Matches[1] } }
::MPPS }
::MPPS foreach ($root in @('HKCU:\Software\Python','HKLM:\Software\Python','HKLM:\Software\WOW6432Node\Python')) {
::MPPS   Get-ChildItem -LiteralPath $root -Recurse | Where-Object { $_.PSChildName -eq 'InstallPath' } | ForEach-Object {
::MPPS     Add-Python $_.GetValue('ExecutablePath')
::MPPS     Add-Python $_.GetValue('')
::MPPS   }
::MPPS }
::MPPS foreach ($root in @('HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall','HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall','HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall')) {
::MPPS   Get-ChildItem -LiteralPath $root | Get-ItemProperty | Where-Object { $_.DisplayName -match 'Python|Anaconda|Miniconda|Miniforge|Mambaforge|WinPython' } | ForEach-Object { Add-Python $_.InstallLocation }
::MPPS }
::MPPS Get-Command conda -All -CommandType Application | ForEach-Object {
::MPPS   $base = Split-Path (Split-Path $_.Source -Parent) -Parent
::MPPS   Add-Python $base
::MPPS   Get-ChildItem -LiteralPath (Join-Path $base 'envs') -Directory | ForEach-Object { Add-Python $_.FullName }
::MPPS }
::MPPS Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.conda\environments.txt') | ForEach-Object { Add-Python $_ }
::MPPS $patterns = @(
::MPPS   "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
::MPPS   "$env:LOCALAPPDATA\Python\pythoncore-*\python.exe",
::MPPS   "$env:ProgramFiles\Python*\python.exe",
::MPPS   "${env:ProgramFiles(x86)}\Python*\python.exe",
::MPPS   "$env:USERPROFILE\.conda\envs\*\python.exe"
::MPPS )
::MPPS foreach ($base in @($env:USERPROFILE,$env:LOCALAPPDATA,$env:ProgramData)) {
::MPPS   foreach ($distribution in @('anaconda3','miniconda3','miniforge3','mambaforge')) {
::MPPS     Add-Python (Join-Path $base $distribution)
::MPPS     $patterns += (Join-Path $base "$distribution\envs\*\python.exe")
::MPPS   }
::MPPS }
::MPPS foreach ($pattern in $patterns) { Get-ChildItem -Path $pattern -File | ForEach-Object { Add-Python $_.FullName } }
::MPPS $found | ForEach-Object { $_ }
:existing_python
set "PYTHON_PATH="
set /p "PYTHON_PATH=Paste the full path to python.exe: "
if not defined PYTHON_PATH (
    echo No path was entered. Paste the full path, then press Enter.
    goto :existing_python
)
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
echo After installation, close this window and double-click 1_Setup_Windows.bat again.
pause
exit /b 1

:failed
echo.
echo Setup failed. Review the error above, then run 1_Setup_Windows.bat again.
pause
exit /b 1
