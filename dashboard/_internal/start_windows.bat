@echo off
setlocal
cd /d "%~dp0.."
if /i "%~1"=="--server" goto :server
if not exist ".venv\Scripts\python.exe" goto :missing_setup
".venv\Scripts\python.exe" -c "import flask, yaml, packaging" >nul 2>&1
if errorlevel 1 goto :missing_setup
start "MatchPlant Dashboard" "%ComSpec%" /d /c ""%~f0" --server"
powershell -NoProfile -Command "$ready=$false; for ($i=0; $i -lt 30; $i++) { try { $response=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5050/' -TimeoutSec 1; if ($response.StatusCode -eq 200) { $ready=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if ($ready) { Start-Process 'http://127.0.0.1:5050/'; exit 0 }; exit 1"
if errorlevel 1 (
    echo The dashboard did not become ready. Check the MatchPlant Dashboard window for errors.
    pause
    exit /b 1
)
exit /b 0

:server
title MatchPlant Dashboard
echo Starting MatchPlant Dashboard...
echo Keep this window open while using the dashboard.
echo Close this window or press Ctrl+C to stop it.
echo.
".venv\Scripts\python.exe" -u "_internal\app.py"
if errorlevel 1 (
    echo.
    echo Dashboard startup failed. Review the error above.
    pause
    exit /b 1
)
exit /b 0

:missing_setup
echo Dashboard setup is missing or incomplete.
echo Double-click 1_Setup_Windows.bat first, then run 2_Start_Windows.bat again.
pause
exit /b 1
