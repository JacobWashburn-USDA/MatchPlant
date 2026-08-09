@echo off
REM Double-click this file to start the dashboard and open it in your
REM browser. No terminal typing required.
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Setup hasn't been run yet.
    echo Double-click setup.bat first, then try this again.
    echo.
    pause
    exit /b 1
)

echo Starting MatchPlant Dashboard...
start "MatchPlant Dashboard" .venv\Scripts\python app.py

timeout /t 3 /nobreak >nul
start http://127.0.0.1:5050

echo.
echo The dashboard is running in a separate window titled "MatchPlant Dashboard".
echo Close that window to stop the dashboard.
pause
