@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
echo.
echo ========================================
echo      PoseFrame Studio - Local App
echo ========================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Setting up Python virtual environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo ERROR: Install Python 3.10+ from https://www.python.org/ and try again.
        pause
        exit /b 1
    )
)
if not exist ".venv\.basic_installed" (
    echo [2/2] Installing web server dependencies...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Flask installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    type nul > ".venv\.basic_installed"
)
echo.
echo Opening http://127.0.0.1:8765
echo Press CTRL+C to stop the server.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
endlocal
