@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run run.bat once to create the environment.
    pause
    exit /b 1
)
echo Installing Ultralytics YOLO26 and PyTorch. This may download large packages.
".venv\Scripts\python.exe" -m pip install -r requirements-yolo.txt
if errorlevel 1 (
    echo Installation failed. Check internet access and Python version.
) else (
    echo Ultralytics installed. Restart run.bat and use Auto-label with YOLO.
)
pause
endlocal
