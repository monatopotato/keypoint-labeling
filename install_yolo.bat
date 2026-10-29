@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run run.bat once to create the environment.
    pause
    exit /b 1
)
echo Installing Ultralytics YOLO26 and PyTorch. This may download large packages.
".venv\Scripts\python.exe" install_pytorch.py %*
if errorlevel 1 (
    echo Installation failed. Check internet access and Python version.
    pause
    exit /b 1
)
echo Checking that PyTorch and Ultralytics can load...
".venv\Scripts\python.exe" -c "import torch; from ultralytics import YOLO; print('YOLO dependencies are ready.')"
if errorlevel 1 (
    echo Dependency check failed. For Windows DLL errors, install or repair the Microsoft Visual C++ x64 runtime:
    echo https://aka.ms/vc14/vc_redist.x64.exe
    echo Then restart Windows if requested and run this installer again.
    pause
    exit /b 1
)
echo Ultralytics installed. Restart run.bat and use Auto-label with YOLO.
pause
endlocal
