@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "run_v2_1.py" %*
) else (
    py -3 "run_v2_1.py" %*
)
if errorlevel 1 (
    echo.
    echo Start failed. See V2_1_GUIDE.md for environment setup.
    pause
)
endlocal
