@echo off
setlocal
cd /d "%~dp0"

echo Local QC Inspector - one-time setup
echo Keep Wi-Fi connected until this window reports SUCCESS.
echo.

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
        echo During installation select Add Python to PATH. Then run this file again.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install --no-input -r requirements.txt
if errorlevel 1 (
    echo.
    echo SETUP FAILED. Check the internet connection, then run this file again.
    pause
    exit /b 1
)

python -c "import streamlit, cv2, numpy, PIL, pandas; print('SUCCESS: offline files are installed.')"
if errorlevel 1 (
    echo.
    echo SETUP FAILED: package import check did not pass.
    pause
    exit /b 1
)

echo.
echo SUCCESS. You can now disconnect Wi-Fi and use launch.bat.
pause
