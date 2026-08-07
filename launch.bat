@echo off
setlocal
cd /d "%~dp0"

rem This file intentionally NEVER installs or updates packages. It is safe to use
rem when the computer has no Wi-Fi. First-time setup is in setup_once_with_wifi.bat.
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo OFFLINE STARTUP NOT READY
    echo This computer has not completed the one-time local installation yet.
    echo Connect to the internet once, run setup_once_with_wifi.bat, wait for SUCCESS,
    echo then use launch.bat forever after -- including with Wi-Fi disconnected.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -c "import streamlit, cv2, numpy, PIL, pandas" >nul 2>&1
if errorlevel 1 (
    echo.
    echo OFFLINE STARTUP NOT READY
    echo Local program files are incomplete. Connect to the internet once and run
    echo setup_once_with_wifi.bat. This inspection app does not need Wi-Fi afterwards.
    echo.
    pause
    exit /b 1
)

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_ADDRESS=127.0.0.1
set NO_PROXY=127.0.0.1,localhost

echo.
echo Starting Local QC Inspector in OFFLINE mode.
echo Leave this window open while using the app. The app is at http://127.0.0.1:8501
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

pause
