@echo off
REM run.bat – Helper launcher for CineMatch on Windows
REM Double-click this file, or run it from PowerShell: .\run.bat

REM Try to find Python
where py >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON=py
    GOTO :found
)

where python >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON=python
    GOTO :found
)

IF EXIST "C:\Users\HP\AppData\Local\Python\bin\python.exe" (
    SET PYTHON=C:\Users\HP\AppData\Local\Python\bin\python.exe
    GOTO :found
)

ECHO ERROR: Python not found. Install Python from https://python.org
PAUSE
EXIT /B 1

:found
ECHO Using Python: %PYTHON%
ECHO.

REM 1. Install requirements if needed
ECHO [1/3] Checking dependencies...
%PYTHON% -c "import streamlit, sklearn, pandas" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt
)

REM 2. Download dataset if missing
IF NOT EXIST "data\movies.csv" (
    ECHO [2/3] Downloading MovieLens dataset...
    %PYTHON% download_data.py
) ELSE (
    ECHO [2/3] Dataset already present.
)

REM 3. Launch the app
ECHO [3/3] Launching CineMatch...
ECHO Open http://localhost:8501 in your browser.
ECHO Press Ctrl+C to stop.
ECHO.
%PYTHON% -m streamlit run app.py

PAUSE
