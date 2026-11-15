@echo off
rem Record on this Windows computer: double-click this file.
rem Serves http://localhost:8000 to this computer only. Close the window or press Ctrl+C to stop.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment ^(.venv^)...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer. Then double-click run_local.bat again.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" launch_local.py %*
if errorlevel 1 pause
