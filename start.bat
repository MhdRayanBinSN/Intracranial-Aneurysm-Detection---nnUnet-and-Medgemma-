@echo off
echo ===================================================
echo Starting Intracranial Aneurysm Detection System
echo ===================================================

:: Ensure we are starting in the project directory
cd /d "%~dp0"

echo [1/3] Starting Main Backend...
start "Main Backend" powershell -NoExit -Command "cd backend; conda activate pretrained_detect; python main.py"

echo [2/3] Starting MedGemma Backend...
start "MedGemma Backend" powershell -NoExit -Command "cd Medgemma\backend; conda activate medgemma; python main.py"

echo [3/3] Starting React Frontend...
start "Frontend" powershell -NoExit -Command "cd frontend; npm run dev"

echo.
echo All services are launching in separate PowerShell windows!
echo - Main Backend terminal opened
echo - MedGemma Backend terminal opened
echo - Frontend terminal opened
echo.
echo You can close this window.
