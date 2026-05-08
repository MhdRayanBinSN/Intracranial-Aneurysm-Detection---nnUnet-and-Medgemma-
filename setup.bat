@echo off
echo ===================================================
echo Setting up Intracranial Aneurysm Detection System
echo ===================================================
echo This script will install all dependencies for the new computer.
echo Make sure you have Anaconda/Miniconda and Node.js installed first!
echo.
pause

:: Ensure we are starting in the project directory
cd /d "%~dp0"

echo.
echo ===================================================
echo [1/3] Setting up Frontend...
echo ===================================================
cd frontend
call npm install
cd ..

echo.
echo ===================================================
echo [2/3] Setting up Main Backend (pretrained_detect)...
echo ===================================================
call conda create -n pretrained_detect python=3.10 -y
call C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat pretrained_detect 2>nul || call conda activate pretrained_detect
cd backend
echo Installing PyTorch with CUDA 11.8...
call pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
echo Installing other requirements...
call pip install -r requirements.txt
cd ..

echo.
echo ===================================================
echo [3/3] Setting up MedGemma Backend (medgemma)...
echo ===================================================
call conda create -n medgemma python=3.10 -y
call C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat medgemma 2>nul || call conda activate medgemma
cd Medgemma\backend
echo Installing PyTorch with CUDA 11.8...
call pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
echo Installing other requirements...
call pip install -r requirements.txt
cd ..\..

echo.
echo ===================================================
echo Setup Complete! 
echo You can now double-click start.bat to launch the app.
echo ===================================================
pause
