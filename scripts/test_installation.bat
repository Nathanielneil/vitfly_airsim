@echo off
REM Test installation script for VitFly-AirSim

echo ========================================
echo VitFly-AirSim Installation Test
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run install_windows.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Testing Python environment...

REM Test Python version
python -c "import sys; print(f'Python version: {sys.version}')"
if errorlevel 1 (
    echo ERROR: Python not working
    pause
    exit /b 1
)

echo.
echo Testing dependencies...

REM Test PyTorch
echo Testing PyTorch...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if errorlevel 1 (
    echo ERROR: PyTorch not working
    pause
    exit /b 1
)

echo.
echo Testing other dependencies...

REM Test other key packages
python -c "import numpy, cv2, pandas, matplotlib, yaml; print('All core dependencies working')"
if errorlevel 1 (
    echo ERROR: Some dependencies missing
    pause
    exit /b 1
)

REM Test AirSim
echo Testing AirSim...
python -c "import airsim; print(f'AirSim version: {airsim.__version__ if hasattr(airsim, \"__version__\") else \"installed\"}')"
if errorlevel 1 (
    echo ERROR: AirSim not working
    pause
    exit /b 1
)

echo.
echo Testing VitFly modules...

REM Test model imports
echo Testing model imports...
python tests\test_models.py
if errorlevel 1 (
    echo ERROR: Model tests failed
    pause
    exit /b 1
)

echo.
echo Testing system integration...
python tests\test_system.py
if errorlevel 1 (
    echo ERROR: System tests failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Test Results
echo ========================================
echo.
echo ✓ Python environment working
echo ✓ PyTorch installed and working
echo ✓ Dependencies installed correctly
echo ✓ AirSim available
echo ✓ VitFly models working
echo ✓ System integration successful
echo.
echo 🎉 Installation test PASSED!
echo.
echo VitFly-AirSim is ready to use.
echo.
echo Next steps:
echo 1. Install and run AirSim/Unreal Engine
echo 2. Use run_training.bat to train models
echo 3. Use run_simulation.bat to run simulations
echo.

pause