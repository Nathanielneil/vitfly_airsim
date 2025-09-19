@echo off
REM VitFly-AirSim Installation Script for Windows
REM This script installs all dependencies and sets up the environment

echo ========================================
echo VitFly-AirSim Installation for Windows
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required
    pause
    exit /b 1
)

echo Python version is compatible.
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated.
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing...
)

echo.

REM Install PyTorch (CPU version by default)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo PyTorch installed successfully.
echo.

REM Install other requirements
echo Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo Dependencies installed successfully.
echo.

REM Install AirSim (if not already installed)
echo Installing AirSim...
pip install airsim
if errorlevel 1 (
    echo ERROR: Failed to install AirSim
    pause
    exit /b 1
)

echo AirSim installed successfully.
echo.

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "data\training_data" mkdir data\training_data
if not exist "models" mkdir models
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs

echo Directories created.
echo.

REM Copy AirSim settings
echo Setting up AirSim configuration...
set AIRSIM_SETTINGS_DIR=%USERPROFILE%\Documents\AirSim
if not exist "%AIRSIM_SETTINGS_DIR%" mkdir "%AIRSIM_SETTINGS_DIR%"

copy config\airsim_settings.json "%AIRSIM_SETTINGS_DIR%\settings.json"
if errorlevel 1 (
    echo WARNING: Failed to copy AirSim settings
) else (
    echo AirSim settings configured.
)

echo.

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo.

REM Installation complete
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To use VitFly-AirSim:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Start AirSim/Unreal Engine environment
echo 3. Run training: python scripts\train.py --config config\train_config.yaml
echo 4. Run simulation: python scripts\simulate.py --config config\simulation_config.yaml
echo.
echo For CUDA support, reinstall PyTorch with CUDA:
echo   pip uninstall torch torchvision torchaudio
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

pause