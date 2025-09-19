@echo off
REM VitFly-AirSim Training Runner for Windows

echo ========================================
echo VitFly-AirSim Training
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

REM Check if data directory exists
if not exist "data\training_data" (
    echo ERROR: Training data directory not found
    echo Please ensure you have training data in data\training_data\
    pause
    exit /b 1
)

REM Check if config file exists
if not exist "config\train_config.yaml" (
    echo ERROR: Training config file not found
    echo Please ensure config\train_config.yaml exists
    pause
    exit /b 1
)

echo.
echo Starting training...
echo Configuration: config\train_config.yaml
echo Data directory: data\training_data
echo.

REM Run training
python scripts\train.py --config config\train_config.yaml

if errorlevel 1 (
    echo.
    echo Training failed!
    pause
    exit /b 1
) else (
    echo.
    echo Training completed successfully!
    echo Check the outputs\ directory for results
)

echo.
pause