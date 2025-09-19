@echo off
REM VitFly-AirSim Simulation Runner for Windows

echo ========================================
echo VitFly-AirSim Simulation
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

REM Check if config file exists
if not exist "config\simulation_config.yaml" (
    echo ERROR: Simulation config file not found
    echo Please ensure config\simulation_config.yaml exists
    pause
    exit /b 1
)

echo.
echo Please ensure AirSim/Unreal Engine is running before starting simulation.
echo.
set /p "ready=Press Enter when AirSim is ready, or Ctrl+C to cancel..."

echo.
echo Starting simulation...
echo Configuration: config\simulation_config.yaml
echo.

REM Run simulation
python scripts\simulate.py --config config\simulation_config.yaml

if errorlevel 1 (
    echo.
    echo Simulation failed or was interrupted!
) else (
    echo.
    echo Simulation completed!
)

echo.
pause