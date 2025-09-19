# VitFly-AirSim Installation Guide

Complete installation guide for VitFly-AirSim on Windows 10.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10 (64-bit)
- **Memory**: 16GB RAM (recommended)
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU (optional, but recommended for training)

### Software Requirements
- **Python 3.8+**: Download from [python.org](https://python.org)
- **Visual Studio 2019/2022**: For C++ development (optional)
- **Git**: For version control
- **AirSim**: Simulation environment
- **Unreal Engine 4.27**: For AirSim environments

## Installation Steps

### Step 1: Install Python

1. Download Python 3.8+ from [python.org](https://python.org)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Clone Repository

```cmd
git clone <repository-url> vitfly_airsim
cd vitfly_airsim
```

### Step 3: Run Installation Script

```cmd
scripts\install_windows.bat
```

This script will:
- Create a Python virtual environment
- Install PyTorch and all dependencies
- Install AirSim Python package
- Set up directory structure
- Configure AirSim settings

### Step 4: Install AirSim and Unreal Engine

#### Option A: Pre-built AirSim Environments (Recommended)

1. Download pre-built environments from [AirSim releases](https://github.com/microsoft/AirSim/releases)
2. Extract to a suitable location (e.g., `C:\AirSim\`)

#### Option B: Build from Source

1. Install Unreal Engine 4.27 from Epic Games Launcher
2. Clone AirSim repository:
   ```cmd
   git clone https://github.com/Microsoft/AirSim.git
   cd AirSim
   ```
3. Follow [AirSim build instructions](https://microsoft.github.io/AirSim/build_windows/)

### Step 5: Test Installation

Run the installation test:
```cmd
scripts\test_installation.bat
```

## Configuration

### AirSim Settings

The installation script automatically copies the AirSim configuration to:
```
%USERPROFILE%\Documents\AirSim\settings.json
```

You can customize camera settings, vehicle configuration, and simulation parameters by editing this file.

### Training Configuration

Edit `config\train_config.yaml` to customize training parameters:
- Model type (ViT, ViTLSTM, ConvNet, etc.)
- Learning rate and batch size
- Data directories
- Device settings (CPU/CUDA)

### Simulation Configuration

Edit `config\simulation_config.yaml` for simulation settings:
- Flight parameters
- Control frequencies
- Safety settings
- Visualization options

## Usage

### Training Models

1. Prepare training data in `data\training_data\`
2. Run training:
   ```cmd
   scripts\run_training.bat
   ```

### Running Simulations

1. Start AirSim environment
2. Run simulation:
   ```cmd
   scripts\run_simulation.bat
   ```

### Manual Commands

Activate the environment and use Python scripts directly:
```cmd
venv\Scripts\activate.bat
python scripts\train.py --config config\train_config.yaml
python scripts\simulate.py --config config\simulation_config.yaml
python scripts\evaluate.py --model-path models\best_model.pth --model-type ViTLSTM --data-dir data\eval_data
```

## GPU Support

### Installing CUDA Support

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

1. Activate virtual environment:
   ```cmd
   venv\Scripts\activate.bat
   ```

2. Uninstall CPU-only PyTorch:
   ```cmd
   pip uninstall torch torchvision torchaudio
   ```

3. Install CUDA version:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Verify CUDA installation:
   ```cmd
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Troubleshooting

### Common Issues

#### Python Not Found
- Ensure Python is added to PATH
- Restart command prompt after Python installation

#### Virtual Environment Creation Failed
```cmd
python -m pip install --upgrade pip
python -m pip install virtualenv
```

#### AirSim Connection Failed
- Ensure AirSim is running
- Check firewall settings
- Verify AirSim settings.json configuration

#### CUDA Out of Memory
- Reduce batch size in training config
- Use gradient accumulation
- Switch to CPU training

#### Module Import Errors
- Ensure virtual environment is activated
- Reinstall requirements:
  ```cmd
  pip install -r requirements.txt
  ```

### Getting Help

1. Check the troubleshooting section in README.md
2. Run diagnostic tests:
   ```cmd
   scripts\test_installation.bat
   ```
3. Create an issue on GitHub with:
   - Error messages
   - System specifications
   - Installation log output

## Advanced Configuration

### Custom Data Paths

Edit configuration files to use custom data directories:
- Training data: `config\train_config.yaml` → `data_dir`
- Model output: `config\train_config.yaml` → `output_dir`
- Simulation data: `config\simulation_config.yaml` → `data_output_dir`

### Multi-GPU Training

For multi-GPU training, modify the training script to use DataParallel:
```python
model = torch.nn.DataParallel(model)
```

### Custom Models

Add new models to `src\models\` and register them in the model factory.

## Performance Optimization

### Training Performance
- Use CUDA if available
- Increase batch size for better GPU utilization
- Use multiple data loader workers
- Enable mixed precision training

### Inference Performance
- Use GPU for real-time inference
- Optimize model for deployment (TensorRT, ONNX)
- Reduce image resolution if needed
- Use model quantization

## Next Steps

After successful installation:

1. **Download or collect training data**
2. **Train your first model** using the provided scripts
3. **Test in simulation** with AirSim
4. **Evaluate performance** using the evaluation tools
5. **Deploy to real hardware** (if available)

For detailed usage instructions, see the main README.md file.