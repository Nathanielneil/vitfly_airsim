# VitFly-AirSim Windows 部署文档 (Conda)

本文档详细说明如何在 Windows 系统上使用 Conda 虚拟环境部署 VitFly-AirSim 项目。

---

## 目录

- [系统要求](#系统要求)
- [准备工作](#准备工作)
- [安装步骤](#安装步骤)
- [配置 AirSim](#配置-airsim)
- [验证安装](#验证安装)
- [使用指南](#使用指南)
- [常见问题](#常见问题)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 64-bit | Windows 10/11 64-bit |
| **CPU** | Intel i5 或 AMD 同等产品 | Intel i7 或更高 |
| **内存** | 8GB RAM | 16GB RAM |
| **显卡** | 集成显卡（仅CPU训练） | NVIDIA GTX 1060 或更高 |
| **存储** | 20GB 可用空间 | 50GB SSD |

### 软件依赖

- **Python**: 3.8 - 3.10
- **Conda**: Miniconda 或 Anaconda
- **Git**: 用于版本控制
- **AirSim**: 1.8.1+
- **Unreal Engine**: 4.27 (可选，仅需自建环境时)
- **Visual Studio**: 2019/2022 (可选，用于编译某些包)

---

## 准备工作

### 1. 安装 Miniconda

#### 下载安装

1. 访问 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html)
2. 下载 **Miniconda3 Windows 64-bit** 安装程序
3. 运行安装程序

#### 安装选项

安装时**建议**配置：
-  **Add Miniconda3 to my PATH environment variable** （重要！）
-  **Register Miniconda3 as my default Python**
- 安装路径：使用默认路径或选择无空格的路径（如 `C:\Miniconda3`）

#### 验证安装

打开 **命令提示符 (CMD)** 或 **Anaconda Prompt**，输入：

```cmd
conda --version
python --version
```

应显示类似：
```
conda 23.10.0
Python 3.11.5
```

---

### 2. 安装 Git

#### 下载安装

1. 访问 [Git 官网](https://git-scm.com/download/win)
2. 下载 **64-bit Git for Windows Setup**
3. 运行安装程序，使用默认选项即可

#### 验证安装

```cmd
git --version
```

应显示类似：
```
git version 2.42.0.windows.1
```

---

### 3. 安装 NVIDIA 驱动（可选但推荐）

如果你有 NVIDIA 显卡：

1. 访问 [NVIDIA 驱动下载页面](https://www.nvidia.com/Download/index.aspx)
2. 选择你的显卡型号
3. 下载并安装最新驱动

#### 验证 GPU

```cmd
nvidia-smi
```

应显示显卡信息和驱动版本。

---

## 安装步骤

### 步骤 1：克隆项目

打开 **命令提示符**，执行以下命令：

```cmd
cd C:\
git clone https://github.com/Nathanielneil/vitfly_airsim.git
cd vitfly_airsim
```

> **提示**：你可以克隆到任何目录，但路径中**不要包含中文或空格**。

---

### 步骤 2：创建 Conda 虚拟环境

```cmd
conda create -n vitfly python=3.8 -y
```

**参数说明**：
- `-n vitfly`：环境名称为 "vitfly"
- `python=3.8`：使用 Python 3.8
- `-y`：自动确认

等待创建完成（约 1-2 分钟）。

---

### 步骤 3：激活虚拟环境

```cmd
conda activate vitfly
```

成功后，命令提示符前会显示 `(vitfly)`：

```
(vitfly) C:\vitfly_airsim>
```

---

### 步骤 4：安装 PyTorch

根据你的硬件配置选择：

#### 选项 A：有 NVIDIA GPU（推荐）

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### 选项 B：仅 CPU 版本

```cmd
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

> **注意**：此步骤会下载较大文件（约 2-3GB），需要 5-15 分钟。

#### 验证 PyTorch 安装

```cmd
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**预期输出**：
- GPU 版本：`CUDA available: True`
- CPU 版本：`CUDA available: False`

---

### 步骤 5：升级 pip

```cmd
python -m pip install --upgrade pip
```

---

### 步骤 6：安装项目依赖

```cmd
pip install -r requirements.txt
```

这将安装以下主要依赖：
- `airsim` - AirSim Python API
- `opencv-python` - 图像处理
- `PyYAML` - 配置文件解析
- `matplotlib`, `seaborn` - 可视化
- `tensorboard` - 训练监控
- `pandas`, `scipy` - 数据处理

安装时间约 3-5 分钟。

#### 国内用户加速（可选）

如果下载速度慢，使用清华镜像：

```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 步骤 7：验证所有依赖

```cmd
python -c "import torch, cv2, yaml, airsim, matplotlib, pandas; print(' All dependencies installed successfully!')"
```

如果没有报错，说明所有依赖安装成功！

---

## 配置 AirSim

### 步骤 1：创建 AirSim 配置目录

```cmd
mkdir %USERPROFILE%\Documents\AirSim
```

---

### 步骤 2：复制配置文件

```cmd
copy config\airsim_settings.json %USERPROFILE%\Documents\AirSim\settings.json
```

验证配置文件：

```cmd
type %USERPROFILE%\Documents\AirSim\settings.json
```

---

### 步骤 3：下载 AirSim 环境

#### 选项 A：使用预编译环境（推荐初学者）

1. 访问 [AirSim Releases](https://github.com/microsoft/AirSim/releases/tag/v1.8.1)
2. 下载预编译环境（推荐）：
   - **Blocks.zip** - 简单的方块环境（约 500MB）
   - **LandscapeMountains.zip** - 山地环境（约 2GB）
   - **CityEnviron.zip** - 城市环境（约 3GB）

3. 解压到 `C:\AirSim\` 目录

示例结构：
```
C:\AirSim\
└── Blocks\
    └── WindowsNoEditor\
        └── Blocks.exe
```

#### 选项 B：从源码构建（高级用户）

需要 Unreal Engine 4.27 和 Visual Studio 2019/2022。

详细步骤参考：[AirSim Build on Windows](https://microsoft.github.io/AirSim/build_windows/)

---

### 步骤 4：测试 AirSim

1. **启动 AirSim 环境**：
   ```cmd
   C:\AirSim\Blocks\WindowsNoEditor\Blocks.exe
   ```

2. **选择飞行器类型**：在弹出窗口选择 **Multirotor**

3. **测试连接**（在新的 CMD 窗口）：
   ```cmd
   conda activate vitfly
   python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection(); print(' AirSim connection successful!')"
   ```

如果显示 ` AirSim connection successful!`，说明配置成功！

---

## 验证安装

### 完整环境测试

运行以下命令进行完整测试：

```cmd
conda activate vitfly
python -c "
import sys
print('Python version:', sys.version)
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
import airsim
print('AirSim installed:', airsim.__version__)
import cv2
print('OpenCV version:', cv2.__version__)
import yaml, matplotlib, pandas, numpy
print(' All core libraries working!')
"
```

### 检查清单

安装验证清单：

- [ ] Conda 已安装（`conda --version`）
- [ ] Git 已安装（`git --version`）
- [ ] 虚拟环境已创建（`conda env list` 能看到 vitfly）
- [ ] PyTorch 已安装（`python -c "import torch"`）
- [ ] GPU 可用（如有显卡，`torch.cuda.is_available()` 返回 True）
- [ ] 项目依赖已安装（`pip list` 能看到 airsim, opencv-python 等）
- [ ] AirSim 配置文件已复制
- [ ] AirSim 环境能正常启动
- [ ] Python 能连接到 AirSim

---

## 使用指南

### 环境管理

#### 激活环境

每次使用前需要激活环境：

```cmd
conda activate vitfly
```

#### 退出环境

```cmd
conda deactivate
```

#### 查看已安装包

```cmd
conda list
```

#### 查看所有环境

```cmd
conda env list
```

#### 删除环境（重新安装时）

```cmd
conda remove -n vitfly --all -y
```

---

### 项目目录结构

```
vitfly_airsim/
├── config/                    # 配置文件
│   ├── airsim_settings.json  # AirSim 配置
│   ├── train_config.yaml     # 训练配置
│   └── simulation_config.yaml # 仿真配置
├── data/                      # 数据目录
│   ├── training_data/        # 训练数据
│   └── eval_data/            # 评估数据
├── environments/              # UE4 环境文件（可选）
├── models/                    # 预训练模型
├── scripts/                   # 运行脚本
│   ├── train.py              # 训练脚本
│   ├── simulate.py           # 仿真脚本
│   ├── evaluate.py           # 评估脚本
│   └── *.bat                 # Windows 批处理脚本
├── src/                       # 源代码
│   ├── models/               # 模型定义
│   ├── airsim_interface/     # AirSim 接口
│   ├── training/             # 训练逻辑
│   └── utils/                # 工具函数
├── tests/                     # 测试代码
└── requirements.txt           # Python 依赖
```

---

### 训练模型

#### 1. 准备训练数据

将训练数据放置在 `data/training_data/` 目录。

#### 2. 配置训练参数

编辑 `config/train_config.yaml`：

```yaml
model:
  type: "ViTLSTM"  # 可选: ViT, ViTLSTM, ConvNet, LSTMNet

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100

data:
  data_dir: "data/training_data"

device:
  use_cuda: true  # 如果有 GPU 设置为 true
```

#### 3. 运行训练

```cmd
conda activate vitfly
python scripts\train.py --config config\train_config.yaml
```

#### 4. 监控训练过程

在浏览器打开 TensorBoard：

```cmd
tensorboard --logdir=runs
```

访问 http://localhost:6006

---

### 运行仿真

#### 1. 启动 AirSim 环境

双击启动 AirSim：
```cmd
C:\AirSim\Blocks\WindowsNoEditor\Blocks.exe
```

选择 **Multirotor** 模式。

#### 2. 配置仿真参数

编辑 `config/simulation_config.yaml`：

```yaml
simulation:
  max_steps: 1000
  speed: 1.0

model:
  checkpoint_path: "models/vitlstm_best.pth"
  model_type: "ViTLSTM"

visualization:
  show_depth: true
  save_video: false
```

#### 3. 运行仿真

```cmd
conda activate vitfly
python scripts\simulate.py --config config\simulation_config.yaml
```

#### 4. 控制快捷键（在 AirSim 窗口）

- **F1**: 查看帮助
- **Backspace**: 重置无人机位置
- **0**: 切换视角
- **;**: 显示/隐藏参数
- **ESC**: 退出

---

### 评估模型

```cmd
conda activate vitfly
python scripts\evaluate.py --model models\vitlstm_best.pth --model-type ViTLSTM --data-dir data\eval_data
```

评估结果会保存在 `results/` 目录。

---

### 创建快速启动脚本

在项目根目录创建 `start_vitfly.bat`：

```batch
@echo off
title VitFly-AirSim Environment
color 0A

echo ========================================
echo  VitFly-AirSim Quick Start
echo ========================================
echo.

REM 激活 conda 环境
call conda activate vitfly

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment
    echo Please ensure conda is installed and vitfly environment exists
    pause
    exit /b 1
)

echo [OK] Environment activated: vitfly
echo.
echo Available commands:
echo   1. Train:    python scripts\train.py --config config\train_config.yaml
echo   2. Simulate: python scripts\simulate.py --config config\simulation_config.yaml
echo   3. Evaluate: python scripts\evaluate.py --model models\best_model.pth
echo.
echo Current directory: %cd%
echo.

REM 保持窗口打开
cmd /k
```

使用方法：双击 `start_vitfly.bat` 即可自动激活环境。

---

## 常见问题

### Q1: 提示 "conda 不是内部或外部命令"

**原因**：Conda 未添加到系统 PATH。

**解决方法**：
1. 使用 **Anaconda Prompt** 而非普通 CMD
2. 或手动添加 Conda 到 PATH：
   - 右键 **此电脑** → **属性** → **高级系统设置** → **环境变量**
   - 在系统变量 Path 中添加：`C:\Miniconda3\Scripts` 和 `C:\Miniconda3`

---

### Q2: pip install 下载很慢

**解决方法**：使用国内镜像源

```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或永久配置：

```cmd
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Q3: torch.cuda.is_available() 返回 False

**可能原因**：
1. 安装了 CPU 版本的 PyTorch
2. NVIDIA 驱动未安装或过旧
3. CUDA 版本不匹配

**解决方法**：

1. 检查显卡驱动：
```cmd
nvidia-smi
```

2. 重新安装 GPU 版 PyTorch：
```cmd
conda activate vitfly
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

---

### Q4: AirSim 连接失败

**错误信息**：
```
ConnectionError: Could not connect to AirSim
```

**解决方法**：

1. **确保 AirSim 正在运行**
   - 启动 AirSim 环境（.exe 文件）
   - 在主菜单选择 **Multirotor**

2. **检查配置文件**
   ```cmd
   type %USERPROFILE%\Documents\AirSim\settings.json
   ```

3. **检查防火墙**
   - 允许 AirSim 通过防火墙
   - 或临时关闭防火墙测试

4. **重启 AirSim**

---

### Q5: 导入模块报错

**错误信息**：
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方法**：

1. **确认环境已激活**：
   ```cmd
   conda activate vitfly
   ```

2. **重新安装依赖**：
   ```cmd
   pip install -r requirements.txt --force-reinstall
   ```

3. **检查 Python 路径**：
   ```cmd
   python -c "import sys; print(sys.executable)"
   ```
   应该指向 conda 环境中的 Python。

---

### Q6: CUDA Out of Memory

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：

1. **减小 batch size**：
   编辑 `config/train_config.yaml`，将 `batch_size` 从 32 改为 16 或 8

2. **使用梯度累积**：
   在训练代码中启用梯度累积

3. **降低图像分辨率**：
   减小输入图像尺寸

4. **使用 CPU 训练**：
   ```yaml
   device:
     use_cuda: false
   ```

---

### Q7: AirSim 启动后崩溃

**解决方法**：

1. **更新显卡驱动**
2. **降低图形设置**：
   - 编辑 `settings.json`，添加：
   ```json
   {
     "SettingsVersion": 1.2,
     "SimMode": "Multirotor",
     "ViewMode": "NoDisplay"
   }
   ```

3. **使用更简单的环境**（如 Blocks 而非 CityEnviron）

---

### Q8: 训练速度很慢

**优化方法**：

1. **确认使用 GPU**：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应为 True
   ```

2. **增大 batch size**（在显存允许的情况下）

3. **使用多线程数据加载**：
   在训练配置中设置 `num_workers: 4`

4. **使用混合精度训练**：
   启用 AMP (Automatic Mixed Precision)

---

## 性能优化

### GPU 加速

#### 检查 GPU 使用情况

训练时在另一个 CMD 窗口运行：

```cmd
nvidia-smi -l 1
```

观察 GPU 利用率，应该在 80-100%。

#### CUDA 版本对应表

| PyTorch 版本 | CUDA 版本 | 命令 |
|-------------|----------|------|
| 2.0+ | 11.8 | `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia` |
| 2.0+ | 12.1 | `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia` |
| 1.13 | 11.7 | `conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia` |

---

### 训练优化

#### 1. 数据加载优化

在 `train_config.yaml` 中：

```yaml
data_loader:
  num_workers: 4  # 使用多进程加载数据
  pin_memory: true  # 锁定内存，加速 GPU 传输
  prefetch_factor: 2  # 预加载批次数
```

#### 2. 混合精度训练

使用 PyTorch AMP 加速训练（节省显存，提速 2-3 倍）：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 3. 梯度累积

显存不足时，模拟大 batch size：

```python
accumulation_steps = 4

for i, data in enumerate(dataloader):
    loss = model(data)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 推理优化

#### 1. 模型量化

减小模型大小，加速推理：

```python
import torch

model = torch.load('models/best_model.pth')
model.eval()

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model, 'models/quantized_model.pth')
```

#### 2. 导出为 ONNX

跨平台部署：

```python
import torch

model = torch.load('models/best_model.pth')
model.eval()

dummy_input = torch.randn(1, 3, 60, 90)
torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=['input'],
    output_names=['output']
)
```

#### 3. TensorRT 加速（NVIDIA GPU）

安装 TensorRT 后：

```python
import torch_tensorrt

model = torch.load('models/best_model.pth')
model.eval()

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 60, 90))],
    enabled_precisions={torch.float16}
)
```

---

### 系统优化

#### Windows 性能设置

1. **电源计划**：设置为 "高性能"
   - 控制面板 → 电源选项 → 高性能

2. **关闭不必要的后台程序**

3. **禁用 Windows 更新**（训练期间）

4. **增加虚拟内存**：
   - 此电脑 → 属性 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存
   - 设置为物理内存的 1.5-2 倍

---

## 故障排除

### 日志文件位置

- **训练日志**：`runs/`
- **AirSim 日志**：`%USERPROFILE%\Documents\AirSim\logs\`
- **Python 日志**：通常在项目根目录 `logs/`

---

### 调试模式

启用详细日志：

```cmd
set PYTHONVERBOSE=1
python scripts\train.py --config config\train_config.yaml --debug
```

---

### 重置环境

如果遇到无法解决的问题，完全重置：

```cmd
# 1. 删除 conda 环境
conda deactivate
conda remove -n vitfly --all -y

# 2. 删除 AirSim 配置
del %USERPROFILE%\Documents\AirSim\settings.json

# 3. 清理 pip 缓存
pip cache purge

# 4. 重新开始安装
conda create -n vitfly python=3.8 -y
conda activate vitfly
...
```

---

### 获取帮助

1. **查看项目文档**：
   - README.md
   - INSTALL.md
   - PROJECT_OVERVIEW.md

2. **检查日志文件**

3. **在 GitHub 提交 Issue**：
   - 包含错误信息
   - 系统配置（Python版本、PyTorch版本、GPU型号等）
   - 复现步骤

4. **社区支持**：
   - AirSim 官方文档：https://microsoft.github.io/AirSim/
   - PyTorch 论坛：https://discuss.pytorch.org/

---

## 环境导出与共享

### 导出 Conda 环境

```cmd
conda activate vitfly
conda env export > environment.yml
```

### 从环境文件创建

```cmd
conda env create -f environment.yml
```

### 导出 pip 依赖（跨平台）

```cmd
pip list --format=freeze > requirements_exact.txt
```

---

## 卸载指南

### 完全卸载

```cmd
# 1. 删除 conda 环境
conda remove -n vitfly --all -y

# 2. 删除项目文件
rmdir /s /q C:\vitfly_airsim

# 3. 删除 AirSim 配置
rmdir /s /q %USERPROFILE%\Documents\AirSim

# 4. 删除 AirSim 环境（如果下载了）
rmdir /s /q C:\AirSim

# 5. （可选）卸载 Miniconda
# 在控制面板 → 程序和功能 → 卸载 Miniconda3
```

---

## 附录

### A. 推荐配置示例

#### 入门配置（CPU训练）

```yaml
# train_config.yaml
model:
  type: "ConvNet"

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 50

device:
  use_cuda: false
```

#### 标准配置（单GPU）

```yaml
# train_config.yaml
model:
  type: "ViTLSTM"

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100

device:
  use_cuda: true
  gpu_id: 0
```

#### 高性能配置（多GPU）

```yaml
# train_config.yaml
model:
  type: "ViTLSTM"

training:
  batch_size: 64
  learning_rate: 0.0003
  num_epochs: 200

device:
  use_cuda: true
  multi_gpu: true
  gpu_ids: [0, 1]
```

---

### B. 环境变量配置

在系统环境变量中添加（可选）：

```cmd
# CUDA 相关
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

# PyTorch 优化
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

---

### C. 常用命令速查表

| 操作 | 命令 |
|------|------|
| 激活环境 | `conda activate vitfly` |
| 退出环境 | `conda deactivate` |
| 查看环境列表 | `conda env list` |
| 查看已安装包 | `conda list` 或 `pip list` |
| 安装新包 | `pip install package_name` |
| 更新包 | `pip install --upgrade package_name` |
| 删除包 | `pip uninstall package_name` |
| 导出环境 | `conda env export > environment.yml` |
| 清理缓存 | `conda clean --all` |
| 检查 GPU | `nvidia-smi` |
| 训练模型 | `python scripts\train.py --config config\train_config.yaml` |
| 运行仿真 | `python scripts\simulate.py --config config\simulation_config.yaml` |
| 评估模型 | `python scripts\evaluate.py --model models\best.pth` |

---

### D. 参考资源

- **项目主页**：https://github.com/Nathanielneil/vitfly_airsim
- **AirSim 文档**：https://microsoft.github.io/AirSim/
- **PyTorch 文档**：https://pytorch.org/docs/stable/index.html
- **Conda 文档**：https://docs.conda.io/
- **Vision Transformer 论文**：https://arxiv.org/abs/2010.11929

---

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.0.0 | 2024-10-01 | 初始版本，完整的 Windows + Conda 部署文档 |

---

## 许可证

本文档遵循 MIT License，与项目主仓库保持一致。

---

## 贡献者

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**文档结束** | 祝你使用愉快！ 
