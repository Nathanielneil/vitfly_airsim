# VitFly-AirSim: Vision Transformer Quadrotor Obstacle Avoidance for Windows

基于Vision Transformers的四旋翼无人机端到端避障系统的Windows移植版本，使用AirSim作为仿真环境。

## 项目概述

VitFly-AirSim是对原始VitFly项目（ICRA 2025）的重构和移植，从Linux ROS + Flightmare架构迁移到Windows 10 + AirSim 1.8.1 + UE4.7.2架构。

### 核心特性

- **视觉Transformer模型**: ViT, ViT+LSTM, ConvNet, LSTMNet, UNet
- **端到端学习**: 直接从深度图像到速度命令
- **AirSim仿真**: 基于Unreal Engine 4的高真实度仿真
- **Windows原生支持**: 完全支持Windows 10操作系统
- **零样本迁移**: 仿真训练模型可直接用于实际部署

## 系统要求

### 硬件要求
- Windows 10 (64位)
- NVIDIA GPU (推荐GTX 1060或更高)
- 16GB RAM (推荐)
- 20GB 可用磁盘空间

### 软件依赖
- Python 3.8+
- PyTorch 1.9+
- AirSim 1.8.1
- Unreal Engine 4.27.2
- Visual Studio 2019/2022

## 目录结构

```
vitfly_airsim/
├── src/                    # 源代码
│   ├── models/            # 深度学习模型
│   ├── airsim_interface/  # AirSim接口层
│   ├── training/          # 训练相关代码
│   ├── inference/         # 推理和控制
│   └── utils/             # 工具函数
├── config/                # 配置文件
├── environments/          # UE4环境文件
├── scripts/               # 部署脚本
├── tests/                 # 测试代码
├── models/                # 预训练模型
└── data/                  # 数据集
```

## 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/Nathanielneil/vitfly_airsim.git
cd vitfly_airsim

# 安装Python依赖
pip install -r requirements.txt

# 配置AirSim
copy config/airsim_settings.json %USERPROFILE%/Documents/AirSim/settings.json
```

### 2. 运行仿真测试

```bash
# 启动训练
python scripts/train.py --config config/train_config.yaml

# 运行仿真评估
python scripts/evaluate.py --model models/vitlstm_best.pth
```

### 3. 实时部署

```bash
# 连接实际无人机(需要深度相机)
python scripts/deploy.py --model models/vitlstm_best.pth --camera realsense
```

## 架构说明

### 原架构 vs 新架构

| 组件 | 原架构 (Linux) | 新架构 (Windows) |
|------|----------------|------------------|
| 仿真器 | Flightmare | AirSim + UE4 |
| 中间件 | ROS | 直接Python API |
| 操作系统 | Ubuntu 20.04 | Windows 10 |
| 图形引擎 | Unity3D | Unreal Engine 4 |

### 主要改进

1. **跨平台兼容性**: 完全支持Windows生态系统
2. **简化部署**: 移除ROS依赖，使用纯Python实现
3. **更强仿真**: AirSim提供更真实的物理仿真
4. **易于扩展**: 模块化架构便于功能扩展

## 技术细节

### 模型架构

- **输入**: 60×90深度图像 + 期望速度 + 四元数姿态
- **输出**: 3D速度命令 (vx, vy, vz)
- **最佳模型**: ViT+LSTM (3.56M参数)

### AirSim集成

- 使用AirSim Python API进行无人机控制
- 实时深度图像获取和处理
- 碰撞检测和安全监控
- 自动重置和评估机制

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目基于MIT许可证开源。

## 致谢

- 原始VitFly项目: GRASP Lab, University of Pennsylvania
- AirSim项目: Microsoft Research
- Vision Transformer: Google Research

## 联系方式

如有问题或建议，请创建Issue或联系项目维护者。
