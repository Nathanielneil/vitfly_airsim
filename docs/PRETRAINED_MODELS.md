# 使用预训练 VitFly 模型

本项目支持使用原始 VitFly 论文（ICRA 2025）中的预训练模型进行避障飞行。

## 📦 下载预训练模型

### 方法 1：通过 Box.com 下载（推荐）

1. 访问 VitFly Datashare：https://upenn.app.box.com/v/ViT-quad-datashare
2. 下载 `pretrained_models.tar` (约 50MB)
3. 解压到项目的 `models` 目录：

```bash
cd /path/to/vitfly_airsim
mkdir -p models
tar -xvf /path/to/pretrained_models.tar -C models
```

### 方法 2：使用下载脚本

```bash
# 1. 下载 pretrained_models.tar 到 /tmp/
# 2. 运行脚本自动解压
bash scripts/download_pretrained_models.sh
```

## 🤖 可用模型

下载后，你将获得以下预训练模型：

| 模型类型 | 文件名 | 描述 | 性能 |
|---------|--------|------|------|
| **ViT+LSTM** | `vitlstm_best.pth` | Vision Transformer + LSTM（最佳） | ⭐⭐⭐⭐⭐ |
| ViT | `vit_best.pth` | 纯 Vision Transformer | ⭐⭐⭐⭐ |
| ConvNet | `convnet_best.pth` | 卷积神经网络基线 | ⭐⭐⭐ |
| LSTMNet | `lstmnet_best.pth` | LSTM 网络基线 | ⭐⭐⭐ |
| UNet | `unet_best.pth` | U-Net 架构 | ⭐⭐⭐ |

**推荐使用 ViT+LSTM 模型**，它在论文中表现最好。

## 🚁 使用预训练模型运行仿真

### 基本用法

```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/vitlstm_best.pth \
  --model-type ViTLSTM
```

### 参数说明

- `--mode model`: 使用模型推理模式（而非专家策略）
- `--model-path`: 预训练模型文件路径
- `--model-type`: 模型类型，可选值：
  - `ViTLSTM` - Vision Transformer + LSTM
  - `ViT` - Vision Transformer
  - `ConvNet` - Convolutional Network
  - `LSTMNet` - LSTM Network
  - `UNet` - U-Net

### 不同模型示例

```bash
# ViT+LSTM (推荐)
python scripts/simulate.py --mode model \
  --model-path models/vitlstm_best.pth --model-type ViTLSTM

# 纯 ViT
python scripts/simulate.py --mode model \
  --model-path models/vit_best.pth --model-type ViT

# ConvNet
python scripts/simulate.py --mode model \
  --model-path models/convnet_best.pth --model-type ConvNet
```

### 高速测试

论文中测试了高达 7 m/s 的速度：

```bash
python scripts/simulate.py --mode model \
  --model-path models/vitlstm_best.pth \
  --model-type ViTLSTM \
  --desired-velocity 7.0 \
  --max-duration 60.0
```

## ⚙️ 配置文件方式

你也可以在 `config/simulation_config.yaml` 中配置：

```yaml
# 模型推理模式
mode: "model"
use_model: true
model_path: "models/vitlstm_best.pth"
model_type: "ViTLSTM"
desired_velocity: 5.0
```

然后直接运行：

```bash
python scripts/simulate.py --config config/simulation_config.yaml
```

## 🔍 模型细节

### 输入
- **深度图像**: 90x60 像素，单通道
- **归一化**: [0, 1] 范围，10m 最大距离

### 输出
- **速度命令**: (vx, vy, vz) 三维向量
- **坐标系**: NED（North-East-Down）

### 训练环境
- **模拟器**: Flightmare (Unity-based)
- **训练数据**: 专家策略行为克隆
- **环境**: 球体障碍物、树木等

### 零样本迁移
这些模型在 Flightmare 中训练，但可以：
- ✅ 零样本迁移到 AirSim
- ✅ 零样本迁移到真实世界
- ✅ 泛化到不同障碍物类型

## 📊 性能对比

根据论文结果：

| 模型 | 成功率 | 平均速度 | 碰撞距离 |
|-----|--------|---------|---------|
| ViT+LSTM | **95%** | **5.2 m/s** | **7.8 m** |
| ViT | 89% | 4.8 m/s | 6.5 m |
| ConvNet | 82% | 4.3 m/s | 5.2 m |
| LSTMNet | 79% | 4.1 m/s | 4.8 m |

## ⚠️ 注意事项

### 模型兼容性
- 预训练模型期望 90x60 输入分辨率
- AirSim 配置应匹配 D435i 相机（848x480），然后预处理到 90x60
- 我们的 `sensor_manager.py` 已自动处理这个调整

### 深度范围
- 模型训练时假设 10m 最大深度
- 超出范围的深度会被裁剪
- 归一化公式: `depth_normalized = clip(depth_meters / 10.0, 0, 1)`

### 控制频率
- 模型训练使用 30 Hz 控制频率
- 建议在配置中设置 `control_frequency: 30.0`

## 🔗 参考资料

- **论文**: [Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance](https://arxiv.org/abs/2405.10391)
- **项目主页**: https://www.anishbhattacharya.com/research/vitfly
- **GitHub**: https://github.com/anish-bhattacharya/vitfly
- **Datashare**: https://upenn.app.box.com/v/ViT-quad-datashare

## 🐛 故障排除

### 模型加载失败
```
RuntimeError: Error(s) in loading state_dict
```
**解决方案**: 确保 `--model-type` 与模型文件匹配

### 性能差
**可能原因**:
1. 控制频率太低 → 增加到 30 Hz
2. 速度太高 → 从 2-3 m/s 开始测试
3. 环境差异太大 → 预训练模型在简单障碍物环境训练

### 立即碰撞
**可能原因**:
1. 深度图像预处理问题 → 检查 sensor_manager 日志
2. 坐标系不匹配 → 确认使用 NED 坐标系
3. 起飞后碰撞检测误报 → 已在代码中修复

## 📝 总结

**优点**:
- ✅ 无需收集训练数据
- ✅ 无需训练模型
- ✅ 立即可用的高性能避障
- ✅ 零样本迁移能力强

**缺点**:
- ⚠️ 可能需要微调以适应特定环境
- ⚠️ 训练数据分布可能与 AirSim 不完全匹配

**建议工作流程**:
1. 先使用预训练模型测试基本功能
2. 如果性能不满意，收集 AirSim 特定数据
3. 使用预训练模型作为初始化，进行微调（transfer learning）
