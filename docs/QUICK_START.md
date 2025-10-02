# VitFly-AirSim 快速入门指南

本指南提供 VitFly-AirSim 项目的完整工作流程。

## 📋 目录

1. [系统要求](#系统要求)
2. [运行专家策略](#运行专家策略)
3. [运行预训练模型](#运行预训练模型)
4. [录制和标注视频](#录制和标注视频)
5. [常见问题](#常见问题)

## 系统要求

- **AirSim**: 已安装并运行（UE4/Unity）
- **Python**: 3.8+ with PyTorch, OpenCV, pandas
- **预训练模型**: 从 [Box.com](https://upenn.app.box.com/v/ViT-quad-datashare) 下载（可选）

## 运行专家策略

使用基于规则的专家避障算法：

```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode expert \
  --desired-velocity 3.0 \
  --max-duration 60.0
```

**参数说明**：
- `--mode expert` - 使用专家策略（基于规则的避障）
- `--desired-velocity 3.0` - 期望速度 3 m/s
- `--max-duration 60.0` - 最大飞行时间 60 秒

## 运行预训练模型

使用 ViT-LSTM 模型进行端到端避障：

```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 3.0 \
  --max-duration 60.0
```

**可用模型类型**：
- `ViTLSTM` - Vision Transformer + LSTM（**推荐**，准确率最高）
- `ViT` - 纯 Vision Transformer
- `ConvNet` - 卷积神经网络基准
- `LSTMNet` - LSTM 基准
- `UNet` - U-Net 架构

详见 [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md)

## 录制和标注视频

### 步骤 1：在 AirSim 中录制视频

1. 启动 AirSim 仿真环境
2. 按 **F9** 开始录制视频
3. 运行仿真（见上方命令）
4. 按 **F9** 停止录制

视频保存位置：
```
Windows: C:\Users\<YourName>\Documents\AirSim\<CurrentDate>\airsim_rec_<timestamp>.mp4
Linux: ~/Documents/AirSim/<CurrentDate>/airsim_rec_<timestamp>.mp4
```

### 步骤 2：同时记录遥测数据

运行仿真时添加 `--record-telemetry` 参数：

```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 3.0 \
  --max-duration 60.0 \
  --record-telemetry flight_data.csv
```

这会生成包含以下数据的 CSV 文件：
- 时间戳
- 位置 (x, y, z)
- 速度 (vx, vy, vz)
- 姿态（四元数）
- 碰撞状态

### 步骤 3：标注视频

使用遥测数据标注录制的视频：

```bash
python scripts/annotate_flight_video.py \
  --video ~/Documents/AirSim/2025-10-02/airsim_rec_20251002_150000.mp4 \
  --data flight_data.csv \
  --output annotated_flight.mp4
```

**标注效果**：
- 速度箭头（蓝色=前进，绿色=侧向，黄色=合成）
- 实时统计面板（速度、位置、时间）
- 轨迹追踪小地图（俯视图）

详见 [VIDEO_ANNOTATION.md](VIDEO_ANNOTATION.md)

### 步骤 4：创建对比视频

使用 FFmpeg 创建专家 vs 模型的并排对比：

```bash
# 左右对比
ffmpeg -i expert_annotated.mp4 -i model_annotated.mp4 \
  -filter_complex hstack comparison.mp4

# 上下对比
ffmpeg -i expert_annotated.mp4 -i model_annotated.mp4 \
  -filter_complex vstack comparison.mp4
```

## 常见问题

### Q1: 模型加载失败 - state_dict 不匹配

**问题**：`Error(s) in loading state_dict for ViTLSTM: Missing key(s)...`

**解决方案**：确保使用正确的模型类型。如果下载的是 `ViTLSTM_model.pth`，使用：
```bash
--model-type ViTLSTM
```

### Q2: 无人机持续下降并坠毁

**问题**：模型推理时无人机高度不断降低

**当前状态**：正在调试中。已添加详细日志来分析模型输出。运行带日志的仿真：
```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 3.0 \
  --log-level DEBUG
```

查看日志中的 "Model Output Statistics" 部分，分析 Z 轴速度是否异常。

### Q3: 可视化窗口无法关闭

**问题**：仿真退出后终端卡住

**解决方案**：按 **Ctrl+C** 两次强制退出。第一次触发正常关闭，第二次强制退出。

### Q4: 视频和数据不同步

**问题**：标注的箭头与实际运动不匹配

**解决方案**：
- 确保 AirSim 录制和仿真同时开始（先按 F9，再运行脚本）
- 检查 CSV 文件时间戳是否从 0 开始递增
- AirSim 录制通常是 30 FPS，确保遥测记录频率 ≥ 30 Hz

### Q5: AirSim 连接失败

**问题**：`Failed to connect to AirSim`

**解决方案**：
1. 确保 AirSim UE4/Unity 环境正在运行
2. 检查 `config/simulation_config.yaml` 中的主机和端口：
   ```yaml
   airsim_host: '127.0.0.1'
   airsim_port: 41451
   ```
3. 确认防火墙未阻止连接

## 🚀 完整示例工作流

创建一个论文级别的对比视频：

```bash
# 1. 启动 AirSim（手动）

# 2. 运行专家策略并录制
# （在 AirSim 中按 F9 开始录制）
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode expert \
  --desired-velocity 3.0 \
  --max-duration 60.0 \
  --record-telemetry expert_data.csv
# （在 AirSim 中按 F9 停止录制）

# 3. 重置环境，运行模型并录制
# （重新定位无人机，按 F9 开始录制）
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 3.0 \
  --max-duration 60.0 \
  --record-telemetry model_data.csv
# （按 F9 停止录制）

# 4. 标注两个视频
python scripts/annotate_flight_video.py \
  --video ~/Documents/AirSim/2025-10-02/airsim_rec_001.mp4 \
  --data expert_data.csv \
  --output expert_annotated.mp4

python scripts/annotate_flight_video.py \
  --video ~/Documents/AirSim/2025-10-02/airsim_rec_002.mp4 \
  --data model_data.csv \
  --output model_annotated.mp4

# 5. 创建并排对比视频
ffmpeg -i expert_annotated.mp4 -i model_annotated.mp4 \
  -filter_complex hstack paper_comparison.mp4

# 完成！paper_comparison.mp4 可用于论文提交
```

## 📚 更多文档

- [预训练模型详细说明](PRETRAINED_MODELS.md)
- [视频标注工具指南](VIDEO_ANNOTATION.md)

## 🔧 开发状态

- ✅ AirSim 集成
- ✅ 专家策略避障
- ✅ 预训练模型加载
- ✅ 实时可视化
- ✅ 视频标注工具
- ✅ 遥测数据记录
- 🚧 模型下降问题调试中
- ⏳ 模型性能优化（待完成）
- ⏳ AirSim 特定数据采集（可选）
- ⏳ 模型微调（可选）

## 💬 获取帮助

如遇问题，请检查：
1. AirSim 是否正常运行
2. Python 依赖是否完整安装
3. 模型文件路径是否正确
4. 配置文件是否有效

提交 issue 时请包含：
- 完整错误信息
- 使用的命令
- 系统信息（Windows/Linux, Python 版本等）
