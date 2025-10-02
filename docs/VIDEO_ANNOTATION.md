# 视频标注工具使用指南

本文档介绍如何为论文和展示制作带有速度向量、轨迹等信息的专业飞行视频。

## 📹 工作流程

### 步骤 1：在 AirSim 中录制视频

在 AirSim 中，按 **F9** 开始录制飞行视频。录制的视频默认保存在：

```
Windows: C:\Users\<YourName>\Documents\AirSim\<CurrentDate>\
Linux: ~/Documents/AirSim/<CurrentDate>/
```

录制文件名格式：`airsim_rec_<timestamp>.mp4`

### 步骤 2：在仿真时记录飞行数据

运行仿真时添加 `--record-telemetry` 参数：

```bash
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 3.0 \
  --record-telemetry flight_data.csv
```

这会生成一个 CSV 文件，包含：
- `timestamp` - 时间戳（秒）
- `position_x`, `position_y`, `position_z` - 位置（米）
- `velocity_x`, `velocity_y`, `velocity_z` - 速度（米/秒）
- `orientation_w`, `orientation_x`, `orientation_y`, `orientation_z` - 姿态四元数
- 其他遥测数据

### 步骤 3：标注视频

使用 `annotate_flight_video.py` 脚本：

```bash
python scripts/annotate_flight_video.py \
  --video ~/Documents/AirSim/2025-10-02/airsim_rec_20251002_172200.mp4 \
  --data flight_data.csv \
  --output annotated_flight.mp4
```

## 🎬 标注效果

生成的视频将包含：

```
┌──────────────────────────────────────────┐
│  原始 AirSim 视频画面                    │
│                                          │
│  ┌────────────────────┐                 │
│  │  Stats Panel       │                 │
│  │  Velocity: X=2.34  │                 │
│  │  Speed: 2.35 m/s   │                 │
│  │  Position: ...     │                 │
│  │  [热力图色条]      │                 │
│  │  蓝→绿→黄→红       │                 │
│  │  0───────8 m/s     │                 │
│  └────────────────────┘                 │
│                                          │
│  速度箭头（热力图颜色）：                │
│    慢速 → 蓝色                           │
│    中速 → 绿色/黄色                      │
│    快速 → 红色                           │
│                                          │
│              ┌─────────────┐             │
│              │ Trajectory  │             │
│              │  (Top View) │             │
│              │  颜色=速度  │             │
│              │    ...●──   │             │
│              └─────────────┘             │
└──────────────────────────────────────────┘
```

**热力图颜色映射**：
- 🔵 蓝色：低速（0 m/s）
- 🔷 青色：低中速（2 m/s）
- 🟢 绿色：中速（4 m/s）
- 🟡 黄色：中高速（6 m/s）
- 🟠 橙色：高速（7 m/s）
- 🔴 红色：最高速（8+ m/s）

## ⚙️ 高级选项

### 自定义热力图速度范围

```bash
python scripts/annotate_flight_video.py \
  --video input.mp4 \
  --data flight_data.csv \
  --output output.mp4 \
  --min-velocity 0.0 \
  --max-velocity 10.0  # 默认: 8.0
```

**推荐速度范围**：
- 慢速飞行（训练数据）：`--max-velocity 5.0`
- 正常飞行：`--max-velocity 8.0`（默认）
- 高速飞行：`--max-velocity 12.0`

### 禁用热力图颜色

使用固定颜色（蓝色=前进，绿色=侧向，黄色=合成）：

```bash
python scripts/annotate_flight_video.py \
  --video input.mp4 \
  --data flight_data.csv \
  --output output.mp4 \
  --no-heatmap
```

### 自定义箭头大小

```bash
python scripts/annotate_flight_video.py \
  --video input.mp4 \
  --data flight_data.csv \
  --output output.mp4 \
  --arrow-scale 50.0  # 默认: 30.0
```

### 禁用轨迹显示

```bash
python scripts/annotate_flight_video.py \
  --video input.mp4 \
  --data flight_data.csv \
  --output output.mp4 \
  --no-trajectory
```

### 禁用统计面板

```bash
python scripts/annotate_flight_video.py \
  --video input.mp4 \
  --data flight_data.csv \
  --output output.mp4 \
  --no-stats
```

## 📊 手动创建遥测数据文件

如果没有使用 `--record-telemetry`，可以手动创建 CSV 文件：

```csv
timestamp,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z
0.00,0.0,0.0,-5.0,0.0,0.0,0.0
0.033,0.1,0.0,-5.0,3.0,0.0,0.0
0.066,0.2,0.0,-5.0,3.0,0.1,0.0
...
```

最少需要的列：
- `timestamp` - 时间戳（与视频帧对应）
- `position_x`, `position_y`, `position_z` - 位置
- `velocity_x`, `velocity_y`, `velocity_z` - 速度

## 🎓 论文使用建议

### 1. 对比视频

创建两个视频对比专家策略和模型推理：

```bash
# 专家策略
python scripts/simulate.py --mode expert --record-telemetry expert_data.csv
python scripts/annotate_flight_video.py \
  --video expert_flight.mp4 --data expert_data.csv --output expert_annotated.mp4

# 模型推理
python scripts/simulate.py --mode model --record-telemetry model_data.csv
python scripts/annotate_flight_video.py \
  --video model_flight.mp4 --data model_data.csv --output model_annotated.mp4
```

### 2. 不同速度对比

测试不同期望速度：

```bash
# 3 m/s
python scripts/simulate.py --desired-velocity 3.0 --record-telemetry v3_data.csv

# 5 m/s
python scripts/simulate.py --desired-velocity 5.0 --record-telemetry v5_data.csv

# 7 m/s
python scripts/simulate.py --desired-velocity 7.0 --record-telemetry v7_data.csv
```

### 3. 视频合并

使用 `ffmpeg` 将多个视频并排显示：

```bash
# 左右对比
ffmpeg -i expert_annotated.mp4 -i model_annotated.mp4 \
  -filter_complex hstack output_comparison.mp4

# 上下对比
ffmpeg -i expert_annotated.mp4 -i model_annotated.mp4 \
  -filter_complex vstack output_comparison.mp4

# 2x2 网格
ffmpeg -i v3_annotated.mp4 -i v5_annotated.mp4 \
       -i v7_annotated.mp4 -i expert_annotated.mp4 \
  -filter_complex "[0][1]hstack[top];[2][3]hstack[bottom];[top][bottom]vstack" \
  output_grid.mp4
```

## 🐛 故障排除

### 问题 1：视频和数据不同步

**原因**：数据记录频率与视频帧率不匹配

**解决方案**：
- AirSim 录制视频通常是 30 FPS
- 确保数据记录频率 ≥ 30 Hz
- 或在标注脚本中添加插值

### 问题 2：箭头太小/太大

**解决方案**：调整 `--arrow-scale` 参数
- 速度快 → 使用较小的 scale（如 20.0）
- 速度慢 → 使用较大的 scale（如 50.0）

### 问题 3：轨迹显示不全

**原因**：轨迹范围太大，超出小地图

**解决方案**：编辑脚本中的 `map_size` 参数（默认 200 像素）

## 📝 数据格式参考

### 完整的 CSV 列（推荐）

```csv
timestamp,
position_x,position_y,position_z,
velocity_x,velocity_y,velocity_z,
orientation_w,orientation_x,orientation_y,orientation_z,
desired_velocity,
collision
```

### JSON 格式（可选）

```json
[
  {
    "timestamp": 0.0,
    "position_x": 0.0,
    "position_y": 0.0,
    "position_z": -5.0,
    "velocity_x": 0.0,
    "velocity_y": 0.0,
    "velocity_z": 0.0
  },
  ...
]
```

## 🎨 自定义标注样式

### 热力图颜色映射

热力图使用 HSV 色彩空间进行平滑过渡：

```python
# 色相范围：240（蓝色）→ 0（红色）
hue = int(240 * (1 - normalized_speed))
```

**自定义颜色范围**：

编辑 `scripts/annotate_flight_video.py` 中的 `velocity_to_heatmap_color()` 函数：

```python
def velocity_to_heatmap_color(self, speed: float) -> Tuple[int, int, int]:
    normalized = (speed - self.min_velocity) / (self.max_velocity - self.min_velocity)
    normalized = np.clip(normalized, 0.0, 1.0)

    # 修改这里以调整颜色范围
    # 240 = 蓝色, 120 = 绿色, 60 = 黄色, 0 = 红色
    hue = int(240 * (1 - normalized))  # 蓝→红

    # 或使用其他范围：
    # hue = int(120 + 60 * (1 - normalized))  # 绿→黄→红

    hsv_color = np.uint8([[[hue, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr_color)
```

### 其他样式设置

编辑 `scripts/annotate_flight_video.py` 中的参数：

```python
# 箭头粗细
cv2.arrowedLine(frame, start, end, color, thickness=4)  # 主箭头
cv2.arrowedLine(frame, start, end, color, thickness=3)  # 分量箭头

# 文字大小
font_scale = 0.7

# 面板透明度
alpha = 0.7  # 0.0=完全透明, 1.0=完全不透明

# 轨迹线粗细
cv2.line(frame, points[i-1], points[i], color, thickness=2)
```

## 📖 示例

完整示例工作流：

```bash
# 1. 在 AirSim 中按 F9 开始录制

# 2. 运行仿真并记录数据
python scripts/simulate.py \
  --config config/simulation_config.yaml \
  --mode model \
  --model-path models/ViTLSTM_model.pth \
  --model-type ViTLSTM \
  --desired-velocity 5.0 \
  --max-duration 60.0 \
  --record-telemetry flight_2025_10_02.csv

# 3. 在 AirSim 中按 F9 停止录制

# 4. 标注视频
python scripts/annotate_flight_video.py \
  --video ~/Documents/AirSim/2025-10-02/airsim_rec_20251002_150000.mp4 \
  --data flight_2025_10_02.csv \
  --output paper_figure_flight_demo.mp4 \
  --arrow-scale 40.0

# 5. 检查输出
vlc paper_figure_flight_demo.mp4
```

## 🔗 相关文档

- [AirSim Recording API](https://microsoft.github.io/AirSim/settings/)
- [OpenCV Video I/O](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
