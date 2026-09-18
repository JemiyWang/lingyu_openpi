# 模型推理阶段数据记录指南

本文档介绍如何在 Teleavatar 机器人运行模型推理时记录数据，用于后续分析、调试或扩充训练数据集。

## 概述

OpenPI 的 Teleavatar 示例提供了一个完整的推理阶段数据记录系统，主要由两个模块组成：

- **`main.py`**: 主运行脚本，通过 `--record` 参数启用数据记录
- **`teleavatar_recorder.py`**: 数据记录器实现，支持键盘控制和 LeRobot 格式输出

### 核心特性

- 键盘控制录制的开始/停止
- 轨迹成功/失败标注
- 输出 LeRobot 兼容的数据集格式
- 自动保存图像、状态和动作数据
- 支持 numpy 格式作为备选输出

## 快速开始

### 1. 启动策略服务器

首先在一个终端启动模型推理服务器：

```bash
cd /home/ubuntu/lingyu/openpi

# 使用训练好的检查点
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/my_experiment/20000
```

### 2. 启动带录制功能的机器人控制

在另一个终端运行 `main.py` 并启用录制：

```bash
cd /home/ubuntu/lingyu/openpi

# 基础用法：启用录制
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record

# 完整配置示例
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir teleavatar_recordings \
    --dataset-name my_dataset \
    --prompt "抓取红色方块" \
    --control-mode soft_inpainting \
    --max-record-frames 5000
```

### 3. 使用键盘控制录制

启动后，终端会显示录制器控制说明。使用以下按键控制：

| 按键 | 功能 |
|------|------|
| `r` | 开始录制 |
| `q` | 停止录制 |
| `s` | 将当前轨迹保存为**成功**轨迹 |
| `f` | 将当前轨迹保存为**失败**轨迹 |
| `d` | 丢弃当前录制 |

## 命令行参数详解

### 录制相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--record` | `False` | 启用 20Hz 数据录制（与控制频率一致） |
| `--record-direct` | `False` | 启用 30fps 数据录制（图像/状态 30fps，动作插值） |
| `--record-dir` | `teleavatar_recordings` | 录制数据保存目录 |
| `--dataset-name` | `teleavatar_dataset` | LeRobot 数据集名称 |
| `--max-record-frames` | `10000` | 单个 episode 最大帧数（安全限制） |

### 录制模式对比

| 模式 | 图像频率 | 状态频率 | 动作频率 | 说明 |
|------|---------|---------|---------|------|
| `--record` | 20Hz | 20Hz | 20Hz | 与控制循环同步，数据在 on_step 回调中采集 |
| `--record-direct` | 20fps | 20fps | 20Hz | 独立 ROS2 订阅采集，动作通过时间戳匹配 |

**选择建议**：
- `--record-direct`：直接订阅 ROS2 话题，时间戳更精确
- `--record`：简单可靠，与控制循环完全同步

### 控制相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote-host` | `0.0.0.0` | 策略服务器 IP 地址 |
| `--remote-port` | `8000` | 策略服务器端口 |
| `--control-frequency` | `20.0` | 控制循环频率 (Hz) |
| `--control-mode` | `soft_inpainting` | 控制模式：`action_chunk`、`receding_horizon` 或 `soft_inpainting` |
| `--prompt` | `Stack the three blocks` | 机器人任务指令 |
| `--num-episodes` | `100` | 运行的 episode 数量 |
| `--max-episode-steps` | `250` | 每个 episode 的最大步数 |

### Soft Inpainting 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--action-horizon` | `30` | 策略返回的动作序列长度 |
| `--actions-to-execute` | `13` | 执行多少步后重新查询策略 |
| `--actions-to-keep` | `2` | 保留多少步用于 inpainting 约束 |

## 数据格式

### LeRobot 格式（默认）

当 LeRobot 库可用时，数据保存为 LeRobot 数据集格式，包含以下字段：

**状态数据 (`observation.state`)**
- 形状：`(48,)`
- 内容：左右臂各 7 个关节位置 + 夹爪位置 + 速度 + 力矩

**动作数据 (`action`)**
- 形状：`(16,)`
- 内容：左右臂各 7 个关节目标位置 + 夹爪目标位置

**图像数据**
- `observation.images.left_color`: 左臂相机 (480×848×3)
- `observation.images.right_color`: 右臂相机 (480×848×3)
- `observation.images.head_camera`: 头部相机 (1080×1920×3)

**标签数据**
- `success`: 轨迹是否成功（布尔值）
- `next.done`: 是否为 episode 最后一帧（布尔值）

### Numpy 格式（备选）

如果 LeRobot 不可用，数据保存为 numpy 格式：

```
teleavatar_recordings/
├── episode_0000/
│   ├── meta.json          # 元数据（episode_id, success, num_frames, fps, task_description）
│   ├── states.npy         # 状态序列 (N, 48)
│   ├── actions.npy        # 动作序列 (N, 16)
│   ├── timestamps.npy     # 时间戳序列 (N,)
│   └── images/
│       ├── left_color/
│       │   ├── frame_00000.npy
│       │   └── ...
│       ├── right_color/
│       │   └── ...
│       └── head_camera/
│           └── ...
├── episode_0001/
│   └── ...
```

## 直接 ROS2 订阅录制架构

使用 `--record-direct` 时，录制器独立于控制循环，直接订阅 ROS2 话题：

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2 话题层                                   │
├─────────────────────────────────────────────────────────────────┤
│  /left/image_raw        ─┐                                      │
│  /right/image_raw        ├─→ 20fps 定时采样 → 图像缓冲          │
│  /head/image_raw        ─┘                                      │
│  /*/joint_states        ────→ 20fps 定时采样 → 状态缓冲         │
└─────────────────────────────────────────────────────────────────┘
                                    ↑
                                    │ 接收动作 (带时间戳)
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    控制循环 (20Hz)                               │
├─────────────────────────────────────────────────────────────────┤
│    推理 → 生成 action → 发送给录制器                            │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    保存时处理                                    │
├─────────────────────────────────────────────────────────────────┤
│  动作匹配: 通过时间戳找到最近的动作 (最近邻)                    │
│                                                                  │
│  帧:    F0 -------- F1 -------- F2                              │
│         0ms        50ms       100ms                              │
│                                                                  │
│  动作:  A0 -------- A1 -------- A2                              │
│         0ms        50ms       100ms                              │
│                                                                  │
│  匹配结果: F0→A0, F1→A1, F2→A2 (按时间戳最近邻匹配)             │
└─────────────────────────────────────────────────────────────────┘
```

## 典型工作流程

### 场景 1：收集成功轨迹扩充数据集（20Hz）

```bash
# 1. 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/exp1/20000

# 2. 启动录制（20Hz，与控制频率一致）
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir training_data \
    --dataset-name successful_demos \
    --prompt "将红色方块放入盒子中" \
    --max-episode-steps 5000

# 3. 操作流程
#    - 按 'r' 开始录制
#    - 观察机器人执行任务
#    - 任务完成后按 'q' 停止录制
#    - 如果成功，按 's' 保存
#    - 如果失败，按 'f' 标记失败或按 'd' 丢弃
#    - 重复以上步骤收集更多数据
```

### 场景 1b：使用直接 ROS2 订阅录制

```bash
# 1. 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/exp1/20000

# 2. 启动录制（直接订阅 ROS2 话题，时间戳更精确）
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record-direct \
    --record-dir training_data \
    --dataset-name successful_demos \
    --prompt "将红色方块放入盒子中" \
    --max-episode-steps 5000

# 录制器独立于控制循环，直接订阅 ROS2 话题
# 图像/状态/动作都是 20fps，通过时间戳匹配
```

### 场景 2：调试模型行为

```bash
# 使用较短的录制限制便于快速迭代
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir debug_runs \
    --dataset-name debug_session \
    --max-record-frames 500 \
    --max-episode-steps 100

# 录制后可以分析：
# - 状态轨迹是否平滑
# - 动作是否合理
# - 图像输入是否正确
```

### 场景 3：对比不同控制模式

```bash
# Action Chunk 模式
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir comparison_data \
    --dataset-name action_chunk_test \
    --control-mode action_chunk

# Receding Horizon 模式
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir comparison_data \
    --dataset-name receding_horizon_test \
    --control-mode receding_horizon

# Soft Inpainting 模式
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir comparison_data \
    --dataset-name soft_inpainting_test \
    --control-mode soft_inpainting
```

## TeleavatarRecorder 类详解

### 状态机

录制器使用三状态状态机：

```
idle (空闲) --[按 'r']--> recording (录制中) --[按 'q']--> pending_label (待标注)
                                                              |
                              +--------------------------------+
                              |
                              v
              [按 's'] --> 保存为成功 --> idle
              [按 'f'] --> 保存为失败 --> idle
              [按 'd'] --> 丢弃数据 --> idle
```

### 初始化参数

```python
TeleavatarRecorder(
    output_dir: str = "teleavatar_recordings",  # 输出目录
    dataset_name: str = "teleavatar_dataset",   # 数据集名称
    fps: int = 20,                               # 帧率（应与控制频率匹配）
    task_description: str = "",                  # 默认任务描述
    max_frames: int = 10000,                     # 单 episode 最大帧数
)
```

### Subscriber 接口

`TeleavatarRecorder` 实现了 `openpi_client.runtime.subscriber.Subscriber` 接口：

- `on_episode_start()`: episode 开始时调用（不自动开始录制）
- `on_step(observation, action)`: 每步调用，缓存数据
- `on_episode_end()`: episode 结束时调用，自动停止录制

### 独立测试

可以独立运行 `teleavatar_recorder.py` 进行测试：

```bash
cd /home/ubuntu/lingyu/openpi
python examples/teleavatar/teleavatar_recorder.py
```

这会使用随机数据模拟录制过程，用于验证录制器功能。

## 注意事项

1. **终端要求**: 键盘控制需要在终端中运行，不支持后台运行或非 TTY 环境
2. **帧数限制**: 达到 `max_frames` 时会自动停止录制
3. **图像分辨率**: 图像保持原始分辨率以匹配训练数据
4. **存储空间**: 图像数据较大，请确保有足够的磁盘空间
5. **LeRobot 依赖**: 如果 LeRobot 未安装，会自动降级为 numpy 格式

## 与训练流程集成

录制的 LeRobot 格式数据可以直接用于模型微调：

```bash
# 1. 合并录制数据与原有数据
uv run scripts/merge_lerobot_datasets.py \
    --input-dirs teleavatar_recordings/teleavatar_dataset \
    --output-dir merged_dataset

# 2. 重新计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar

# 3. 继续训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar \
    --exp-name=finetuned_with_new_data
```

## 常见问题

### Q: 按键没有响应？
A: 确保在终端中运行（不是 IDE 的运行按钮），并且终端窗口处于激活状态。

### Q: 录制的数据在哪里？
A: 默认在 `teleavatar_recordings/` 目录下，可通过 `--record-dir` 参数修改。

### Q: 如何查看录制的数据？
A: LeRobot 格式可使用 LeRobot 工具查看，numpy 格式可直接用 Python 加载：
```python
import numpy as np
import json

# 加载元数据
with open("teleavatar_recordings/episode_0000/meta.json") as f:
    meta = json.load(f)

# 加载轨迹数据
states = np.load("teleavatar_recordings/episode_0000/states.npy")
actions = np.load("teleavatar_recordings/episode_0000/actions.npy")
```

### Q: 如何只录制成功的轨迹？
A: 使用键盘控制，在每次录制结束后根据任务执行结果选择 `s`（成功）或 `d`（丢弃）。

### Q: 录制时机器人控制会变慢吗？
A: 录制操作在主线程中进行，但数据缓存操作非常轻量。保存操作在停止录制后进行，不影响实时控制。
