# Teleavatar 数据记录功能修改方案

## 1. 需求分析

### 1.1 功能目标
- 在策略推理过程中，记录**未经预处理的原始图像**（3个视角）
- 记录机器人本体状态数据（48维）
- 记录策略推理输出的动作（16维）
- 通过键盘交互控制数据记录的启停
- 支持轨迹成功/失败标记
- 输出格式与 LeRobot 兼容

### 1.2 键盘交互流程
```
策略推理运行中...
  ↓
按 'r' → 开始记录当前轨迹
  ↓
记录数据中... (显示帧计数)
  ↓
按 'q' → 停止记录
  ↓
等待标记: 按 's' 保存为成功 / 按 'f' 保存为失败 / 按 'd' 丢弃
  ↓
继续等待下一次记录...
```

## 2. 架构设计

### 2.1 组件关系图
```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Environment  │  │    Agent     │  │ TeleavatarRecorder  │   │
│  │  (env.py)    │  │ (PolicyAgent)│  │   (Subscriber)      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘   │
│         │                  │                     │               │
│         │    observation   │    action           │               │
│         └────────────────→ ├────────────────────→│               │
│                            │                     │               │
└────────────────────────────┼─────────────────────┼───────────────┘
                             │                     │
                             │                     ▼
                             │         ┌──────────────────────┐
                             │         │   KeyboardListener   │
                             │         │   (separate thread)  │
                             │         └──────────┬───────────┘
                             │                    │
                             │                    ▼
                             │         ┌──────────────────────┐
                             │         │  LeRobotDataWriter   │
                             │         │  (保存时调用)         │
                             │         └──────────────────────┘
```

### 2.2 数据流
```
ROS2 Topics
    │
    ▼
TeleavatarROS2Interface (ros2_interface.py)
    │
    ├─→ latest_images (原始图像, dict)
    │     • left_color: (480, 848, 3) RGB
    │     • right_color: (480, 848, 3) RGB
    │     • head_camera: (1080, 1920, 3) RGB
    │
    └─→ latest_joint_states (关节状态, dict)
          • left_arm, right_arm: JointState
          • left_gripper, right_gripper: JointState
    │
    ▼
TeleavatarEnvironment.get_observation() (env.py)
    │
    └─→ observation dict
          • observation/state: (48,) float32
          • observation/images/left_color: uint8
          • observation/images/right_color: uint8
          • observation/images/head_camera: uint8
          • prompt: str
    │
    ▼
Runtime._step() → subscriber.on_step(observation, action)
    │
    └─→ TeleavatarRecorder 记录数据
```

## 3. 文件修改清单

### 3.1 新建文件
| 文件 | 功能 |
|------|------|
| `examples/teleavatar/teleavatar_recorder.py` | 核心记录器 (替代 recording_subscriber.py) |

### 3.2 修改文件
| 文件 | 修改内容 |
|------|----------|
| `examples/teleavatar/main.py` | 添加 `--record` 参数，集成 TeleavatarRecorder |

## 4. 详细设计

### 4.1 `teleavatar_recorder.py` 结构

```python
class TeleavatarRecorder(Subscriber):
    """
    Teleavatar 数据记录器

    功能:
    1. 实现 Subscriber 接口，接收 observation 和 action
    2. 键盘监听（独立线程）
    3. 内存缓冲当前轨迹数据
    4. LeRobot 格式保存
    """

    def __init__(
        self,
        output_dir: str,           # 输出目录
        dataset_name: str,         # 数据集名称
        fps: int = 20,             # 记录帧率
        task_description: str = "" # 任务描述
    ):
        # 状态机
        self._state: Literal["idle", "recording", "pending_label"] = "idle"

        # 当前轨迹缓冲
        self._current_episode_buffer: List[dict] = []
        self._episode_count: int = 0

        # 键盘监听线程
        self._keyboard_thread: Thread

        # LeRobot 数据集 (延迟初始化)
        self._dataset: Optional[LeRobotDataset] = None

    # === Subscriber 接口实现 ===
    def on_episode_start(self) -> None:
        """Episode 开始时调用"""
        pass  # 不自动开始记录

    def on_step(self, observation: dict, action: dict) -> None:
        """每一步调用，记录数据"""
        if self._state != "recording":
            return
        self._buffer_frame(observation, action)

    def on_episode_end(self) -> None:
        """Episode 结束时调用"""
        if self._state == "recording":
            self._stop_recording()

    # === 键盘控制 ===
    def _keyboard_listener(self) -> None:
        """键盘监听循环 (独立线程)"""
        # 使用 pynput 或 termios 实现非阻塞键盘监听

    def _start_recording(self) -> None:
        """开始记录"""
        self._state = "recording"
        self._current_episode_buffer = []
        print("[RECORDER] Started recording...")

    def _stop_recording(self) -> None:
        """停止记录，等待标记"""
        self._state = "pending_label"
        print(f"[RECORDER] Stopped. {len(self._current_episode_buffer)} frames recorded.")
        print("[RECORDER] Press 's' for success, 'f' for failure, 'd' to discard")

    def _save_episode(self, success: bool) -> None:
        """保存当前轨迹到 LeRobot 格式"""
        if not self._current_episode_buffer:
            print("[RECORDER] No data to save")
            return

        self._ensure_dataset_initialized()
        self._write_episode_to_dataset(success)
        self._episode_count += 1
        self._current_episode_buffer = []
        self._state = "idle"

    def _discard_episode(self) -> None:
        """丢弃当前轨迹"""
        self._current_episode_buffer = []
        self._state = "idle"
        print("[RECORDER] Episode discarded")

    # === 数据处理 ===
    def _buffer_frame(self, observation: dict, action: dict) -> None:
        """缓冲一帧数据"""
        frame = {
            "timestamp": time.time(),
            "observation": {
                "state": observation["observation/state"].copy(),
                "images": {
                    "left_color": observation["observation/images/left_color"].copy(),
                    "right_color": observation["observation/images/right_color"].copy(),
                    "head_camera": observation["observation/images/head_camera"].copy(),
                },
            },
            "action": action["actions"].copy(),
            "prompt": observation.get("prompt", ""),
        }
        self._current_episode_buffer.append(frame)

        # 显示进度
        if len(self._current_episode_buffer) % 20 == 0:
            print(f"\r[RECORDER] Recording... {len(self._current_episode_buffer)} frames", end="")

    # === LeRobot 格式写入 ===
    def _ensure_dataset_initialized(self) -> None:
        """延迟初始化 LeRobot 数据集"""
        if self._dataset is not None:
            return

        features = self._setup_features()
        self._dataset = LeRobotDataset.create(
            repo_id=self._dataset_name,
            fps=self._fps,
            features=features,
            robot_type="teleavatar_dual_arm",
            use_videos=True
        )

    def _setup_features(self) -> dict:
        """定义 LeRobot 数据集特征"""
        features = {
            # 动作 (16维: 左臂7+夹爪1, 右臂7+夹爪1)
            "action": {
                "dtype": "float32",
                "shape": (16,),
                "names": {"motors": [
                    "left_j1", "left_j2", "left_j3", "left_j4",
                    "left_j5", "left_j6", "left_j7", "left_gripper",
                    "right_j1", "right_j2", "right_j3", "right_j4",
                    "right_j5", "right_j6", "right_j7", "right_gripper"
                ]}
            },
            # 状态 (48维)
            "observation.state": {
                "dtype": "float32",
                "shape": (48,),
                "names": {"motors": [...]}  # 详见 convert_teleavatar_data_to_lerobot.py
            },
            # 成功标记
            "success": {
                "dtype": "bool",
                "shape": (1,),
            },
            # 图像 (3个视角)
            "observation.images.left_color": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "video_info": {"video.fps": 20.0, ...}
            },
            "observation.images.right_color": {...},
            "observation.images.head_camera": {
                "dtype": "video",
                "shape": (1080, 1920, 3),
                ...
            },
        }
        return features

    def _write_episode_to_dataset(self, success: bool) -> None:
        """将缓冲数据写入 LeRobot 数据集"""
        episode_buffer = self._dataset.create_episode_buffer(self._episode_count)

        for i, frame in enumerate(self._current_episode_buffer):
            is_last = (i == len(self._current_episode_buffer) - 1)
            frame_data = {
                "action": frame["action"],
                "observation.state": frame["observation"]["state"],
                "observation.images.left_color": frame["observation"]["images"]["left_color"],
                "observation.images.right_color": frame["observation"]["images"]["right_color"],
                "observation.images.head_camera": frame["observation"]["images"]["head_camera"],
                "success": np.array([success and is_last], dtype=bool),
                "next.done": np.array([is_last], dtype=bool),
            }
            self._dataset.add_frame(frame_data, frame["prompt"])

        self._dataset.save_episode()
```

### 4.2 键盘监听方案

**方案 A: 使用 pynput (推荐)**
```python
from pynput import keyboard

def _keyboard_listener(self):
    def on_press(key):
        try:
            if key.char == 'r' and self._state == "idle":
                self._start_recording()
            elif key.char == 'q' and self._state == "recording":
                self._stop_recording()
            elif key.char == 's' and self._state == "pending_label":
                self._save_episode(success=True)
            elif key.char == 'f' and self._state == "pending_label":
                self._save_episode(success=False)
            elif key.char == 'd' and self._state == "pending_label":
                self._discard_episode()
        except AttributeError:
            pass  # Special keys

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
```

**方案 B: 使用 termios (纯终端，无额外依赖)**
```python
import sys
import termios
import tty
import select

def _keyboard_listener(self):
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while self._running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                self._handle_key(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
```

### 4.3 main.py 修改

```python
# 添加参数
@dataclasses.dataclass
class Args:
    # ... 现有参数 ...

    # 数据记录设置
    record: bool = False
    """是否启用数据记录"""

    record_dir: str = "teleavatar_recordings"
    """记录数据保存目录"""

    dataset_name: str = "teleavatar_dataset"
    """LeRobot 数据集名称"""

def main(args: Args) -> None:
    # ... 现有代码 ...

    # 创建 subscribers 列表
    subscribers = []

    if args.record:
        from examples.teleavatar.teleavatar_recorder import TeleavatarRecorder
        recorder = TeleavatarRecorder(
            output_dir=args.record_dir,
            dataset_name=args.dataset_name,
            fps=int(args.control_frequency),
            task_description=args.prompt
        )
        subscribers.append(recorder)
        logging.info("Data recording enabled")
        logging.info("  Press 'r' to start recording")
        logging.info("  Press 'q' to stop recording")
        logging.info("  Press 's' to save as success, 'f' as failure, 'd' to discard")

    # 创建 runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=subscribers,  # 传入 recorder
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
```

## 5. LeRobot 数据集输出结构

```
teleavatar_recordings/
└── teleavatar_dataset/
    ├── meta/
    │   ├── info.json           # 数据集元信息
    │   ├── episodes.jsonl      # Episode 索引
    │   ├── stats.json          # 统计信息
    │   └── tasks.jsonl         # 任务描述
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   └── ...
    │   └── ...
    └── videos/
        ├── chunk-000/
        │   ├── observation.images.left_color/
        │   │   ├── episode_000000.mp4
        │   │   └── ...
        │   ├── observation.images.right_color/
        │   │   └── ...
        │   └── observation.images.head_camera/
        │       └── ...
        └── ...
```

## 6. 使用方式

```bash
# 启动策略服务器 (终端1)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/my_exp/20000

# 运行机器人并启用记录 (终端2)
python examples/teleavatar/main.py \
    --remote-host 127.0.0.1 \
    --record \
    --record-dir ./my_recordings \
    --dataset-name my_teleavatar_data \
    --prompt "Pick up the red block"

# 交互控制:
# - 按 'r' 开始记录
# - 按 'q' 停止记录
# - 按 's' 保存为成功轨迹
# - 按 'f' 保存为失败轨迹
# - 按 'd' 丢弃当前记录
```

## 7. 实现步骤

### Phase 1: 基础记录器
1. 创建 `teleavatar_recorder.py`
2. 实现 Subscriber 接口
3. 实现内存缓冲
4. 实现键盘监听 (pynput)

### Phase 2: LeRobot 集成
1. 实现 `_setup_features()`
2. 实现 `_write_episode_to_dataset()`
3. 测试视频编码

### Phase 3: main.py 集成
1. 添加命令行参数
2. 集成 recorder 到 runtime
3. 端到端测试

## 8. 注意事项

1. **线程安全**: 键盘监听在独立线程，需要用锁保护共享状态
2. **内存管理**: 长轨迹可能占用大量内存，考虑设置最大帧数限制
3. **视频编码**: LeRobot 使用 ffmpeg 编码视频，需确保环境有 ffmpeg
4. **图像格式**: 确保保存的是原始 RGB uint8 图像，不是预处理后的
5. **时间戳**: 每帧记录时间戳，用于后续分析和调试

## 9. 依赖

```toml
# pyproject.toml 新增
"pynput>=1.7.6"  # 键盘监听 (可选)
```

或使用纯 termios 方案，无需额外依赖。
