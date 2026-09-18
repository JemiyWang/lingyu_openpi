#!/usr/bin/env python3
"""
Teleavatar High-Frequency Data Recorder - Records robot data directly from ROS2.

This recorder decouples image/state capture from the control loop:
- Images and joint states: Recorded at target fps directly from ROS2 callbacks
- Actions: Received from control loop and matched to frames by timestamp

Features:
- Keyboard-controlled recording start/stop
- Success/failure trajectory labeling
- LeRobot-compatible dataset output format
- Direct ROS2 subscription for precise timing

Usage:
    Integrated with main.py via --record-direct flag.

    Keyboard controls:
    - 'r': Start recording
    - 'q': Stop recording
    - 's': Save as success trajectory
    - 'f': Save as failure trajectory
    - 'd': Discard current recording
"""

import logging
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

# ROS2 imports
try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image, JointState
    from geometry_msgs.msg import Pose
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 not available. Direct ROS2 recorder requires ROS2.")

# LeRobot path setup - add common lerobot locations before import
import os
_lerobot_paths = [
    "/home/lingyu/project/lerobot",
    "/home/lingyu/VLA-project/lerobot",
    "/home/lingyu/anaconda3/envs/lerobot/lib/python3.10/site-packages",
    os.path.expanduser("~/project/lerobot"),
]
for _p in _lerobot_paths:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# LeRobot imports - try multiple path patterns for different versions
# v0.4.x uses lerobot.datasets.*, newer versions use lerobot.common.datasets.*
LEROBOT_AVAILABLE = False
VIDEO_ENCODING_AVAILABLE = False
encode_video_frames = None  # Will be set if available

# Try new path first (lerobot >= 0.5)
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames
    LEROBOT_AVAILABLE = True
    VIDEO_ENCODING_AVAILABLE = True
    logging.info("LeRobot loaded (new path: lerobot.common.datasets)")
except ImportError:
    pass

# Try old path (lerobot 0.4.x)
if not LEROBOT_AVAILABLE:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LEROBOT_AVAILABLE = True
        logging.info("LeRobot loaded (old path: lerobot.datasets)")

        # Try to get video encoding from different locations
        try:
            from lerobot.datasets.video_utils import encode_video_frames
            VIDEO_ENCODING_AVAILABLE = True
        except ImportError:
            try:
                from lerobot.common.datasets.video_utils import encode_video_frames
                VIDEO_ENCODING_AVAILABLE = True
            except ImportError:
                logging.warning("Video encoding not available, will use default codec")
    except ImportError:
        pass

if not LEROBOT_AVAILABLE:
    logging.warning("LeRobot not available. Recording will use numpy format.")



@dataclass
class TimestampedFrame:
    """A single frame with timestamp."""
    timestamp: float
    images: Dict[str, np.ndarray]
    state: np.ndarray


@dataclass
class TimestampedAction:
    """An action with timestamp from control loop."""
    timestamp: float
    action: np.ndarray


class ROS2RecorderNode(Node):
    """ROS2 node that subscribes to topics and buffers data at 30fps."""

    def __init__(
        self,
        target_fps: float = 30.0,
        buffer_callback=None,
    ):
        super().__init__('teleavatar_30fps_recorder')

        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # ~33.3ms for 30fps
        self.buffer_callback = buffer_callback

        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        # Latest data cache (updated by callbacks)
        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_states: Dict[str, JointState] = {}
        self.latest_ee_poses: Dict[str, Pose] = {}  # End-effector poses

        # Joint name mappings
        self.left_joint_names = ['l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7']
        self.right_joint_names = ['r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7']

        # Setup subscribers
        self._setup_subscribers()

        # Timer for 30fps sampling
        self.is_recording = False
        self.sample_timer = self.create_timer(self.frame_interval, self._sample_callback)

        self.get_logger().info(f"ROS2RecorderNode initialized at {target_fps} fps")

    def _setup_subscribers(self):
        """Setup ROS2 subscribers."""
        # Image subscribers - 4 cameras matching your actual ROS2 topics:
        # /xr_video_topic/ffmpeg -> head_camera (头部双目) - ffmpeg格式，需特殊处理
        # /head/image_raw -> chest_camera (胸部)
        # /left/image_raw -> left_color (左腕)
        # /right/image_raw -> right_color (右腕)

        # Standard image subscribers (sensor_msgs/Image)
        self.create_subscription(
            Image, '/head/image_raw',
            lambda msg: self._image_callback(msg, 'chest_camera'), 10
        )
        self.create_subscription(
            Image, '/left/image_raw',
            lambda msg: self._image_callback(msg, 'left_color'), 10
        )
        self.create_subscription(
            Image, '/right/image_raw',
            lambda msg: self._image_callback(msg, 'right_color'), 10
        )

        # XR head camera uses ffmpeg format - try to subscribe if available
        try:
            from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
            self.create_subscription(
                FFMPEGPacket, '/xr_video_topic/ffmpeg',
                self._ffmpeg_callback, 10
            )
            self._ffmpeg_available = True
            self.get_logger().info("FFMPEGPacket subscriber enabled for /xr_video_topic/ffmpeg")
        except ImportError:
            self._ffmpeg_available = False
            self.get_logger().warning(
                "ffmpeg_image_transport_msgs not available. "
                "XR head camera will use placeholder images. "
                "Install with: sudo apt install ros-humble-ffmpeg-image-transport-msgs"
            )

        # Joint state subscribers
        self.create_subscription(
            JointState, '/left_arm/joint_states',
            lambda msg: self._joint_callback(msg, 'left_arm'), 10
        )
        self.create_subscription(
            JointState, '/right_arm/joint_states',
            lambda msg: self._joint_callback(msg, 'right_arm'), 10
        )
        self.create_subscription(
            JointState, '/left_gripper/joint_states',
            lambda msg: self._joint_callback(msg, 'left_gripper'), 10
        )
        self.create_subscription(
            JointState, '/right_gripper/joint_states',
            lambda msg: self._joint_callback(msg, 'right_gripper'), 10
        )

        # End-effector pose subscribers (for 62-dim state)
        self.create_subscription(
            Pose, '/left_arm/current_ee_pose',
            lambda msg: self._ee_pose_callback(msg, 'left_ee'), 10
        )
        self.create_subscription(
            Pose, '/right_arm/current_ee_pose',
            lambda msg: self._ee_pose_callback(msg, 'right_ee'), 10
        )

    def _image_callback(self, msg: Image, camera_name: str):
        """Update latest image cache."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self.lock:
                self.latest_images[camera_name] = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f"Image callback error ({camera_name}): {e}")

    def _ffmpeg_callback(self, msg):
        """Handle ffmpeg compressed video packets for XR head camera."""
        try:
            import av
            import io

            # Decode ffmpeg packet using PyAV
            # FFMPEGPacket contains: data (bytes), width, height, encoding, pts, flags
            packet_data = bytes(msg.data)

            # Create in-memory container for decoding
            if not hasattr(self, '_ffmpeg_decoder'):
                # Initialize decoder on first packet
                self._ffmpeg_codec_ctx = av.CodecContext.create('hevc', 'r')
                self._ffmpeg_decoder = True

            # Decode the packet
            packet = av.Packet(packet_data)
            try:
                frames = self._ffmpeg_codec_ctx.decode(packet)
                for frame in frames:
                    # Convert to numpy array (RGB)
                    img = frame.to_ndarray(format='rgb24')
                    with self.lock:
                        self.latest_images['head_camera'] = img.copy()
            except av.AVError:
                # May need more packets to decode (B-frames, etc.)
                pass

        except ImportError:
            # PyAV not available, use placeholder
            if not hasattr(self, '_ffmpeg_warning_shown'):
                self.get_logger().warning(
                    "PyAV not available for ffmpeg decoding. "
                    "Install with: pip install av"
                )
                self._ffmpeg_warning_shown = True
        except Exception as e:
            self.get_logger().error(f"FFmpeg decode error: {e}")

    def _joint_callback(self, msg: JointState, joint_group: str):
        """Update latest joint state cache."""
        with self.lock:
            self.latest_joint_states[joint_group] = msg

    def _ee_pose_callback(self, msg: Pose, ee_name: str):
        """Update latest end-effector pose cache."""
        with self.lock:
            self.latest_ee_poses[ee_name] = msg

    def _sample_callback(self):
        """Timer callback - sample data at 30fps when recording."""
        if not self.is_recording or self.buffer_callback is None:
            return

        with self.lock:
            # Check if we have required data
            # 4 cameras: head_camera (头部双目), chest_camera (胸部), left_color (左腕), right_color (右腕)
            # head_camera uses ffmpeg format which may not be available, use placeholder if needed
            required_images = ['chest_camera', 'left_color', 'right_color']  # head_camera is optional
            required_joints = ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']
            required_ee_poses = ['left_ee', 'right_ee']

            if not all(cam in self.latest_images for cam in required_images):
                return
            if not all(joint in self.latest_joint_states for joint in required_joints):
                return
            if not all(ee in self.latest_ee_poses for ee in required_ee_poses):
                return

            # Build images dict
            # head_camera: use actual image if available, otherwise create placeholder
            if 'head_camera' in self.latest_images:
                head_img = self.latest_images['head_camera'].copy()
            else:
                # Create placeholder image (2160x4320 black image with warning text)
                head_img = np.zeros((2160, 4320, 3), dtype=np.uint8)
                if not hasattr(self, '_head_camera_warning_shown'):
                    self.get_logger().warning(
                        "XR head camera not available (ffmpeg decoding). "
                        "Using placeholder images for head_camera."
                    )
                    self._head_camera_warning_shown = True

            images = {
                'head_camera': head_img,
                'chest_camera': self.latest_images['chest_camera'].copy(),
                'left_color': self.latest_images['left_color'].copy(),
                'right_color': self.latest_images['right_color'].copy(),
            }

            # Build 62-dimensional state vector (matching example dataset)
            state = self._build_state_vector()

            # Create timestamped frame
            frame = TimestampedFrame(
                timestamp=time.time(),
                images=images,
                state=state,
            )

            # Send to buffer
            self.buffer_callback(frame)

    def _build_state_vector(self) -> np.ndarray:
        """Build 62-dim state vector matching example dataset format.

        Layout:
        - [0:16] Joint positions (left_joint1-7, left_gripper, right_joint1-7, right_gripper)
        - [16:32] Joint velocities
        - [32:48] Joint efforts
        - [48:55] Left end-effector pose (position_xyz, orientation_xyzw)
        - [55:62] Right end-effector pose (position_xyz, orientation_xyzw)
        """
        state = np.zeros(62, dtype=np.float32)

        left_arm = self.latest_joint_states['left_arm']
        right_arm = self.latest_joint_states['right_arm']
        left_gripper = self.latest_joint_states['left_gripper']
        right_gripper = self.latest_joint_states['right_gripper']
        left_ee = self.latest_ee_poses['left_ee']
        right_ee = self.latest_ee_poses['right_ee']

        # Positions (0-15)
        state[0:7] = self._extract_field(left_arm, 'position', 7)
        state[7] = self._extract_field(left_gripper, 'position', 1)[0]
        state[8:15] = self._extract_field(right_arm, 'position', 7)
        state[15] = self._extract_field(right_gripper, 'position', 1)[0]

        # Velocities (16-31)
        state[16:23] = self._extract_field(left_arm, 'velocity', 7)
        state[23] = self._extract_field(left_gripper, 'velocity', 1)[0]
        state[24:31] = self._extract_field(right_arm, 'velocity', 7)
        state[31] = self._extract_field(right_gripper, 'velocity', 1)[0]

        # Efforts (32-47)
        state[32:39] = self._extract_field(left_arm, 'effort', 7)
        state[39] = self._extract_field(left_gripper, 'effort', 1)[0]
        state[40:47] = self._extract_field(right_arm, 'effort', 7)
        state[47] = self._extract_field(right_gripper, 'effort', 1)[0]

        # Left end-effector pose (48-54)
        state[48] = left_ee.position.x
        state[49] = left_ee.position.y
        state[50] = left_ee.position.z
        state[51] = left_ee.orientation.x
        state[52] = left_ee.orientation.y
        state[53] = left_ee.orientation.z
        state[54] = left_ee.orientation.w

        # Right end-effector pose (55-61)
        state[55] = right_ee.position.x
        state[56] = right_ee.position.y
        state[57] = right_ee.position.z
        state[58] = right_ee.orientation.x
        state[59] = right_ee.orientation.y
        state[60] = right_ee.orientation.z
        state[61] = right_ee.orientation.w

        return state

    def _extract_field(self, msg: JointState, field: str, n: int) -> np.ndarray:
        """Extract field from JointState message."""
        data = getattr(msg, field, [])
        if len(data) >= n:
            return np.array(data[:n], dtype=np.float32)
        result = np.zeros(n, dtype=np.float32)
        result[:len(data)] = data
        return result

    def start_recording(self):
        """Start recording at 30fps."""
        self.is_recording = True

    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False


class TeleavatarRecorder30fps:
    """
    Teleavatar high-frequency data recorder with direct ROS2 subscription.

    Records images and joint states directly from ROS2 at target fps,
    and matches actions to frames by timestamp (nearest neighbor).
    """

    def __init__(
        self,
        output_dir: str = "teleavatar_recordings",
        dataset_name: str = "teleavatar_dataset",
        target_fps: int = 20,
        task_description: str = "",
        max_frames: int = 10000,
        **kwargs,  # Accept but ignore extra parameters for backwards compatibility
    ):
        """
        Initialize the high-frequency recorder.

        Args:
            output_dir: Base directory for recordings
            dataset_name: Name for the LeRobot dataset
            target_fps: Target recording fps (default 20, matching control frequency)
            task_description: Default task description
            max_frames: Maximum frames per episode (safety limit)
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 is required for direct ROS2 recorder")

        self._output_dir = Path(output_dir)
        # Only create parent directory, let LeRobotDataset.create handle the dataset root
        # to avoid conflict with LeRobot's exist_ok=False check
        self._output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_name = dataset_name
        self._target_fps = target_fps
        self._task_description = task_description
        self._max_frames = max_frames

        # State machine
        self._state: Literal["idle", "recording", "pending_label"] = "idle"
        self._lock = threading.Lock()

        # Data buffers
        self._frame_buffer: List[TimestampedFrame] = []
        self._action_buffer: List[TimestampedAction] = []
        self._episode_count = 0
        self._recording_start_time = 0.0

        # LeRobot dataset
        self._dataset: Optional[LeRobotDataset] = None
        self._dataset_path = self._output_dir / self._dataset_name

        # ROS2 node (will be created in separate thread)
        self._ros_node: Optional[ROS2RecorderNode] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._ros_ready = threading.Event()

        # Start ROS2 recorder node
        self._start_ros2_node()

        # Keyboard listener
        self._running = True
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()

        self._print_instructions()

    def _start_ros2_node(self):
        """Start ROS2 recorder node in background thread.

        Note: Assumes ROS2 is already initialized by the environment.
        The recorder node will be added to a new executor in a separate thread.
        """
        def ros_spin():
            # Check if ROS2 is already initialized (by environment)
            try:
                if not rclpy.ok():
                    rclpy.init()
                    self._owns_rclpy = True
                else:
                    self._owns_rclpy = False
            except Exception:
                # rclpy not initialized yet
                rclpy.init()
                self._owns_rclpy = True

            self._ros_node = ROS2RecorderNode(
                target_fps=self._target_fps,
                buffer_callback=self._on_frame_received,
            )
            executor = MultiThreadedExecutor()
            executor.add_node(self._ros_node)
            self._ros_ready.set()

            try:
                executor.spin()
            finally:
                executor.shutdown()
                self._ros_node.destroy_node()
                # Only shutdown rclpy if we initialized it
                if self._owns_rclpy:
                    rclpy.shutdown()

        self._owns_rclpy = False
        self._ros_thread = threading.Thread(target=ros_spin, daemon=True)
        self._ros_thread.start()

        # Wait for node to be ready
        if not self._ros_ready.wait(timeout=10.0):
            raise RuntimeError("Failed to start ROS2 recorder node")

        logging.info(f"ROS2 recorder node started (target {self._target_fps} fps)")

    def _on_frame_received(self, frame: TimestampedFrame):
        """Callback when a new frame is sampled at 30fps."""
        with self._lock:
            if self._state != "recording":
                return

            if len(self._frame_buffer) >= self._max_frames:
                logging.warning(f"Max frames ({self._max_frames}) reached, auto-stopping")
                self._stop_recording_internal()
                return

            self._frame_buffer.append(frame)

            # Progress indicator
            frame_count = len(self._frame_buffer)
            if frame_count % 30 == 0:
                elapsed = time.time() - self._recording_start_time
                print(f"\r[RECORDER] Recording... {frame_count} frames ({elapsed:.1f}s)", end="", flush=True)

    def add_action(self, action: np.ndarray, timestamp: Optional[float] = None):
        """
        Add an action from the control loop (called at 20Hz).

        Args:
            action: 16-dim action array
            timestamp: Optional timestamp (defaults to current time)
        """
        with self._lock:
            if self._state != "recording":
                return

            ts = timestamp if timestamp is not None else time.time()
            self._action_buffer.append(TimestampedAction(timestamp=ts, action=action.copy()))

    def _print_instructions(self):
        """Print keyboard control instructions."""
        print("\n" + "=" * 60)
        print("TELEAVATAR RECORDER READY (Direct ROS2)")
        print("=" * 60)
        print("Keyboard controls:")
        print("  'r' - Start recording")
        print("  'q' - Stop recording")
        print("  's' - Save as SUCCESS trajectory")
        print("  'f' - Save as FAILURE trajectory")
        print("  'd' - Discard current recording")
        print("=" * 60)
        print(f"Output: {self._dataset_path}")
        print(f"Target FPS: {self._target_fps}")
        print(f"Action matching: nearest neighbor by timestamp")
        print(f"Video encoding: HEVC (H.265) - matching put_trash dataset")
        print(f"Dataset format: LeRobot v2.1")
        print("=" * 60 + "\n")

    # =========================================================================
    # Keyboard Listener
    # =========================================================================

    def _keyboard_listener(self):
        """Background thread for keyboard input handling."""
        if not sys.stdin.isatty():
            logging.warning("Not running in a terminal, keyboard controls disabled")
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self._handle_key(key)
        except Exception as e:
            logging.error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str):
        """Handle a single key press."""
        with self._lock:
            if key == "r" and self._state == "idle":
                self._start_recording_internal()
            elif key == "q" and self._state == "recording":
                self._stop_recording_internal()
            elif key == "s" and self._state == "pending_label":
                self._save_episode_internal(success=True)
            elif key == "f" and self._state == "pending_label":
                self._save_episode_internal(success=False)
            elif key == "d" and self._state == "pending_label":
                self._discard_episode_internal()

    # =========================================================================
    # Recording Control
    # =========================================================================

    def _start_recording_internal(self):
        """Start recording (internal, lock must be held)."""
        self._state = "recording"
        self._frame_buffer = []
        self._action_buffer = []
        self._recording_start_time = time.time()

        # Tell ROS2 node to start recording
        if self._ros_node:
            self._ros_node.start_recording()

        print(f"\n[RECORDER] Started recording episode {self._episode_count}...")

    def _stop_recording_internal(self):
        """Stop recording (internal, lock must be held)."""
        self._state = "pending_label"

        # Tell ROS2 node to stop recording
        if self._ros_node:
            self._ros_node.stop_recording()

        duration = time.time() - self._recording_start_time
        frame_count = len(self._frame_buffer)
        action_count = len(self._action_buffer)

        print(f"\n[RECORDER] Stopped. {frame_count} frames, {action_count} actions ({duration:.1f}s)")
        print(f"[RECORDER] Effective fps: {frame_count / duration:.1f}")
        print("[RECORDER] Press 's' for SUCCESS, 'f' for FAILURE, 'd' to DISCARD")

    def _save_episode_internal(self, success: bool):
        """Save episode with action matching."""
        if not self._frame_buffer:
            print("[RECORDER] No data to save")
            self._state = "idle"
            return

        label = "SUCCESS" if success else "FAILURE"
        print(f"[RECORDER] Saving as {label}...")
        print(f"[RECORDER] Matching {len(self._action_buffer)} actions to {len(self._frame_buffer)} frames...")

        try:
            # Match actions to frames by timestamp (nearest neighbor)
            matched_actions = self._match_actions_to_frames()

            if LEROBOT_AVAILABLE:
                self._save_to_lerobot(success, matched_actions)
            else:
                self._save_to_numpy(success, matched_actions)

            print(f"[RECORDER] Episode {self._episode_count} saved as {label}")
            self._episode_count += 1
        except Exception as e:
            logging.error(f"Failed to save episode: {e}")
            print(f"[RECORDER] ERROR: Failed to save - {e}")
            import traceback
            traceback.print_exc()

        self._frame_buffer = []
        self._action_buffer = []
        self._state = "idle"
        print("[RECORDER] Ready for next recording (press 'r' to start)")

    def _discard_episode_internal(self):
        """Discard episode (internal, lock must be held)."""
        frame_count = len(self._frame_buffer)
        self._frame_buffer = []
        self._action_buffer = []
        self._state = "idle"
        print(f"[RECORDER] Discarded {frame_count} frames")
        print("[RECORDER] Ready for next recording (press 'r' to start)")

    # =========================================================================
    # Action Matching
    # =========================================================================

    def _match_actions_to_frames(self) -> List[np.ndarray]:
        """
        Match actions to frames by timestamp using nearest neighbor.

        Returns:
            List of matched actions, one per frame
        """
        if not self._frame_buffer:
            return []

        if not self._action_buffer:
            # No actions recorded - return zeros
            logging.warning("No actions recorded, using zeros")
            return [np.zeros(16, dtype=np.float32) for _ in self._frame_buffer]

        # Sort actions by timestamp
        sorted_actions = sorted(self._action_buffer, key=lambda a: a.timestamp)
        action_times = np.array([a.timestamp for a in sorted_actions])
        action_values = np.array([a.action for a in sorted_actions])  # (N, 16)

        # Get frame timestamps
        frame_times = np.array([f.timestamp for f in self._frame_buffer])

        # Match each frame to nearest action by timestamp
        matched = []
        for t in frame_times:
            # Find nearest action by timestamp
            idx = np.argmin(np.abs(action_times - t))
            matched.append(action_values[idx].copy())

        return matched

    # =========================================================================
    # LeRobot Dataset Writing
    # =========================================================================

    def _ensure_dataset_initialized(self):
        """Initialize LeRobot dataset if not already done."""
        if self._dataset is not None:
            return

        # Check if dataset already exists
        # LeRobot stores datasets at: root/repo_id/meta/...
        dataset_path = self._output_dir / self._dataset_name
        meta_path = dataset_path / "meta"

        if meta_path.exists():
            # Load existing dataset for appending
            try:
                self._dataset = LeRobotDataset(
                    repo_id=self._dataset_name,
                    root=self._output_dir,
                )
                logging.info(f"Loaded existing LeRobot dataset from {dataset_path} with {self._dataset.num_episodes} episodes")
                return
            except Exception as e:
                logging.warning(f"Failed to load existing dataset, creating new one: {e}")

        features = self._setup_features()

        # Use root parameter to store dataset in local directory (not HuggingFace cache)
        # Note: use_videos=True enables video encoding during save_episode()
        # batch_encoding_size=1 ensures videos are encoded immediately (not batched)
        self._dataset = LeRobotDataset.create(
            repo_id=self._dataset_name,
            fps=self._target_fps,
            root=self._output_dir,  # Store in --record-dir location
            features=features,
            robot_type="teleavatar",  # v2.1 format uses "teleavatar" (not "teleavatar_dual_arm")
            use_videos=True,
        )
        logging.info(f"Created LeRobot dataset at {self._output_dir / self._dataset_name}")

    def _setup_features(self) -> dict:
        """Define LeRobot dataset features matching example dataset (62-dim)."""
        # 62-dim state/action names matching example dataset:
        # - Joint positions (16): left_joint1-7_position, left_gripper_position, right_joint1-7_position, right_gripper_position
        # - Joint velocities (16): same with _velocity
        # - Joint efforts (16): same with _effort
        # - End-effector poses (14): left_ee_position_x/y/z, left_ee_orientation_x/y/z/w, right_ee_...
        state_action_names = []

        # Joint positions (0-15)
        for prefix in ["left", "right"]:
            for i in range(1, 8):
                state_action_names.append(f"{prefix}_joint{i}_position")
            state_action_names.append(f"{prefix}_gripper_position")

        # Joint velocities (16-31)
        for prefix in ["left", "right"]:
            for i in range(1, 8):
                state_action_names.append(f"{prefix}_joint{i}_velocity")
            state_action_names.append(f"{prefix}_gripper_velocity")

        # Joint efforts (32-47)
        for prefix in ["left", "right"]:
            for i in range(1, 8):
                state_action_names.append(f"{prefix}_joint{i}_effort")
            state_action_names.append(f"{prefix}_gripper_effort")

        # End-effector poses (48-61)
        for prefix in ["left", "right"]:
            state_action_names.append(f"{prefix}_ee_position_x")
            state_action_names.append(f"{prefix}_ee_position_y")
            state_action_names.append(f"{prefix}_ee_position_z")
            state_action_names.append(f"{prefix}_ee_orientation_x")
            state_action_names.append(f"{prefix}_ee_orientation_y")
            state_action_names.append(f"{prefix}_ee_orientation_z")
            state_action_names.append(f"{prefix}_ee_orientation_w")

        features = {
            "action": {
                "dtype": "float32",
                "shape": (62,),
                "names": state_action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (62,),
                "names": state_action_names,
            },
            "success": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            # 4 cameras matching put_trash dataset format:
            # - head_camera: 头部双目 (2160x4320)
            # - chest_camera: 胸部 (480x848)
            # - left_color: 左腕 (480x848)
            # - right_color: 右腕 (480x848)
            # Video codec: hevc (H.265) to match put_trash dataset
            "observation.images.head_camera": {
                "dtype": "video",
                "shape": (2160, 4320, 3),  # High-res head stereo camera
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 2160,
                    "video.width": 4320,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self._target_fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.chest_camera": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 848,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self._target_fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.left_color": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 848,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self._target_fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.right_color": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 848,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self._target_fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
        }
        return features

    def _save_to_lerobot(self, success: bool, interpolated_actions: List[np.ndarray]):
        """Save buffered data to LeRobot dataset with HEVC encoding (matching put_trash format)."""
        self._ensure_dataset_initialized()

        # Create episode buffer
        self._dataset.episode_buffer = self._dataset.create_episode_buffer()

        total_frames = len(self._frame_buffer)

        logging.info(f"Adding {total_frames} frames to dataset...")
        add_start = time.time()

        for i, (frame, action) in enumerate(zip(self._frame_buffer, interpolated_actions)):
            is_last = i == len(self._frame_buffer) - 1

            # Expand action to 62-dim to match example dataset format
            # Input action is 16-dim (joint positions), expand using state for remaining dims
            action_62d = np.zeros(62, dtype=np.float32)
            if len(action) >= 16:
                action_62d[0:16] = action[:16]  # Joint positions from action
            else:
                action_62d[0:len(action)] = action
            # Copy velocity, effort, and EE pose from state (indices 16-61)
            action_62d[16:62] = frame.state[16:62]

            frame_data = {
                "action": action_62d,
                "observation.state": frame.state.astype(np.float32),
                "success": np.array([success and is_last], dtype=bool),
                "next.done": np.array([is_last], dtype=bool),
                "observation.images.head_camera": frame.images['head_camera'],
                "observation.images.chest_camera": frame.images['chest_camera'],
                "observation.images.left_color": frame.images['left_color'],
                "observation.images.right_color": frame.images['right_color'],
                "task": self._task_description,
            }

            self._dataset.add_frame(frame_data)

            # Progress indicator for long episodes
            if (i + 1) % 100 == 0:
                print(f"\r[RECORDER] Adding frames... {i + 1}/{total_frames}", end="", flush=True)

        add_elapsed = time.time() - add_start
        logging.info(f"Added {total_frames} frames in {add_elapsed:.1f}s ({total_frames/add_elapsed:.1f} fps)")

        # Save episode - this writes parquet data and encodes images to videos
        logging.info("Saving episode (encoding videos with HEVC)...")
        save_start = time.time()

        # Save episode data (parquet) first, but skip default video encoding
        # We'll encode videos ourselves with HEVC codec
        self._dataset.save_episode()

        # Get episode index
        episode_idx = self._dataset.num_episodes - 1

        # Re-encode videos with HEVC codec to match put_trash dataset
        if VIDEO_ENCODING_AVAILABLE:
            self._reencode_episode_videos_hevc(episode_idx)

        save_elapsed = time.time() - save_start
        logging.info(f"Episode saved in {save_elapsed:.1f}s")

        # Update episodes.jsonl with success/failure metadata
        self._update_episode_metadata(episode_idx, total_frames, success)

        # Verify video files were created
        videos_dir = self._output_dir / self._dataset_name / "videos"
        if videos_dir.exists():
            video_files = list(videos_dir.rglob(f"*episode_{episode_idx:06d}.mp4"))
            logging.info(f"Video files created: {len(video_files)}")
            for vf in video_files:
                logging.info(f"  - {vf.relative_to(self._output_dir)}")
        else:
            logging.warning(f"Videos directory not found: {videos_dir}")

    def _reencode_episode_videos_hevc(self, episode_idx: int):
        """Re-encode episode videos with HEVC codec to match put_trash dataset format."""
        import shutil

        dataset_path = self._output_dir / self._dataset_name

        # Video keys to re-encode
        video_keys = [
            "observation.images.head_camera",
            "observation.images.chest_camera",
            "observation.images.left_color",
            "observation.images.right_color",
        ]

        for video_key in video_keys:
            # Find the video file (v2.1 format: videos/chunk-000/{video_key}/episode_{idx}.mp4)
            video_path = dataset_path / "videos" / "chunk-000" / video_key / f"episode_{episode_idx:06d}.mp4"

            if not video_path.exists():
                # Try alternative path pattern (v3.0 format)
                video_path_alt = dataset_path / "videos" / video_key / "chunk-000" / f"file-{episode_idx:03d}.mp4"
                if video_path_alt.exists():
                    video_path = video_path_alt
                else:
                    logging.warning(f"Video not found for re-encoding: {video_key}")
                    continue

            # Get corresponding images directory (v2.1 format: images/{image_key}/episode_{idx}/)
            img_dir = dataset_path / "images" / video_key / f"episode_{episode_idx:06d}"
            if not img_dir.exists():
                # Try alternative path patterns
                alt_paths = [
                    dataset_path / "images" / video_key / "chunk-000",
                    dataset_path / "images" / video_key,
                ]
                img_dir = None
                for alt_path in alt_paths:
                    if alt_path.exists() and list(alt_path.glob("frame_*.png")):
                        img_dir = alt_path
                        break
                if img_dir is None:
                    logging.warning(f"Images directory not found for {video_key}, skipping HEVC re-encoding")
                    continue

            # Create temp file for new video
            temp_video = video_path.with_suffix('.hevc.mp4')

            try:
                logging.info(f"Re-encoding {video_key} with HEVC codec...")
                encode_video_frames(
                    imgs_dir=img_dir,
                    video_path=temp_video,
                    fps=self._target_fps,
                    vcodec="hevc",  # Use HEVC (H.265) codec
                    pix_fmt="yuv420p",
                    crf=30,
                    overwrite=True,
                )

                # Replace original with HEVC version
                shutil.move(str(temp_video), str(video_path))
                logging.info(f"  Re-encoded: {video_path.name}")

            except Exception as e:
                logging.error(f"Failed to re-encode {video_key}: {e}")
                if temp_video.exists():
                    temp_video.unlink()

    def _update_episode_metadata(self, episode_idx: int, length: int, success: bool):
        """Update episodes.jsonl to include success/failure metadata (v2.1 format)."""
        import json

        dataset_path = self._output_dir / self._dataset_name
        episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"

        if not episodes_jsonl.exists():
            logging.warning(f"episodes.jsonl not found at {episodes_jsonl}")
            return

        # Read existing episodes
        episodes = []
        with open(episodes_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

        # Find and update the target episode, or append if new
        episode_found = False
        for ep in episodes:
            if ep.get('episode_index') == episode_idx:
                ep['success'] = success
                ep['length'] = length
                if 'tasks' not in ep:
                    ep['tasks'] = [self._task_description] if self._task_description else []
                episode_found = True
                break

        if not episode_found:
            # Append new episode entry
            episodes.append({
                'episode_index': episode_idx,
                'tasks': [self._task_description] if self._task_description else [],
                'length': length,
                'success': success,
            })

        # Write back to episodes.jsonl
        with open(episodes_jsonl, 'w') as f:
            for ep in episodes:
                f.write(json.dumps(ep) + '\n')

        logging.info(f"Updated episodes.jsonl: episode {episode_idx} marked as {'SUCCESS' if success else 'FAILURE'}")

    def _save_to_numpy(self, success: bool, interpolated_actions: List[np.ndarray]):
        """Save to numpy format (fallback)."""
        episode_dir = self._output_dir / f"episode_{self._episode_count:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        import json

        meta = {
            "episode_id": self._episode_count,
            "success": success,
            "num_frames": len(self._frame_buffer),
            "fps": self._target_fps,
            "task_description": self._task_description,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "original_action_count": len(self._action_buffer),
            "interpolated": True,
        }
        with open(episode_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save data arrays
        states = np.array([f.state for f in self._frame_buffer])
        actions = np.array(interpolated_actions)
        timestamps = np.array([f.timestamp for f in self._frame_buffer])

        np.save(episode_dir / "states.npy", states)
        np.save(episode_dir / "actions.npy", actions)
        np.save(episode_dir / "timestamps.npy", timestamps)

        # Save images - 4 cameras
        images_dir = episode_dir / "images"
        for cam_key in ["head_camera", "chest_camera", "left_color", "right_color"]:
            cam_dir = images_dir / cam_key
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(self._frame_buffer):
                img = frame.images.get(cam_key)
                if img is not None:
                    np.save(cam_dir / f"frame_{i:05d}.npy", img)

        logging.info(f"Saved episode to {episode_dir}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def stop(self):
        """Stop the recorder and cleanup."""
        self._running = False
        if self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=1.0)

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()


# =========================================================================
# Standalone test
# =========================================================================

if __name__ == "__main__":
    print("Testing TeleavatarRecorder30fps...")
    print("This test requires ROS2 topics to be publishing.")
    print("Press Ctrl+C to exit.\n")

    try:
        recorder = TeleavatarRecorder30fps(
            output_dir="test_recordings_30fps",
            dataset_name="test_dataset_30fps",
            target_fps=30,
            task_description="Test task",
        )

        # Simulate actions from control loop at 20Hz
        def simulate_actions():
            while True:
                if recorder._state == "recording":
                    action = np.random.randn(16).astype(np.float32)
                    recorder.add_action(action)
                time.sleep(0.05)  # 20Hz

        action_thread = threading.Thread(target=simulate_actions, daemon=True)
        action_thread.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
