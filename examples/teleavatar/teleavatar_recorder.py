#!/usr/bin/env python3
"""
Teleavatar Data Recorder - Records robot data during policy inference.

Features:
- Keyboard-controlled recording start/stop
- Success/failure trajectory labeling
- LeRobot-compatible dataset output format

Usage:
    Integrated with main.py via --record flag.

    Keyboard controls:
    - 'r': Start recording
    - 'q': Stop recording
    - 's': Save as success trajectory
    - 'f': Save as failure trajectory
    - 'd': Discard current recording
"""

import logging
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Literal

import numpy as np
from openpi_client.runtime import subscriber as _subscriber

# LeRobot imports (try both new and old paths for compatibility)
LEROBOT_AVAILABLE = False
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LEROBOT_AVAILABLE = True
    except ImportError:
        logging.warning("LeRobot not available. Recording will use numpy format.")


class TeleavatarRecorder(_subscriber.Subscriber):
    """
    Teleavatar data recorder implementing the Subscriber interface.

    Records raw observations (images, state) and actions during policy inference,
    with keyboard-controlled start/stop and success/failure labeling.
    """

    def __init__(
        self,
        output_dir: str = "teleavatar_recordings",
        dataset_name: str = "teleavatar_dataset",
        fps: int = 20,
        task_description: str = "",
        max_frames: int = 10000,
    ):
        """
        Initialize the recorder.

        Args:
            output_dir: Base directory for recordings
            dataset_name: Name for the LeRobot dataset
            fps: Recording frame rate (should match control frequency)
            task_description: Default task description for recordings
            max_frames: Maximum frames per episode (safety limit)
        """
        self._output_dir = Path(output_dir)
        # Only create parent directory, let LeRobotDataset.create handle the dataset root
        # to avoid conflict with LeRobot's exist_ok=False check
        self._output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_name = dataset_name
        self._fps = fps
        self._task_description = task_description
        self._max_frames = max_frames

        # State machine: idle -> recording -> pending_label -> idle
        self._state: Literal["idle", "recording", "pending_label"] = "idle"
        self._lock = threading.Lock()

        # Episode buffer
        self._current_episode_buffer: list[dict] = []
        self._episode_count: int = 0
        self._recording_start_time: float = 0.0

        # LeRobot dataset (lazy initialization)
        self._dataset: LeRobotDataset | None = None
        self._dataset_path = self._output_dir / self._dataset_name

        # Keyboard listener
        self._running = True
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()

        # Print instructions
        self._print_instructions()

    def _print_instructions(self) -> None:
        """Print keyboard control instructions."""
        print("\n" + "=" * 60)
        print("TELEAVATAR RECORDER READY")
        print("=" * 60)
        print("Keyboard controls:")
        print("  'r' - Start recording")
        print("  'q' - Stop recording")
        print("  's' - Save as SUCCESS trajectory")
        print("  'f' - Save as FAILURE trajectory")
        print("  'd' - Discard current recording")
        print("=" * 60)
        print(f"Output: {self._dataset_path}")
        print(f"FPS: {self._fps}, Max frames: {self._max_frames}")
        print("=" * 60 + "\n")

    # =========================================================================
    # Subscriber Interface Implementation
    # =========================================================================

    def on_episode_start(self) -> None:
        """Called when a runtime episode starts."""
        # Don't auto-start recording; wait for keyboard trigger

    def on_step(self, observation: dict, action: dict) -> None:
        """Called each step with observation and action data."""
        with self._lock:
            if self._state != "recording":
                return

            # Check frame limit
            if len(self._current_episode_buffer) >= self._max_frames:
                logging.warning(f"Max frames ({self._max_frames}) reached, auto-stopping")
                self._stop_recording_internal()
                return

            self._buffer_frame(observation, action)

    def on_episode_end(self) -> None:
        """Called when a runtime episode ends."""
        with self._lock:
            if self._state == "recording":
                logging.info("Episode ended while recording, stopping...")
                self._stop_recording_internal()

    # =========================================================================
    # Keyboard Listener
    # =========================================================================

    def _keyboard_listener(self) -> None:
        """Background thread for keyboard input handling."""
        # Check if running in a terminal
        if not sys.stdin.isatty():
            logging.warning("Not running in a terminal, keyboard controls disabled")
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                # Non-blocking read with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self._handle_key(key)
        except Exception as e:
            logging.error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> None:
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
    # Recording Control (Internal, must hold lock)
    # =========================================================================

    def _start_recording_internal(self) -> None:
        """Start recording (internal, lock must be held)."""
        self._state = "recording"
        self._current_episode_buffer = []
        self._recording_start_time = time.time()
        print(f"\n[RECORDER] Started recording episode {self._episode_count}...")

    def _stop_recording_internal(self) -> None:
        """Stop recording (internal, lock must be held)."""
        self._state = "pending_label"
        duration = time.time() - self._recording_start_time
        frame_count = len(self._current_episode_buffer)
        print(f"\n[RECORDER] Stopped. {frame_count} frames recorded ({duration:.1f}s)")
        print("[RECORDER] Press 's' for SUCCESS, 'f' for FAILURE, 'd' to DISCARD")

    def _save_episode_internal(self, success: bool) -> None:
        """Save episode (internal, lock must be held)."""
        if not self._current_episode_buffer:
            print("[RECORDER] No data to save")
            self._state = "idle"
            return

        label = "SUCCESS" if success else "FAILURE"
        print(f"[RECORDER] Saving as {label}...")

        try:
            if LEROBOT_AVAILABLE:
                self._save_to_lerobot(success)
            else:
                self._save_to_numpy(success)

            print(f"[RECORDER] Episode {self._episode_count} saved as {label}")
            self._episode_count += 1
        except Exception as e:
            logging.error(f"Failed to save episode: {e}")
            print(f"[RECORDER] ERROR: Failed to save - {e}")

        self._current_episode_buffer = []
        self._state = "idle"
        print("[RECORDER] Ready for next recording (press 'r' to start)")

    def _discard_episode_internal(self) -> None:
        """Discard episode (internal, lock must be held)."""
        frame_count = len(self._current_episode_buffer)
        self._current_episode_buffer = []
        self._state = "idle"
        print(f"[RECORDER] Discarded {frame_count} frames")
        print("[RECORDER] Ready for next recording (press 'r' to start)")

    # =========================================================================
    # Data Buffering
    # =========================================================================

    def _buffer_frame(self, observation: dict, action: dict) -> None:
        """Buffer a single frame of data."""
        frame = {
            "timestamp": time.time(),
            "observation": {
                "state": self._safe_copy(observation.get("observation/state")),
                "images": {
                    # 4 cameras: head_camera (头部双目), chest_camera (胸部), left_color (左腕), right_color (右腕)
                    "head_camera": self._safe_copy(observation.get("observation/images/head_camera")),
                    "chest_camera": self._safe_copy(observation.get("observation/images/chest_camera")),
                    "left_color": self._safe_copy(observation.get("observation/images/left_color")),
                    "right_color": self._safe_copy(observation.get("observation/images/right_color")),
                },
            },
            "action": self._safe_copy(action.get("actions")),
            "prompt": observation.get("prompt", self._task_description),
        }
        self._current_episode_buffer.append(frame)

        # Progress indicator
        frame_count = len(self._current_episode_buffer)
        if frame_count % 20 == 0:
            elapsed = time.time() - self._recording_start_time
            print(f"\r[RECORDER] Recording... {frame_count} frames ({elapsed:.1f}s)", end="", flush=True)

    def _safe_copy(self, data):
        """Safely copy data (handles numpy arrays and None)."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data.copy()
        return data

    # =========================================================================
    # LeRobot Dataset Writing
    # =========================================================================

    def _ensure_dataset_initialized(self) -> None:
        """Initialize LeRobot dataset if not already done."""
        if self._dataset is not None:
            return

        # Check if dataset already exists
        meta_path = self._dataset_path / "meta"

        if meta_path.exists():
            # Load existing dataset for appending
            try:
                self._dataset = LeRobotDataset(
                    repo_id=str(self._dataset_path),
                )
                logging.info(f"Loaded existing LeRobot dataset from {self._dataset_path} with {self._dataset.num_episodes} episodes")
                return
            except Exception as e:
                logging.warning(f"Failed to load existing dataset, creating new one: {e}")

        features = self._setup_features()

        self._dataset = LeRobotDataset.create(
            repo_id=str(self._dataset_path),
            fps=self._fps,
            features=features,
            robot_type="teleavatar_dual_arm",
            use_videos=True,
        )
        logging.info(f"Created LeRobot dataset at {self._dataset_path}")

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
            # Action (62-dim matching example dataset)
            "action": {
                "dtype": "float32",
                "shape": (62,),
                "names": state_action_names,
            },
            # State (62-dim matching example dataset)
            "observation.state": {
                "dtype": "float32",
                "shape": (62,),
                "names": state_action_names,
            },
            # Success flag
            "success": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            # Episode done flag
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
            # 4 cameras matching example dataset:
            # - head_camera: 头部双目 (2160x4320)
            # - chest_camera: 胸部 (480x848)
            # - left_color: 左腕 (480x848)
            # - right_color: 右腕 (480x848)
            "observation.images.head_camera": {
                "dtype": "video",
                "shape": (2160, 4320, 3),  # High-res head stereo camera
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": float(self._fps),
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.chest_camera": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": float(self._fps),
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.left_color": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": float(self._fps),
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.right_color": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": float(self._fps),
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        }
        return features

    def _save_to_lerobot(self, success: bool) -> None:
        """Save buffered data to LeRobot dataset."""
        self._ensure_dataset_initialized()

        # Create episode buffer
        self._dataset.episode_buffer = self._dataset.create_episode_buffer(self._episode_count)

        # Get task description from first frame
        task_desc = self._current_episode_buffer[0].get("prompt", self._task_description)
        start_time = self._current_episode_buffer[0]["timestamp"]

        for i, frame in enumerate(self._current_episode_buffer):
            is_last = i == len(self._current_episode_buffer) - 1

            # Get state (62-dim)
            state_62d = self._ensure_array(frame["observation"]["state"], (62,), np.float32)

            # Expand action to 62-dim to match example dataset format
            # Input action may be 16-dim (joint positions), expand using state for remaining dims
            raw_action = frame["action"]
            action_62d = np.zeros(62, dtype=np.float32)
            if raw_action is not None:
                raw_action = np.asarray(raw_action, dtype=np.float32)
                if len(raw_action) >= 62:
                    action_62d = raw_action[:62]
                elif len(raw_action) >= 16:
                    action_62d[0:16] = raw_action[:16]  # Joint positions from action
                    action_62d[16:62] = state_62d[16:62]  # Copy rest from state
                else:
                    action_62d[0:len(raw_action)] = raw_action
                    action_62d[16:62] = state_62d[16:62]
            else:
                action_62d = state_62d.copy()  # Use state as action if no action provided

            # Prepare frame data
            frame_data = {
                "action": action_62d,
                "observation.state": state_62d,
                "success": np.array([success and is_last], dtype=bool),
                "next.done": np.array([is_last], dtype=bool),
            }

            # Add images - 4 cameras with different resolutions
            camera_shapes = {
                "head_camera": (2160, 4320, 3),  # High-res head stereo
                "chest_camera": (480, 848, 3),
                "left_color": (480, 848, 3),
                "right_color": (480, 848, 3),
            }
            for cam_key, shape in camera_shapes.items():
                img = frame["observation"]["images"].get(cam_key)
                if img is not None:
                    frame_data[f"observation.images.{cam_key}"] = img
                else:
                    # Create black frame as fallback
                    frame_data[f"observation.images.{cam_key}"] = np.zeros(shape, dtype=np.uint8)

            # Add task to frame_data (new LeRobot API)
            frame_data["task"] = task_desc
            self._dataset.add_frame(frame_data)

        # Save episode
        self._dataset.save_episode()

    def _ensure_array(self, data, shape: tuple, dtype) -> np.ndarray:
        """Ensure data is a numpy array with correct shape and dtype."""
        if data is None:
            return np.zeros(shape, dtype=dtype)
        arr = np.asarray(data, dtype=dtype)
        if arr.shape != shape:
            result = np.zeros(shape, dtype=dtype)
            min_len = min(arr.size, result.size)
            result.flat[:min_len] = arr.flat[:min_len]
            return result
        return arr

    # =========================================================================
    # Numpy Fallback (when LeRobot not available)
    # =========================================================================

    def _save_to_numpy(self, success: bool) -> None:
        """Save buffered data to numpy format (fallback)."""
        episode_dir = self._output_dir / f"episode_{self._episode_count:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        import json

        meta = {
            "episode_id": self._episode_count,
            "success": success,
            "num_frames": len(self._current_episode_buffer),
            "fps": self._fps,
            "task_description": self._current_episode_buffer[0].get("prompt", self._task_description),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        with open(episode_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save data
        states = []
        actions = []
        timestamps = []

        for frame in self._current_episode_buffer:
            states.append(frame["observation"]["state"])
            actions.append(frame["action"])
            timestamps.append(frame["timestamp"])

        np.save(episode_dir / "states.npy", np.array(states))
        np.save(episode_dir / "actions.npy", np.array(actions))
        np.save(episode_dir / "timestamps.npy", np.array(timestamps))

        # Save images as separate files
        images_dir = episode_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(self._current_episode_buffer):
            # 4 cameras
            for cam_key in ["head_camera", "chest_camera", "left_color", "right_color"]:
                img = frame["observation"]["images"].get(cam_key)
                if img is not None:
                    cam_dir = images_dir / cam_key
                    cam_dir.mkdir(exist_ok=True)
                    np.save(cam_dir / f"frame_{i:05d}.npy", img)

        logging.info(f"Saved episode to {episode_dir}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def stop(self) -> None:
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
    # Simple test without actual robot
    import time

    print("Testing TeleavatarRecorder...")
    recorder = TeleavatarRecorder(
        output_dir="test_recordings",
        dataset_name="test_dataset",
        fps=20,
        task_description="Test task",
    )

    # Simulate some steps
    print("\nSimulating data...")
    for _ in range(100):
        obs = {
            "observation/state": np.random.randn(62).astype(np.float32),  # 62-dim state
            # 4 cameras: head_camera (2160x4320), chest_camera, left_color, right_color (480x848)
            "observation/images/head_camera": np.random.randint(0, 255, (2160, 4320, 3), dtype=np.uint8),
            "observation/images/chest_camera": np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8),
            "observation/images/left_color": np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8),
            "observation/images/right_color": np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8),
            "prompt": "Test task",
        }
        action = {
            "actions": np.random.randn(16).astype(np.float32),
        }
        recorder.on_step(obs, action)
        time.sleep(0.05)

    print("\nTest complete. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
        print("\nDone.")
