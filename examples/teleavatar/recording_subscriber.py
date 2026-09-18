import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from openpi_client.runtime import subscriber as _subscriber


class TeleavatarRecordingSubscriber(_subscriber.Subscriber):
    """Records raw observations and actions for Teleavatar with CLI control."""

    def __init__(self, base_dir: str = "teleavatar_records", prompt: str | None = None) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._prompt = prompt

        self._recording = False
        self._current_dir: Optional[Path] = None
        self._step_idx = 0
        self._label: Optional[str] = None
        self._lock = threading.Lock()

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _input_loop(self) -> None:
        """Simple stdin loop for start/stop/label commands."""
        instructions = (
            "Recorder ready. Commands: start | stop | success | fail | status"
        )
        print(instructions)
        for line in sys.stdin:
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd == "start":
                self._start_recording()
            elif cmd == "stop":
                self._stop_recording()
            elif cmd in {"success", "fail"}:
                self._set_label_and_stop(cmd)
            elif cmd == "status":
                self._print_status()
            else:
                print(f"Unknown command: {cmd}")

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording:
                print("Recording already active")
                return
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._current_dir = self._base_dir / f"traj_{ts}"
            self._current_dir.mkdir(parents=True, exist_ok=True)
            meta = {"start_time": ts, "prompt": self._prompt}
            (self._current_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            self._step_idx = 0
            self._label = None
            self._recording = True
            print(f"Started recording -> {self._current_dir}")

    def _stop_recording(self) -> None:
        with self._lock:
            if not self._recording:
                print("Recording not active")
                return
            self._recording = False
            self._write_label()
            print("Stopped recording")

    def _set_label_and_stop(self, label: str) -> None:
        with self._lock:
            if not self._recording:
                print("Recording not active")
                return
            self._label = label
            self._recording = False
            self._write_label()
            print(f"Stopped recording with label={label}")

    def _write_label(self) -> None:
        if self._current_dir is None:
            return
        meta_path = self._current_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        if self._label:
            meta["label"] = self._label
        meta_path.write_text(json.dumps(meta, indent=2))

    def _print_status(self) -> None:
        with self._lock:
            state = "on" if self._recording else "off"
            print(f"Recording {state}; last dir={self._current_dir}")

    def on_episode_start(self) -> None:  # pragma: no cover - interface hook
        return

    def on_step(self, observation: dict, action: dict) -> None:  # pragma: no cover - interface hook
        with self._lock:
            if not self._recording or self._current_dir is None:
                return
            path = self._current_dir / f"step_{self._step_idx:05d}.npy"
            self._step_idx += 1
        # Save outside lock to minimize blocking
        payload = {"observation": observation, "action": action}
        np.save(path, payload, allow_pickle=True)

    def on_episode_end(self) -> None:  # pragma: no cover - interface hook
        with self._lock:
            if self._recording:
                self._recording = False
                self._write_label()
                print("Episode ended; recording stopped")
