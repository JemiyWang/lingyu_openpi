"""Compute SED state/action normalization without decoding the video streams.

The SED adapter leaves the 16-D state and absolute 16-D actions unchanged
before normalization, so reading the two parquet columns is equivalent to the
generic OpenPI statistics pass while avoiding an unnecessary image decode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

import openpi.shared.normalize as normalize

from sed_robot_config import TASK_PROMPTS, DATA_ROOT, stats_path


def _fixed_list_array(column) -> np.ndarray:
    column = column.combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(column), -1)


def compute(task: str, max_frames: int | None = None, force: bool = False) -> Path:
    output = stats_path(task)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    task_root = DATA_ROOT / task
    parquet_files = sorted((task_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found below {task_root / 'data'}")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    seen = 0
    for parquet_path in parquet_files:
        table = parquet.read_table(parquet_path, columns=["observation.state", "action"])
        state = _fixed_list_array(table["observation.state"])
        actions = _fixed_list_array(table["action"])
        if max_frames is not None:
            remaining = max_frames - seen
            if remaining <= 0:
                break
            state = state[:remaining]
            actions = actions[:remaining]
        stats["state"].update(state)
        stats["actions"].update(actions)
        seen += len(state)
        if max_frames is not None and seen >= max_frames:
            break

    if seen < 2:
        raise ValueError(f"Only {seen} frames available; at least two are required")
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    normalize.save(output.parent, norm_stats)
    print(f"Wrote {output} ({seen} frames, task={task})")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASK_PROMPTS))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    compute(args.task, max_frames=args.max_frames, force=args.force)


if __name__ == "__main__":
    main()
