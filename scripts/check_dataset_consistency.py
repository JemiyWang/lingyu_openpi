#!/usr/bin/env python3
"""
检查数据集中每个episode的parquet帧数与视频帧数是否一致。
不一致的episode将被移动到discarded文件夹。

Usage:
    python scripts/check_dataset_consistency.py --dataset-path /path/to/dataset [--dry-run]
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


def get_video_frame_count(video_path: str) -> int | None:
    """使用ffprobe获取视频帧数"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError) as e:
        print(f"  Warning: Failed to get frame count for {video_path}: {e}")
    return None


def get_parquet_row_count(parquet_path: str) -> int | None:
    """获取parquet文件的行数"""
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"  Warning: Failed to read parquet {parquet_path}: {e}")
    return None


def check_episode_consistency(
    dataset_path: Path,
    episode_index: int,
    chunk_index: int,
    video_keys: list[str],
) -> tuple[bool, dict]:
    """
    检查单个episode的一致性。

    Returns:
        (is_consistent, details_dict)
    """
    episode_name = f"episode_{episode_index:06d}"
    chunk_name = f"chunk-{chunk_index:03d}"

    # 获取parquet行数
    parquet_path = dataset_path / "data" / chunk_name / f"{episode_name}.parquet"
    parquet_rows = get_parquet_row_count(str(parquet_path))

    if parquet_rows is None:
        return False, {"error": f"Cannot read parquet: {parquet_path}"}

    details = {
        "parquet_rows": parquet_rows,
        "video_frames": {},
        "mismatches": [],
    }

    # 检查每个视频的帧数
    is_consistent = True
    for video_key in video_keys:
        video_path = dataset_path / "videos" / chunk_name / video_key / f"{episode_name}.mp4"
        if not video_path.exists():
            details["video_frames"][video_key] = None
            details["mismatches"].append(f"{video_key}: video not found")
            is_consistent = False
            continue

        frame_count = get_video_frame_count(str(video_path))
        details["video_frames"][video_key] = frame_count

        if frame_count is None:
            details["mismatches"].append(f"{video_key}: cannot read frame count")
            is_consistent = False
        elif frame_count != parquet_rows:
            details["mismatches"].append(
                f"{video_key}: {frame_count} frames vs {parquet_rows} parquet rows"
            )
            is_consistent = False

    return is_consistent, details


def discard_episode(
    dataset_path: Path,
    discarded_path: Path,
    episode_index: int,
    chunk_index: int,
    video_keys: list[str],
    dry_run: bool = False,
) -> None:
    """将不一致的episode移动到discarded文件夹"""
    episode_name = f"episode_{episode_index:06d}"
    chunk_name = f"chunk-{chunk_index:03d}"

    # 移动parquet文件
    parquet_src = dataset_path / "data" / chunk_name / f"{episode_name}.parquet"
    parquet_dst = discarded_path / "data" / chunk_name / f"{episode_name}.parquet"

    if parquet_src.exists():
        if dry_run:
            print(f"  [DRY-RUN] Would move: {parquet_src} -> {parquet_dst}")
        else:
            parquet_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(parquet_src), str(parquet_dst))
            print(f"  Moved: {parquet_src} -> {parquet_dst}")

    # 移动视频文件
    for video_key in video_keys:
        video_src = dataset_path / "videos" / chunk_name / video_key / f"{episode_name}.mp4"
        video_dst = discarded_path / "videos" / chunk_name / video_key / f"{episode_name}.mp4"

        if video_src.exists():
            if dry_run:
                print(f"  [DRY-RUN] Would move: {video_src} -> {video_dst}")
            else:
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(video_src), str(video_dst))
                print(f"  Moved: {video_src} -> {video_dst}")


def main():
    parser = argparse.ArgumentParser(description="检查数据集episode一致性")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/mnt/data/lingyu/rollout_build_blocks/rollout_build_blocks",
        help="数据集路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查不实际移动文件",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    discarded_path = dataset_path.parent / f"{dataset_path.name}_discarded"

    # 读取数据集信息
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]

    # 获取视频keys
    video_keys = [
        key for key, val in info["features"].items()
        if isinstance(val, dict) and val.get("dtype") == "video"
    ]
    print(f"Dataset: {dataset_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Video keys: {video_keys}")
    print(f"Discarded path: {discarded_path}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)

    # 检查每个episode
    consistent_count = 0
    inconsistent_count = 0
    inconsistent_episodes = []

    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size

        is_consistent, details = check_episode_consistency(
            dataset_path, episode_index, chunk_index, video_keys
        )

        if is_consistent:
            consistent_count += 1
        else:
            inconsistent_count += 1
            inconsistent_episodes.append((episode_index, details))
            print(f"Episode {episode_index:06d}: INCONSISTENT")
            if "error" in details:
                print(f"  Error: {details['error']}")
            else:
                print(f"  Parquet rows: {details['parquet_rows']}")
                for mismatch in details["mismatches"]:
                    print(f"  - {mismatch}")

    print("-" * 60)
    print(f"Consistent episodes: {consistent_count}")
    print(f"Inconsistent episodes: {inconsistent_count}")

    # 丢弃不一致的episode
    if inconsistent_episodes:
        print("-" * 60)
        print("Discarding inconsistent episodes...")
        for episode_index, details in inconsistent_episodes:
            chunk_index = episode_index // chunks_size
            discard_episode(
                dataset_path,
                discarded_path,
                episode_index,
                chunk_index,
                video_keys,
                dry_run=args.dry_run,
            )

        if not args.dry_run:
            # 更新meta信息
            print("-" * 60)
            print("Note: meta files (episodes.jsonl, info.json) need manual update")
            print("after discarding episodes to reflect the new episode count and indices.")

    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
