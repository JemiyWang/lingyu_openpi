#!/usr/bin/env python3
"""
重新索引数据集：重命名文件并更新元数据。
在丢弃不一致的episode后运行此脚本。

Usage:
    python scripts/reindex_dataset.py --dataset-path /path/to/dataset [--dry-run]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def get_existing_episodes(dataset_path: Path) -> list[int]:
    """获取现有的episode索引列表"""
    data_dir = dataset_path / "data" / "chunk-000"
    episodes = []
    for f in data_dir.glob("episode_*.parquet"):
        idx = int(f.stem.split("_")[1])
        episodes.append(idx)
    return sorted(episodes)


def rename_file(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """重命名文件"""
    if not src.exists():
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would rename: {src.name} -> {dst.name}")
    else:
        shutil.move(str(src), str(dst))
        print(f"  Renamed: {src.name} -> {dst.name}")
    return True


def update_parquet_episode_index(parquet_path: Path, new_index: int, dry_run: bool = False):
    """更新parquet文件中的episode_index列"""
    if dry_run:
        return

    import pyarrow as pa

    # 读取parquet文件
    table = pq.read_table(parquet_path)

    # 检查是否有episode_index列
    if "episode_index" in table.column_names:
        # 创建新的episode_index列
        num_rows = table.num_rows
        new_episode_index = pa.array([new_index] * num_rows, type=pa.int64())

        # 替换列
        col_idx = table.column_names.index("episode_index")
        table = table.set_column(col_idx, "episode_index", new_episode_index)

        # 写回文件
        pq.write_table(table, parquet_path)


def main():
    parser = argparse.ArgumentParser(description="重新索引数据集")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/mnt/data/lingyu/rollout_build_blocks/rollout_build_blocks",
        help="数据集路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要进行的操作，不实际执行",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    # 读取数据集信息
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    # 获取视频keys
    video_keys = [
        key for key, val in info["features"].items()
        if isinstance(val, dict) and val.get("dtype") == "video"
    ]

    # 获取现有的episode索引
    existing_episodes = get_existing_episodes(dataset_path)
    new_total_episodes = len(existing_episodes)

    print(f"Dataset: {dataset_path}")
    print(f"Existing episodes: {len(existing_episodes)} (indices {existing_episodes[0]} to {existing_episodes[-1]})")
    print(f"Video keys: {video_keys}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)

    # 创建旧索引到新索引的映射
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(existing_episodes)}

    # 读取原始的episodes.jsonl和episodes_stats.jsonl
    episodes_jsonl_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_stats_jsonl_path = dataset_path / "meta" / "episodes_stats.jsonl"

    episodes_data = {}
    with open(episodes_jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            episodes_data[ep["episode_index"]] = ep

    episodes_stats_data = {}
    with open(episodes_stats_jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            episodes_stats_data[ep["episode_index"]] = ep

    # 第一步：重命名文件（使用临时名称避免冲突）
    print("Step 1: Renaming files to temporary names...")
    chunk_name = "chunk-000"

    for old_idx in existing_episodes:
        old_name = f"episode_{old_idx:06d}"
        tmp_name = f"episode_tmp_{old_idx:06d}"

        # 重命名parquet文件
        old_parquet = dataset_path / "data" / chunk_name / f"{old_name}.parquet"
        tmp_parquet = dataset_path / "data" / chunk_name / f"{tmp_name}.parquet"
        rename_file(old_parquet, tmp_parquet, args.dry_run)

        # 重命名视频文件
        for video_key in video_keys:
            old_video = dataset_path / "videos" / chunk_name / video_key / f"{old_name}.mp4"
            tmp_video = dataset_path / "videos" / chunk_name / video_key / f"{tmp_name}.mp4"
            rename_file(old_video, tmp_video, args.dry_run)

    print("-" * 60)
    print("Step 2: Renaming files to final names and updating parquet content...")

    for old_idx, new_idx in index_mapping.items():
        tmp_name = f"episode_tmp_{old_idx:06d}"
        new_name = f"episode_{new_idx:06d}"

        # 重命名parquet文件
        tmp_parquet = dataset_path / "data" / chunk_name / f"{tmp_name}.parquet"
        new_parquet = dataset_path / "data" / chunk_name / f"{new_name}.parquet"
        rename_file(tmp_parquet, new_parquet, args.dry_run)

        # 更新parquet文件中的episode_index
        if not args.dry_run and new_parquet.exists():
            update_parquet_episode_index(new_parquet, new_idx, args.dry_run)

        # 重命名视频文件
        for video_key in video_keys:
            tmp_video = dataset_path / "videos" / chunk_name / video_key / f"{tmp_name}.mp4"
            new_video = dataset_path / "videos" / chunk_name / video_key / f"{new_name}.mp4"
            rename_file(tmp_video, new_video, args.dry_run)

    # 第三步：更新元数据文件
    print("-" * 60)
    print("Step 3: Updating metadata files...")

    # 计算新的总帧数
    new_total_frames = 0
    new_episodes_list = []
    new_episodes_stats_list = []

    for old_idx, new_idx in index_mapping.items():
        if old_idx in episodes_data:
            ep = episodes_data[old_idx].copy()
            ep["episode_index"] = new_idx
            new_episodes_list.append(ep)
            new_total_frames += ep["length"]

        if old_idx in episodes_stats_data:
            ep_stats = episodes_stats_data[old_idx].copy()
            ep_stats["episode_index"] = new_idx
            new_episodes_stats_list.append(ep_stats)

    # 按新索引排序
    new_episodes_list.sort(key=lambda x: x["episode_index"])
    new_episodes_stats_list.sort(key=lambda x: x["episode_index"])

    # 更新info.json
    new_info = info.copy()
    new_info["total_episodes"] = new_total_episodes
    new_info["total_frames"] = new_total_frames
    new_info["total_videos"] = new_total_episodes * len(video_keys)
    new_info["splits"] = {"train": f"0:{new_total_episodes}"}

    if args.dry_run:
        print(f"  [DRY-RUN] Would update info.json:")
        print(f"    total_episodes: {info['total_episodes']} -> {new_total_episodes}")
        print(f"    total_frames: {info['total_frames']} -> {new_total_frames}")
        print(f"    total_videos: {info['total_videos']} -> {new_info['total_videos']}")
        print(f"    splits: {info['splits']} -> {new_info['splits']}")
        print(f"  [DRY-RUN] Would update episodes.jsonl with {len(new_episodes_list)} episodes")
        print(f"  [DRY-RUN] Would update episodes_stats.jsonl with {len(new_episodes_stats_list)} episodes")
    else:
        # 备份原文件
        shutil.copy(info_path, info_path.with_suffix(".json.bak"))
        shutil.copy(episodes_jsonl_path, episodes_jsonl_path.with_suffix(".jsonl.bak"))
        shutil.copy(episodes_stats_jsonl_path, episodes_stats_jsonl_path.with_suffix(".jsonl.bak"))
        print("  Backed up original metadata files")

        # 写入新的info.json
        with open(info_path, "w") as f:
            json.dump(new_info, f, indent=4)
        print(f"  Updated info.json: total_episodes={new_total_episodes}, total_frames={new_total_frames}")

        # 写入新的episodes.jsonl
        with open(episodes_jsonl_path, "w") as f:
            for ep in new_episodes_list:
                f.write(json.dumps(ep) + "\n")
        print(f"  Updated episodes.jsonl with {len(new_episodes_list)} episodes")

        # 写入新的episodes_stats.jsonl
        with open(episodes_stats_jsonl_path, "w") as f:
            for ep in new_episodes_stats_list:
                f.write(json.dumps(ep) + "\n")
        print(f"  Updated episodes_stats.jsonl with {len(new_episodes_stats_list)} episodes")

    print("-" * 60)
    print("Done!")
    print(f"New dataset has {new_total_episodes} episodes with {new_total_frames} total frames.")


if __name__ == "__main__":
    main()
