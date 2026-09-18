"""Merge multiple LeRobot v2.1 datasets into one."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def merge_datasets(source_dirs: list[Path], output_dir: Path, chunks_size: int = 1000):
    """Merge multiple LeRobot datasets into one.

    Args:
        source_dirs: List of source dataset directories
        output_dir: Output directory for merged dataset
        chunks_size: Number of episodes per chunk
    """
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Please remove it first or choose a different name.")
        return

    # Create output directory structure
    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "data").mkdir()
    (output_dir / "videos").mkdir()

    # Load first dataset's info as template
    first_info = json.load(open(source_dirs[0] / "meta" / "info.json"))

    # Collect all tasks and create task mapping
    all_tasks = {}  # task_text -> new_task_index
    task_index_counter = 0

    # Track totals
    total_episodes = 0
    total_frames = 0
    total_videos = 0

    # Episodes and stats to write
    all_episodes = []
    all_episodes_stats = []

    print(f"Merging {len(source_dirs)} datasets into {output_dir}")

    for src_dir in source_dirs:
        print(f"\nProcessing: {src_dir.name}")

        # Load source info
        src_info = json.load(open(src_dir / "meta" / "info.json"))
        src_episodes = [json.loads(line) for line in open(src_dir / "meta" / "episodes.jsonl")]
        src_tasks = [json.loads(line) for line in open(src_dir / "meta" / "tasks.jsonl")]

        # Build task index mapping for this source
        src_task_map = {}  # old_task_index -> new_task_index
        for task_entry in src_tasks:
            task_text = task_entry["task"]
            old_idx = task_entry["task_index"]
            if task_text not in all_tasks:
                all_tasks[task_text] = task_index_counter
                task_index_counter += 1
            src_task_map[old_idx] = all_tasks[task_text]

        # Load episodes_stats if exists
        episodes_stats_path = src_dir / "meta" / "episodes_stats.jsonl"
        src_episodes_stats = []
        if episodes_stats_path.exists():
            src_episodes_stats = [json.loads(line) for line in open(episodes_stats_path)]

        # Process each episode
        for ep in src_episodes:
            old_ep_idx = ep["episode_index"]
            new_ep_idx = total_episodes
            old_chunk = old_ep_idx // chunks_size
            new_chunk = new_ep_idx // chunks_size

            # Create new chunk directories if needed
            new_data_chunk_dir = output_dir / "data" / f"chunk-{new_chunk:03d}"
            new_video_chunk_dir = output_dir / "videos" / f"chunk-{new_chunk:03d}"
            new_data_chunk_dir.mkdir(exist_ok=True)
            new_video_chunk_dir.mkdir(exist_ok=True)

            # Copy and update parquet file
            old_parquet = src_dir / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_ep_idx:06d}.parquet"
            new_parquet = new_data_chunk_dir / f"episode_{new_ep_idx:06d}.parquet"

            if old_parquet.exists():
                # Read, update indices, and write
                df = pd.read_parquet(old_parquet)
                df["episode_index"] = new_ep_idx
                # Update global index
                if "index" in df.columns:
                    df["index"] = df["index"] - df["index"].min() + total_frames
                # Update task_index if present
                if "task_index" in df.columns:
                    df["task_index"] = df["task_index"].map(lambda x: src_task_map.get(x, x))
                df.to_parquet(new_parquet)

            # Copy video files
            old_video_chunk_dir = src_dir / "videos" / f"chunk-{old_chunk:03d}"
            if old_video_chunk_dir.exists():
                for video_key_dir in old_video_chunk_dir.iterdir():
                    if video_key_dir.is_dir():
                        new_video_key_dir = new_video_chunk_dir / video_key_dir.name
                        new_video_key_dir.mkdir(exist_ok=True)

                        old_video = video_key_dir / f"episode_{old_ep_idx:06d}.mp4"
                        new_video = new_video_key_dir / f"episode_{new_ep_idx:06d}.mp4"

                        if old_video.exists():
                            shutil.copy2(old_video, new_video)
                            total_videos += 1

            # Update episode entry
            new_ep = {
                "episode_index": new_ep_idx,
                "tasks": ep["tasks"],
                "length": ep["length"],
            }
            all_episodes.append(new_ep)

            # Update episode stats if exists
            if src_episodes_stats and old_ep_idx < len(src_episodes_stats):
                stats = src_episodes_stats[old_ep_idx].copy()
                stats["episode_index"] = new_ep_idx
                all_episodes_stats.append(stats)

            total_frames += ep["length"]
            total_episodes += 1

            if total_episodes % 10 == 0:
                print(f"  Processed {total_episodes} episodes...")

    # Write tasks.jsonl
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for task_text, task_idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_text}) + "\n")

    # Write episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write episodes_stats.jsonl if we have stats
    if all_episodes_stats:
        with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
            for stats in all_episodes_stats:
                f.write(json.dumps(stats) + "\n")

    # Write info.json
    num_chunks = (total_episodes + chunks_size - 1) // chunks_size
    merged_info = first_info.copy()
    merged_info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": total_videos,
        "total_chunks": num_chunks,
        "chunks_size": chunks_size,
        "splits": {"train": f"0:{total_episodes}"},
    })

    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)

    print(f"\n=== Merge Complete ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Total videos: {total_videos}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot v2.1 datasets")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Base directory containing dataset folders",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to match dataset folders (default: *)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=1000,
        help="Episodes per chunk (default: 1000)",
    )

    args = parser.parse_args()

    source_base = Path(args.source_dir).expanduser()
    source_dirs = sorted(source_base.glob(args.pattern))

    # Filter to only directories with meta/info.json
    source_dirs = [d for d in source_dirs if d.is_dir() and (d / "meta" / "info.json").exists()]

    if not source_dirs:
        print(f"No valid datasets found in {source_base} matching pattern '{args.pattern}'")
        return

    print(f"Found {len(source_dirs)} datasets to merge:")
    for d in source_dirs:
        print(f"  - {d.name}")

    output_dir = Path(args.output).expanduser()
    merge_datasets(source_dirs, output_dir, args.chunks_size)


if __name__ == "__main__":
    main()
