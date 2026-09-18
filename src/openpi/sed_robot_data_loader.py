"""SED-specific LeRobot dataset construction.

The SED videos are nominally 45 Hz, but some HEVC streams have timestamps
that differ from the parquet timestamps by one or two video ticks.  The
upstream LeRobot default tolerance (1e-4 s) rejects those otherwise usable
frames.  This module keeps the compatibility adjustment isolated from the
legacy OpenPI data loader.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import Any

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

SED_VIDEO_TOLERANCE_S = 0.05


_decoder_local = threading.local()


def _get_video_decoder(video_path: str):
    """Reuse a small, thread-local decoder cache; decoders are not thread safe."""

    from torchcodec.decoders import VideoDecoder

    if not hasattr(_decoder_local, "decoders"):
        _decoder_local.decoders = OrderedDict()
    cache = _decoder_local.decoders
    if video_path not in cache:
        cache[video_path] = VideoDecoder(video_path, device="cpu", seek_mode="approximate", num_ffmpeg_threads=1)
        # Full-resolution HEVC decoder state is large. Bound memory across all
        # decode threads, rather than retaining 128 decoders per thread.
        while len(cache) > 6:
            cache.popitem(last=False)
    cache.move_to_end(video_path)
    return cache[video_path]


def _decode_cached_video_frames(video_path: str, timestamps: list[float], tolerance_s: float) -> torch.Tensor:
    """Decode frames using a per-worker decoder cache.

    LeRobot's default path constructs a new VideoDecoder for every sample and
    camera. SED has long HEVC streams on shared storage, so that repeated open
    and metadata/seek work can starve the accelerator. The cache is process
    local, so workers never share decoder state.
    """

    decoder = _get_video_decoder(video_path)
    average_fps = decoder.metadata.average_fps
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)
    loaded_frames = list(frames_batch.data)
    loaded_ts = torch.as_tensor(frames_batch.pts_seconds, dtype=torch.float64)
    query_ts = torch.as_tensor(timestamps, dtype=torch.float64)
    distances = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_distances, argmin = distances.min(1)
    if not torch.all(min_distances < tolerance_s):
        bad = min_distances >= tolerance_s
        raise AssertionError(
            f"Video timestamp mismatch for {video_path}: "
            f"{min_distances[bad]} > tolerance_s={tolerance_s}"
        )
    return torch.stack([loaded_frames[i] for i in argmin.tolist()]).to(torch.float32) / 255


class _CachedSedLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobot dataset with parallel samples and thread-local decoder reuse."""

    def get_transformed_batch(self, indices, transform):
        # Lazy creation keeps executors out of the dataset sent to spawn/
        # forkserver workers. Four processes x 24 threads uses 96 CPU cores.
        workers = max(1, int(os.environ.get("OPENPI_SED_DECODE_THREADS", "24")))
        if workers == 1:
            return [transform(self[index]) for index in indices]
        if not hasattr(self, "_decode_pool"):
            # Import the extension once before threads start decoding.
            import torchcodec.decoders  # noqa: F401

            self._decode_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sed_decode")
        return list(self._decode_pool.map(lambda index: transform(self[index]), indices))

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            item[vid_key] = _decode_cached_video_frames(str(video_path), query_ts, self.tolerance_s).squeeze(0)
        return item


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    tolerance_s: float = SED_VIDEO_TOLERANCE_S,
) -> Any:
    """Create a LeRobot dataset with SED's measured video timestamp tolerance."""

    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = _CachedSedLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        tolerance_s=tolerance_s,
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset
