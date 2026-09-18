"""Short, full-model SED PI05 smoke test using all eight GPUs."""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import openpi.models.pi0_config as pi0_config
import openpi.training.config as training_config

from sed_robot_config import TASK_PROMPTS, make_config, stats_ready
from train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASK_PROMPTS), default="stack_blocks", nargs="?")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Global batch size for the smoke run (must be divisible by the 8-device mesh).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override data-loader workers for a diagnostic run (0 disables worker processes).",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Load real batches and report host-side latency without initializing the model.",
    )
    parser.add_argument("--batches", type=int, default=2, help="Number of batches for --data-only.")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the EMA parameter copy to reduce peak initialization memory.",
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="Use PI05 LoRA variants for a memory-safe end-to-end smoke test.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use shape-compatible synthetic batches to smoke-test model/FSDP without video decoding.",
    )
    args = parser.parse_args()
    if not stats_ready(args.task):
        raise RuntimeError(f"Missing norm stats for {args.task}")

    base = make_config(args.task)
    # Match the formal LoRA loader's CPU/video-decoding settings during smoke tests.
    base = dataclasses.replace(
        base,
        num_workers=min(8, max(1, (os.cpu_count() or 8) // 14)),
        prefetch_factor=2,
        pin_memory=True,
    )
    if args.workers is not None:
        base = dataclasses.replace(base, num_workers=args.workers)
    if args.data_only:
        from openpi.training import data_loader as _data_loader
        from train_sed_robot_lora import _install_sed_video_tolerance

        _install_sed_video_tolerance()
        base = dataclasses.replace(base, batch_size=args.batch_size)
        loader = _data_loader.create_data_loader(base, shuffle=True)
        iterator = iter(loader)
        for batch_index in range(args.batches):
            started = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - started
            sample_shape = next(iter(batch[0].images.values())).shape
            print(f"DATA_BATCH index={batch_index} wait_s={elapsed:.3f} image_shape={sample_shape}", flush=True)
        return
    data = training_config.FakeDataConfig() if args.fake else base.data
    if not args.fake:
        from train_sed_robot_lora import _install_sed_video_tolerance

        _install_sed_video_tolerance()
    model = base.model
    freeze_filter = base.freeze_filter
    ema_decay = base.ema_decay
    if args.low_mem:
        model = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=base.model.action_dim,
            action_horizon=base.model.action_horizon,
            discrete_state_input=True,
        )
        freeze_filter = model.get_freeze_filter()
        ema_decay = None
    smoke = dataclasses.replace(
        base,
        model=model,
        freeze_filter=freeze_filter,
        data=data,
        exp_name=f"smoke_{args.task}",
        checkpoint_base_dir="/mnt/mnt/data/lingyu/openpi/checkpoints/sed_robot_smoke",
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        ema_decay=None if args.no_ema else ema_decay,
        save_interval=max(1, args.steps),
        keep_period=None,
        overwrite=True,
        resume=False,
        wandb_enabled=not args.no_wandb,
    )
    train_main(smoke)


if __name__ == "__main__":
    main()
