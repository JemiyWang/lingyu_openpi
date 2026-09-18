"""Run one formal 20k-step SED Robot PI05 LoRA fine-tuning job."""

from __future__ import annotations

import argparse
import dataclasses
import logging

from sed_robot_config import TASK_PROMPTS, stats_ready
from sed_robot_lora_config import make_lora_config
from train import main as train_main


def _install_sed_video_tolerance() -> None:
    """Patch only this process to use SED's non-default video timestamp tolerance."""

    import openpi.training.data_loader as data_loader
    from openpi.sed_robot_data_loader import create_torch_dataset

    data_loader.create_torch_dataset = create_torch_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASK_PROMPTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B only for local diagnostics; do not use for the formal run.",
    )
    args = parser.parse_args()

    if not stats_ready(args.task):
        raise RuntimeError(
            f"Normalization stats are missing or invalid for {args.task}. "
            f"Run compute_sed_norm_stats.py {args.task} first."
        )

    _install_sed_video_tolerance()
    config = make_lora_config(args.task, resume=args.resume, overwrite=args.overwrite)
    if args.no_wandb:
        config = dataclasses.replace(config, wandb_enabled=False)
    logging.info(
        "Starting SED PI05 LoRA task=%s, horizon=%d, batch=%d, steps=%d, fsdp=%d",
        args.task,
        config.model.action_horizon,
        config.batch_size,
        config.num_train_steps,
        config.fsdp_devices,
    )
    train_main(config)


if __name__ == "__main__":
    main()
