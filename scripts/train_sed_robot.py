"""Run one isolated 20k-step PI05 SED fine-tuning job."""

from __future__ import annotations

import argparse
import logging

from sed_robot_config import TASK_PROMPTS, make_config, stats_ready
from train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASK_PROMPTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not stats_ready(args.task):
        raise RuntimeError(
            f"Normalization stats are missing or invalid for {args.task}. "
            f"Run compute_sed_norm_stats.py {args.task} first."
        )
    config = make_config(args.task, resume=args.resume, overwrite=args.overwrite)
    logging.info(
        "Starting SED PI05 task=%s, horizon=%d, batch=%d, steps=%d, fsdp=%d",
        args.task,
        config.model.action_horizon,
        config.batch_size,
        config.num_train_steps,
        config.fsdp_devices,
    )
    train_main(config)


if __name__ == "__main__":
    main()
