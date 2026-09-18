"""Configs for the six SED TeleAvatar PI05 fine-tuning jobs.

Kept as a new file so the legacy ``training/config.py`` and policy modules are
not changed.  The regular OpenPI training and norm-statistics scripts consume
the returned ``TrainConfig`` objects directly.
"""

from __future__ import annotations

from pathlib import Path

import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as normalize
import openpi.training.config as config
import openpi.training.optimizer as optimizer
import openpi.training.weight_loaders as weight_loaders

from openpi.sed_robot_adapter import SedRobotDataConfig


DATA_ROOT = Path("/mnt/mnt/data/FPF_workspace/datasets/sed_robot_20260906")
BASE_PARAMS = Path("/mnt/mnt/data/FPF_workspace/checkpoints/pi05_base/params")
CHECKPOINT_ROOT = Path("/mnt/mnt/data/lingyu/openpi/checkpoints/sed_robot_20260906")
ASSETS_ROOT = DATA_ROOT

TASK_PROMPTS = {
    "collect_food": "Collect the food.",
    "fold_towels": "Fold the towels.",
    "pick_flowers": "Pick the flowers.",
    "pick_up_paper_rolls": "Pick up the paper rolls.",
    "pick_up_trash": "Pick up the trash.",
    "stack_blocks": "Stack the blocks.",
}


def make_config(task: str, *, resume: bool = False, overwrite: bool = False) -> config.TrainConfig:
    """Build the isolated 20k-step PI05 config for one task."""

    if task not in TASK_PROMPTS:
        raise ValueError(f"Unknown SED task {task!r}; choose from {sorted(TASK_PROMPTS)}")
    return config.TrainConfig(
        name="pi05_sed_robot",
        exp_name=task,
        project_name="openpi",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            # PI05's native input format encodes normalized state in the prompt.
            discrete_state_input=True,
        ),
        data=SedRobotDataConfig(
            repo_id=str(DATA_ROOT / task),
            assets=config.AssetsConfig(assets_dir=str(ASSETS_ROOT), asset_id=task),
            default_prompt=TASK_PROMPTS[task],
            use_delta_joint_actions=False,
        ),
        assets_base_dir=str(Path("/mnt/mnt/data/lingyu/openpi/assets/sed_robot_20260906")),
        checkpoint_base_dir=str(CHECKPOINT_ROOT),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=500_000,
            decay_lr=5e-5,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        batch_size=64,
        num_workers=16,
        num_train_steps=20_000,
        log_interval=100,
        save_interval=10_000,
        # Keep only the latest checkpoint; periodic saves still provide crash recovery during training.
        keep_period=None,
        fsdp_devices=8,
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=True,
    )


def stats_path(task: str) -> Path:
    return DATA_ROOT / task / "norm_stats.json"


def stats_ready(task: str) -> bool:
    path = stats_path(task)
    if not path.exists():
        return False
    try:
        loaded = normalize.load(path.parent)
    except Exception:
        return False
    return set(loaded) >= {"state", "actions"}
