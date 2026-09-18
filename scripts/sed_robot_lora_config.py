"""LoRA configs for the six SED Robot PI05 fine-tuning jobs.

This is intentionally a separate module.  The legacy OpenPI configuration
registry is left untouched; the wrapper imports the existing SED data config
and replaces only the model/freeze settings needed for LoRA training.
"""

from __future__ import annotations

import dataclasses
import os

import openpi.models.pi0_config as pi0_config

from sed_robot_config import TASK_PROMPTS, make_config


def make_lora_config(task: str, *, resume: bool = False, overwrite: bool = False):
    """Build a 20k-step full-PI05-architecture LoRA config for one task."""

    base = make_config(task, resume=resume, overwrite=overwrite)
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=base.model.action_dim,
        action_horizon=base.model.action_horizon,
        discrete_state_input=True,
    )
    return dataclasses.replace(
        base,
        name="pi05_sed_robot_lora",
        model=model,
        # Each process decodes independent samples concurrently. Eight
        # processes x 14 decode threads keeps all 112 logical CPUs busy while
        # avoiding the old one-sample-per-process startup explosion.
        num_workers=min(8, max(1, (os.cpu_count() or 8) // 14)),
        prefetch_factor=2,
        pin_memory=True,
        log_interval=20,
        freeze_filter=model.get_freeze_filter(),
        # OpenPI's LoRA recipe disables EMA to reduce memory and checkpoint size.
        ema_decay=None,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=True,
    )


__all__ = ["TASK_PROMPTS", "make_lora_config"]
