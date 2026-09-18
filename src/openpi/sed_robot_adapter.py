"""Standalone SED TeleAvatar data adapter for PI05 fine-tuning.

This module is intentionally separate from the legacy Teleavatar policy and
configuration files.  The SED dataset uses the newer 16-D schema:

* cameras: ``top_head``, ``hand_left``, ``hand_right``;
* state/action: 7 joints + gripper for each arm (16 values);
* actions are absolute joint targets (no delta conversion by default).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import ClassVar

import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class SedTeleAvatarInputs(transforms.DataTransformFn):
    """Map SED's 16-D LeRobot samples into PI model inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    mask_state: bool = False

    required_rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        images_in = data["images"]
        missing = set(self.required_rename_map) - set(images_in)
        if missing:
            raise ValueError(f"Missing SED TeleAvatar cameras: {sorted(missing)}")

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        for source_name, model_name in self.required_rename_map.items():
            image = images_in[source_name]
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            # TorchCodec normally returns [C,H,W], while some LeRobot/video
            # versions add a singleton frame dimension [1,C,H,W]. Normalize
            # both layouts before the HWC-only PIL resize transform.
            while image.ndim > 3 and image.shape[0] == 1:
                image = image[0]
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            images[model_name] = image
            image_masks[model_name] = np.True_

        # PI05 encodes state in the discrete prompt.  The subsequent model
        # transform pads the 16-D state/action to the model's 32-D space.
        state = np.asarray(data["state"])
        if self.model_type != _model.ModelType.PI05:
            state = transforms.pad_to_dim(state, self.action_dim)
        state = np.squeeze(state)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.zeros_like(state) if self.mask_state else state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.model_type != _model.ModelType.PI05:
                actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = np.squeeze(actions)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SedTeleAvatarOutputs(transforms.DataTransformFn):
    """Convert padded PI outputs back to the 16 physical SED actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}


@dataclasses.dataclass(frozen=True)
class SedRobotDataConfig(_config.DataConfigFactory):
    """Data factory for one SED task directory."""

    use_delta_joint_actions: bool = False
    default_prompt: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    mask_state: bool = False

    @property
    def _repack_transform(self) -> transforms.Group:
        return transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        # Imported lazily to avoid making this adapter alter the legacy policy
        # module's public classes.
        data_transforms = transforms.Group(
            inputs=[
                SedTeleAvatarInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[SedTeleAvatarOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_mask = transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_mask)],
                outputs=[transforms.AbsoluteActions(delta_mask)],
            )

        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self._repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_sed_video_loader=True,
        )
