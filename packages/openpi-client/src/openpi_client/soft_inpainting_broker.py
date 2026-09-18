"""Soft Inpainting Broker for smooth action transitions.

This module implements the Soft Inpainting control strategy from the
BEHAVIOR-1K challenge winning solution.

Reference:
    Larchenko, Zarin, and Karnatak (2025)
    "Task adaptation of Vision-Language-Action model:
    1st Place Solution for the 2025 BEHAVIOR Challenge"
    https://arxiv.org/abs/2512.06951
"""

from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class SoftInpaintingBroker(_base_policy.BasePolicy):
    """Wraps a policy to implement Soft Inpainting control.

    This implements the Soft Inpainting approach:
    - Model outputs action_horizon steps (e.g., 30)
    - Execute actions_to_execute steps (e.g., 26)
    - Keep the last actions_to_keep steps (e.g., 4) for next prediction
    - On next inference, pass kept actions as initial_actions for inpainting constraint

    The server-side model uses these initial_actions to:
    1. Apply hard constraint during early denoising (t > 0.3)
    2. Optionally propagate corrections via correlation matrix

    Args:
        policy: The underlying policy that returns action chunks.
        action_horizon: Number of actions the model outputs (e.g., 30).
        actions_to_execute: Number of actions to execute before re-querying (e.g., 26).
        actions_to_keep: Number of actions to keep for inpainting (e.g., 4).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        actions_to_execute: int,
        actions_to_keep: int,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._actions_to_execute = actions_to_execute
        self._actions_to_keep = actions_to_keep
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._kept_actions: np.ndarray | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Re-query when: first call OR executed actions_to_execute steps
        if self._last_results is None or self._cur_step >= self._actions_to_execute:
            # Add kept_actions to obs for server-side inpainting
            if self._kept_actions is not None:
                obs = dict(obs)  # Make a copy to avoid modifying the original
                obs["initial_actions"] = self._kept_actions

            # Query the policy
            self._last_results = self._policy.infer(obs)

            # Extract and keep the last actions_to_keep steps for next prediction
            self._kept_actions = self._extract_kept_actions(self._last_results)

            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        return results

    def _extract_kept_actions(
        self, results: Dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Extract the last actions_to_keep steps for next prediction.

        Args:
            results: Dictionary containing 'actions' key with shape (action_horizon, action_dim)

        Returns:
            Kept actions with shape (actions_to_keep, action_dim), or None if not available
        """
        if "actions" not in results:
            return None

        actions = results["actions"]
        if not isinstance(actions, np.ndarray):
            return None

        # Calculate indices
        start_idx = self._actions_to_execute
        end_idx = start_idx + self._actions_to_keep

        # Ensure we have enough actions
        if len(actions) < end_idx:
            # If not enough actions, take what we can from the end
            available = len(actions) - start_idx
            if available > 0:
                return actions[start_idx:].copy()
            return None

        return actions[start_idx:end_idx].copy()

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._kept_actions = None
        self._cur_step = 0
