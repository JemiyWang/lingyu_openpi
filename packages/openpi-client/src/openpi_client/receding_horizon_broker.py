from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class RecedingHorizonBroker(_base_policy.BasePolicy):
    """Wraps a policy to implement Receding Horizon control.

    Unlike ActionChunkBroker which executes all action_horizon steps before
    re-querying, RecedingHorizonBroker re-queries after open_loop_horizon steps,
    discarding the remaining actions.

    This implements a Model Predictive Control (MPC) style approach where:
    - The model outputs action_horizon steps (e.g., 50)
    - Only open_loop_horizon steps are executed (e.g., 24)
    - After open_loop_horizon steps, a new inference is made with fresh observations
    - The remaining (action_horizon - open_loop_horizon) steps are discarded

    Args:
        policy: The underlying policy that returns action chunks.
        action_horizon: Number of actions the model outputs (e.g., 50).
        open_loop_horizon: Number of actions to execute before re-querying (e.g., 24).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        open_loop_horizon: int,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._open_loop_horizon = open_loop_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Re-query when: first call OR executed open_loop_horizon steps
        if self._last_results is None or self._cur_step >= self._open_loop_horizon:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
