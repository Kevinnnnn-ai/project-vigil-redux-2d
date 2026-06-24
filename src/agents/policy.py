# src/agents/policy.py
"""Policy interface shared by scripted (M1), neural (M2), and human (M3) players.
Every policy maps an (OBS_DIM,) observation to an (ACTION_DIM,) action:
[throttle in [0, 1], gimbal in [-1, 1]]. OBS_DIM=11, ACTION_DIM=2 — see
src/env/spaces.py for the obs/action contract."""
from __future__ import annotations

import abc

import numpy as np


class Policy(abc.ABC):
    @abc.abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map an (OBS_DIM=11,) observation to an (ACTION_DIM=2,) action
        [throttle, gimbal]."""

    def reset(self) -> None:
        """Hook for stateful policies. No-op by default."""
        return None
