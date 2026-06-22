# src/env/spaces.py
# <agent_context>
#   [ARCH]: Single source of truth for observation and action layout. All
#           downstream modules (env step, policy nets, the PD pilot's decoder)
#           import OBS_DIM / ACTION_DIM / VEL_REF / OMEGA_REF from here — never
#           hard-code 10 or 2 or the normalization refs elsewhere.
#   [GOTCHA]: theta enters the obs as (sin, cos), never raw — avoids the ±pi
#             wrap discontinuity. Decode with atan2(obs[4], obs[5]).
#   [GOTCHA]: VEL_REF / OMEGA_REF are code constants, not config — they are
#             part of the obs contract, so changing them invalidates models
#             just like a world edit would, WITHOUT changing the world hash.
#             Treat them as frozen.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Do NOT add action-encoding or reward logic here (YAGNI). Those
#               live in separate modules under src/env/.
#   [CRITICAL]: encodeObs() accepts a WorldConfig, NOT a full Config. Never
#               widen the signature.
#   [VALIDATION]: python -m pytest tests/test_spaces.py -v
# </agent_guardrail>
"""Single source of truth for the observation and action layout.

Observation (10-D float32):

  index  component               normalization
  0      x (offset from pad)     / (width/2)
  1      y (altitude)            / ceiling
  2, 3   vx, vy                  / VEL_REF
  4, 5   sin(theta), cos(theta)  already in [-1, 1]
  6      omega                   / OMEGA_REF
  7      fuel                    already in [0, 1]
  8      spool (actual throttle) already in [0, 1]
  9      ignitionsRemaining      (2 - engineTransitions)/2, in [0,1]

Action (2-D): [throttle in [0, 1], gimbal in [-1, 1]]. The env scales gimbal by
world.maxGimbal; throttle commands the engine, whose spooled force is applied in
physics. mass is NOT in the obs — it is a function of fuel, which is present.
"""
from __future__ import annotations

import math

import numpy as np

OBS_DIM = 10
ACTION_DIM = 2
VEL_REF = 20.0     # m/s — velocity normalization reference
OMEGA_REF = 3.0    # rad/s — spin normalization reference


def toEnvAction(a):
    """Map a tanh-space action in (-1, 1)^2 (neural-net output) to the env
    action [throttle in [0, 1], gimbal in [-1, 1]]. Affine on throttle only, so
    PPO log-prob math (which lives entirely in pre-squash space) is untouched.
    Works on a single (2,) action or a batch (..., 2)."""
    out = np.array(a, dtype=np.float64, copy=True)
    out[..., 0] = (out[..., 0] + 1.0) / 2.0
    return out


def encodeObs(state, world):
    """Build the 10-D float32 observation from a BoosterState. `world` is a
    WorldConfig (used only for the geometric normalizers)."""
    return np.array(
        [
            state.x / (world.width / 2.0),
            state.y / world.ceiling,
            state.vx / VEL_REF,
            state.vy / VEL_REF,
            math.sin(state.theta),
            math.cos(state.theta),
            state.omega / OMEGA_REF,
            state.fuel,
            state.spool,
            # <agent_context>
            #   [ARCH]: index 9 — ignitionsRemaining = (2 - engineTransitions)/2.
            #           1.0=fresh, 0.5=one used, 0.0=locked. In analog mode
            #           engineTransitions is always 0, so this is a constant 1.0.
            #           Both discrete (suicide-burn) and analog models share 10-D obs.
            # </agent_context>
            (2 - state.engineTransitions) / 2.0,
        ],
        dtype=np.float32,
    )
