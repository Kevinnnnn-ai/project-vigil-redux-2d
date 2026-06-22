# src/agents/scripted.py
# <agent_context>
#   [ARCH]: Scripted PD-controller pilot. Three nested loops: lateral offset ->
#           target tilt; tilt error -> gimbal; descent-speed error -> throttle.
#           Serves as (1) proof the env is solvable, (2) a test fixture, and
#           (3) the forever-baseline RL must beat.
#   [GOTCHA]: Decodes the NORMALIZED observation back to physical units via the
#             spaces.py refs — it deliberately consumes the same obs the neural
#             net sees, not env.state, so it exercises the full obs contract.
#   [GOTCHA]: gimbal demand divides by the actual control authority, which in
#             the M5 force model is max(spool,0.1)*maxThrustForce*maxGimbal*
#             gimbalArm/(momentInertiaCoef*mass). It uses the ACTUAL spool (the
#             engine lags the command) and the CURRENT mass (lighter = more
#             responsive), both read from the obs (indices 8 and 7).
#   [GOTCHA]: hover throttle is gravity*mass/maxThrustForce, recomputed each step
#             because mass shrinks as fuel burns — a fixed hover value would
#             over-thrust a near-empty booster.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: The gain constants below are TUNED AGAINST tests/test_scripted.py
#               success-rate thresholds. Retune them only with those tests open
#               in the other hand.
#   [VALIDATION]: python -m pytest tests/test_scripted.py -v
# </agent_guardrail>
"""PdPilot: a hand-tuned PD landing controller over the normalized obs."""
from __future__ import annotations

import math

import numpy as np

from src.agents.policy import Policy
from src.env.spaces import VEL_REF, OMEGA_REF

KX = 0.04                    # lateral offset -> tilt demand
KVX = 0.12                   # lateral velocity -> tilt demand
TILT_MAX = 0.25              # rad, max commanded tilt
KP = 8.0                     # tilt error -> angular accel demand
KD = 4.0                     # spin damping
KT = 0.15                    # descent-speed error -> throttle
MIN_AUTHORITY_THROTTLE = 0.15  # keep the engine warm for gimbal authority


class PdPilot(Policy):
    def __init__(self, world):
        self.world = world

    def act(self, obs: np.ndarray) -> np.ndarray:
        world = self.world
        x = float(obs[0]) * (world.width / 2.0)
        y = float(obs[1]) * world.ceiling
        vx = float(obs[2]) * VEL_REF
        vy = float(obs[3]) * VEL_REF
        theta = math.atan2(float(obs[4]), float(obs[5]))
        omega = float(obs[6]) * OMEGA_REF
        fuel = float(obs[7])
        spool = float(obs[8])

        mass = world.dryMass + world.fuelMass * fuel
        # Throttle that exactly cancels gravity at the CURRENT mass (shrinks as
        # fuel burns and the booster lightens).
        hover = world.gravity * mass / world.maxThrustForce

        # Lateral loop: tilt INTO the offset/drift to cancel it.
        thetaTarget = -min(max(KX * x + KVX * vx, -TILT_MAX), TILT_MAX)

        # Vertical loop: descend faster when high, creep near the ground.
        vyTarget = -min(max(y / 4.0, 0.5), 8.0)
        throttle = min(max(hover + KT * (vyTarget - vy), 0.0), 1.0)
        if y > 2.0:
            throttle = max(throttle, MIN_AUTHORITY_THROTTLE)

        # Attitude loop: gimbal for the desired angular accel given the
        # spool- and mass-scaled authority (alpha = -thrustForce * maxGimbal *
        # gimbalArm / inertia * gimbal). Use the ACTUAL spool (the engine may lag
        # the command); floor it for divide safety.
        alphaDesired = KP * (thetaTarget - theta) - KD * omega
        inertia = world.momentInertiaCoef * mass
        authority = (
            max(spool, 0.1) * world.maxThrustForce * world.maxGimbal
            * world.gimbalArm / inertia
        )
        gimbal = min(max(-alphaDesired / authority, -1.0), 1.0)

        return np.array([throttle, gimbal], dtype=np.float64)
