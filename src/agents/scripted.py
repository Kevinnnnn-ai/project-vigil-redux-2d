# src/agents/scripted.py
# <agent_context>
#   [ARCH]: Scripted controller pilot — a deliberately WEAK binary baseline (not a
#           solver). Three loops: lateral offset -> target tilt; tilt error ->
#           gimbal; and a kinematic SINGLE-BURN vertical loop for the binary
#           suicide-burn engine. Serves as (1) proof the env is solvable from the
#           easy stages, (2) a test fixture, and (3) the forever-baseline RL beats.
#   [GOTCHA]: Decodes the NORMALIZED observation back to physical units via the
#             spaces.py refs — it deliberately consumes the same obs the neural
#             net sees, not env.state, so it exercises the full obs contract.
#   [GOTCHA]: gimbal demand divides by the actual control authority, which in
#             the M5 force model is max(spool,0.1)*maxThrustForce*maxGimbal*
#             gimbalArm/(momentInertiaCoef*mass). It uses the ACTUAL spool (the
#             engine lags the command) and the CURRENT mass (lighter = more
#             responsive), both read from the obs (indices 8 and 7).
#   [GOTCHA]: The engine is BINARY (fires at full above the env's 0.5 throttle
#             threshold or not at all) and locks after one ignite + one cut, so
#             the vertical loop must NOT modulate throttle — it commits to ONE
#             braking burn. It reads obs[9] (ignitionsRemaining: 1.0 pre-burn /
#             0.5 burning / 0.0 locked) for the phase and stays STATELESS (reset()
#             is a no-op), so the controller is fully determined by the obs.
#   [GOTCHA]: The burn ignites a fixed lead-distance EARLY (IGNITE_LEAD seconds of
#             fall) because the engine spools up over ~0.25 s; a bare kinematic
#             brake-distance trigger fires too late to arrest in time.
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
# Binary suicide-burn gains (vertical loop). The engine fires at full or not at
# all and locks after one ignite + one cut, so the controller commits to a
# SINGLE braking burn instead of modulating throttle. Tuned against the
# tests/test_scripted.py success-rate floors (retune them together).
IGNITE_LEAD = 0.32           # s — ignite this much sooner than the bare kinematic
                             # brake point, so the burn fires while the descent is
                             # still arrestable DESPITE the engine's spool-up lag
                             # (the spool ramps over ~0.25 s, during which the
                             # booster keeps falling). Larger = ignite earlier.
CUT_SPEED = 2.0              # m/s — once the burn has slowed the descent to within
                             # this of a standstill, CUT so the engine is OFF
                             # before contact (the success gate requires engine-off).
                             # The remaining descent is bled off by the spool tail-
                             # off plus the short fall to the pad.


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
        ignitionsRemaining = float(obs[9])   # 1.0 pre-burn / 0.5 burning / 0.0 locked

        mass = world.dryMass + world.fuelMass * fuel

        # Lateral loop: tilt INTO the offset/drift to cancel it.
        thetaTarget = -min(max(KX * x + KVX * vx, -TILT_MAX), TILT_MAX)

        # Vertical loop — binary suicide burn: ONE ignition, ONE cutoff. The engine
        # fires at full or not at all, so the controller commits to a single braking
        # burn instead of modulating throttle. obs[9] gives the phase (1.0 pre-burn /
        # 0.5 burning / 0.0 locked). The burn MUST be cut BEFORE contact — the
        # success gate requires the engine commanded OFF as the booster touches down.
        baseAboveRest = max(y - world.legDrop, 0.0)          # base height above its resting height
        brakeAccel = max(world.maxThrustForce / mass - world.gravity, 0.1)
        brakeDist = (vy * vy) / (2.0 * brakeAccel)           # distance to arrest the current descent
        # Lead distance: how far the booster falls during IGNITE_LEAD seconds at the
        # current descent rate. Adding it brings the ignition forward to cover the
        # engine's spool-up lag (a multiplicative margin cannot — the lag is a fixed
        # distance that dominates low/slow spawns yet stays small against a tall drop).
        leadDist = -vy * IGNITE_LEAD if vy < 0.0 else 0.0
        if ignitionsRemaining <= 0.0:
            throttle = 0.0                                   # engine locked off — coast
        elif ignitionsRemaining >= 1.0:
            # Pre-burn: ignite once the (lead-padded) brake distance no longer fits in
            # the remaining height. Coast otherwise — many low/slow spawns land gently
            # with no burn at all.
            mustBurn = vy < 0.0 and (brakeDist + leadDist) >= baseAboveRest
            throttle = 1.0 if mustBurn else 0.0
        else:
            # Burning: hold full thrust until the descent is nearly arrested, then CUT
            # so the engine is off before touchdown.
            throttle = 0.0 if vy >= -CUT_SPEED else 1.0

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
