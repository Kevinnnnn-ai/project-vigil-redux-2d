# src/env/rewards.py
# <agent_context>
#   [ARCH]: Isolated, config-driven reward module — the ONE place reward
#           arithmetic lives. The environment calls into this module and
#           contains no reward logic of its own. Every number derives from
#           cfg.reward.* (and cfg.training.gamma for the shaping discount,
#           kept equal to the GAE discount intentionally).
#   [API]: computeReward is the public entry point (called by the env each
#          step); computePotential is the potential function Phi used by the
#          shaping term.
#   [GOTCHA]: shapingScale is the curriculum/anneal factor passed by the train
#             loop (1.0 default). It multiplies shapingCoef; scaling a
#             potential-based term preserves policy invariance per-step but the
#             ANNEAL itself is a curriculum choice — log changes in reward-log.
#   [GOTCHA]: timeout earns the full flat terminalCrash on purpose: without it,
#             hovering until the clock runs out strictly dominates a risky
#             landing attempt, and the policy learns to stall.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: The (1 - done) factor in the shaping term MUST NOT be removed.
#               It zeroes Phi at the terminal state, REQUIRED for policy
#               invariance (Ng et al. 1999) — guarded by
#               test_shapingTelescopesToInitialPotential.
#   [CRITICAL]: Actions MUST be clipped (throttle [0,1], gimbal [-1,1]) before
#               the control cost. Out-of-range actions must not be over-penalized.
#   [CRITICAL]: Keep computePotential in sync with the obs scales (VEL_REF from
#               spaces.py) — do not introduce new magic normalizers here.
#   [VALIDATION]: python -m pytest tests/test_rewards.py -v
# </agent_guardrail>
"""Isolated, config-driven reward. The ONE place reward logic lives.

Per-step reward = terminal payout (on an outcome) + potential-based shaping
+ control-effort cost. See docs/reward-log.md for the experiment trail.
"""
from __future__ import annotations

import math

from src.env.spaces import VEL_REF


def computePotential(state, world) -> float:
    """Phi(s): higher is better — close to the pad, slow, upright.

        Phi = -( dist(pad)/ceiling + speed/VEL_REF + |theta|/pi )

    All three terms are dimensionless and O(1) over the playable envelope, so
    no component dominates the shaping signal."""
    dist = math.hypot(state.x, state.y) / world.ceiling
    speed = math.hypot(state.vx, state.vy) / VEL_REF
    tilt = abs(state.theta) / math.pi
    return -(dist + speed + tilt)


def computeReward(cfg, prevState, state, action, outcome, impactSpeed, shapingScale=1.0):
    """Assemble the scalar reward for one transition.

    Args:
        cfg:          Full Config (reads cfg.reward.* and cfg.training.gamma).
        prevState:    BoosterState at s (before the step).
        state:        BoosterState at s' (after the step).
        action:       [throttle, gimbal] as commanded (clipped here).
        outcome:      None while flying, else 'success'|'crash'|'timeout'
                      (there is no 'oob' — the world box physically confines the
                      booster, so leaving it is impossible).
        impactSpeed:  hypot(vx, vy) at touchdown; 0.0 when not a touchdown.
        shapingScale: anneal factor in [0, 1] from the train loop (1.0 default).

    Returns:
        float reward.
    """
    reward = cfg.reward
    world = cfg.world
    throttle = min(max(float(action[0]), 0.0), 1.0)
    gimbal = min(max(float(action[1]), -1.0), 1.0)
    done = outcome is not None

    total = 0.0

    # Terminal payout — graded for touchdowns, flat crash penalty for timeout.
    if outcome == 'success':
        gentleness = max(0.0, 1.0 - impactSpeed / world.maxLandingSpeed)
        centering = max(0.0, 1.0 - abs(state.x) / (world.padWidth / 2.0))
        total += (
            reward.terminalSuccess
            + reward.gentlenessBonus * gentleness
            + reward.centeringBonus * centering
        )
    elif outcome == 'crash':
        severity = min(impactSpeed / (4.0 * world.maxLandingSpeed), 1.0)
        total += reward.terminalCrash * (0.5 + 0.5 * severity)
    elif outcome == 'timeout':
        total += reward.terminalCrash

    # Potential-based shaping; (1 - done) zeroes Phi at the terminal state.
    if reward.shapingCoef != 0.0 and shapingScale != 0.0:
        gamma = cfg.training.gamma
        notDone = 0.0 if done else 1.0
        phiPrev = computePotential(prevState, world)
        phiNext = computePotential(state, world)
        total += reward.shapingCoef * shapingScale * (gamma * phiNext * notDone - phiPrev)

    # Control-effort cost: a GENUINE cost (not shaping) — discourages hovering
    # on a hot engine and gimbal thrash.
    total -= reward.controlCost * (throttle * throttle + gimbal * gimbal)

    return total
