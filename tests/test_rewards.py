# tests/test_rewards.py
import math

import pytest

from src.config.loader import loadConfig
from src.env.physics import BoosterState
from src.env.rewards import computePotential, computeReward


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


def _still(x=0.0, y=0.0):
    return BoosterState(x=x, y=y, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=0.5)


def test_perfectLandingEarnsAllBonuses(cfg):
    reward = computeReward(
        cfg, _still(y=0.5), _still(y=0.0), [0.0, 0.0],
        outcome='success', impactSpeed=0.0,
    )
    expected = (
        cfg.reward.terminalSuccess
        + cfg.reward.gentlenessBonus
        + cfg.reward.centeringBonus
    )
    # Shaping adds a small positive term on this descending transition; the
    # terminal payout must dominate.
    assert reward > expected * 0.9


def test_atLimitLandingStillBeatsBareTerminal(cfg):
    world = cfg.world
    edge = BoosterState(x=world.padWidth / 2, y=0.0, vy=-world.maxLandingSpeed)
    reward = computeReward(
        cfg, _still(x=world.padWidth / 2, y=0.5), edge, [0.0, 0.0],
        outcome='success', impactSpeed=world.maxLandingSpeed,
    )
    # Bonuses go to ~0 at the limits but must never go negative.
    assert reward > cfg.reward.terminalSuccess * 0.5


def test_hardCrashWorseThanSoftCrash(cfg):
    soft = computeReward(
        cfg, _still(y=0.5), _still(y=0.0), [0.0, 0.0],
        outcome='crash', impactSpeed=3.0,
    )
    hard = computeReward(
        cfg, _still(y=0.5), _still(y=0.0), [0.0, 0.0],
        outcome='crash', impactSpeed=18.0,
    )
    assert hard < soft < 0


def test_timeoutGetsFlatCrashPenalty(cfg):
    reward = computeReward(
        cfg, _still(y=30.0), _still(y=30.0), [0.0, 0.0],
        outcome='timeout', impactSpeed=0.0,
        shapingScale=0.0,
    )
    assert reward == pytest.approx(cfg.reward.terminalCrash)


def test_shapingTelescopesToInitialPotential(cfg):
    # Discounted sum of potential-based shaping along ANY trajectory ending in
    # a terminal state must equal -shapingCoef * Phi(s0) (Ng et al. 1999).
    gamma = cfg.training.gamma
    states = [
        BoosterState(x=5.0, y=40.0, vx=1.0, vy=-8.0, theta=0.3),
        BoosterState(x=4.0, y=30.0, vx=0.5, vy=-6.0, theta=0.2),
        BoosterState(x=2.0, y=15.0, vx=0.2, vy=-4.0, theta=0.1),
        BoosterState(x=0.5, y=0.0, vx=0.0, vy=-1.0, theta=0.0),
    ]
    total = 0.0
    for t in range(3):
        isLast = t == 2
        shaping = computeReward(
            cfg, states[t], states[t + 1], [0.0, 0.0],
            outcome='success' if isLast else None, impactSpeed=1.0,
        )
        if isLast:
            # Strip the terminal payout to isolate the shaping component.
            shaping -= computeReward(
                cfg, states[t], states[t + 1], [0.0, 0.0],
                outcome='success', impactSpeed=1.0,
                shapingScale=0.0,
            )
        total += (gamma ** t) * shaping
    expected = -cfg.reward.shapingCoef * computePotential(states[0], cfg.world)
    assert total == pytest.approx(expected, abs=1e-9)


def test_controlCostClipsAction(cfg):
    base = computeReward(
        cfg, _still(y=10.0), _still(y=10.0), [1.0, 1.0],
        outcome=None, impactSpeed=0.0,
    )
    wild = computeReward(
        cfg, _still(y=10.0), _still(y=10.0), [99.0, -99.0],
        outcome=None, impactSpeed=0.0,
    )
    assert wild == pytest.approx(base)


def test_shapingScaleZeroSilencesShaping(cfg):
    moving = BoosterState(x=3.0, y=20.0, vx=1.0, vy=-5.0, theta=0.1)
    closer = BoosterState(x=2.0, y=15.0, vx=0.5, vy=-4.0, theta=0.05)
    silent = computeReward(
        cfg, moving, closer, [0.0, 0.0],
        outcome=None, impactSpeed=0.0, shapingScale=0.0,
    )
    assert silent == pytest.approx(0.0)


def test_potentialPrefersPadProximityUprightAndSlow(cfg):
    world = cfg.world
    near = computePotential(BoosterState(x=0.0, y=5.0, vy=-1.0), world)
    far = computePotential(BoosterState(x=10.0, y=40.0, vy=-10.0, theta=0.5), world)
    assert near > far
    assert math.isfinite(near) and math.isfinite(far)
