# tests/test_episode.py
import math

import numpy as np
import pytest

from src.config.loader import loadConfig, CurriculumStage
from src.env.episode import LandingEnv
from src.env.physics import BoosterState
from src.env.spaces import OBS_DIM


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


_TIGHT_STAGE = CurriculumStage(
    name='tight',
    altitude=(15.0, 18.0),
    xOffset=(-3.0, 3.0),
    vx=(-1.0, 1.0),
    vy=(-3.0, -1.0),
    tilt=(-0.1, 0.1),
    omega=(-0.05, 0.05),
)


def test_resetSamplesWithinStageRanges(cfg):
    env = LandingEnv(cfg, stage=_TIGHT_STAGE)
    rng = np.random.default_rng(7)
    for _ in range(100):
        obs = env.reset(rng)
        assert obs.shape == (OBS_DIM,)
        state = env.state
        assert 15.0 <= state.y <= 18.0
        assert -3.0 <= state.x <= 3.0
        assert -1.0 <= state.vx <= 1.0
        assert -3.0 <= state.vy <= -1.0
        assert -0.1 <= state.theta <= 0.1
        assert -0.05 <= state.omega <= 0.05
        assert state.fuel == 1.0
        assert env.t == 0


def test_defaultStageIsLastCurriculumStage(cfg):
    env = LandingEnv(cfg)
    assert env.stage is cfg.curriculum.stages[-1]


# @TAG[gentle-touchdown]: success path — body placed so the toe is just above
# ground; slow descent, upright, on-pad. The Pymunk solver lets it settle on its
# legs; once it comes to rest (REST_SPEED/REST_OMEGA) the verdict is 'success'.
def test_gentleTouchdownOnPadIsSuccess(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    # Just above the pad (toe near ground), descending slowly and upright.
    env.state = BoosterState(x=0.5, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.6, theta=0.0, omega=0.0, fuel=0.5)
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['outcome'] == 'success'
    assert reward > 0


def test_fastTouchdownIsCrash(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.2, vx=0.0, vy=-12.0, theta=0.0, omega=0.0, fuel=0.5)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated
    assert info['outcome'] == 'crash'
    assert reward < 0


def test_touchdownOffPadIsCrash(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    offPad = cfg.world.padWidth / 2 + 2.0
    env.state = BoosterState(x=offPad, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.5, theta=0.0, omega=0.0, fuel=0.5)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated
    assert info['outcome'] == 'crash'


# @TAG[settle-topple]: topple tests — a tilted or spinning booster that falls past
# the balance point physically TOPPLES under the Pymunk solver (it ends lying down,
# |theta| >> the stand threshold) and is classified a crash at rest.
def test_tiltedPastBalanceTopplesDuringSettle(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    tilt = math.atan2(cfg.world.legSpan, cfg.world.bodyHalfLen) + 0.25
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.5, theta=tilt, omega=0.0, fuel=0.5)
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated
    assert info['outcome'] == 'crash'


def test_spinTowardToeTopplesDuringSettle(cfg):
    # @TAG[substeps]: a gentle, upright contact with enough angular momentum
    # physically topples past the balance threshold under the sub-stepped Pymunk
    # solver. The stiffer/sub-stepped contact damps spin more, so it takes a
    # larger omega to topple than the pre-substep model: omega=8.0 reliably
    # produces theta > standTilt at rest (omega~5 now self-rights). Verified
    # empirically; this asserts a TRUE physical topple, not an extrapolation.
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.5, theta=0.0, omega=8.0, fuel=0.5)
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated
    assert info['outcome'] == 'crash'


def test_fastContactCrashesWithoutSettling(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.2, vx=0.0, vy=-12.0, theta=0.0, omega=0.0, fuel=0.5)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated
    assert info['outcome'] == 'crash'
    assert info['impactSpeed'] == pytest.approx(12.0, abs=0.5)


def test_contactTriggersOnToeNotBase(cfg):
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    # Body base is at legDrop + 1.0, so the toe (legDrop below base) is at ~1.0 m
    # above ground — too high to trigger contact in one step at vy=-0.4.
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 1.0, vx=0.0, vy=-0.4, theta=0.0, omega=0.0, fuel=0.5)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert not terminated and not truncated
    assert info['outcome'] is None


def test_fastTouchdownStillCrashDespiteFloorClampingVelocity(cfg):
    # Regression: the physical floor clamp zeroes the post-step downward vy, but
    # the impact verdict reads the APPROACH speed (prevState velocity), so a fast
    # descent is still classified a crash — not a (clamped, vy~0) "gentle" landing.
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.2, vx=0.0, vy=-12.0, theta=0.0, omega=0.0, fuel=0.5)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated
    assert info['outcome'] == 'crash'
    assert info['impactSpeed'] == pytest.approx(12.0, abs=0.5)   # approach speed, not ~0


def test_sideWallDoesNotEndEpisode(cfg):
    # @TAG[pymunk-wall-episode]: Pymunk walls are inelastic segments without CCD.
    # At realistic speeds the contact solver confines the booster but allows up to
    # ~vx*dt penetration before pushing back. The episode does NOT terminate from
    # being at the wall (no out-of-bounds outcome), and the booster reverses within
    # a few steps. Assert: not terminated, and |x| stays within width/2 + wallEps.
    # vx=10 -> max penetration ~0.5 m (= 10 * dt = 10 * 0.05), use wallEps=0.5.
    wallEps = 0.5
    halfW = cfg.world.width / 2
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(
        x=halfW - 0.1, y=30.0, vx=10.0, vy=0.0,
        theta=0.0, omega=0.0, fuel=0.5,
    )
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert not terminated and not truncated
    assert info['outcome'] is None
    # Physical wall confinement: the booster may penetrate by at most vx*dt.
    assert env.state.x <= halfW + wallEps, f'Wall penetration too deep: x={env.state.x:.3f}'


def test_timeoutTruncates(cfg, tmp_path):
    # Tiny maxSteps via a config override file so truncation triggers quickly.
    text = (tmp_path / 'config.yaml')
    text.write_text('world: {maxSteps: 3}\n', encoding='utf-8')
    smallCfg = loadConfig(str(text))
    env = LandingEnv(smallCfg, stage=_TIGHT_STAGE)
    env.reset(np.random.default_rng(0))
    # Full throttle keeps it aloft well past 3 steps from the 15-18 m spawn,
    # so the episode ends on the maxSteps=3 timeout, not a touchdown.
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert truncated and not terminated
    assert info['outcome'] == 'timeout'
    assert reward < 0


# <agent_context>
#   [ARCH]: End-to-end integration test for suicideBurn engine mode. Validates the
#           full episode loop: reset -> toggle ignition -> forced cutoff -> terminal.
#   [GOTCHA]: Uses dataclasses.replace (not direct field mutation) to avoid corrupting
#             the shared cfg fixture — Config and WorldConfig are frozen dataclasses.
#   [GOTCHA]: engineTransitions is tracked on BoosterState (not WorldConfig). The
#             suicideBurn physics branch in stepPhysics() increments it on each
#             on->off or off->on change. The test asserts <= 2 because suicideBurn
#             permits at most one ignition and one cutoff.
#   [GOTCHA]: The step loop uses `burnCfg.world.maxSteps + 1` as the safety ceiling.
#             Real termination comes from the physics + episode classify() path
#             (touchdown or timeout), NOT from the test's own loop counter.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Do NOT assert obs.shape == (9,). The 10-D obs contract is INTENTIONAL
#               (index 9 = ignitionsRemaining). Any test asserting shape (9,) is stale.
#   [CRITICAL]: Do NOT modify production code (physics.py, spaces.py, episode.py)
#               to satisfy this test. If this test fails, the production code has a
#               genuine regression.
#   [VALIDATION]: python -m pytest tests/test_episode.py::test_suicideBurnEpisodeRunsEndToEnd -v
# </agent_guardrail>
def test_suicideBurnEpisodeRunsEndToEnd():
    import dataclasses
    cfg = loadConfig('config.yaml')
    burnCfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, engineMode='suicideBurn'))
    env = LandingEnv(burnCfg)
    obs = env.reset(np.random.default_rng(0))
    assert obs.shape == (10,)
    terminated = truncated = False
    steps = 0
    # fire once, then cut, then coast — exercise the toggle path to a terminal
    while not (terminated or truncated) and steps < burnCfg.world.maxSteps + 1:
        engineCmd = 1.0 if steps < 30 else 0.0
        obs, reward, terminated, truncated, info = env.step([engineCmd, 0.0])
        steps += 1
    assert terminated or truncated
    assert env.state.engineTransitions <= 2
    assert obs.shape == (10,)


# @TAG[second-leg-stand]: regression test — a near-upright, gentle landing
# must rock down onto its second leg and score 'success', NOT wrap through
# the ground to a crash (the bug this test pins).
def test_nearUprightSettlesOntoBothLegs(cfg):
    # A near-upright, slow, low-spin touchdown must rock down onto its second leg
    # and STAND — not rotate the free toe through the ground and wrap to a crash.
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(x=0.0, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.7, theta=0.013, omega=-0.01, fuel=0.5)
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['outcome'] == 'success'


def test_stepDeterministic(cfg):
    a = LandingEnv(cfg, stage=_TIGHT_STAGE)
    b = LandingEnv(cfg, stage=_TIGHT_STAGE)
    a.reset(np.random.default_rng(3))
    b.reset(np.random.default_rng(3))
    for _ in range(20):
        resA = a.step([0.7, 0.1])
        resB = b.step([0.7, 0.1])
        np.testing.assert_array_equal(resA[0], resB[0])
        assert resA[1:] == resB[1:]
        if resA[2] or resA[3]:
            break


def test_settlingDeterministic(cfg):
    # @TAG[pymunk-determinism]: Two envs hand-placed at the same gentle contact
    # state with the same action sequence must produce IDENTICAL obs/reward/
    # terminated/truncated through the full episode, including ground contact and
    # settling. The Pymunk solver is deterministic given identical initial state.
    # We no longer check state.contact (removed from BoosterState); instead we
    # verify the episode runs to a terminal state and both envs agree on every step.
    import copy
    a = LandingEnv(cfg)
    b = LandingEnv(cfg)
    a.reset(np.random.default_rng(0))
    b.reset(np.random.default_rng(0))
    start = BoosterState(
        x=0.3, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.5,
        theta=0.05, omega=0.2, fuel=0.4,
    )
    a.state = copy.deepcopy(start)
    b.state = copy.deepcopy(start)
    reached_terminal = False
    for _ in range(cfg.world.settleStepCap + 5):
        resA = a.step([0.0, 0.0])
        resB = b.step([0.0, 0.0])
        np.testing.assert_array_equal(resA[0], resB[0])
        assert resA[1:] == resB[1:]
        if resA[2] or resA[3]:
            reached_terminal = True
            break
    assert reached_terminal   # the trajectory actually completed (success or crash)
