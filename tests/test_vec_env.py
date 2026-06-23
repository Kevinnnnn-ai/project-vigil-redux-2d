# tests/test_vec_env.py
import numpy as np
import pytest

from src.config.loader import loadConfig, CurriculumStage
from src.env.spaces import OBS_DIM
from src.train.vec_env import VecLandingEnv


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


def test_resetAndStepShapes(cfg):
    vec = VecLandingEnv(cfg, numEnvs=4, seed=0)
    obs = vec.reset()
    assert obs.shape == (4, OBS_DIM)
    actions = np.tile([0.5, 0.0], (4, 1))
    obs, rewards, terminated, truncated, outcomes = vec.step(actions)
    assert obs.shape == (4, OBS_DIM)
    assert rewards.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert len(outcomes) == 4
    assert all(o is None for o in outcomes)   # nobody lands in one step


def test_sameSeedReproduces(cfg):
    a = VecLandingEnv(cfg, numEnvs=3, seed=42)
    b = VecLandingEnv(cfg, numEnvs=3, seed=42)
    np.testing.assert_array_equal(a.reset(), b.reset())
    actions = np.tile([0.7, 0.1], (3, 1))
    for _ in range(50):
        oa = a.step(actions)
        ob = b.step(actions)
        np.testing.assert_array_equal(oa[0], ob[0])
        np.testing.assert_array_equal(oa[1], ob[1])


def test_autoResetGivesFreshEpisode(cfg):
    # Free-fall from a low stage forces quick terminations.
    lowStage = CurriculumStage(
        name='low', altitude=(0.5, 1.0), xOffset=(0.0, 0.0),
        vx=(0.0, 0.0), vy=(-5.0, -5.0), tilt=(0.0, 0.0), omega=(0.0, 0.0),
    )
    vec = VecLandingEnv(cfg, numEnvs=2, seed=1, stage=lowStage)
    vec.reset()
    actions = np.tile([0.0, 0.0], (2, 1))
    sawDone = False
    for _ in range(20):
        obs, rewards, terminated, truncated, outcomes = vec.step(actions)
        for i in range(2):
            if terminated[i] or truncated[i]:
                sawDone = True
                assert outcomes[i] is not None
                assert vec.envs[i].t == 0                  # auto-reset happened
                assert vec.envs[i].state.y >= 0.5          # fresh spawn altitude
    assert sawDone


def test_setStageAppliesOnNextReset(cfg):
    high = CurriculumStage(
        name='high', altitude=(50.0, 55.0), xOffset=(0.0, 0.0),
        vx=(0.0, 0.0), vy=(-1.0, -1.0), tilt=(0.0, 0.0), omega=(0.0, 0.0),
    )
    vec = VecLandingEnv(cfg, numEnvs=2, seed=2, stage=cfg.curriculum.stages[0])
    vec.reset()
    assert all(env.state.y <= 12.0 for env in vec.envs)    # hop altitudes
    vec.setStage(high)
    vec.reset()
    assert all(env.state.y >= 50.0 for env in vec.envs)


def test_setShapingScaleFansOut(cfg):
    vec = VecLandingEnv(cfg, numEnvs=3, seed=3)
    vec.setShapingScale(0.25)
    assert all(env.shapingScale == 0.25 for env in vec.envs)
