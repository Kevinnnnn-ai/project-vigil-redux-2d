# tests/test_rollout.py
import numpy as np
import pytest
import torch

from src.config.loader import loadConfig
from src.env.spaces import OBS_DIM, ACTION_DIM
from src.agents.mlp import MLPPolicy
from src.train.vec_env import VecLandingEnv
from src.train.rollout import computeGae, collectRollout, computeBatchAdvantages


def test_gaeHandComputedTerminalStep():
    # Single step ending the episode: A = r - V(s); bootstrap suppressed.
    adv, ret = computeGae(
        rewards=[2.0], values=[0.5], dones=[1.0],
        lastValue=99.0, gamma=0.9, lam=0.95,
    )
    assert adv[0] == pytest.approx(2.0 - 0.5)
    assert ret[0] == pytest.approx(2.0)


def test_gaeBootstrapsTailWhenNotDone():
    adv, ret = computeGae(
        rewards=[1.0], values=[0.5], dones=[0.0],
        lastValue=2.0, gamma=0.9, lam=0.95,
    )
    assert adv[0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)


def test_gaeThreeStepRecursion():
    gamma, lam = 0.9, 0.8
    rewards = np.array([1.0, 0.0, -1.0])
    values = np.array([0.2, 0.4, 0.6])
    dones = np.array([0.0, 0.0, 1.0])
    adv, ret = computeGae(rewards, values, dones, lastValue=5.0, gamma=gamma, lam=lam)
    d2 = -1.0 - 0.6                                  # terminal: no bootstrap
    d1 = 0.0 + gamma * 0.6 - 0.4
    d0 = 1.0 + gamma * 0.4 - 0.2
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-12)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-12)


def test_collectRolloutShapesAndDones():
    cfg = loadConfig('config.yaml')
    torch.manual_seed(0)
    vec = VecLandingEnv(cfg, numEnvs=2, seed=0, stage=cfg.curriculum.stages[0])
    learner = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(32,))
    batch, lastValues, outcomes = collectRollout(vec, learner, rolloutSteps=64)
    assert batch['obs'].shape == (64, 2, OBS_DIM)
    assert batch['u'].shape == (64, 2, ACTION_DIM)
    for key in ('logp', 'value', 'reward', 'done'):
        assert batch[key].shape == (64, 2)
    assert lastValues.shape == (2,)
    assert set(batch['done'].ravel()) <= {0.0, 1.0}
    # A random policy dropped from the hop stage finishes episodes within 64
    # steps — outcomes must tally those episodes.
    assert sum(outcomes.values()) == int(batch['done'].sum())
    adv, ret = computeBatchAdvantages(batch, lastValues, gamma=0.99, lam=0.95)
    assert adv.shape == (64, 2)
    assert np.isfinite(adv).all() and np.isfinite(ret).all()
