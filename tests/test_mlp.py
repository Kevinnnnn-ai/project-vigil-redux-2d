# tests/test_mlp.py
import numpy as np
import pytest
import torch

from src.env.spaces import OBS_DIM, ACTION_DIM
from src.agents.mlp import MLPPolicy


@pytest.fixture
def policy():
    torch.manual_seed(0)
    return MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(32, 32))


def test_actReturnsEnvSpaceAction(policy):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = policy.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert 0.0 <= action[0] <= 1.0          # throttle in env space
    assert -1.0 <= action[1] <= 1.0


def test_actIsDeterministic(policy):
    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    np.testing.assert_array_equal(policy.act(obs), policy.act(obs))


def test_sampleShapesAndFiniteLogp(policy):
    obs = torch.zeros((16, OBS_DIM))
    a, u, logp, value = policy.sample(obs)
    assert a.shape == (16, ACTION_DIM)
    assert u.shape == (16, ACTION_DIM)
    assert logp.shape == (16,)
    assert value.shape == (16,)
    assert torch.all(a > -1.0) and torch.all(a < 1.0)   # tanh space
    assert torch.isfinite(logp).all()


def test_evaluateActionsMatchesSampleLogp(policy):
    obs = torch.zeros((8, OBS_DIM))
    _, u, logp, _ = policy.sample(obs)
    logp2, entropy, value = policy.evaluateActions(obs, u)
    torch.testing.assert_close(logp, logp2)
    assert torch.isfinite(entropy).all()


def test_saveLoadRoundTrip(policy, tmp_path):
    path = str(tmp_path / 'ckpt.pt')
    policy.save(path, worldHash='abc123', stageName='hop')
    loaded, meta = MLPPolicy.load(path)
    assert meta == {'worldHash': 'abc123', 'stageName': 'hop'}
    obs = np.random.default_rng(2).normal(size=OBS_DIM).astype(np.float32)
    np.testing.assert_allclose(policy.act(obs), loaded.act(obs), atol=1e-6)
