# tests/test_ppo.py
import numpy as np
import pytest
import torch

from src.config.loader import loadConfig
from src.env.spaces import OBS_DIM, ACTION_DIM
from src.agents.mlp import MLPPolicy
from src.train.ppo import ppoUpdate, explainedVariance


def test_explainedVarianceCases():
    returns = np.array([1.0, 2.0, 3.0])
    assert explainedVariance(returns, returns) == pytest.approx(1.0)
    assert explainedVariance(returns, returns.mean() * np.ones(3)) == pytest.approx(0.0)
    assert explainedVariance(np.ones(3), np.array([0.0, 5.0, -5.0])) == 0.0


def test_ppoUpdateChangesParamsAndReportsFiniteStats():
    cfg = loadConfig('config.yaml')
    torch.manual_seed(0)
    policy = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(32,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    n = 128
    obs = torch.randn(n, OBS_DIM)
    with torch.no_grad():
        _, u, logp, _ = policy.sample(obs)
    advantages = torch.randn(n)
    returns = torch.randn(n)

    before = [p.detach().clone() for p in policy.parameters()]
    stats = ppoUpdate(policy, optimizer, obs, u, logp, advantages, returns, cfg.training)
    after = list(policy.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    for key in ('policyLoss', 'valueLoss', 'entropy', 'approxKl', 'clipFrac'):
        assert np.isfinite(stats[key]), key
