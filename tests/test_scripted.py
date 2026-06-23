# tests/test_scripted.py
import numpy as np
import pytest

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.agents.scripted import PdPilot


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


def _successRate(cfg, stage, episodes=25, seed=0):
    env = LandingEnv(cfg, stage=stage)
    pilot = PdPilot(cfg.world)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(episodes):
        obs = env.reset(rng)
        pilot.reset()
        while True:
            action = pilot.act(obs)
            assert 0.0 <= action[0] <= 1.0
            assert -1.0 <= action[1] <= 1.0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info['outcome'] == 'success':
                    wins += 1
                break
    return wins / episodes


def test_pdPilotLandsHopStage(cfg):
    # The PD pilot proves the environment is solvable: >= 80% from low drops.
    rate = _successRate(cfg, cfg.curriculum.stages[0])
    assert rate >= 0.8, f'hop success rate {rate}'


def test_pdPilotManagesDropStage(cfg):
    # A dumb controller should still cope sometimes from mid difficulty;
    # RL is expected to beat this handily.
    rate = _successRate(cfg, cfg.curriculum.stages[1])
    assert rate >= 0.4, f'drop success rate {rate}'
