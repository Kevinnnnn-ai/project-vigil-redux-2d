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


def test_pdPilotLandsTouchdownStage(cfg):
    # PdPilot is now a WEAK binary suicide-burn baseline (single ignite + cut),
    # not the old continuous-throttle pilot. On the easiest stage (touchdown,
    # low + slow spawns) it lands a solid fraction of the time. Measured ~0.40 at
    # this seed; floor well below that so it is meaningful yet not flaky.
    rate = _successRate(cfg, cfg.curriculum.stages[0])
    assert rate >= 0.25, f'touchdown success rate {rate}'


def test_pdPilotManagesHopStage(cfg):
    # The hop stage spawns higher and faster, so the single-burn baseline lands
    # less often, but still SOMETIMES — proof it genuinely flies, not just rides
    # gentle spawns down. Measured ~0.20 at this seed; RL must beat this handily.
    rate = _successRate(cfg, cfg.curriculum.stages[1])
    assert rate >= 0.08, f'hop success rate {rate}'
