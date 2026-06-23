# tests/test_evaluate.py
import numpy as np
import pytest

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.agents.scripted import PdPilot
from src.runtime.evaluate import runEvaluation


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


def test_pdPilotEvaluationOnTouchdownStage(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    result = runEvaluation(env, PdPilot(cfg.world), episodes=10, rng=np.random.default_rng(0))
    assert result['episodes'] == 10
    assert result['successRate'] == 1.0
    assert sum(result['outcomes'].values()) == 10
    assert result['meanImpactSpeed'] <= cfg.world.maxLandingSpeed
    assert result['meanSteps'] > 0


def test_evaluationDeterministicAcrossSameSeed(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[1])
    pilot = PdPilot(cfg.world)
    a = runEvaluation(env, pilot, episodes=5, rng=np.random.default_rng(7))
    b = runEvaluation(env, pilot, episodes=5, rng=np.random.default_rng(7))
    assert a == b
