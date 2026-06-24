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
    # PdPilot is a weak binary suicide-burn baseline: it lands a fraction of the
    # touchdown spawns, not all of them. This test's real job is exercising
    # runEvaluation end-to-end; the success floor just confirms PdPilot genuinely
    # lands sometimes (measured ~0.4 at this seed).
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    result = runEvaluation(env, PdPilot(cfg.world), episodes=10, rng=np.random.default_rng(0))
    assert result['episodes'] == 10
    assert result['successRate'] >= 0.2
    assert sum(result['outcomes'].values()) == 10
    assert result['meanImpactSpeed'] <= cfg.world.maxLandingSpeed
    assert result['meanSteps'] > 0


def test_evaluationDeterministicAcrossSameSeed(cfg):
    # Same seed + a fresh env -> bit-identical evaluation. Each run gets its OWN
    # LandingEnv on purpose: the env holds a persistent Pymunk Space whose solver/
    # contact warm-start caches are NOT cleared by reset() (it only repositions the
    # body), so REUSING one env across two evaluations couples them through residual
    # solver state and drifts continuous metrics (e.g. meanImpactSpeed) at the ~1e-6
    # level. Determinism is a per-fresh-env property; train-time promotion consumes
    # only the discrete success rate, which is robust to that carry-over.
    pilot = PdPilot(cfg.world)
    stage = cfg.curriculum.stages[1]
    a = runEvaluation(LandingEnv(cfg, stage=stage), pilot, episodes=5, rng=np.random.default_rng(7))
    b = runEvaluation(LandingEnv(cfg, stage=stage), pilot, episodes=5, rng=np.random.default_rng(7))
    assert a == b
