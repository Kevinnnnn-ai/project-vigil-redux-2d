# tests/test_loop.py
import os
import textwrap

import numpy as np
import pytest

from src.config.loader import loadConfig
from src.train.loop import trainLanding, shapingScaleFor, evaluateSuccessRate
from src.env.episode import LandingEnv
from src.agents.scripted import PdPilot


def _tinyConfig(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(textwrap.dedent('''
        training:
          numEnvs: 2
          rolloutSteps: 64
          epochs: 2
          minibatchSize: 32
          evalEpisodes: 2
          totalIters: 2
          hidden: [16]
    '''), encoding='utf-8')
    return loadConfig(str(path))


def test_shapingScaleSchedules(tmp_path):
    cfg = _tinyConfig(tmp_path)
    assert shapingScaleFor(cfg, 0, 10) == pytest.approx(1.0)
    assert shapingScaleFor(cfg, 5, 10) == pytest.approx(0.5)


def test_shapingScaleForNoneIsConstant(tmp_path):
    # shapingAnneal: none -> the anneal factor is a constant 1.0 at every iter,
    # so the PBRS shaping signal stays fully on through the late curriculum
    # stages (the linear branch decays it to ~0 over totalIters).
    path = tmp_path / 'config.yaml'
    path.write_text(textwrap.dedent('''
        reward:
          shapingAnneal: none
    '''), encoding='utf-8')
    cfg = loadConfig(str(path))
    assert cfg.reward.shapingAnneal == 'none'
    for it in (0, 5, 300, 599):
        assert shapingScaleFor(cfg, it, 600) == pytest.approx(1.0)


def test_evaluateSuccessRateWithPdPilot(tmp_path):
    # _tinyConfig only overrides `training:`, so `curriculum` falls back to the
    # dataclass default — a SINGLE 'full' stage (altitude 40-52, vy -12..-4). That
    # is the hardest spawn in the game, and the weak binary suicide-burn PdPilot
    # baseline cannot land it (measured 0.0). So this test exercises the
    # evaluateSuccessRate harness and asserts it returns a valid rate rather than a
    # positive floor PdPilot cannot reach on the full stage. (On the easy stages
    # PdPilot does land — see tests/test_scripted.py.)
    cfg = _tinyConfig(tmp_path)
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    rate = evaluateSuccessRate(env, PdPilot(cfg.world), 5, np.random.default_rng(0))
    assert 0.0 <= rate <= 1.0
    assert (rate * 5) == round(rate * 5)  # harness ran 5 episodes; rate must be a multiple of 1/5


def test_trainLandingSmoke(tmp_path):
    cfg = _tinyConfig(tmp_path)
    savePath = str(tmp_path / 'ckpt.pt')
    csvPath = str(tmp_path / 'metrics.csv')
    history = trainLanding(
        cfg, seed=0, savePath=savePath, csvPath=csvPath,
        stage=cfg.curriculum.stages[0],
    )
    assert len(history) == 2
    for record in history:
        for key in ('successRate', 'rolloutSuccess', 'explainedVariance', 'entropy'):
            assert key in record
    assert os.path.exists(savePath)        # best checkpoint written
    assert os.path.exists(csvPath)
    with open(csvPath, encoding='utf-8') as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) == 3                 # header + 2 iterations
