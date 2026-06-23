# tests/test_curriculum.py
import os
import textwrap

import pytest

from src.config.loader import loadConfig
from src.train.curriculum import trainCurriculum


def _tinyConfig(tmp_path, totalIters, evalEvery, promoteAt=0.8):
    path = tmp_path / 'config.yaml'
    path.write_text(textwrap.dedent(f'''
        training:
          numEnvs: 2
          rolloutSteps: 32
          epochs: 1
          minibatchSize: 32
          evalEpisodes: 2
          evalEvery: {evalEvery}
          totalIters: {totalIters}
          hidden: [8]
        curriculum:
          promoteAt: {promoteAt}
          stages:
            - name: a
              altitude: [1.0, 2.0]
              xOffset: [-0.5, 0.5]
              vx: [-0.1, 0.1]
              vy: [-1.0, -0.2]
              tilt: [-0.02, 0.02]
              omega: [-0.02, 0.02]
            - name: b
              altitude: [3.0, 5.0]
              xOffset: [-1.0, 1.0]
              vx: [-0.2, 0.2]
              vy: [-1.5, -0.5]
              tilt: [-0.03, 0.03]
              omega: [-0.03, 0.03]
            - name: c
              altitude: [6.0, 9.0]
              xOffset: [-2.0, 2.0]
              vx: [-0.3, 0.3]
              vy: [-2.0, -0.5]
              tilt: [-0.05, 0.05]
              omega: [-0.05, 0.05]
    '''), encoding='utf-8')
    return loadConfig(str(path))


def _scriptedEval(rates):
    """evaluateFn double: returns the next scripted rate per call."""
    rates = list(rates)

    def evaluateFn(env, policy, episodes, rng):
        return rates.pop(0)
    return evaluateFn


def test_promotesOnThresholdAndTracksStages(tmp_path):
    cfg = _tinyConfig(tmp_path, totalIters=6, evalEvery=2)
    savePath = str(tmp_path / 'out.pt')
    # Evals at iters 0,2,4,5(last): 0.9 promotes a->b, 0.5 stays, 0.9 promotes
    # b->c, 0.4 stays on final.
    history = trainCurriculum(
        cfg, seed=0, savePath=savePath,
        evaluateFn=_scriptedEval([0.9, 0.5, 0.9, 0.4]),
    )
    # Evals land on iters 0, 2, 4 and 5 (last). 0.9 at iter 0 promotes a->b
    # (stage b from iter 1), 0.5 at iter 2 stays, 0.9 at iter 4 promotes b->c.
    stages = [record['stage'] for record in history]
    assert stages == ['a', 'b', 'b', 'b', 'b', 'c']
    promotions = [record['iter'] for record in history if record.get('promoted')]
    assert promotions == [0, 4]
    assert os.path.exists(savePath)


def test_staysWhenBelowThreshold(tmp_path):
    cfg = _tinyConfig(tmp_path, totalIters=4, evalEvery=1)
    history = trainCurriculum(
        cfg, seed=0, savePath=str(tmp_path / 'out.pt'),
        evaluateFn=_scriptedEval([0.2, 0.3, 0.1, 0.2]),
    )
    assert all(record['stage'] == 'a' for record in history)


def test_savesBestOnFinalStage(tmp_path):
    cfg = _tinyConfig(tmp_path, totalIters=5, evalEvery=1)
    savePath = str(tmp_path / 'out.pt')
    history = trainCurriculum(
        cfg, seed=0, savePath=savePath,
        evaluateFn=_scriptedEval([0.9, 0.9, 0.3, 0.8, 0.6]),
    )
    # Promotions at iters 0 (a->b) and 1 (b->c); final-stage evals 0.3, 0.8, 0.6.
    from src.agents.mlp import MLPPolicy
    _, meta = MLPPolicy.load(savePath)
    assert meta['stageName'] == 'c'
    finalRates = [
        record['successRate'] for record in history
        if record['stage'] == 'c' and record.get('successRate', -1) >= 0
    ]
    assert max(finalRates) == 0.8


def test_csvHasStableHeader(tmp_path):
    cfg = _tinyConfig(tmp_path, totalIters=3, evalEvery=2)
    csvPath = str(tmp_path / 'metrics.csv')
    trainCurriculum(
        cfg, seed=0, savePath=str(tmp_path / 'out.pt'), csvPath=csvPath,
        evaluateFn=_scriptedEval([0.0, 0.0]),
    )
    with open(csvPath, encoding='utf-8') as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) == 4                      # header + 3 iters
    assert 'successRate' in lines[0] and 'stage' in lines[0]
