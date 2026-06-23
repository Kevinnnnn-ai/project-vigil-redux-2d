# tests/test_parallel.py
import os
import textwrap

import pytest

from src.config.loader import loadConfig
from src.train.parallel import SeedTask, resolveSeedWorkers, runSeeds


def _tinyConfig(tmp_path, seedWorkers='auto', evalSeeds=(0, 1)):
    """A minimal single-stage config that trains in a blink (few envs/iters)."""
    seeds = ', '.join(str(s) for s in evalSeeds)
    path = tmp_path / 'config.yaml'
    path.write_text(textwrap.dedent(f'''
        training:
          numEnvs: 2
          rolloutSteps: 32
          epochs: 1
          minibatchSize: 32
          evalEpisodes: 2
          evalEvery: 1
          totalIters: 2
          hidden: [8]
          evalSeeds: [{seeds}]
          seedWorkers: {seedWorkers}
        curriculum:
          promoteAt: 0.8
          stages:
            - name: only
              altitude: [1.0, 2.0]
              xOffset: [-0.5, 0.5]
              vx: [-0.1, 0.1]
              vy: [-1.0, -0.2]
              tilt: [-0.02, 0.02]
              omega: [-0.02, 0.02]
    '''), encoding='utf-8')
    return loadConfig(str(path)), str(path)


def _tasksFor(cfg, configPath, tmp_path, stageName='only'):
    """One SeedTask per evalSeed pointing at scratch checkpoint/csv paths."""
    tasks = []
    for seed in cfg.training.evalSeeds:
        tasks.append(SeedTask(
            configPath=configPath,
            seed=seed,
            savePath=str(tmp_path / f'seed{seed}.pt'),
            csvPath=str(tmp_path / f'seed{seed}.csv'),
            stageName=stageName,        # single-stage path (trainLanding)
            modelName='lux',
            envName='baseline',
        ))
    return tasks


# ── resolveSeedWorkers ──────────────────────────────────────────────────────

def test_resolveSerialFlagForcesOne(tmp_path):
    cfg, _ = _tinyConfig(tmp_path, seedWorkers='auto', evalSeeds=(0, 1, 2))
    assert resolveSeedWorkers(cfg, serial=True) == 1


def test_resolveAutoCapsAtSeedCountAndCpu(tmp_path):
    cfg, _ = _tinyConfig(tmp_path, seedWorkers='auto', evalSeeds=(0, 1))
    expected = min(2, os.cpu_count() or 1)
    assert resolveSeedWorkers(cfg, serial=False) == expected


def test_resolveIntClampsToSeedCount(tmp_path):
    cfg, _ = _tinyConfig(tmp_path, seedWorkers=99, evalSeeds=(0, 1))
    assert resolveSeedWorkers(cfg, serial=False) == 2     # clamped to 2 seeds


def test_resolveIntBelowSeedCountKept(tmp_path):
    cfg, _ = _tinyConfig(tmp_path, seedWorkers=1, evalSeeds=(0, 1, 2))
    assert resolveSeedWorkers(cfg, serial=False) == 1


# ── runSeeds (serial path; deterministic, no pool) ──────────────────────────

def test_serialReturnsOneResultPerSeedInSeedOrder(tmp_path):
    cfg, configPath = _tinyConfig(tmp_path, evalSeeds=(2, 0, 1))
    tasks = _tasksFor(cfg, configPath, tmp_path)
    results = runSeeds(tasks, maxWorkers=1)
    assert [r.seed for r in results] == [0, 1, 2]          # sorted by seed
    for r in results:
        assert 0.0 <= r.bestRate <= 1.0
        assert os.path.exists(r.savePath)


def test_workerExceptionPropagates(tmp_path):
    cfg, configPath = _tinyConfig(tmp_path)
    tasks = _tasksFor(cfg, configPath, tmp_path)
    tasks[0].stageName = 'does-not-exist'                  # stageByName raises
    with pytest.raises(SystemExit):
        runSeeds(tasks, maxWorkers=1)


# ── parity: parallel result == serial result (the core invariant) ───────────

def test_parallelMatchesSerialPerSeed(tmp_path):
    """A seed's bestRate is identical whether run in-process or in a child
    process — seeds are independent and self-seed their own RNG/torch."""
    serialDir = tmp_path / 'serial'
    parDir = tmp_path / 'par'
    serialDir.mkdir()
    parDir.mkdir()
    cfgS, pathS = _tinyConfig(serialDir, evalSeeds=(0, 1))
    cfgP, pathP = _tinyConfig(parDir, evalSeeds=(0, 1))
    serial = runSeeds(_tasksFor(cfgS, pathS, serialDir), maxWorkers=1)
    parallel = runSeeds(_tasksFor(cfgP, pathP, parDir), maxWorkers=2)
    serialBySeed = {r.seed: r.bestRate for r in serial}
    parallelBySeed = {r.seed: r.bestRate for r in parallel}
    assert serialBySeed == parallelBySeed
