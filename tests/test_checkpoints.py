# tests/test_checkpoints.py
import os

import numpy as np
import pytest
import torch

from src.env.spaces import OBS_DIM, ACTION_DIM
from src.agents.mlp import MLPPolicy
from src.agents.checkpoints import resolveModelPath, loadCheckpoint


@pytest.fixture
def modelsDir(tmp_path):
    torch.manual_seed(0)
    policy = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(16,))
    policy.save(str(tmp_path / 'best.pt'), worldHash='hash-a', stageName='full')
    policy.save(str(tmp_path / 'seed0.pt'), worldHash='hash-a', stageName='hop')
    return str(tmp_path)


def test_resolveNamedModels(modelsDir):
    assert resolveModelPath(modelsDir, 'best') == os.path.join(modelsDir, 'best.pt')
    assert resolveModelPath(modelsDir, 'seed0') == os.path.join(modelsDir, 'seed0.pt')


def test_resolveWithinNestedModelDir(tmp_path):
    luxDir = tmp_path / 'lux'
    luxDir.mkdir()
    torch.manual_seed(0)
    policy = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(16,))
    policy.save(str(luxDir / 'best.pt'), worldHash='hash-a', stageName='full')
    assert resolveModelPath(str(luxDir), 'best') == os.path.join(str(luxDir), 'best.pt')


def test_resolveWithinModelEnvSubdir(tmp_path):
    envDir = tmp_path / 'lux' / 'baseline'
    envDir.mkdir(parents=True)
    torch.manual_seed(0)
    policy = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=(16,))
    policy.save(str(envDir / 'best.pt'), worldHash='hash-a', stageName='full')
    assert resolveModelPath(str(envDir), 'best') == os.path.join(str(envDir), 'best.pt')


def test_resolveExplicitPathPassesThrough(modelsDir):
    path = os.path.join(modelsDir, 'best.pt')
    assert resolveModelPath(modelsDir, path) == path


def test_resolveMissingListsWhatExists(modelsDir):
    with pytest.raises(FileNotFoundError) as err:
        resolveModelPath(modelsDir, 'seed9')
    assert 'best.pt' in str(err.value)      # tells the user what IS there


def test_loadCheckpointHappyPath(modelsDir):
    policy, meta = loadCheckpoint(os.path.join(modelsDir, 'best.pt'), 'hash-a')
    assert meta['stageName'] == 'full'
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = policy.act(obs)
    assert action.shape == (ACTION_DIM,)


def test_loadCheckpointHashMismatchNamesBothHashes(modelsDir):
    with pytest.raises(ValueError) as err:
        loadCheckpoint(os.path.join(modelsDir, 'best.pt'), 'hash-B')
    message = str(err.value)
    assert 'hash-a' in message and 'hash-B' in message