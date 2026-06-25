# tests/test_showcase_train_all.py
"""train_all builds the right training command and records a sane registry without
running PPO (the actual training is a documented manual step)."""
import os
import sys

SHOWCASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp', 'showcase'))
sys.path.insert(0, SHOWCASE_DIR)

import train_all
from milestones import MILESTONES


def test_buildTrainCommandExactArgs():
    m = [x for x in MILESTONES if x['name'] == 'm4-suicide-run1'][0]
    cmd = train_all.buildTrainCommand(m)
    assert cmd[1:] == ['-m', 'scripts.train', '--config',
                       os.path.join('tmp', 'configs', 'm4-suicide-run1.yaml'),
                       '--run', '7004']
    assert train_all.buildTrainCommand(m, isSerial=True)[-1] == '--serial'


def test_worldHashOkForCommittedConfig():
    m = [x for x in MILESTONES if x['name'] == 'm6-anneal-none'][0]
    assert train_all._worldHashOk(m) is True


def test_bestSuccessOfFiltersSentinelsAndHandlesEmpty(monkeypatch):
    monkeypatch.setattr(train_all, 'readSeedHistories', lambda logsDir: {
        0: [{'iter': 0, 'successRate': -1.0}, {'iter': 5, 'successRate': 0.4}],
        1: [{'iter': 0, 'successRate': -1.0}, {'iter': 5, 'successRate': 0.7}],
    })
    assert train_all._bestSuccessOf(7001) == 0.7
    monkeypatch.setattr(train_all, 'readSeedHistories', lambda logsDir: {})
    assert train_all._bestSuccessOf(7001) is None
    monkeypatch.setattr(train_all, 'readSeedHistories', lambda logsDir: {
        0: [{'iter': 0, 'successRate': -1.0}],
    })
    assert train_all._bestSuccessOf(7001) is None


def test_registryMdRendersAllMilestones(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, 'REGISTRY_MD', str(tmp_path / 'REGISTRY.md'))
    train_all._writeRegistryMd({'m6-anneal-none': {'trainedAt': '2026-06-25T00:00:00',
                                                   'bestSuccess': 0.0, 'worldHashOk': True}})
    text = (tmp_path / 'REGISTRY.md').read_text()
    for m in MILESTONES:
        assert m['name'] in text
    assert '| m6-anneal-none |' in text
