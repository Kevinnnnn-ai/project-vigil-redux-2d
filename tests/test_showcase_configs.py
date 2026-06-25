# tests/test_showcase_configs.py
"""Showcase configs MUST share the current world hash (so every retrained model
co-views) and MUST faithfully encode each milestone's reward/curriculum deltas."""
import os
import sys

import yaml

SHOWCASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp', 'showcase'))
sys.path.insert(0, SHOWCASE_DIR)

from src.config.loader import loadConfig
import gen_configs
from milestones import MILESTONES


def _baseRaw():
    with open('config.yaml', 'r', encoding='utf-8') as a:
        return yaml.safe_load(a)


def test_everyMilestoneSharesCurrentWorldHash(tmp_path):
    baseHash = loadConfig('config.yaml').computeWorldHash()
    baseRaw = _baseRaw()
    for milestone in MILESTONES:
        raw = gen_configs.buildConfigDict(milestone, baseRaw)
        path = tmp_path / milestone['file']
        with open(path, 'w', encoding='utf-8') as a:
            yaml.safe_dump(raw, a, sort_keys=False, default_flow_style=None)
        cfg = loadConfig(str(path))
        assert cfg.computeWorldHash() == baseHash, milestone['name']


def test_milestoneDeltasApplied():
    baseRaw = _baseRaw()
    built = {m['name']: gen_configs.buildConfigDict(m, baseRaw) for m in MILESTONES}
    names1 = [s['name'] for s in built['m1-original-shaping']['curriculum']['stages']]
    assert names1 == ['hop', 'drop', 'full']
    names2 = [s['name'] for s in built['m2-walls-touchdown']['curriculum']['stages']]
    assert 'glide' not in names2 and 'touchdown' in names2
    assert built['m6-anneal-none']['reward']['shapingAnneal'] == 'none'
    assert built['m5-run2']['reward']['shapingAnneal'] == 'linear'
    full4 = [s for s in built['m4-suicide-run1']['curriculum']['stages'] if s['name'] == 'full'][0]
    assert list(full4['altitude']) == [40.0, 52.0]
    assert built['m1-original-shaping']['training']['entCoef'] == 0.01
    assert built['m4-suicide-run1']['training']['totalIters'] == 300


def test_fastModeCapsItersAndSeeds():
    baseRaw = _baseRaw()
    raw = gen_configs.buildConfigDict(MILESTONES[0], baseRaw, isFast=True)
    assert raw['training']['totalIters'] == gen_configs.FAST_ITERS
    assert raw['training']['evalSeeds'] == list(gen_configs.FAST_SEEDS)


def test_committedConfigsExistAndMatchHash():
    """Drift guard on the COMMITTED artifacts: editing config.yaml's world without
    regenerating tmp/configs/ fails here."""
    baseHash = loadConfig('config.yaml').computeWorldHash()
    for milestone in MILESTONES:
        path = os.path.join('tmp', 'configs', milestone['file'])
        assert os.path.exists(path), f'missing {path} — run gen_configs.py'
        assert loadConfig(path).computeWorldHash() == baseHash, milestone['name']
