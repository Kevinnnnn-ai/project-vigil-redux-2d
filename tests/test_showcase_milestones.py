# tests/test_showcase_milestones.py
"""The milestone table is the single source of truth for the showcase; guard its
shape so gen_configs/train_all/gallery can trust it."""
import os
import sys

SHOWCASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp', 'showcase'))
sys.path.insert(0, SHOWCASE_DIR)

from milestones import MILESTONES, BASE_STAGES


def test_sixMilestonesWithUniqueRunsAndFiles():
    assert len(MILESTONES) == 6
    runs = [m['run'] for m in MILESTONES]
    files = [m['file'] for m in MILESTONES]
    assert sorted(runs) == [7001, 7002, 7003, 7004, 7005, 7006]
    assert len(set(files)) == 6


def test_stagesAreSubsetsEndingInFull():
    for m in MILESTONES:
        assert set(m['stages']).issubset(set(BASE_STAGES)), m['name']
        assert m['stages'][-1] == 'full', m['name']


def test_rewardAnnealValuesValid():
    for m in MILESTONES:
        assert m['reward']['shapingAnneal'] in ('linear', 'none'), m['name']
