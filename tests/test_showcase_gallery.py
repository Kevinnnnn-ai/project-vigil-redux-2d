# tests/test_showcase_gallery.py
"""gallery builds the right watch command and supports headless --print without
opening a window."""
import os
import sys

SHOWCASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp', 'showcase'))
sys.path.insert(0, SHOWCASE_DIR)

import gallery
from milestones import MILESTONES


def test_buildWatchCommandExactArgs():
    m = [x for x in MILESTONES if x['name'] == 'm4-suicide-run1'][0]
    cmd = gallery.buildWatchCommand(m)
    assert cmd[1:] == ['-m', 'scripts.watch', '--config',
                       os.path.join('tmp', 'configs', 'm4-suicide-run1.yaml'),
                       '--run', '7004']


def test_printModeShowsCommandAndDoesNotLaunch(capsys):
    m = [x for x in MILESTONES if x['name'] == 'm4-suicide-run1'][0]
    code = gallery._watch(m, isPrintOnly=True)
    assert code == 0
    out = capsys.readouterr().out
    assert 'scripts.watch' in out and 'm4-suicide-run1.yaml' in out


def test_unknownMilestoneLookup():
    assert gallery.MILESTONE_BY_NAME.get('nope') is None
