# tmp/showcase/gallery.py
"""View the showcase milestones. Run from repo root:
    python tmp/showcase/gallery.py                          # list milestones + status
    python tmp/showcase/gallery.py --milestone m4-suicide-run1
    python tmp/showcase/gallery.py --all                    # watch each trained one in turn
    python tmp/showcase/gallery.py --milestone m4-suicide-run1 --print   # command only

Each launches: python -m scripts.watch --config <file> --run <run>
(watching with the milestone's OWN config -> world hash matches AND its own spawn stage)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.metrics.live import runCheckpointDir

from milestones import MILESTONES


MILESTONE_BY_NAME = {milestone['name']: milestone for milestone in MILESTONES}


def buildWatchCommand(milestone):
    """Exact argv to watch one milestone's best checkpoint in its own world/stage."""
    configPath = os.path.join('tmp', 'configs', milestone['file'])
    return [
        sys.executable, '-m', 'scripts.watch',
        '--config', configPath, '--run', str(milestone['run']),
    ]


def _isTrained(milestone):
    return os.path.exists(
        os.path.join(REPO_ROOT, runCheckpointDir(milestone['run']), 'best.pt'),
    )


def _printTable():
    print('Showcase milestones:')
    for milestone in MILESTONES:
        mark = 'trained' if _isTrained(milestone) else 'NOT trained'
        print(f"  {milestone['name']:22} run-{milestone['run']}  [{milestone['fidelity']:6}]  {mark}")
    print('\nWatch one:  python tmp/showcase/gallery.py --milestone <name>')


def _watch(milestone, isPrintOnly):
    command = buildWatchCommand(milestone)
    if isPrintOnly:
        print(' '.join(command))
        return 0
    if not _isTrained(milestone):
        print(f"WARNING: {milestone['name']} has no best.pt at run-{milestone['run']} — train it first.")
        return 1
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--milestone', default=None, help='milestone name to watch')
    parser.add_argument('--all', action='store_true', help='watch each trained milestone in turn')
    parser.add_argument('--print', action='store_true', dest='isPrintOnly', help='print the watch command, do not launch')
    args = parser.parse_args()
    if args.all:
        for milestone in MILESTONES:
            if args.isPrintOnly or _isTrained(milestone):
                _watch(milestone, args.isPrintOnly)
        return
    if args.milestone is None:
        _printTable()
        return
    milestone = MILESTONE_BY_NAME.get(args.milestone)
    if milestone is None:
        raise SystemExit(f'no milestone {args.milestone!r}; names: {list(MILESTONE_BY_NAME)}')
    raise SystemExit(_watch(milestone, args.isPrintOnly))


if __name__ == '__main__':
    main()
