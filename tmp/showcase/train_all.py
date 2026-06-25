# tmp/showcase/train_all.py
"""Train every showcase milestone (or one via --only) on its reserved run number,
then record run status to registry.json + REGISTRY.md.

Run from repo root:
    python tmp/showcase/train_all.py                 # all 6 (HOURS at full fidelity)
    python tmp/showcase/train_all.py --only m6-anneal-none
    python tmp/showcase/train_all.py --serial        # seeds sequential per run

Each milestone trains via: python -m scripts.train --config <file> --run <run>.
Use `gen_configs.py --fast` first for quick (unconverged) models."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config.loader import loadConfig
from src.metrics.live import runLogsDir, readSeedHistories

from milestones import MILESTONES


CONFIGS_DIR = os.path.join(REPO_ROOT, 'tmp', 'configs')
REGISTRY_JSON = os.path.join(os.path.dirname(__file__), 'registry.json')
REGISTRY_MD = os.path.join(os.path.dirname(__file__), 'REGISTRY.md')


def buildTrainCommand(milestone, isSerial=False):
    """Exact argv for training one milestone via scripts.train."""
    configPath = os.path.join('tmp', 'configs', milestone['file'])
    command = [
        sys.executable, '-m', 'scripts.train',
        '--config', configPath, '--run', str(milestone['run']),
    ]
    if isSerial:
        command.append('--serial')
    return command


def _bestSuccessOf(run):
    """Max successRate across the run's seed CSVs (ignoring -1.0 sentinels), or None."""
    histories = readSeedHistories(runLogsDir(run))
    rates = [
        record['successRate']
        for records in histories.values()
        for record in records
        if record['successRate'] >= 0
    ]
    return max(rates) if rates else None


def _worldHashOk(milestone):
    base = loadConfig(os.path.join(REPO_ROOT, 'config.yaml')).computeWorldHash()
    path = os.path.join(CONFIGS_DIR, milestone['file'])
    return loadConfig(path).computeWorldHash() == base


def _loadRegistry():
    if os.path.exists(REGISTRY_JSON):
        with open(REGISTRY_JSON, 'r', encoding='utf-8') as a:
            return json.load(a)
    return {}


def _writeRegistryMd(registry):
    lines = [
        '# Showcase Registry',
        '',
        '| Milestone | Run | Config | Fidelity | Trained | bestSuccess | worldHashOk |',
        '|---|---|---|---|---|---|---|',
    ]
    for milestone in MILESTONES:
        entry = registry.get(milestone['name'], {})
        isTrained = 'yes' if entry.get('trainedAt') else 'no'
        best = entry.get('bestSuccess')
        bestStr = f'{best:.2f}' if isinstance(best, (int, float)) else '-'
        lines.append(
            f"| {milestone['name']} | {milestone['run']} | {milestone['file']} | "
            f"{milestone['fidelity']} | {isTrained} | {bestStr} | {entry.get('worldHashOk', '-')} |"
        )
    with open(REGISTRY_MD, 'w', encoding='utf-8') as a:
        a.write('\n'.join(lines) + '\n')


def _writeRegistry(registry):
    with open(REGISTRY_JSON, 'w', encoding='utf-8') as a:
        json.dump(registry, a, indent=2, sort_keys=True)
    _writeRegistryMd(registry)


def trainOne(milestone, isSerial=False):
    """Run scripts.train for one milestone, update the registry, return the exit code."""
    command = buildTrainCommand(milestone, isSerial=isSerial)
    print(f"=== training {milestone['name']} -> run-{milestone['run']} ===")
    print('  ' + ' '.join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    isOk = result.returncode == 0
    registry = _loadRegistry()
    registry[milestone['name']] = {
        'run': milestone['run'],
        'config': os.path.join('tmp', 'configs', milestone['file']),
        'trainedAt': datetime.datetime.now().isoformat(timespec='seconds') if isOk else None,
        'returnCode': result.returncode,
        'bestSuccess': _bestSuccessOf(milestone['run']) if isOk else None,
        'worldHashOk': _worldHashOk(milestone),
    }
    _writeRegistry(registry)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', action='store_true', help='train each milestone seeds sequentially')
    parser.add_argument('--only', default=None, help='train only this milestone name')
    args = parser.parse_args()
    chosen = [m for m in MILESTONES if args.only is None or m['name'] == args.only]
    if not chosen:
        raise SystemExit(f'no milestone named {args.only!r}; names: {[m["name"] for m in MILESTONES]}')
    for milestone in chosen:
        trainOne(milestone, isSerial=args.serial)


if __name__ == '__main__':
    main()
