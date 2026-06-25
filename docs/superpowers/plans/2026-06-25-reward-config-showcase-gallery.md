# Reward-Config Showcase & Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct each documented past training milestone as a standalone config under `tmp/configs/`, pinned to the current fixed world, so every retrained model shares the world hash and can be trained ("train them all") and viewed ("view them all") through a small `tmp/showcase/` kit.

**Architecture:** A single source of truth (`tmp/showcase/milestones.py`) defines 6 milestones as reward/training/curriculum deltas. `gen_configs.py` copies `config.yaml`'s `world:` block verbatim into each milestone config (guaranteeing world-hash identity). `train_all.py` trains each at a reserved run number and records a registry; `gallery.py` launches the viewer per milestone. No core code changes — `--config` and the reward-excluded world hash already exist.

**Tech Stack:** Python 3.14.5, PyYAML, pytest, the existing `src/` PPO/env/config packages and `scripts/{train,watch}.py`.

## Global Constraints

- **Run from repo root** with the `.env.local` venv (`.env.local\Scripts\python.exe`); `src.` is not pip-installed. Invoke pytest as `python -m pytest`.
- **Python 3.14.5** (`.env.local`, gitignored).
- **Code conventions (`.claude/AGENTS.md`):** camelCase variables and functions (functions start with a verb); booleans prefixed `is/has/can`; PascalCase classes; SCREAMING_SNAKE constants; single-quote strings only; helper functions/classes prefixed `_`; 3 blank lines between top-level sections; comments with `#` only.
- **Never edit** `config.yaml`, `src/config/loader.py`, `src/agents/checkpoints.py`/`mlp.py` (checkpoint format), or any core `scripts/` — the kit only *reads/uses* them.
- **World block copied verbatim** from `config.yaml`; every generated config MUST hash to the current world hash `f5c82b420d2a6ebc`.
- **Git-track** `tmp/configs/*.yaml` and `tmp/showcase/*`; run artifacts (`checkpoints/run-*`, `stdout/logs/run-*`, plots) stay gitignored.
- **Reserved run band:** `run-7001 … run-7006`, one per milestone (in `milestones.py`).
- **TDD, commit per task, never push.** Work stays on branch `feat/reward-config-showcase`.
- **YAML keys are camelCase 1:1 with dataclass fields; unknown keys raise `TypeError`** at load (so generated configs use only existing keys). `validateConfig` requires every stage `altitude[1] < world.ceiling` (60) and `reward.shapingAnneal ∈ {linear, none}`.

---

### Task 1: Milestone definitions (single source of truth)

**Files:**
- Create: `tmp/showcase/milestones.py`
- Test: `tests/test_showcase_milestones.py`

**Interfaces:**
- Produces: `MILESTONES` — a list of dicts, each with keys `name, file, run, reward, training, stages, fullOverride, source, fidelity, note`. `reward`/`training` are partial override dicts; `stages` is a list of base stage names; `fullOverride` is a dict (`altitude`, `xOffset`) or `None`. Consumed by Tasks 2, 4, 5.
- Also produces `BASE_STAGES` (tuple of the 5 base stage names) and `WIDE_FULL` (the pre-narrowing full-stage spawn dict).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_showcase_milestones.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'milestones'`.

- [ ] **Step 3: Write the milestone definitions**

```python
# tmp/showcase/milestones.py
"""Single source of truth for the reward/curriculum showcase milestones.

Each milestone reconstructs a documented past training system (docs/REWARD_LOG.md +
git history) on TODAY's world, so every retrained model shares the current world
hash and co-views. gen_configs/train_all/gallery all import MILESTONES.
Each milestone applies its reward/training deltas on top of config.yaml and selects
a subset of the base curriculum stages (optionally overriding the 'full' spawn)."""


BASE_STAGES = ('touchdown', 'hop', 'drop', 'glide', 'full')
# The pre-narrowing 'full' spawn (git 08fcc4d): wide drop window, full lateral span.
WIDE_FULL = {'altitude': [40.0, 52.0], 'xOffset': [-14.0, 14.0]}


MILESTONES = [
    {
        'name': 'm1-original-shaping',
        'file': 'm1-original-shaping.yaml',
        'run': 7001,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.01, 'totalIters': 220},
        'stages': ['hop', 'drop', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-12 original/rev1 (approx; oobPenalty + no-walls excluded)',
        'fidelity': 'approx',
        'note': 'No touchdown/glide rung -> success unsamplable; faithful failure.',
    },
    {
        'name': 'm2-walls-touchdown',
        'file': 'm2-walls-touchdown.yaml',
        'run': 7002,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.01, 'totalIters': 220},
        'stages': ['touchdown', 'hop', 'drop', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-12 rev2 (approx)',
        'fidelity': 'approx',
        'note': 'Touchdown rung, no glide; entCoef 0.01.',
    },
    {
        'name': 'm3-m5-glide',
        'file': 'm3-m5-glide.yaml',
        'run': 7003,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 260},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-13 M5 (approx; ~= m4 minus 40 iters)',
        'fidelity': 'approx',
        'note': 'Glide rung + entCoef 0.02 are today baseline; near-duplicate of m4.',
    },
    {
        'name': 'm4-suicide-run1',
        'file': 'm4-suicide-run1.yaml',
        'run': 7004,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 300},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'git 08fcc4d (exact); represents 06-15 timing & 06-16 pymunk',
        'fidelity': 'exact',
        'note': 'Suicide run-1 baseline: wide full, 300 iters, anneal linear.',
    },
    {
        'name': 'm5-run2',
        'file': 'm5-run2.yaml',
        'run': 7005,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 600},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': None,
        'source': 'git 36d58ce (exact)',
        'fidelity': 'exact',
        'note': 'Run-2: narrow full [52,52], 600 iters, anneal linear.',
    },
    {
        'name': 'm6-anneal-none',
        'file': 'm6-anneal-none.yaml',
        'run': 7006,
        'reward': {'shapingAnneal': 'none'},
        'training': {'entCoef': 0.02, 'totalIters': 600},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': None,
        'source': 'HEAD config.yaml (exact)',
        'fidelity': 'exact',
        'note': 'Current shipped config: anneal none.',
    },
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_showcase_milestones.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tmp/showcase/milestones.py tests/test_showcase_milestones.py
git commit -m "feat(showcase): milestone definitions (single source of truth)"
```

---

### Task 2: Config generator (`gen_configs.py`) + world-hash identity logic test

**Files:**
- Create: `tmp/showcase/gen_configs.py`
- Test: `tests/test_showcase_configs.py`

**Interfaces:**
- Consumes: `MILESTONES` (Task 1); `src.config.loader.loadConfig`.
- Produces: `buildConfigDict(milestone, baseRaw, isFast=False) -> dict` (pure: returns a complete config dict with `world:` copied verbatim and deltas applied) and `generateAll(outDir=CONFIGS_DIR, isFast=False) -> list[str]` (writes the yaml files). Consumed by Task 3 and the tests.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_showcase_configs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gen_configs'`.

- [ ] **Step 3: Write the generator**

```python
# tmp/showcase/gen_configs.py
"""Generate tmp/configs/*.yaml for each showcase milestone, copying config.yaml's
world: block VERBATIM so every config shares the current world hash.

Run from repo root:
    python tmp/showcase/gen_configs.py            # full-fidelity (committed set)
    python tmp/showcase/gen_configs.py --fast     # quick: capped iters + 1 seed

--fast OVERWRITES the same files with throwaway quick configs; restore the committed
set with `git checkout -- tmp/configs` or by re-running without --fast."""
from __future__ import annotations

import argparse
import copy
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml

from milestones import MILESTONES


BASE_CONFIG_PATH = os.path.join(REPO_ROOT, 'config.yaml')
CONFIGS_DIR = os.path.join(REPO_ROOT, 'tmp', 'configs')
FAST_ITERS = 60
FAST_SEEDS = [0]


def _loadBaseRaw():
    with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as a:
        return yaml.safe_load(a) or {}


def buildConfigDict(milestone, baseRaw, isFast=False):
    """Return a complete config dict = baseRaw with the milestone's reward/training/
    curriculum deltas applied. world: is copied verbatim (guarantees hash identity)."""
    raw = copy.deepcopy(baseRaw)
    raw['reward'].update(milestone['reward'])
    raw['training'].update(milestone['training'])
    stageByName = {stage['name']: stage for stage in baseRaw['curriculum']['stages']}
    stages = [copy.deepcopy(stageByName[name]) for name in milestone['stages']]
    if milestone['fullOverride']:
        for stage in stages:
            if stage['name'] == 'full':
                stage.update(milestone['fullOverride'])
    raw['curriculum']['stages'] = stages
    if isFast:
        raw['training']['totalIters'] = FAST_ITERS
        raw['training']['evalSeeds'] = list(FAST_SEEDS)
    return raw


def _header(milestone, isFast):
    tag = '  [FAST: capped iters / 1 seed]' if isFast else ''
    return (
        f"# GENERATED by tmp/showcase/gen_configs.py — DO NOT hand-edit.{tag}\n"
        f"# Milestone: {milestone['name']}  ({milestone['fidelity']})\n"
        f"# Source: {milestone['source']}\n"
        f"# Note: {milestone['note']}\n"
        f"# world: copied verbatim from config.yaml (shares the current world hash).\n"
    )


def generateAll(outDir=CONFIGS_DIR, isFast=False):
    """Write every milestone's yaml into outDir; return the paths written."""
    os.makedirs(outDir, exist_ok=True)
    baseRaw = _loadBaseRaw()
    paths = []
    for milestone in MILESTONES:
        raw = buildConfigDict(milestone, baseRaw, isFast=isFast)
        path = os.path.join(outDir, milestone['file'])
        with open(path, 'w', encoding='utf-8') as a:
            a.write(_header(milestone, isFast))
            yaml.safe_dump(raw, a, sort_keys=False, default_flow_style=None)
        paths.append(path)
        print(f"wrote {path}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='cap iters + single seed for quick models')
    args = parser.parse_args()
    generateAll(isFast=args.fast)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_showcase_configs.py -v -k "not committed"`
Expected: PASS (3 tests: `everyMilestoneSharesCurrentWorldHash`, `milestoneDeltasApplied`, `fastModeCapsItersAndSeeds`).

- [ ] **Step 5: Commit**

```bash
git add tmp/showcase/gen_configs.py tests/test_showcase_configs.py
git commit -m "feat(showcase): config generator with world-hash identity guarantee"
```

---

### Task 3: Generate + commit the 6 configs, with a drift guard on the committed artifacts

**Files:**
- Create (generated): `tmp/configs/m1-original-shaping.yaml` … `tmp/configs/m6-anneal-none.yaml`
- Modify: `tests/test_showcase_configs.py` (add the committed-artifact drift guard)

**Interfaces:**
- Consumes: `generateAll` (Task 2).
- Produces: the committed `tmp/configs/*.yaml` set that `train_all`/`gallery` consume.

- [ ] **Step 1: Add the failing drift-guard test**

Append to `tests/test_showcase_configs.py`:

```python
def test_committedConfigsExistAndMatchHash():
    """Drift guard on the COMMITTED artifacts: editing config.yaml's world without
    regenerating tmp/configs/ fails here."""
    baseHash = loadConfig('config.yaml').computeWorldHash()
    for milestone in MILESTONES:
        path = os.path.join('tmp', 'configs', milestone['file'])
        assert os.path.exists(path), f'missing {path} — run gen_configs.py'
        assert loadConfig(path).computeWorldHash() == baseHash, milestone['name']
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_showcase_configs.py::test_committedConfigsExistAndMatchHash -v`
Expected: FAIL — `AssertionError: missing tmp/configs/m1-original-shaping.yaml — run gen_configs.py`.

- [ ] **Step 3: Generate the committed configs**

Run: `python tmp/showcase/gen_configs.py`
Expected stdout: six `wrote tmp/configs/m*.yaml` lines.

- [ ] **Step 4: Verify the drift guard now passes**

Run: `python -m pytest tests/test_showcase_configs.py -v`
Expected: PASS (4 tests). Also confirm the hash by eye:

Run: `python -c "from src.config.loader import loadConfig; print(loadConfig('tmp/configs/m6-anneal-none.yaml').computeWorldHash())"`
Expected: `f5c82b420d2a6ebc`.

- [ ] **Step 5: Commit**

```bash
git add tmp/configs/ tests/test_showcase_configs.py
git commit -m "feat(showcase): generate 6 world-pinned milestone configs + drift guard"
```

---

### Task 4: Batch trainer (`train_all.py`) + registry

**Files:**
- Create: `tmp/showcase/train_all.py`
- Test: `tests/test_showcase_train_all.py`

**Interfaces:**
- Consumes: `MILESTONES`; `src.metrics.live.{runLogsDir, readSeedHistories}`; `src.config.loader.loadConfig`; `scripts.train` (as a subprocess).
- Produces: `buildTrainCommand(milestone, isSerial=False) -> list[str]`; `trainOne(milestone, isSerial=False) -> int`; writes `tmp/showcase/registry.json` + `tmp/showcase/REGISTRY.md`. `_worldHashOk(milestone) -> bool`, `_bestSuccessOf(run) -> float|None`, `_writeRegistryMd(registry)`.

- [ ] **Step 1: Write the failing test**

```python
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


def test_registryMdRendersAllMilestones(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, 'REGISTRY_MD', str(tmp_path / 'REGISTRY.md'))
    train_all._writeRegistryMd({'m6-anneal-none': {'trainedAt': '2026-06-25T00:00:00',
                                                   'bestSuccess': 0.0, 'worldHashOk': True}})
    text = (tmp_path / 'REGISTRY.md').read_text()
    for m in MILESTONES:
        assert m['name'] in text
    assert '| m6-anneal-none |' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_showcase_train_all.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_all'`.

- [ ] **Step 3: Write the batch trainer**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_showcase_train_all.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Manual end-to-end smoke (optional, fast; cleans up)**

This confirms a generated config trains and produces a checkpoint, without touching the showcase run band. Run from repo root:

```bash
python tmp/showcase/gen_configs.py --fast
python -m scripts.train --config tmp/configs/m6-anneal-none.yaml --run 9002 --serial
ls checkpoints/run-9002/best.pt
```
Expected: `checkpoints/run-9002/best.pt` exists. Then clean up the sentinel and restore the committed configs:

```bash
rm -rf checkpoints/run-9002 stdout/logs/run-9002 stdout/convergence-plots/run-9002.png
python tmp/showcase/gen_configs.py
git checkout -- tmp/configs
```

- [ ] **Step 6: Commit**

```bash
git add tmp/showcase/train_all.py tests/test_showcase_train_all.py
git commit -m "feat(showcase): batch trainer + run/reward registry"
```

---

### Task 5: Gallery launcher (`gallery.py`)

**Files:**
- Create: `tmp/showcase/gallery.py`
- Test: `tests/test_showcase_gallery.py`

**Interfaces:**
- Consumes: `MILESTONES`; `src.metrics.live.runCheckpointDir`; `scripts.watch` (as a subprocess).
- Produces: `buildWatchCommand(milestone) -> list[str]`; `MILESTONE_BY_NAME`; `_isTrained(milestone) -> bool`; `_watch(milestone, isPrintOnly) -> int`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_showcase_gallery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gallery'`.

- [ ] **Step 3: Write the gallery**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_showcase_gallery.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tmp/showcase/gallery.py tests/test_showcase_gallery.py
git commit -m "feat(showcase): gallery launcher (per-milestone watch + --print)"
```

---

### Task 6: Documentation + full-suite verification

**Files:**
- Modify: `docs/CHANGELOG.md`, `docs/REWARD_LOG.md`, `.claude/agent-memory/context.md`, `.claude/agent-memory/decisions.md`, `.claude/agent-memory/notes.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Run the full suite (baseline green before doc edits)**

Run: `python -m pytest -q`
Expected: all prior tests + the 4 new test files pass (no failures).

- [ ] **Step 2: Append the CHANGELOG entry**

Read the top of `docs/CHANGELOG.md` first to match its exact entry format, then add an entry covering: the `tmp/showcase/` kit (`milestones.py`, `gen_configs.py`, `train_all.py`, `gallery.py`), the 6 world-pinned `tmp/configs/*.yaml`, the world-hash identity guard test, and the reserved run band `7001-7006`. State plainly that this adds **no core code change** (uses the existing `--config` flag + reward-excluded world hash) and that models from all configs co-view because they share world hash `f5c82b420d2a6ebc`.

- [ ] **Step 3: Append the REWARD_LOG entry**

Read the entry format at the top of `docs/REWARD_LOG.md`, then add ONE entry dated 2026-06-25 documenting the reproduction effort:
- **Hypothesis:** past models can be re-shown in one world by retraining each documented milestone's reward+curriculum on today's fixed world (reward changes never invalidate models).
- **Config:** the two reproducible reward variants — A (`shapingAnneal: linear`) and B (`none`) — across 6 milestone configs in `tmp/configs/`; mapping to runs `7001-7006` lives in `tmp/showcase/registry.json`.
- **Result:** PENDING — training is launched by the user; "failure-OK" reproductions, not new reward designs.
- **Verdict:** ITERATE — record per-milestone outcomes in `tmp/showcase/REGISTRY.md`.

- [ ] **Step 4: Update agent-memory**

- `context.md`: under "Scripts / commands", add the showcase kit (`python tmp/showcase/{gen_configs,train_all,gallery}.py`) and note `tmp/configs/` holds world-pinned reward/curriculum milestone configs (git-tracked); add to "Current state" that the showcase feature exists.
- `decisions.md`: append a 2026-06-25 entry — the reward log yields only 2 reproducible reward variants (linear/none); historical model diversity was world/curriculum; chose Approach A (generated standalone configs, zero core change); reserved run band 7001-7006; world-hash identity guard test. Reference the spec + this plan.
- `notes.md`: add a "Showcase kit" section — run as `python tmp/showcase/<name>.py` from repo root (each bootstraps repo root onto `sys.path`); workflow is gen_configs → train_all → gallery; world drift → `gen_configs.py` re-syncs then retrain; `--fast` configs are throwaway (`git checkout -- tmp/configs` to restore).

- [ ] **Step 5: Final full-suite verification**

Run: `python -m pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add docs/CHANGELOG.md docs/REWARD_LOG.md .claude/agent-memory/context.md .claude/agent-memory/decisions.md .claude/agent-memory/notes.md
git commit -m "docs(showcase): CHANGELOG, REWARD_LOG, agent-memory for the showcase kit"
```

---

## Self-Review

**1. Spec coverage:**
- §4 Approach A (generated standalone YAMLs, zero core change) → Tasks 2-3. ✓
- §5 the 6 configs with exact deltas → `milestones.py` (Task 1) + generator (Task 2) + artifacts (Task 3). ✓
- §6 world-identity guarantee + guard test → Task 2 (logic) + Task 3 (committed-artifact drift guard). ✓
- §7 directory layout + git tracking → Tasks 1-5 create files under `tmp/showcase/` + `tmp/configs/`; commits track them; run artifacts stay gitignored. ✓
- §8 tooling (gen/train_all/gallery, reserved band, run-as-module bootstrap, `--fast`) → Tasks 2,4,5. ✓
- §9 provenance registry (json canonical + md) → Task 4. ✓
- §10 compute + showcase-speed → `--fast` (Task 2) + Task 4 Step 5. ✓
- §11 testing plan (4 test files, guard, smoke + cleanup) → Tasks 1-5 tests + Task 4 Step 5. ✓
- §12 doc updates → Task 6. ✓
- §13 risks (drift guard, run-band caveat, m3≈m4, comment loss, import path) → encoded in guard test, `milestones.py` notes, and `notes.md` (Task 6). ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; doc steps (Task 6) name exact sections/content to write (CHANGELOG/REWARD_LOG bodies are summarized because they must match each file's existing entry format, read at edit time — the required content is fully enumerated). ✓

**3. Type consistency:** `buildConfigDict(milestone, baseRaw, isFast=False)`, `generateAll(outDir, isFast)`, `FAST_ITERS`/`FAST_SEEDS`, `buildTrainCommand(milestone, isSerial=False)`, `buildWatchCommand(milestone)`, `_isTrained`, `_watch(milestone, isPrintOnly)`, `_worldHashOk`, `_bestSuccessOf(run)`, `_writeRegistryMd(registry)`, `REGISTRY_MD`/`REGISTRY_JSON` are referenced identically across tasks and tests. Milestone dict keys (`name/file/run/reward/training/stages/fullOverride/source/fidelity/note`) are consistent between `milestones.py` and every consumer. ✓
