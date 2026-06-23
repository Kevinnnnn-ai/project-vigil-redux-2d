# src/metrics/live.py
# <agent_context>
#   [ARCH]: Run-artifact layout + the live-convergence bridge. Owns where a run's
#           checkpoints/metrics/plot live (checkpoints/run-N/, stdout/logs/run-N/,
#           stdout/convergence-plots/run-N.png) and turns the per-seed metrics CSVs
#           (flushed every iteration by metrics/logger.py:CsvLogger) into the
#           {seed: history} dict that metrics/plot.py:plotConvergence already draws.
#   [API]:  resolveNextRun, run*/seed* path helpers, readSeedHistories(logsDir),
#           renderConvergence(logsDir, outPath, rolloutSteps, numEnvs, title).
#   [GOTCHA]: readSeedHistories tolerates a partially-written final row (a CSV read
#             mid-flush) and keeps the curriculum -1.0 sentinels verbatim — the
#             plotter, not the reader, filters them out.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: rolloutSteps/numEnvs handed to renderConvergence MUST be the values
#               training ran with (cfg.training.{rolloutSteps,numEnvs}); they set the
#               step x-axis (see metrics/plot.py). plot.py is reused UNCHANGED.
#   [VALIDATION]: python -m pytest tests/test_live.py -v
# </agent_guardrail>
"""Run-artifact paths and the CSV -> histories -> convergence-PNG live bridge."""
from __future__ import annotations

import csv
import glob
import os
import re

from src.metrics.plot import plotConvergence


CHECKPOINTS_ROOT = 'checkpoints'
LOGS_ROOT = os.path.join('stdout', 'logs')
PLOTS_DIR = os.path.join('stdout', 'convergence-plots')
RUN_PREFIX = 'run-'
# @ANCHOR[seed-csv]: per-seed metrics file under a run's logs dir; the seed index is
# recovered from this name by readSeedHistories and written by scripts/train.py.
_SEED_CSV_GLOB = 'seed*.csv'
_SEED_INDEX = re.compile(r'seed(\d+)')


def resolveNextRun(checkpointsRoot=CHECKPOINTS_ROOT):
    """Next run number: one past the highest existing checkpoints/run-<int>, or 1 if
    the root is absent or holds no run-<int> dirs. Non-matching names are ignored."""
    if not os.path.isdir(checkpointsRoot):
        return 1
    highest = 0
    for entry in os.listdir(checkpointsRoot):
        if not entry.startswith(RUN_PREFIX):
            continue
        suffix = entry[len(RUN_PREFIX):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def runCheckpointDir(run, checkpointsRoot=CHECKPOINTS_ROOT):
    """checkpoints/run-N — holds seed<seed>.pt and best.pt for one session."""
    return os.path.join(checkpointsRoot, f'{RUN_PREFIX}{run}')


def runLogsDir(run, logsRoot=LOGS_ROOT):
    """stdout/logs/run-N — holds the per-seed live metrics CSVs for one session."""
    return os.path.join(logsRoot, f'{RUN_PREFIX}{run}')


def runPlotPath(run, plotsDir=PLOTS_DIR):
    """stdout/convergence-plots/run-N.png — the live + final convergence figure."""
    return os.path.join(plotsDir, f'{RUN_PREFIX}{run}.png')


def seedCheckpointPath(run, seed, checkpointsRoot=CHECKPOINTS_ROOT):
    return os.path.join(runCheckpointDir(run, checkpointsRoot), f'seed{seed}.pt')


def seedCsvPath(run, seed, logsRoot=LOGS_ROOT):
    return os.path.join(runLogsDir(run, logsRoot), f'seed{seed}.csv')


def _seedOf(path):
    """Recover the integer seed from a 'seed<seed>.csv' file name, or None."""
    match = _SEED_INDEX.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def _readOneCsv(path):
    """Parse one per-seed metrics CSV into [{'iter': int, 'successRate': float}].
    Rows missing or non-coercible on either field are skipped — this tolerates a CSV
    read mid-flush, whose final line may be only partially written."""
    records = []
    with open(path, newline='', encoding='utf-8') as a:
        for row in csv.DictReader(a):
            try:
                records.append({
                    'iter': int(row['iter']),
                    'successRate': float(row['successRate']),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return records


def readSeedHistories(logsDir):
    """{seed: [record, ...]} for every seed<seed>.csv in logsDir; {} if logsDir is
    absent or empty. Sentinel successRate -1.0 rows are retained verbatim (the
    plotter filters them), so this mirrors trainLanding/trainCurriculum histories."""
    histories = {}
    for path in glob.glob(os.path.join(logsDir, _SEED_CSV_GLOB)):
        seed = _seedOf(path)
        if seed is None:
            continue
        records = _readOneCsv(path)
        if records:
            histories[seed] = records
    return histories


def renderConvergence(logsDir, outPath, rolloutSteps, numEnvs, title=None):
    """Read the run's per-seed CSVs and (re)render the convergence PNG at outPath via
    the shared plotConvergence, creating outPath's directory. Returns the seeds drawn;
    an empty run still writes an axes-only 'warming up' frame (plotConvergence does)."""
    os.makedirs(os.path.dirname(outPath) or '.', exist_ok=True)
    histories = readSeedHistories(logsDir)
    return plotConvergence(histories, outPath, rolloutSteps, numEnvs, title=title)
