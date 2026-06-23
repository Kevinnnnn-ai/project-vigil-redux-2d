# scripts/train.py
"""Train the booster lander across all configured seeds, save the best checkpoint.

Usage:
    python -m scripts.train                              # curriculum: touchdown -> full
    python -m scripts.train --stage hop                  # single-stage (no promotion)
    python -m scripts.train --run 4                      # force the run number (default: auto)

Reads everything from the --config file. Trains once per seed in training.evalSeeds
(seeds run CONCURRENTLY, one process per seed — see src/train/parallel.py;
training.seedWorkers / --serial control the concurrency). Each session is a numbered
RUN: per-seed checkpoints go to checkpoints/run-N/seed<seed>.pt (best across seeds copied
to checkpoints/run-N/best.pt), per-iteration metrics to stdout/logs/run-N/seed<seed>.csv,
and a convergence figure to stdout/convergence-plots/run-N.png that UPDATES LIVE during
training (a scripts.live_convergence subprocess re-renders it from the metrics CSVs) and
is finalized when the run ends. For curriculum runs the score is the best success rate ON
THE FINAL STAGE (0 if the ladder never got there). Low spread across seeds is the
acceptance signal (stability), not a single lucky run."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from src.config.loader import loadConfig
from src.train.parallel import (
    SeedTask,
    resolveSeedWorkers,
    runSeeds,
    stageByName,
)
from src.metrics.plot import plotConvergence
from src.metrics.live import (
    resolveNextRun,
    runCheckpointDir,
    runLogsDir,
    runPlotPath,
    seedCheckpointPath,
    seedCsvPath,
)

# @CONFIG[live-refresh]: seconds between live convergence re-renders by the spawned
# scripts.live_convergence subprocess. Eval points arrive every evalEvery iters, so a
# few seconds keeps the PNG fresh without churning.
LIVE_REFRESH_SECONDS = 5


def _startLiveRefresher(run, configPath):
    """Spawn the background live-convergence renderer for this run, or return None if it
    cannot start. Best-effort: training must proceed even without a live plot."""
    command = [
        sys.executable, '-m', 'scripts.live_convergence',
        '--run', str(run), '--config', configPath,
        '--interval', str(LIVE_REFRESH_SECONDS),
    ]
    try:
        return subprocess.Popen(command)
    except Exception as exc:                       # noqa: BLE001 — live plot is best-effort
        print(f"WARNING: live convergence refresher did not start ({exc}); training continues.")
        return None


def _stopLiveRefresher(process):
    """Terminate the live-convergence subprocess, killing it if it will not exit.
    Guarded so refresher cleanup never masks a training error or KeyboardInterrupt."""
    if process is None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as exc:                       # noqa: BLE001 — teardown is best-effort
        print(f"WARNING: could not cleanly stop live convergence refresher ({exc}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default=None, help='curriculum stage name (default: final stage)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--run', type=int, default=None, help='run number -> checkpoints/run-N/ (default: auto-increment)')
    parser.add_argument('--model', default=None, help='optional label folded into the plot title')
    parser.add_argument('--env', default='baseline', help='optional label folded into the plot title')
    parser.add_argument('--serial', action='store_true', help='train seeds sequentially (overrides training.seedWorkers; for debug/repro)')
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    run = args.run if args.run is not None else resolveNextRun()
    checkpointDir = runCheckpointDir(run)
    logsDir = runLogsDir(run)
    plotPath = runPlotPath(run)
    os.makedirs(checkpointDir, exist_ok=True)
    os.makedirs(logsDir, exist_ok=True)
    os.makedirs(os.path.dirname(plotPath), exist_ok=True)
    finalName = cfg.curriculum.stages[-1].name

    if args.stage:                       # fail fast in the PARENT on a bad stage name,
        stageByName(cfg, args.stage)     # before any worker process is spawned

    # @ANCHOR[seed-tasks]: one picklable unit of work per seed. stageName=None
    # selects the curriculum path; a name selects single-stage (trainLanding).
    tasks = [
        SeedTask(
            configPath=args.config,
            seed=seed,
            savePath=seedCheckpointPath(run, seed),
            csvPath=seedCsvPath(run, seed),
            stageName=args.stage,
            modelName=args.model or f'run-{run}',
            envName=args.env,
        )
        for seed in cfg.training.evalSeeds
    ]
    maxWorkers = resolveSeedWorkers(cfg, serial=args.serial)
    print(f"run-{run}: training {len(tasks)} seed(s) {list(cfg.training.evalSeeds)} with {maxWorkers} worker(s)")
    print(f"  checkpoints -> {checkpointDir}{os.sep}   metrics -> {logsDir}{os.sep}   live plot -> {plotPath}")

    # @ANCHOR[live-plot]: re-render plotPath from the per-seed CSVs every few seconds
    # WHILE training runs. Spawn before collecting; ALWAYS tear down (finally) so the
    # final authoritative frame below never races a half-written live frame.
    liveProc = _startLiveRefresher(run, args.config)
    try:
        results = runSeeds(tasks, maxWorkers=maxWorkers)       # sorted by seed
    finally:
        _stopLiveRefresher(liveProc)

    bests = []
    histories = {}
    for result in results:
        bests.append((result.bestRate, result.savePath))
        histories[result.seed] = result.history
        print(f"[seed {result.seed}] best success {result.bestRate:.2f} -> {result.savePath}")

    rates = np.array([rate for rate, _ in bests])
    bestRate, bestPath = max(bests, key=lambda pair: pair[0])
    shutil.copyfile(bestPath, os.path.join(checkpointDir, 'best.pt'))
    scope = args.stage if args.stage else f'curriculum->{finalName}'
    print(
        f"\n{scope}: success across seeds {cfg.training.evalSeeds} = "
        f"{rates.mean():.2f} +/- {rates.std():.2f}  (min {rates.min():.2f})\n"
        f"best.pt <- {bestPath} ({bestRate:.2f})  [{checkpointDir}{os.sep}]",
    )

    # @ANCHOR[convergence-plot]: final authoritative frame from the COMPLETE in-memory
    # histories (the live subprocess is already stopped). Warn-and-continue — a plotting
    # glitch must not discard a finished run.
    title = f'Convergence run-{run}: {scope}'
    if args.model:
        title += f' [{args.model}/{args.env}]'
    try:
        plotConvergence(
            histories, plotPath,
            rolloutSteps=cfg.training.rolloutSteps,
            numEnvs=cfg.training.numEnvs,
            title=title,
        )
        print(f"convergence plot -> {plotPath}")
    except Exception as exc:                      # noqa: BLE001 — plotting is best-effort
        print(f"WARNING: convergence plot failed ({exc}); checkpoints are unaffected.")


if __name__ == '__main__':
    main()
