# scripts/train.py
"""Train the booster lander across all configured seeds, save the best checkpoint.

Usage:
    python -m scripts.train                              # curriculum: touchdown -> full
    python -m scripts.train --stage hop                  # single-stage (no promotion)
    python -m scripts.train --model solis --env baseline # -> models/solis/baseline/

Reads everything from the --config file. Trains once per seed in training.evalSeeds
(seeds run CONCURRENTLY, one process per seed — see src/train/parallel.py;
training.seedWorkers / --serial control the concurrency). Saves each run's best
checkpoint to models/<model>/<env>/seed<seed>.pt, then copies the best across seeds
to models/<model>/<env>/best.pt. For curriculum runs the score is the best success
rate ON THE FINAL STAGE (0 if the ladder never got there). Low spread across seeds
is the acceptance signal (stability), not a single lucky run."""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from src.config.loader import loadConfig
from src.train.parallel import (
    SeedTask,
    resolveSeedWorkers,
    runSeeds,
    stageByName,
)
from src.metrics.plot import plotConvergence

# @CONFIG[metrics-dir]: per-iteration training CSVs land here (one file per seed).
# stdout/ (replaced the former 'runs/') is organized by artifact type — metrics/
# for these CSVs, alongside checkpoints/, configs/, scripts/ for one-off probes.
# Gitignored except a tracked stdout/.gitkeep, so the tree exists on a fresh clone
# without committing run artifacts; os.makedirs creates metrics/ on first train.
METRICS_DIR = os.path.join('stdout', 'metrics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default=None, help='curriculum stage name (default: final stage)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', default=None, help='thrust profile -> models/<model>/<env>/ (default: runtime.model)')
    parser.add_argument('--env', default='baseline', help='environment subdir -> models/<model>/<env>/ (default: baseline)')
    parser.add_argument('--serial', action='store_true', help='train seeds sequentially (overrides training.seedWorkers; for debug/repro)')
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    modelName = args.model or cfg.runtime.model
    modelsDir = os.path.join('models', modelName, args.env)
    os.makedirs(modelsDir, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    finalName = cfg.curriculum.stages[-1].name

    if args.stage:                       # fail fast in the PARENT on a bad stage name,
        stageByName(cfg, args.stage)     # before any worker process is spawned

    # @ANCHOR[seed-tasks]: one picklable unit of work per seed. stageName=None
    # selects the curriculum path; a name selects single-stage (trainLanding).
    tasks = [
        SeedTask(
            configPath=args.config,
            seed=seed,
            savePath=os.path.join(modelsDir, f'seed{seed}.pt'),
            csvPath=os.path.join(METRICS_DIR, f'seed{seed}_metrics.csv'),
            stageName=args.stage,
            modelName=modelName,
            envName=args.env,
        )
        for seed in cfg.training.evalSeeds
    ]
    maxWorkers = resolveSeedWorkers(cfg, serial=args.serial)
    print(f"training {len(tasks)} seed(s) {list(cfg.training.evalSeeds)} with {maxWorkers} worker(s)")
    results = runSeeds(tasks, maxWorkers=maxWorkers)       # sorted by seed

    bests = []
    histories = {}
    for result in results:
        bests.append((result.bestRate, result.savePath))
        histories[result.seed] = result.history
        print(f"[seed {result.seed}] best success {result.bestRate:.2f} -> {result.savePath}")

    rates = np.array([rate for rate, _ in bests])
    bestRate, bestPath = max(bests, key=lambda pair: pair[0])
    shutil.copyfile(bestPath, os.path.join(modelsDir, 'best.pt'))
    scope = args.stage if args.stage else f'curriculum->{finalName}'
    print(
        f"\n{scope}: success across seeds {cfg.training.evalSeeds} = "
        f"{rates.mean():.2f} +/- {rates.std():.2f}  (min {rates.min():.2f})\n"
        f"best.pt <- {bestPath} ({bestRate:.2f})  [models/{modelName}/{args.env}/]",
    )

    # @ANCHOR[convergence-plot]: store the success-rate-vs-steps figure next to the
    # checkpoints. Warn-and-continue — a plotting glitch must not discard a finished run.
    plotPath = os.path.join(modelsDir, 'convergence.png')
    try:
        plotConvergence(
            histories, plotPath,
            rolloutSteps=cfg.training.rolloutSteps,
            numEnvs=cfg.training.numEnvs,
            title=f'Convergence: {modelName}/{args.env} ({scope})',
        )
        print(f"convergence plot -> {plotPath}")
    except Exception as exc:                      # noqa: BLE001 — plotting is best-effort
        print(f"WARNING: convergence plot failed ({exc}); checkpoints are unaffected.")


if __name__ == '__main__':
    main()
