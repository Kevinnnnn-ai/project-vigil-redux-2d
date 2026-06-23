# scripts/evaluate.py
"""Score a trained net against the PD-pilot baseline, headlessly.

Usage:
    python -m scripts.evaluate                  # runtime.watchModel, final stage
    python -m scripts.evaluate --checkpoint seed2 --stage drop --episodes 200

Prints success rate, outcome breakdown, mean impact speed and episode length
for the checkpoint AND for PdPilot on the same seeds — the baseline the net
must beat."""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.agents.checkpoints import resolveModelPath, loadCheckpoint
from src.metrics.live import resolveNextRun, runCheckpointDir
from src.agents.scripted import PdPilot
from src.runtime.evaluate import runEvaluation
from scripts.play import stageByName


def _printResult(label, result):
    outcomes = '  '.join(f'{k}: {v}' for k, v in sorted(result['outcomes'].items()))
    print(f"  {label:12} success {result['successRate']:.2%}   {outcomes}")
    print(f"  {'':12} impact mean {result['meanImpactSpeed']:.2f} m/s   "
          f"episode mean {result['meanSteps']:.0f} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--run', type=int, default=None, help='run number -> checkpoints/run-N/ (default: latest run)')
    parser.add_argument('--checkpoint', default=None, help="checkpoint within the model dir: 'best', 'seed<N>' or a path (default: runtime.watchModel)")
    parser.add_argument('--stage', default=None)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    stage = stageByName(cfg, args.stage)
    episodes = args.episodes or cfg.runtime.evaluateEpisodes
    run = args.run if args.run is not None else resolveNextRun() - 1
    if run < 1:
        raise SystemExit('no runs found in checkpoints/ — train a model first or pass --run N')
    modelsDir = runCheckpointDir(run)
    checkpoint = args.checkpoint or cfg.runtime.watchModel
    path = resolveModelPath(modelsDir, checkpoint)
    net, meta = loadCheckpoint(path, cfg.computeWorldHash())

    env = LandingEnv(cfg, stage=stage)
    print(f"evaluate: stage {stage.name}, {episodes} episodes, seed {args.seed}")
    netResult = runEvaluation(env, net, episodes, np.random.default_rng(args.seed))
    _printResult(os.path.basename(path), netResult)
    pdResult = runEvaluation(env, PdPilot(cfg.world), episodes, np.random.default_rng(args.seed))
    _printResult('PdPilot', pdResult)


if __name__ == '__main__':
    main()
