# scripts/live_convergence.py
"""Live convergence plot: re-render stdout/convergence-plots/run-N.png from a run's
per-seed metrics CSVs on a fixed interval, so the curve updates while training runs.

Usage (standalone, e.g. in a second terminal):
    python -m scripts.live_convergence --run 3
    python -m scripts.live_convergence --run 3 --interval 10
    python -m scripts.live_convergence --run 3 --once          # one frame, then exit

scripts/train.py spawns this as a background subprocess for the run it is training and
terminates it when the run finishes; the same tool is runnable by hand. The step x-axis
factors (rolloutSteps, numEnvs) come from --config unless given explicitly."""
from __future__ import annotations

import argparse
import time

from src.config.loader import loadConfig
from src.metrics.live import runLogsDir, runPlotPath, renderConvergence


def parseArgs():
    parser = argparse.ArgumentParser(description='Live convergence curve from per-seed metrics CSVs.')
    parser.add_argument('--run', type=int, required=True, help='run number -> stdout/logs/run-N -> stdout/convergence-plots/run-N.png')
    parser.add_argument('--config', default='config.yaml', help='config for the step x-axis factors (rolloutSteps, numEnvs)')
    parser.add_argument('--logs-dir', default=None, help='override the run logs dir (default: stdout/logs/run-N)')
    parser.add_argument('--out', default=None, help='override the output PNG (default: stdout/convergence-plots/run-N.png)')
    parser.add_argument('--rollout-steps', type=int, default=None, help='override rolloutSteps (with --num-envs, skips loading --config)')
    parser.add_argument('--num-envs', type=int, default=None, help='override numEnvs (with --rollout-steps, skips loading --config)')
    parser.add_argument('--interval', type=float, default=5.0, help='seconds between re-renders')
    parser.add_argument('--once', action='store_true', help='render a single frame and exit')
    return parser.parse_args()


def main():
    args = parseArgs()
    logsDir = args.logs_dir if args.logs_dir is not None else runLogsDir(args.run)
    outPath = args.out if args.out is not None else runPlotPath(args.run)
    if args.rollout_steps is not None and args.num_envs is not None:
        rolloutSteps, numEnvs = args.rollout_steps, args.num_envs
    else:
        training = loadConfig(args.config).training
        rolloutSteps, numEnvs = training.rolloutSteps, training.numEnvs
    title = f'Training convergence (run-{args.run})'

    while True:
        # @SIDEFX: best-effort re-render; a glitch (or a not-yet-written CSV) must
        # never kill the loop — mirror scripts/train.py's plot try/except.
        try:
            plotted = renderConvergence(logsDir, outPath, rolloutSteps, numEnvs, title=title)
            print(f'live convergence -> {outPath} ({len(plotted)} seed(s))', flush=True)
        except Exception as exc:                  # noqa: BLE001 — best-effort plotting
            print(f'live convergence skipped (no data yet?): {exc}', flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
