# scripts/watch.py
"""Watch a controller land the booster.

Usage:
    python -m scripts.watch                     # trained net from runtime.watchModel
    python -m scripts.watch --checkpoint seed1
    python -m scripts.watch --stage drop
    python -m scripts.watch --pilot pd          # scripted PD pilot — NO checkpoint needed

Loads a checkpoint through the worldHash guard (a model trained in different
physics refuses to load), UNLESS --pilot pd is given, which flies the scripted
PdPilot instead — handy for watching the landing/settling physics with no trained
model. Keys: space=pause  n=step  r=reset  -/= speed  esc=quit."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.agents.checkpoints import resolveModelPath, loadCheckpoint
from src.agents.scripted import PdPilot
from src.runtime.render import Renderer, FPS
from src.runtime.loop import runEpisodeLoop
from scripts.play import stageByName


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    # <agent_context>
    #   [ARCH]: --model is the thrust profile, --env the environment subdir -> models/<model>/<env>/;
    #           --checkpoint selects within that dir. --env defaults to 'baseline'.
    #   [GOTCHA]: the <env> level is organizational only — compatibility is enforced by the worldHash guard, not the path.
    # </agent_context>
    parser.add_argument('--model', default=None, help='thrust profile -> models/<model>/<env>/ (default: runtime.model)')
    parser.add_argument('--env', default='baseline', help='environment subdir -> models/<model>/<env>/ (default: baseline)')
    parser.add_argument('--checkpoint', default=None, help="checkpoint within the model dir: 'best', 'seed<N>' or a path (default: runtime.watchModel)")
    parser.add_argument('--pilot', default=None, choices=['pd'], help="fly the scripted PD pilot instead of a trained net (no checkpoint needed)")
    parser.add_argument('--stage', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=None, dest='maxFrames')
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    stage = stageByName(cfg, args.stage)

    # @TAG[watch-source]: the action source is either the scripted PD pilot
    # (--pilot pd, no checkpoint) or a trained net loaded through the worldHash
    # guard. Both expose .act(obs) -> env action, so the loop is source-agnostic.
    if args.pilot == 'pd':
        source = PdPilot(cfg.world)
        sourceLabel = 'PD pilot'
        print(f'watch: scripted PD pilot (no checkpoint) flying stage {stage.name}')
    else:
        modelsDir = os.path.join('models', args.model or cfg.runtime.model, args.env)
        checkpoint = args.checkpoint or cfg.runtime.watchModel
        path = resolveModelPath(modelsDir, checkpoint)
        source, meta = loadCheckpoint(path, cfg.computeWorldHash())
        sourceLabel = os.path.basename(path)
        print(f"watch: {sourceLabel} (trained on stage {meta['stageName']}) "
              f"flying stage {stage.name}")

    env = LandingEnv(cfg, stage=stage)
    renderer = Renderer(cfg.world)
    record = {'success': 0, 'crash': 0, 'timeout': 0}

    def onEpisodeEnd(outcome):
        record[outcome] = record.get(outcome, 0) + 1

    def hud(isPaused, speed):
        state = env.state
        total = max(sum(record.values()), 1)
        return [
            f"WATCH  {sourceLabel}  stage {stage.name}"
            f"{'   PAUSED' if isPaused else ''}   speed x{speed:g}",
            f"landed {record['success']}/{sum(record.values())}  "
            f"({record['success'] / total:.0%})   crashed {record['crash']}  "
            f"timeout {record['timeout']}",
            # <agent_context>
            #   [ARCH]: ignitions-remaining is HUD-only; computed inline from state.engineTransitions (max 2).
            #   [GOTCHA]: cfg is captured from main() outer scope — do NOT move this closure above cfg definition (line 41).
            # </agent_context>
            f"fuel {state.fuel * 100:3.0f}%  engine {state.spool * 100:3.0f}%  "
            f"ign {2 - state.engineTransitions}  "
            f"vx {state.vx:+5.1f}  vy {state.vy:+5.1f} m/s  tilt {math.degrees(state.theta):+5.1f} deg",
            'space=pause  n=step  r=reset  -/= speed  esc=quit',
        ]

    runEpisodeLoop(
        env, renderer, lambda obs, intents: source.act(obs),
        np.random.default_rng(args.seed),
        fps=FPS, onEpisodeEnd=onEpisodeEnd, hudFn=hud, maxFrames=args.maxFrames,
    )


if __name__ == '__main__':
    main()
