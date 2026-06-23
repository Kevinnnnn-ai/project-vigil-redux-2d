# scripts/play.py
"""Fly the booster yourself from the keyboard.

Usage:
    python -m scripts.play                  # full-difficulty spawns
    python -m scripts.play --stage hop      # easier spawns by stage name

Controls: W/UP ramp throttle up, S/DOWN ramp down; A/LEFT and D/RIGHT rotate
the nose (gimbal); space=pause  n=step  r=reset  -/= speed  esc/q=quit.
The HUD shows fuel, velocity, tilt, throttle, and your landing record."""
from __future__ import annotations

import argparse
import math

import numpy as np

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.runtime.render import Renderer, FPS
from src.runtime.loop import runEpisodeLoop

THROTTLE_RAMP = 2.0     # full sweep in 0.5 s of held key


def stageByName(cfg, name):
    if name is None:
        return cfg.curriculum.stages[-1]
    for stage in cfg.curriculum.stages:
        if stage.name == name:
            return stage
    names = [stage.name for stage in cfg.curriculum.stages]
    raise SystemExit(f'unknown stage {name!r}; config has {names}')


class HumanSource:
    """Stateful keyboard pilot: holds the ramped throttle between frames."""

    def __init__(self, dt):
        self.dt = dt
        self.throttle = 0.0

    def __call__(self, obs, intents):
        self.throttle = min(max(
            self.throttle + intents.throttleDir * THROTTLE_RAMP * self.dt, 0.0,
        ), 1.0)
        return np.array([self.throttle, intents.gimbal])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--stage', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=None, dest='maxFrames')
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    stage = stageByName(cfg, args.stage)
    env = LandingEnv(cfg, stage=stage)
    renderer = Renderer(cfg.world)
    source = HumanSource(cfg.world.dt)
    record = {'success': 0, 'crash': 0, 'timeout': 0}
    lastOutcome = {'value': '-'}

    def onEpisodeEnd(outcome):
        record[outcome] = record.get(outcome, 0) + 1
        lastOutcome['value'] = outcome
        source.throttle = 0.0

    def hud(isPaused, speed):
        state = env.state
        return [
            f"PLAY  stage {stage.name}   landed {record['success']}  "
            f"crashed {record['crash']}  timeout {record['timeout']}   last: {lastOutcome['value']}"
            f"{'   PAUSED' if isPaused else ''}",
            f"fuel {state.fuel * 100:3.0f}%  cmd {source.throttle * 100:3.0f}%  "
            f"engine {state.spool * 100:3.0f}%  "
            f"vx {state.vx:+5.1f}  vy {state.vy:+5.1f} m/s  "
            f"tilt {math.degrees(state.theta):+5.1f} deg",
            f"limits: speed <= {cfg.world.maxLandingSpeed} m/s  "
            f"tip-over > {math.degrees(math.atan2(cfg.world.legSpan, cfg.world.bodyHalfLen + cfg.world.legDrop)):.0f} deg  on the pad",
            'w/s throttle  a/d rotate  space=pause  r=reset  -/= speed  esc=quit',
        ]

    runEpisodeLoop(
        env, renderer, source, np.random.default_rng(args.seed),
        fps=FPS, onEpisodeEnd=onEpisodeEnd, hudFn=hud, maxFrames=args.maxFrames,
    )


if __name__ == '__main__':
    main()
