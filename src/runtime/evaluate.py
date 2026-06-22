# src/runtime/evaluate.py
"""Deterministic policy evaluation with outcome breakdown — the stats the
training CSV never shows. Used by scripts/evaluate.py for net-vs-baseline
tables; train-time promotion uses the lighter train.loop.evaluateSuccessRate."""
from __future__ import annotations

import numpy as np


def runEvaluation(env, policy, episodes, rng):
    """Roll `episodes` deterministic episodes (policy.act = squashed mean for
    nets). Returns {episodes, successRate, outcomes, meanImpactSpeed, meanSteps};
    meanImpactSpeed averages only over touchdown episodes (success or crash)."""
    outcomes = {}
    impacts = []
    steps = []
    wins = 0
    for _ in range(episodes):
        obs = env.reset(rng)
        policy.reset()
        while True:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            if terminated or truncated:
                outcome = info['outcome']
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                wins += int(outcome == 'success')
                if outcome in ('success', 'crash'):
                    impacts.append(info['impactSpeed'])
                steps.append(env.t)
                break
    return {
        'episodes': episodes,
        'successRate': wins / episodes,
        'outcomes': outcomes,
        'meanImpactSpeed': float(np.mean(impacts)) if impacts else 0.0,
        'meanSteps': float(np.mean(steps)),
    }
