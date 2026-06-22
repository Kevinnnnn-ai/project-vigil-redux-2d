# src/train/loop.py
# <agent_context>
#   [ARCH]: Single-stage PPO training driver: collect -> GAE -> update -> eval ->
#           save-best, with the shaping anneal applied per iteration. The
#           curriculum loop that ADVANCES stages automatically lands in M4 and
#           will call this module's pieces; trainLanding itself trains one stage.
#   [GOTCHA]: Saves the BEST policy by eval success rate, not the final weights —
#             the policy keeps sharpening past its peak (tag-simulation finding).
#   [GOTCHA]: The eval env's shapingScale is irrelevant (success rate ignores
#             rewards), so it is never touched.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: shapingScale must be set on the VEC env each iteration BEFORE
#               collecting — the anneal schedule is part of the reward design
#               (see docs/reward-log.md).
#   [VALIDATION]: python -m pytest tests/test_loop.py -v
# </agent_guardrail>
"""trainLanding: PPO on LandingEnv for one curriculum stage."""
from __future__ import annotations

import numpy as np
import torch

from src.env.episode import LandingEnv
from src.env.spaces import OBS_DIM, ACTION_DIM
from src.agents.mlp import MLPPolicy
from src.train.device import resolveDevice
from src.train.vec_env import VecLandingEnv
from src.train.rollout import collectRollout, computeBatchAdvantages
from src.train.ppo import ppoUpdate, explainedVariance
from src.metrics.logger import CsvLogger

# @LOG[prefix-width]: fixed column width for the bracketed "[stage seed N]" tag
# that opens every progress line. Widest tags are the literal 'curriculum' and
# the 'touchdown' stage, so a single seed-digit line is ~21 chars; pad to 24 to
# keep the iter/success/... columns aligned across stages and the PROMOTED line.
LOG_PREFIX_WIDTH = 24


def shapingScaleFor(cfg, it, totalIters):
    """Anneal factor for iteration `it`: linear -> 1 at it 0 down to ~0 at the
    last iter; none -> constant 1.0."""
    if cfg.reward.shapingAnneal == 'linear':
        return 1.0 - it / max(totalIters, 1)
    return 1.0


def evaluateSuccessRate(env, policy, episodes, rng):
    """Deterministic eval (policy.act = squashed mean): fraction of episodes
    ending with outcome == 'success'."""
    wins = 0
    for _ in range(episodes):
        obs = env.reset(rng)
        policy.reset()
        while True:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            if terminated or truncated:
                wins += int(info['outcome'] == 'success')
                break
    return wins / episodes


def trainLanding(cfg, seed, savePath, csvPath=None, stage=None, policy=None):
    """Run PPO for cfg.training.totalIters iterations with cfg.training.numEnvs
    parallel envs on ONE curriculum stage (default: the final/full stage).
    Pass `policy` to continue training existing weights (curriculum handoff).
    Returns a list of per-iteration stat dicts. Saves the best-by-successRate
    policy to savePath."""
    training = cfg.training
    stage = stage if stage is not None else cfg.curriculum.stages[-1]
    torch.manual_seed(seed)
    np.random.seed(seed)

    vecEnv = VecLandingEnv(cfg, numEnvs=training.numEnvs, seed=seed, stage=stage)
    evalEnv = LandingEnv(cfg, stage=stage)
    evalRng = np.random.default_rng(seed + 10_000)
    # @CONFIG[training.device]: GPU-primary, CPU fallback. Move the net BEFORE
    # building the optimizer so its state lands on the same device.
    device = resolveDevice(training.device)
    learner = policy if policy is not None else MLPPolicy(
        OBS_DIM, ACTION_DIM, hidden=training.hidden,
    )
    learner.to(device)
    devicePrefix = f"[train seed {seed}]"
    print(f"{devicePrefix:<{LOG_PREFIX_WIDTH}} device: {device}")
    optimizer = torch.optim.Adam(learner.parameters(), lr=training.lr)
    logger = CsvLogger(csvPath) if csvPath else None

    history = []
    bestRate = -1.0
    for it in range(training.totalIters):
        vecEnv.setShapingScale(shapingScaleFor(cfg, it, training.totalIters))
        batch, lastValues, outcomes = collectRollout(
            vecEnv, learner, training.rolloutSteps, device=device,
        )
        adv, ret = computeBatchAdvantages(
            batch, lastValues, gamma=training.gamma, lam=training.gaeLambda,
        )

        obsT = torch.as_tensor(batch['obs'].reshape(-1, OBS_DIM), dtype=torch.float32, device=device)
        uT = torch.as_tensor(batch['u'].reshape(-1, ACTION_DIM), dtype=torch.float32, device=device)
        oldLogpT = torch.as_tensor(batch['logp'].reshape(-1), dtype=torch.float32, device=device)
        advT = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        retT = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=device)

        stats = ppoUpdate(learner, optimizer, obsT, uT, oldLogpT, advT, retT, training)
        stats['explainedVariance'] = explainedVariance(
            ret.reshape(-1), batch['value'].reshape(-1),
        )
        episodesDone = sum(outcomes.values())
        stats['rolloutSuccess'] = (
            outcomes.get('success', 0) / episodesDone if episodesDone else 0.0
        )
        stats['successRate'] = evaluateSuccessRate(
            evalEnv, learner, training.evalEpisodes, evalRng,
        )
        stats['iter'] = it
        stats['stage'] = stage.name
        history.append(stats)
        if logger:
            logger.log(stats)

        isImproved = stats['successRate'] > bestRate
        if isImproved:
            bestRate = stats['successRate']
            learner.save(
                savePath,
                worldHash=cfg.computeWorldHash(),
                stageName=stage.name,
            )

        prefix = f"[{stage.name} seed {seed}]"
        print(
            f"{prefix:<{LOG_PREFIX_WIDTH}} iter {it:4d}  success {stats['successRate']:.2f}  "
            f"rollout {stats['rolloutSuccess']:.2f}  EV {stats['explainedVariance']:+.2f}  "
            f"entropy {stats['entropy']:.3f}  kl {stats['approxKl']:+.4f}"
            f"{'  <- saved best' if isImproved else ''}",
        )

    if logger:
        logger.close()
    return history
