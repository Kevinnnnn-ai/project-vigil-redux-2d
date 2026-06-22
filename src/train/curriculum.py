# src/train/curriculum.py
# <agent_context>
#   [ARCH]: The automatic stage ladder — trainLanding's iteration body plus a
#           stage pointer. Promotion happens only at eval iters (it % evalEvery
#           == 0, plus the final iter): successRate >= curriculum.promoteAt
#           advances to the next stage with the SAME policy, optimizer, and
#           anneal clock (global linear over totalIters). This formalizes the
#           hand-rolled ladder that fixed M2 (see observations:
#           M2_DISCOUNT_PROCRASTINATION_EXPLOITS).
#   [GOTCHA]: evaluateFn is injectable so tests drive promotion with scripted
#             rates instead of real training. Default is the real
#             evaluateSuccessRate from train.loop.
#   [GOTCHA]: savePath gets the best-by-success policy ON THE FINAL STAGE. If
#             the run never reaches the final stage, the latest promotion-time
#             snapshot is saved there instead (logged loudly) so a checkpoint
#             always exists.
#   [GOTCHA]: Non-eval iters write successRate -1 to keep the CSV header stable
#             (CsvLogger derives columns from the FIRST record).
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Do not reset the policy, the optimizer, or the anneal clock at
#               a promotion — skill transfer across stages is the whole point.
#   [VALIDATION]: python -m pytest tests/test_curriculum.py -v
# </agent_guardrail>
"""trainCurriculum: PPO up the spawn-difficulty ladder, promoting on eval
success rate."""
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
from src.train.loop import shapingScaleFor, evaluateSuccessRate, LOG_PREFIX_WIDTH
from src.metrics.logger import CsvLogger


def trainCurriculum(cfg, seed, savePath, csvPath=None, evaluateFn=None):
    """Run PPO for cfg.training.totalIters, climbing curriculum stages on
    promotion evals. Returns the per-iteration history. Saves the best
    final-stage policy (or the last promotion snapshot) to savePath."""
    training = cfg.training
    stages = cfg.curriculum.stages
    promoteAt = cfg.curriculum.promoteAt
    evaluateFn = evaluateFn if evaluateFn is not None else evaluateSuccessRate
    torch.manual_seed(seed)
    np.random.seed(seed)

    stageIdx = 0
    vecEnv = VecLandingEnv(cfg, numEnvs=training.numEnvs, seed=seed, stage=stages[0])
    evalEnv = LandingEnv(cfg, stage=stages[0])
    evalRng = np.random.default_rng(seed + 10_000)
    # @CONFIG[training.device]: GPU-primary, CPU fallback. Move the net BEFORE
    # building the optimizer so its state lands on the same device.
    device = resolveDevice(training.device)
    learner = MLPPolicy(OBS_DIM, ACTION_DIM, hidden=training.hidden)
    learner.to(device)
    devicePrefix = f"[curriculum seed {seed}]"
    print(f"{devicePrefix:<{LOG_PREFIX_WIDTH}} device: {device}")
    optimizer = torch.optim.Adam(learner.parameters(), lr=training.lr)
    logger = CsvLogger(csvPath) if csvPath else None

    history = []
    bestFinalRate = -1.0
    hasSavedCheckpoint = False

    def saveSnapshot():
        learner.save(
            savePath,
            worldHash=cfg.computeWorldHash(),
            stageName=stages[stageIdx].name,
        )

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
        stats['iter'] = it
        stats['stage'] = stages[stageIdx].name
        stats['successRate'] = -1.0      # sentinel; eval iters overwrite
        stats['promoted'] = 0

        isEvalIter = (it % training.evalEvery == 0) or (it == training.totalIters - 1)
        if isEvalIter:
            rate = evaluateFn(evalEnv, learner, training.evalEpisodes, evalRng)
            stats['successRate'] = rate
            isFinalStage = stageIdx == len(stages) - 1

            if isFinalStage:
                if rate > bestFinalRate:
                    bestFinalRate = rate
                    saveSnapshot()
                    hasSavedCheckpoint = True
            elif rate >= promoteAt:
                saveSnapshot()           # stage-complete snapshot (overwritten later)
                hasSavedCheckpoint = True
                stageIdx += 1
                vecEnv.setStage(stages[stageIdx])
                evalEnv = LandingEnv(cfg, stage=stages[stageIdx])
                stats['promoted'] = 1
                # @LOG[aligned-prefix]: left-pad the bracketed tag to a fixed
                # width so the iter/success/... columns line up across stages
                # (stage names vary in length: hop..touchdown..curriculum).
                promotePrefix = f"[curriculum seed {seed}]"
                print(
                    f"{promotePrefix:<{LOG_PREFIX_WIDTH}} iter {it:4d}  PROMOTED -> "
                    f"stage {stages[stageIdx].name} (rate {rate:.2f})",
                )

            evalPrefix = f"[{stats['stage']} seed {seed}]"
            print(
                f"{evalPrefix:<{LOG_PREFIX_WIDTH}} iter {it:4d}  success {rate:.2f}  "
                f"rollout {stats['rolloutSuccess']:.2f}  "
                f"EV {stats['explainedVariance']:+.2f}  entropy {stats['entropy']:.3f}",
            )

        history.append(stats)
        if logger:
            logger.log(stats)

    if not hasSavedCheckpoint:
        saveSnapshot()
        fallbackPrefix = f"[curriculum seed {seed}]"
        print(f"{fallbackPrefix:<{LOG_PREFIX_WIDTH}} no promotion reached — saved final weights")
    if logger:
        logger.close()
    return history
