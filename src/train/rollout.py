# src/train/rollout.py
# <agent_context>
#   [ARCH]: Vectorized rollout collection + GAE advantage estimation. Ported
#           from tag-simulation minus the opponent plumbing (single agent).
#   [GOTCHA]: Terminal semantics: terminated (touchdown/oob) AND truncated
#             (timeout) are BOTH true MDP terminals here — timeout pays the full
#             terminalCrash by reward design, so it is a real outcome, not a
#             horizon artifact. GAE therefore bootstraps 0 at any done;
#             lastValue bootstraps ONLY a rollout cut mid-episode.
#   [GOTCHA]: The learner samples in tanh space; actions cross into the env
#             through spaces.toEnvAction. Buffers store the PRE-SQUASH u (for
#             PPO ratio recomputation), never the env action.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: computeGae must remain a pure function over ONE env's flat
#               trajectory. Envs are independent — never mix columns in one
#               GAE sweep (computeBatchAdvantages loops per env).
#   [VALIDATION]: python -m pytest tests/test_rollout.py -v
# </agent_guardrail>
"""Vectorized rollout collection and GAE advantage estimation."""
from __future__ import annotations

import numpy as np
import torch

from src.env.spaces import OBS_DIM, ACTION_DIM, toEnvAction


def computeGae(rewards, values, dones, lastValue, gamma, lam):
    """Generalized Advantage Estimation over ONE flat trajectory.

    rewards, values, dones: 1-D arrays of equal length T (dones in {0,1}).
    lastValue: V(s_after_last_step), used only if dones[-1] == 0.
    Returns (advantages, returns), both length T. returns = advantages + values.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    lastAdv = 0.0
    for t in reversed(range(T)):
        nextValue = lastValue if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextValue * nonterminal - values[t]
        lastAdv = delta + gamma * lam * nonterminal * lastAdv
        adv[t] = lastAdv
    returns = adv + values
    return adv, returns


def collectRollout(vecEnv, learner, rolloutSteps, device=None):
    """Collect `rolloutSteps` transitions across all parallel envs with the
    learner sampling stochastically.

    `device` is the torch.device the learner lives on (None -> CPU, the
    fallback). Inference tensors are built there; every result crosses back to
    the env via .cpu().numpy() (a no-op on CPU) so the GPU and CPU paths share
    one code path.

    Returns (batch, lastValues, outcomes): batch holds (rolloutSteps, numEnvs[,dim])
    numpy arrays; lastValues is (numEnvs,) = V(last obs) for the GAE tail
    bootstrap; outcomes counts finished episodes by outcome string (training-time
    success telemetry).
    """
    numEnvs = vecEnv.numEnvs
    obsBuf = np.zeros((rolloutSteps, numEnvs, OBS_DIM), dtype=np.float32)
    uBuf = np.zeros((rolloutSteps, numEnvs, ACTION_DIM), dtype=np.float32)
    logpBuf = np.zeros((rolloutSteps, numEnvs), dtype=np.float32)
    valueBuf = np.zeros((rolloutSteps, numEnvs), dtype=np.float32)
    rewardBuf = np.zeros((rolloutSteps, numEnvs), dtype=np.float32)
    doneBuf = np.zeros((rolloutSteps, numEnvs), dtype=np.float32)
    outcomeCounts = {}

    obs = vecEnv.reset()
    for t in range(rolloutSteps):
        with torch.no_grad():
            # @INVARIANT: inference tensors live on the learner's device;
            # .cpu().numpy() (no-op on CPU) is REQUIRED before crossing back to
            # the NumPy env — Tensor.numpy() raises on a CUDA tensor.
            obsT = torch.as_tensor(obs, dtype=torch.float32, device=device)
            aT, uT, logpT, valueT = learner.sample(obsT)
        envActions = toEnvAction(aT.cpu().numpy())

        nextObs, rewards, terminated, truncated, outcomes = vecEnv.step(envActions)

        obsBuf[t] = obs
        uBuf[t] = uT.cpu().numpy()
        logpBuf[t] = logpT.cpu().numpy()
        valueBuf[t] = valueT.cpu().numpy()
        rewardBuf[t] = rewards
        doneBuf[t] = np.logical_or(terminated, truncated).astype(np.float32)
        for outcome in outcomes:
            if outcome is not None:
                outcomeCounts[outcome] = outcomeCounts.get(outcome, 0) + 1

        obs = nextObs

    with torch.no_grad():
        obsT = torch.as_tensor(obs, dtype=torch.float32, device=device)
        lastValues = learner.valueOf(obsT).cpu().numpy()

    batch = {
        'obs': obsBuf,
        'u': uBuf,
        'logp': logpBuf,
        'value': valueBuf,
        'reward': rewardBuf,
        'done': doneBuf,
    }
    return batch, lastValues, outcomeCounts


def computeBatchAdvantages(batch, lastValues, gamma, lam):
    """Run GAE independently per env column, then return (advantages, returns)
    each shaped (rolloutSteps, numEnvs)."""
    rolloutSteps, numEnvs = batch['reward'].shape
    adv = np.zeros((rolloutSteps, numEnvs), dtype=np.float64)
    ret = np.zeros((rolloutSteps, numEnvs), dtype=np.float64)
    for i in range(numEnvs):
        a, r = computeGae(
            batch['reward'][:, i], batch['value'][:, i], batch['done'][:, i],
            lastValues[i], gamma, lam,
        )
        adv[:, i] = a
        ret[:, i] = r
    return adv, ret
