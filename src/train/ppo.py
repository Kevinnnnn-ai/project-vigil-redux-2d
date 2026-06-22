# src/train/ppo.py
# <agent_context>
#   [ARCH]: Hand-written PPO update — clipped surrogate + value loss + entropy
#           bonus, with explained-variance and KL/clip diagnostics. Ported
#           verbatim from tag-simulation (task-agnostic: consumes tensors).
#   [GOTCHA]: Advantages are normalized PER UPDATE (over the whole batch), not
#             per minibatch — minibatch-level normalization adds noise.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Do not reintroduce task-specific logic here; this module reads
#               hyperparameters off cfg.training and nothing else.
#   [VALIDATION]: python -m pytest tests/test_ppo.py -v
# </agent_guardrail>
"""Hand-written PPO update. Reads hyperparameters off a config object exposing
epochs, minibatchSize, clipEps, vfCoef, entCoef, maxGradNorm (cfg.training)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def explainedVariance(returns, values):
    """1 - Var(returns - values) / Var(returns). Scale-invariant critic quality
    (survives the reward-shaping anneal). ~1 perfect, ~0 useless, <0 worse than
    predicting the mean. Returns 0.0 if returns has zero variance."""
    returns = np.asarray(returns, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    var = returns.var()
    if var == 0:
        return 0.0
    return float(1.0 - (returns - values).var() / var)


def ppoUpdate(policy, optimizer, obs, u, oldLogp, advantages, returns, train):
    """One PPO update over `train.epochs` passes of minibatches.

    obs:(N,obsDim) u:(N,actDim) oldLogp:(N,) advantages:(N,) returns:(N,) tensors.
    Advantages are normalized per update. Returns a dict of averaged stats.
    """
    n = obs.shape[0]
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {
        'policyLoss': 0.0,
        'valueLoss': 0.0,
        'entropy': 0.0,
        'approxKl': 0.0,
        'clipFrac': 0.0,
    }
    nUpdates = 0

    for _ in range(train.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, train.minibatchSize):
            idx = perm[start:start + train.minibatchSize]
            newLogp, entropy, value = policy.evaluateActions(obs[idx], u[idx])

            ratio = torch.exp(newLogp - oldLogp[idx])
            mbAdv = adv[idx]
            unclipped = ratio * mbAdv
            clipped = torch.clamp(ratio, 1 - train.clipEps, 1 + train.clipEps) * mbAdv
            policyLoss = -torch.min(unclipped, clipped).mean()

            valueLoss = 0.5 * (returns[idx] - value).pow(2).mean()
            ent = entropy.mean()

            loss = policyLoss + train.vfCoef * valueLoss - train.entCoef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), train.maxGradNorm)
            optimizer.step()

            with torch.no_grad():
                approxKl = (oldLogp[idx] - newLogp).mean().item()
                clipFrac = ((ratio - 1.0).abs() > train.clipEps).float().mean().item()
            stats['policyLoss'] += policyLoss.item()
            stats['valueLoss'] += valueLoss.item()
            stats['entropy'] += ent.item()
            stats['approxKl'] += approxKl
            stats['clipFrac'] += clipFrac
            nUpdates += 1

    for key in stats:
        stats[key] /= max(nUpdates, 1)
    return stats
