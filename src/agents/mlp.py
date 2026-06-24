# src/agents/mlp.py
# <agent_context>
#   [ARCH]: Actor-critic MLP with a tanh-squashed diagonal Gaussian policy and a
#           learned state-independent log-std. Direct port of tag-simulation's
#           MLPPolicy with ONE boundary change: act() returns the ENV-space
#           action ([throttle 0..1, gimbal -1..1]) via spaces.toEnvAction, so
#           every Policy consumer sees one action convention.
#   [GOTCHA]: sample()/evaluateActions() stay entirely in tanh/pre-squash space —
#             PPO ratios are computed on u, never on env actions. The affine
#             throttle map is applied only where actions cross into the env
#             (rollout collection, act()).
#   [GOTCHA]: logStd is a free nn.Parameter (state-independent). Init 0.0 ->
#             std 1.0 for ample early exploration.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: _squashedLogProb includes the log(1 - tanh(u)^2) Jacobian
#               correction. Do not remove the eps, change the sum axis, or flip
#               the sign — exact bounded-action densities depend on it.
#   [CRITICAL]: act() must stay deterministic (squashed MEAN, no sampling) —
#               eval/watch/play all rely on it.
#   [VALIDATION]: python -m pytest tests/test_mlp.py -v
# </agent_guardrail>
"""Actor-critic MLP with a tanh-squashed Gaussian policy.

Implements the Policy interface (act, env-space) for inference plus training
methods (sample, evaluateActions, valueOf) in tanh/pre-squash space."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.agents.policy import Policy
from src.env.spaces import toEnvAction

_LOG_STD_INIT = 0.0    # std = exp(0) = 1.0 at init -> ample early exploration
_TANH_EPS = 1e-6       # guards log(1 - tanh^2) against log(0)


class MLPPolicy(nn.Module, Policy):
    def __init__(self, obsDim, actDim, hidden=(64, 64)):
        nn.Module.__init__(self)
        self.obsDim = obsDim
        self.actDim = actDim
        self.hidden = tuple(hidden)

        layers = []
        last = obsDim
        for h in self.hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.meanHead = nn.Linear(last, actDim)
        self.valueHead = nn.Linear(last, 1)
        self.logStd = nn.Parameter(torch.full((actDim,), _LOG_STD_INIT))

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.meanHead(h)
        value = self.valueHead(h).squeeze(-1)
        return mean, value

    def _policyParams(self, obs):
        """Return (mean, std, value). std broadcasts over the batch."""
        h = self.trunk(obs)
        mean = self.meanHead(h)
        value = self.valueHead(h).squeeze(-1)
        std = self.logStd.exp()
        return mean, std, value

    def _squashedLogProb(self, u, mean, std):
        """log p(a) for a = tanh(u), summed over action dims.
        log p(a) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2 + eps)."""
        base = Normal(mean, std).log_prob(u).sum(-1)
        correction = torch.log(1.0 - torch.tanh(u) ** 2 + _TANH_EPS).sum(-1)
        return base - correction

    @torch.no_grad()
    def act(self, obs):
        """Policy interface: deterministic inference for eval/watch. Acts on the
        squashed MEAN (no sampling) so rendered flight isn't jittery.
        numpy (OBS_DIM=11,) in -> numpy (ACTION_DIM=2,) ENV action out
        ([throttle 0..1, gimbal -1..1])."""
        # @INVARIANT: build the input on the MODEL's own device so act() stays
        # self-contained for watch/play/eval regardless of where the net lives;
        # .cpu() before .numpy() (no-op on CPU) guards the CUDA-tensor .numpy() trap.
        device = next(self.parameters()).device
        obsT = torch.as_tensor(
            np.asarray(obs), dtype=torch.float32, device=device,
        ).unsqueeze(0)
        mean, _ = self.forward(obsT)
        return toEnvAction(torch.tanh(mean).squeeze(0).cpu().numpy())

    def sample(self, obs):
        """Stochastic TANH-SPACE action for rollout collection. obs: (N, obsDim)
        tensor. Returns (a, u, logp, value); a = tanh(u) in (-1, 1)^2 — callers
        convert with toEnvAction before stepping the env."""
        mean, std, value = self._policyParams(obs)
        dist = Normal(mean, std)
        u = dist.sample()
        a = torch.tanh(u)
        logp = self._squashedLogProb(u, mean, std)
        return a, u, logp, value

    def evaluateActions(self, obs, u):
        """For the PPO update: recompute (logp, entropy, value) for stored
        pre-squash samples u under CURRENT params. Gradients flow."""
        mean, std, value = self._policyParams(obs)
        logp = self._squashedLogProb(u, mean, std)
        entropy = Normal(mean, std).entropy().sum(-1)
        return logp, entropy, value

    @torch.no_grad()
    def valueOf(self, obs):
        """Critic value for a batch of observations: (N, obsDim) -> (N,).
        Used for the GAE bootstrap of a rollout cut mid-episode."""
        _, _, value = self._policyParams(obs)
        return value

    def save(self, path, *, worldHash, stageName):
        """Checkpoint: weights + architecture + provenance. worldHash is the
        compatibility guard the M3 watch path checks against the live config."""
        torch.save({
            'stateDict': self.state_dict(),
            'obsDim': self.obsDim,
            'actDim': self.actDim,
            'hidden': list(self.hidden),
            'worldHash': worldHash,
            'stageName': stageName,
        }, path)

    @classmethod
    def load(cls, path):
        """Return (policy, meta) where meta has worldHash and stageName."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        net = cls(ckpt['obsDim'], ckpt['actDim'], hidden=tuple(ckpt['hidden']))
        net.load_state_dict(ckpt['stateDict'])
        net.eval()
        return net, {'worldHash': ckpt['worldHash'], 'stageName': ckpt['stageName']}
