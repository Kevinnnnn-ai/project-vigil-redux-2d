# src/train/vec_env.py
# <agent_context>
#   [ARCH]: VecLandingEnv — a batch of independent LandingEnvs stepped together.
#           Parallel envs average out single-env gradient noise (the stability
#           fix proven in tag-simulation M2).
#   [GOTCHA]: On step, any env that finished is auto-reset and its returned
#             next-obs is the FRESH episode's initial obs; the reward/terminated/
#             truncated/outcome are for the transition that just ENDED.
#   [GOTCHA]: Each sub-env owns its own numpy Generator spawned from one root
#             seed, so auto-resets are reproducible regardless of which envs
#             finish when.
#   [GOTCHA]: setStage()/setShapingScale() fan out to all sub-envs. Stage takes
#             effect per-env at that env's NEXT reset (LandingEnv semantics) —
#             in-flight episodes finish under their spawn stage.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Keep the step return order (obs, rewards, terminated, truncated,
#               outcomes) — rollout.py unpacks positionally.
#   [VALIDATION]: python -m pytest tests/test_vec_env.py -v
# </agent_guardrail>
"""VecLandingEnv: a batch of LandingEnvs with auto-reset and seeded RNG streams."""
from __future__ import annotations

import numpy as np

from src.env.episode import LandingEnv
from src.env.spaces import OBS_DIM


class VecLandingEnv:
    def __init__(self, cfg, numEnvs, seed, stage=None):
        self.cfg = cfg
        self.numEnvs = numEnvs
        self.envs = [LandingEnv(cfg, stage=stage) for _ in range(numEnvs)]
        self.rngs = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(seed).spawn(numEnvs)
        ]

    def setStage(self, stage):
        for env in self.envs:
            env.setStage(stage)

    def setShapingScale(self, scale):
        for env in self.envs:
            env.shapingScale = scale

    def reset(self):
        obs = np.zeros((self.numEnvs, OBS_DIM), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs[i] = env.reset(self.rngs[i])
        return obs

    def step(self, actions):
        """actions: (numEnvs, 2) ENV-space actions. Returns
        (obs, rewards, terminated, truncated, outcomes); outcomes[i] is the
        ended episode's outcome string for done envs, else None."""
        obs = np.zeros((self.numEnvs, OBS_DIM), dtype=np.float32)
        rewards = np.zeros(self.numEnvs, dtype=np.float32)
        terminated = np.zeros(self.numEnvs, dtype=bool)
        truncated = np.zeros(self.numEnvs, dtype=bool)
        outcomes = [None] * self.numEnvs
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            outcomes[i] = info['outcome']
            if term or trunc:
                o = env.reset(self.rngs[i])   # auto-reset -> fresh-episode obs
            obs[i] = o
        return obs, rewards, terminated, truncated, outcomes
