# src/agents/checkpoints.py
# <agent_context>
#   [ARCH]: The one place checkpoint paths are resolved and the worldHash
#           compatibility guard is enforced (the M0 contract: a model is
#           loadable iff its stored world hash matches the live config's).
#           scripts/{watch,play,evaluate}.py all load through here.
#   [GOTCHA]: resolveModelPath accepts 'best', 'seed<N>' (bare names get .pt
#             appended) or an explicit existing path, resolved against a SINGLE
#             model dir. Models are namespaced TWO levels deep:
#             models/<model>/<env>/ (e.g. models/lux/baseline/); callers pass
#             models/<model>/<env> as the dir. The <env> level is organizational
#             only — compatibility is the worldHash guard's job, not the path.
#             A miss raises FileNotFoundError that LISTS the dir contents — the
#             error message is part of the UX, keep it informative.
# </agent_context>
# <agent_guardrail>
#   [CRITICAL]: Never skip or soften the hash check here. Watching a model in
#               the wrong physics silently produces nonsense flight.
#   [VALIDATION]: python -m pytest tests/test_checkpoints.py -v
# </agent_guardrail>
"""Hash-guarded checkpoint resolution and loading."""
from __future__ import annotations

import os

from src.agents.mlp import MLPPolicy


def resolveModelPath(modelsDir, name):
    """Turn a model selector into a real path. `name` is 'best', 'seed<N>',
    a bare filename, or an explicit path."""
    if os.path.exists(name):
        return name
    fileName = name if name.endswith('.pt') else f'{name}.pt'
    path = os.path.join(modelsDir, fileName)
    if not os.path.exists(path):
        existing = sorted(
            entry for entry in os.listdir(modelsDir) if entry.endswith('.pt')
        ) if os.path.isdir(modelsDir) else []
        listing = ', '.join(existing) if existing else '(none)'
        raise FileNotFoundError(
            f'no checkpoint {fileName!r} in {modelsDir!r}; available: {listing}',
        )
    return path


def loadCheckpoint(path, expectedWorldHash):
    """Load (policy, meta) and enforce world compatibility: meta['worldHash']
    must equal the live config's hash, else ValueError naming both."""
    policy, meta = MLPPolicy.load(path)
    if meta['worldHash'] != expectedWorldHash:
        raise ValueError(
            f'checkpoint {path!r} was trained in world {meta["worldHash"]!r} but '
            f'the current config.yaml world hashes to {expectedWorldHash!r} — '
            f'the physics differ; retrain or restore the old world settings',
        )
    return policy, meta
