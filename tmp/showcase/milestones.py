# tmp/showcase/milestones.py
"""Single source of truth for the reward/curriculum showcase milestones.

Each milestone reconstructs a documented past training system (docs/REWARD_LOG.md +
git history) on TODAY's world, so every retrained model shares the current world
hash and co-views. gen_configs/train_all/gallery all import MILESTONES.
Each milestone applies its reward/training deltas on top of config.yaml and selects
a subset of the base curriculum stages (optionally overriding the 'full' spawn)."""


BASE_STAGES = ('touchdown', 'hop', 'drop', 'glide', 'full')
# The pre-narrowing 'full' spawn (git 08fcc4d): wide drop window, full lateral span.
WIDE_FULL = {'altitude': [40.0, 52.0], 'xOffset': [-14.0, 14.0]}


MILESTONES = [
    {
        'name': 'm1-original-shaping',
        'file': 'm1-original-shaping.yaml',
        'run': 7001,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.01, 'totalIters': 220},
        'stages': ['hop', 'drop', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-12 original/rev1 (approx; oobPenalty + no-walls excluded)',
        'fidelity': 'approx',
        'note': 'No touchdown/glide rung -> success unsamplable; faithful failure.',
    },
    {
        'name': 'm2-walls-touchdown',
        'file': 'm2-walls-touchdown.yaml',
        'run': 7002,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.01, 'totalIters': 220},
        'stages': ['touchdown', 'hop', 'drop', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-12 rev2 (approx)',
        'fidelity': 'approx',
        'note': 'Touchdown rung, no glide; entCoef 0.01.',
    },
    {
        'name': 'm3-m5-glide',
        'file': 'm3-m5-glide.yaml',
        'run': 7003,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 260},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'REWARD_LOG 2026-06-13 M5 (approx; ~= m4 minus 40 iters)',
        'fidelity': 'approx',
        'note': 'Glide rung + entCoef 0.02 are today baseline; near-duplicate of m4.',
    },
    {
        'name': 'm4-suicide-run1',
        'file': 'm4-suicide-run1.yaml',
        'run': 7004,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 300},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': WIDE_FULL,
        'source': 'git 08fcc4d (exact); represents 06-15 timing & 06-16 pymunk',
        'fidelity': 'exact',
        'note': 'Suicide run-1 baseline: wide full, 300 iters, anneal linear.',
    },
    {
        'name': 'm5-run2',
        'file': 'm5-run2.yaml',
        'run': 7005,
        'reward': {'shapingAnneal': 'linear'},
        'training': {'entCoef': 0.02, 'totalIters': 600},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': None,
        'source': 'git 36d58ce (exact)',
        'fidelity': 'exact',
        'note': 'Run-2: narrow full [52,52], 600 iters, anneal linear.',
    },
    {
        'name': 'm6-anneal-none',
        'file': 'm6-anneal-none.yaml',
        'run': 7006,
        'reward': {'shapingAnneal': 'none'},
        'training': {'entCoef': 0.02, 'totalIters': 600},
        'stages': ['touchdown', 'hop', 'drop', 'glide', 'full'],
        'fullOverride': None,
        'source': 'HEAD config.yaml (exact)',
        'fidelity': 'exact',
        'note': 'Current shipped config: anneal none.',
    },
]
