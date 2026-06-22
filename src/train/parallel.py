# src/train/parallel.py
# <agent_context>
#   [ARCH]: Runs the evalSeeds training runs CONCURRENTLY, one OS process per
#           seed (ProcessPoolExecutor). Each seed already builds its own env,
#           net, optimizer, and RNG streams (see train/loop.py,
#           train/curriculum.py), so seeds are naturally process-isolated — this
#           module only fans them out and gathers per-seed results. It does NOT
#           touch trainLanding/trainCurriculum; the per-seed math is unchanged
#           from the old sequential loop in scripts/train.py.
#   [GOTCHA]: maxWorkers == 1 runs IN-PROCESS (no pool) — that is the --serial /
#             seedWorkers=1 path AND what tests use to stay deterministic. The
#             same _runSeed() body runs either way, so results are identical;
#             only console line ORDERING interleaves under a real pool.
#   [GOTCHA]: On Windows multiprocessing uses 'spawn', so _runSeed must be a
#             module-level function and every SeedTask field must be picklable —
#             hence we pass the config PATH (a str) and reload in the child, not
#             the frozen Config object.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Results are returned SORTED BY SEED regardless of completion
#               order — the downstream best.pt pick, summary, and convergence
#               plot in scripts/train.py rely on a stable seed ordering.
#   [CRITICAL]: Fail-fast — the first worker exception propagates. A crashed
#               seed is a real error, not something to silently skip.
#   [VALIDATION]: python -m pytest tests/test_parallel.py -v
# </agent_guardrail>
"""Parallel per-seed PPO training: one process per evalSeed, results by seed."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from src.config.loader import loadConfig


# @ANCHOR[seed-task]: the picklable unit of work handed to one child process.
# Only primitives (str/int) — see the 'spawn' GOTCHA in the header.
@dataclass
class SeedTask:
    configPath: str
    seed: int
    savePath: str
    csvPath: str | None
    stageName: str | None      # single-stage name -> trainLanding; None -> curriculum
    modelName: str             # carried for log/debug context, not load logic
    envName: str


@dataclass
class SeedResult:
    seed: int
    bestRate: float
    savePath: str
    history: list


def stageByName(cfg, name):
    """Resolve a curriculum stage by name, or raise with the known names. Shared
    by the worker and scripts/train.py so single-stage resolution lives once."""
    for stage in cfg.curriculum.stages:
        if stage.name == name:
            return stage
    names = [stage.name for stage in cfg.curriculum.stages]
    raise SystemExit(f'unknown stage {name!r}; config has {names}')


def _bestCurriculumRate(history, finalStageName):
    """Best success rate recorded ON THE FINAL STAGE (eval iters only); 0.0 if
    the ladder never reached it. Matches the old scripts/train.py logic."""
    return max(
        (
            record['successRate'] for record in history
            if record['stage'] == finalStageName and record['successRate'] >= 0
        ),
        default=0.0,
    )


# @ANCHOR[run-seed]: the worker body. Module-level + primitives-only so it
# pickles under 'spawn'. Imports the trainers lazily to keep torch out of the
# parent until a child needs it.
def _runSeed(task: SeedTask) -> SeedResult:
    """Train ONE seed end to end and return its best success rate. Runs in a
    child process under a real pool, or in-process on the serial path."""
    import torch

    from src.train.loop import trainLanding
    from src.train.curriculum import trainCurriculum

    # @INVARIANT: cap BLAS/torch threads to 1 per child so N seed-processes do
    # not each spawn cpu_count threads and oversubscribe the machine.
    torch.set_num_threads(1)

    cfg = loadConfig(task.configPath)
    if task.stageName is not None:
        stage = stageByName(cfg, task.stageName)
        history = trainLanding(cfg, task.seed, task.savePath, task.csvPath, stage=stage)
        bestRate = max(record['successRate'] for record in history)
    else:
        history = trainCurriculum(cfg, task.seed, task.savePath, task.csvPath)
        bestRate = _bestCurriculumRate(history, cfg.curriculum.stages[-1].name)
    return SeedResult(task.seed, bestRate, task.savePath, history)


def resolveSeedWorkers(cfg, serial: bool) -> int:
    """How many seeds to train at once. --serial wins; then config.seedWorkers:
    'auto' caps at min(numSeeds, cpu_count), an int clamps to [1, numSeeds]."""
    numSeeds = len(cfg.training.evalSeeds)
    if serial:
        return 1
    workers = cfg.training.seedWorkers
    if workers == 'auto':
        return max(1, min(numSeeds, os.cpu_count() or 1))
    return max(1, min(int(workers), numSeeds))


def runSeeds(tasks: list[SeedTask], maxWorkers: int) -> list[SeedResult]:
    """Train every task's seed, returning one SeedResult per seed SORTED BY SEED.
    maxWorkers == 1 runs in-process (deterministic, no pool); > 1 fans out across
    processes. The first worker exception propagates (fail-fast)."""
    if maxWorkers <= 1:
        results = [_runSeed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=maxWorkers) as pool:
            # .map preserves submission order and re-raises the first exception.
            results = list(pool.map(_runSeed, tasks))
    return sorted(results, key=lambda r: r.seed)
