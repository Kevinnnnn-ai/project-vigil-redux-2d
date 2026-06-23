# Workflows

The exact commands for the common loops in this repo. Run **everything from the repo root** —
`src` is a namespace package (`import src.env.episode`), not pip-installed, so the cwd must be
the repo root for imports and relative paths (`models/…`, `stdout/metrics/…`) to resolve. See
[AGENTS.md](AGENTS.md) for the docs overview and `docs/personal/commands.md` for the author's
short cheat-sheet.

> **Configuration is `config.yaml` only** — there is no `configs/` directory and no
> `--model`/`--env` world-selection axis. The world hash comes from `config.yaml` and gates
> checkpoint loading. See [CONVENTIONS.md](CONVENTIONS.md) §4.

## Setup

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate     Unix: . .venv/bin/activate
pip install -r requirements.txt       # torch, numpy, pyyaml, pygame, pymunk
python -m pytest -q                   # full suite, headless
```

`pymunk` is a hard dependency — a fresh clone fails `pytest` until `pip install` runs.

## Run the tests

```sh
python -m pytest -q                       # full suite (testpaths=tests via pytest.ini)
python -m pytest tests/test_rewards.py -v  # one file
```

The load-bearing guards must stay green: `test_shapingTelescopesToInitialPotential` (the PBRS
`(1−done)` invariance), `tests/test_scripted.py` (the PdPilot solvability thresholds), and the
worldHash guard tests.

## Train a model

```sh
python -m scripts.train                    # full curriculum: touchdown -> full
python -m scripts.train --stage hop        # single stage, no promotion
python -m scripts.train --run 4            # force a specific run number (default: auto-increment)
python -m scripts.train --serial           # train seeds one-at-a-time (debug/repro)
```

- `--stage` = train one curriculum rung instead of climbing the ladder (omit for production).
- `--run N` = force the run number (default: auto-increment).
- `--serial` = sequential seeds (default is concurrent — see gotcha 2 below).

Output: `checkpoints/run-N/seed<seed>.pt` per seed, `checkpoints/run-N/best.pt` (best across
seeds). Per-iteration metrics CSVs land in `stdout/logs/run-N/`. A live-updating convergence
PNG is written to `stdout/convergence-plots/run-N.png`. All of `checkpoints/` and `stdout/`
are gitignored (except `.gitkeep`).

## Watch a trained model land (pygame window)

```sh
python -m scripts.watch                                        # latest run, default checkpoint
python -m scripts.watch --run 3 --checkpoint best              # specific run + checkpoint
python -m scripts.watch --run 3 --checkpoint seed1 --stage drop  # specific seed/stage
python -m scripts.watch --pilot pd                             # scripted PD pilot, no checkpoint
```

`--run N` selects the run (default: latest). `--checkpoint` selects within `checkpoints/run-N/`
(`best`, `seed<N>`, or a path). Controls: `space`=pause, `n`=step, `r`=reset, `-`/`=`=speed,
`esc`=quit.

## Evaluate vs the PD-pilot baseline (headless)

```sh
python -m scripts.evaluate                                       # latest run, default checkpoint
python -m scripts.evaluate --run 3 --stage full --episodes 200
python -m scripts.evaluate --run 3 --checkpoint seed0
```

Prints, for both the trained net AND `PdPilot` on the same seeds: success rate, outcome
breakdown (success/crash/timeout), mean impact speed, mean episode length. Eval is
**deterministic** (uses the squashed mean — see [CONVENTIONS.md](CONVENTIONS.md) §5).

## Fly it yourself (human pilot — no model)

```sh
python -m scripts.play
python -m scripts.play --stage hop
```

Controls: `w`/`s` (up/down) = throttle, `a`/`d` (left/right) = gimbal, `space`=pause, `n`=step,
`r`=reset, `-`/`=`=speed, `esc`/`q`=quit. HUD shows fuel, throttle, spool, velocity, tilt, and
the landing record.

## Add or tune a reward term (end to end)

Owned by the `reward-shaper` subagent. The loop:

1. **Edit the reward** in `src/env/rewards.py` — `computePotential` for shaping, `computeReward`
   for terminal/control-cost. Keep shaping potential-based and keep the `(1 − done)` factor.
2. **Add config keys** to `config.yaml:reward`, plus the `RewardConfig` dataclass in
   `src/config/loader.py`.
3. **Test** — `python -m pytest tests/test_rewards.py -v`. Do **not** break
   `test_shapingTelescopesToInitialPotential`.
4. **Annotate** — header block + `@TAG[id]` landmarks (`code-annotation` skill).
5. **Train + inspect** — `scripts.train`, then `scripts.watch` (look for reward hacking:
   hovering, oscillation, exploiting a term) and `scripts.evaluate --episodes 200`.
6. **Log it** — a new [REWARD_LOG.md](REWARD_LOG.md) entry (hypothesis / config / result /
   verdict). This is a **hard rule**: every reward version is logged. Cross-link the rationale
   in [CHANGELOG.md](CHANGELOG.md) and any finding in [OBSERVATIONS.md](OBSERVATIONS.md).

## Command gotchas (the ones that bite)

1. **CPU beats GPU here** — configs default to `training.device: cpu`; CPU is ~2.8× *faster*
   for this launch-bound 64×64 stack. To try GPU, set `device: auto` in the config (no CLI
   flag). Device never enters the world hash, so switching doesn't invalidate models. See
   `CPU_BEATS_GPU_FOR_THIS_PPO` in [OBSERVATIONS.md](OBSERVATIONS.md).
2. **Parallel ≡ serial per-seed** — seeds train concurrently (one process per seed; capped by
   `training.seedWorkers: auto` = `min(#seeds, cpu_count)`). Parallel and serial yield
   **identical** per-seed results; only console line ordering interleaves. Use `--serial` (or
   `seedWorkers: 1`) only for debugging.
3. **`config.yaml` is the world-hash source** — the world hash comes from `config.yaml`.
   A config that does not match the checkpoint's stored hash rejects it at load (`ValueError`).
4. **No `gymnasium.make()`** — `LandingEnv` is gym-*style* but not registered; instantiate it
   directly.

## Before committing

- `python -m pytest -q` is green.
- The `code-annotation` skill applied to any edited `src/` file (header block + `@TAG[id]`).
- Docs updated **as part of the diff** ([CONVENTIONS.md](CONVENTIONS.md) §7): [CHANGELOG.md](CHANGELOG.md)
  for behavior, [REWARD_LOG.md](REWARD_LOG.md) for reward changes, [OBSERVATIONS.md](OBSERVATIONS.md)
  for findings.
- For an RL-correctness-sensitive change, the `rl-reviewer` subagent has seen it.
