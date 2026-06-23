# Project Context

Durable, concurrent facts about `project-vigil-redux-2d`. This is a snapshot of the
present state and stated direction — not a history. Update in place; do not append a log here
(logs go in `decisions.md`, observations in `notes.md`).

## What this project is

A 2D reinforcement-learning sandbox: a **from-scratch PPO** agent (no stable-baselines3, no
gymnasium registration) learns to fly and land a single-stage **gimbaled rocket booster** in a
**Pymunk (Chipmunk2D) rigid-body** simulation. The booster hull + two legs are one rigid body
that physically collides with a static ground; landing, settling, and tip-over **emerge from the
solver**, not from a scripted verdict. Lineage: replicated from `project-vigil-redux-2d.zip`
(upstream name `2d-ppo-booster-landing`, author Kevin Jie); a separate `../project-vigil-redux/`
is the prior iteration with extensive training runs/checkpoints.

## Goal / direction — the true suicide burn

The repository is re-aimed at ONE task: **train a model to land the booster with a true
suicide-burn thrust profile — ignite the engine once, cut it once, touch down.** The single-burn
rewire is **DONE** (the analog world was removed; success now requires the engine to be cut before
touchdown — see below). Per the user's scope choice, the objective is a **safe single-burn landing**
(upright, on-pad, gentle, engine cut before contact); fuel-optimality and tightened-precision
reward terms were deliberately NOT added (a possible future direction, not current scope).

## Current state

- Runnable + tested. `src/` (27 modules) + the scaffold (`scripts/{train,watch,play,evaluate}.py`,
  `config.yaml`, `tests/`, `docs/`) are present; full suite green (`python -m pytest -q`, 150 passed).
- **Numbered-run artifacts + live convergence** (directory-agent feature): each `scripts.train`
  session is a run N: checkpoints → `checkpoints/run-N/{seed<seed>.pt,best.pt}`, per-iteration
  metrics → `stdout/logs/run-N/seed<seed>.csv`, convergence figure → `stdout/convergence-plots/run-N.png`,
  which **updates live during training** (a `scripts/live_convergence.py` subprocess re-renders it
  from the CSVs every 5 s) and is finalized at the end. Run number auto-increments
  (`src/metrics/live.py:resolveNextRun`; `--run` overrides). Renderer is the unchanged
  `src/metrics/plot.py:plotConvergence`, fed by `src/metrics/live.py`. (Replaced the upstream
  `models/<model>/<env>/` layout.) Artifacts are gitignored; `.gitkeep` keeps the skeleton.
- Local env: `.env.local/` venv on **Python 3.14.5** (gitignored). Deps: pymunk 7.3, torch 2.12.1,
  pygame-ce 2.5.7, numpy 2.5, pyyaml 6.0.3, matplotlib, pytest. Run from repo root as `python -m ...`
  (the `src.` package is not pip-installed).
- `.claude/AGENTS.md` documents the project (foundation-era blurb/quickstart/structure; some of it
  predates the rewire). Editing `AGENTS.md` was a one-off human-authorized override of the `CLAUDE.md`
  Hard Rule — the rule still stands by default (see `decisions.md`). There is now a root `README.md`
  (single-world) plus the recovered `docs/` tree.

## The world: a single binary suicide-burn engine

There is **one world** (the analog/`lux` world and the `solis` name are gone). The engine fires at
**FULL** when the env-action throttle `> SUICIDE_ON_THRESHOLD` (0.5), else **OFF**; at most **two
state-changes** (one ignition, one cutoff) then it locks (`engineTransitions` capped at 2 in
`src/env/physics.py`). Spool lag (`throttleResponse`) and fuel burn still apply.
`obs[9] = ignitionsRemaining = (2 − engineTransitions)/2` (1.0 fresh / 0.5 burning / 0.0 locked) is
the only ignition signal reaching the policy. `engineMode`, `minThrottle`, `throttleCutoff` were
removed from `WorldConfig`; `config.yaml` is the single control panel and there is no `configs/` dir.

## Success rule — cut before touchdown (`src/env/episode.py`)

`success = isUpright AND isOnPad AND isGentle AND isCutOff`, where `isCutOff = not engineOnAtTouchdown`
and `engineOnAtTouchdown` is latched from `prevState.engineCommandedOn` at first toe contact (mirrors
the `impactSpeed` latch). A booster still commanded-on at touchdown is a `crash`. `info` exposes
`engineOnAtTouchdown` and `engineTransitions`. The reward module (`src/env/rewards.py`) is
**unchanged** — the gate rides the existing `terminalCrash` payout.

## Load-bearing contracts (do not silently break)

1. **Frozen obs/action contract** — `src/env/spaces.py`: `OBS_DIM=10`, `ACTION_DIM=2`, `VEL_REF=20.0`,
   `OMEGA_REF=3.0`. θ enters only as `(sinθ, cosθ)`. ⚠️ Changing the obs layout/refs invalidates models
   **without** changing the world hash.
2. **World-hash checkpoint guard** — `computeWorldHash` (`config/loader.py`) over `world:` fields +
   `PHYSICS_MODEL_VERSION='suicide-1'`; enforced by `loadCheckpoint` (`agents/checkpoints.py`).
3. **Single gamma** — `training.gamma` is the only discount; shared by GAE (`train/rollout.py`) and reward shaping (`env/rewards.py`).
4. **PBRS `(1 − done)` potential invariance** — `env/rewards.py` (Ng et al. 1999).
5. **Config = single control panel** — `config.yaml` → frozen dataclasses (`config/loader.py`); camelCase keys map 1:1 to fields. No per-world configs.
6. **Outcome emerges, episode only observes** — `env/physics.py` runs flight + contact in one solver (`_SUBSTEPS=4`); `LandingEnv.step` classifies success/crash/timeout from settled state.
7. **Reward arithmetic in one place** — `env/rewards.py:computeReward`; `LandingEnv.step` holds zero reward logic.

## Scripts / commands (run from repo root, `.env.local` venv)

- `python -m scripts.train [--stage <name>] [--run N] [--serial]` — train; checkpoints → `checkpoints/run-N/`.
- `python -m scripts.watch [--run N] [--checkpoint best|seed<N>] [--stage <name>] [--pilot pd]` — watch a checkpoint (or the PD pilot) fly.
- `python -m scripts.evaluate [--run N] [--checkpoint ...] [--episodes K]` — headless eval vs the PdPilot baseline.
- `python -m scripts.play` — fly it yourself.
- No `--model`/`--env` selection axis (one world). `train` keeps `--model`/`--env` only as optional cosmetic plot-title labels.

## Baseline

`PdPilot` (`src/agents/scripted.py`) is now a **weak binary single-burn baseline** (coast → ignite once
→ cut before touchdown; lands ~40% on the easy touchdown stage). Still the forever-baseline RL must beat.
