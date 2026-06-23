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

## Stated goal / direction

Re-aim the whole repository at **specifically targeting and tweaking models to perfectly execute
the IDEAL SUICIDE BURN — a "true hover slam"**: a fuel-optimal, (near-)single-burn descent that
arrests velocity to ~0 *exactly* at the pad, upright and centered. The existing `solis` world
already implements the binary suicide-burn engine; the rewire focuses the reward/success/eval
stack on burn quality. **This rewire has NOT started** — see "Current state".

## Current state (foundation phase)

- Only `src/` (27 `.py` modules across `env/ agents/ train/ runtime/ config/ metrics/`) and a
  populated `requirements.txt` have been replicated, verbatim, from the zip. Foundation is
  **verified importable**: 22/22 modules import from repo root, all files `py_compile` clean.
- **Runnable end-to-end.** The scaffold (`scripts/{train,watch,play,evaluate}.py`, `config.yaml`,
  `configs/{lux,solis}/*.yaml`, `tests/`, `docs/`) was recovered from the zip (commit `ed2d76a`).
  Train via `python -m scripts.train [--stage <name>] [--run N] [--serial]` from repo root.
- **Numbered-run artifacts + live convergence.** Each `scripts.train` session is a run N:
  checkpoints → `checkpoints/run-N/{seed<seed>.pt,best.pt}`, per-iteration metrics →
  `stdout/logs/run-N/seed<seed>.csv`, convergence figure → `stdout/convergence-plots/run-N.png`,
  which **updates live during training** (a `scripts/live_convergence.py` subprocess re-renders it
  from the CSVs every 5 s) and is finalized at the end. Run number auto-increments
  (`src/metrics/live.py:resolveNextRun`; `--run` overrides). The renderer is the unchanged
  `src/metrics/plot.py:plotConvergence`, fed by `src/metrics/live.py`. (Replaces the upstream
  `models/<model>/<env>/` layout.) Artifacts are gitignored; `.gitkeep` keeps the skeleton.
- Local env: `.env.local/` venv on **Python 3.14.5** (gitignored). Deps installed: pymunk 7.3,
  torch 2.12.1, pygame-ce 2.5.7, numpy 2.5, pyyaml 6.0.3, matplotlib, pytest. Run code with the
  repo root on `PYTHONPATH` / as `python -m ...` from root (the `src.` package is not pip-installed).
- **`.claude/AGENTS.md` now documents the foundation:** its `# Project Vigil Redux 2D` blurb,
  `## Quickstart and Commands` (setup + import smoke test + notes), and `## Project Structure`
  (annotated `src/` tree) are filled. There is still **no root `README.md`**. Note for future agents:
  editing `AGENTS.md` was a one-off human-authorized override of the `CLAUDE.md` Hard Rule — the rule
  still stands by default (see `decisions.md` 2026-06-22).

## Two worlds (one config switch: `WorldConfig.engineMode`)

| World | `engineMode` | Throttle behavior |
|-------|--------------|-------------------|
| **lux**   | `analog`      | Continuous spooled throttle in `[0, 1]` |
| **solis** | `suicideBurn` | Binary — command > `SUICIDE_ON_THRESHOLD` (0.5) fires at full; ≤ **2 ignition transitions** then the engine locks |

`engineMode` is a `world:` field, so it is folded into `computeWorldHash` — analog and suicide-burn
checkpoints are mutually incompatible.

## Load-bearing contracts (a rewire must not silently break these)

1. **Frozen obs/action contract** — `src/env/spaces.py`. `OBS_DIM=10`, `ACTION_DIM=2`,
   `VEL_REF=20.0`, `OMEGA_REF=3.0`. θ enters obs only as `(sinθ, cosθ)` (decode `atan2(obs[4],obs[5])`).
   Obs index 9 = `ignitionsRemaining = (2 − engineTransitions)/2` is the **only** ignition signal
   reaching the policy. ⚠️ Changing the obs layout/refs invalidates models **without** changing the world hash.
2. **World-hash checkpoint guard** — `computeWorldHash` (`config/loader.py`) over `world:` fields +
   `PHYSICS_MODEL_VERSION='pymunk-2'`; enforced by `loadCheckpoint` (`agents/checkpoints.py`).
3. **Single gamma** — `training.gamma` is the only discount; shared by GAE (`train/rollout.py`) and reward shaping (`env/rewards.py`).
4. **PBRS `(1 − done)` potential invariance** — `env/rewards.py`; required for policy invariance (Ng et al. 1999).
5. **Config = single control panel** — `config.yaml` + `configs/{lux,solis}/<env>.yaml` → frozen dataclasses (`config/loader.py`); camelCase keys map 1:1 to fields.
6. **Outcome emerges, episode only observes** — `env/physics.py` runs flight + contact in one solver (`_SUBSTEPS=4`); `LandingEnv.step` classifies success/crash/timeout by reading settled state. Impact speed latched once from approach velocity.
7. **Reward arithmetic in one place** — `env/rewards.py:computeReward`; `LandingEnv.step` holds zero reward logic.

## Known gap vs the hover-slam goal (confirmed in source)

No reward term, success predicate, or logged metric currently expresses **velocity-arrested-to-~0**,
**ignition-economy (≤2 / single-burn)**, or **fuel-optimality** — these exist only implicitly via the
gentleness gate (`impactSpeed ≤ maxLandingSpeed`) and the control-effort cost. Success today =
`isUpright AND isOnPad AND isGentle`. The new objective is not yet expressed anywhere.

## Most-likely rewire targets (ranked) — see `notes.md` for why

`env/rewards.py` → `env/episode.py` → `config/loader.py` → `env/physics.py` →
`agents/scripted.py` → `runtime/evaluate.py` → `metrics/logger.py` → (last, high blast radius) `env/spaces.py`.
