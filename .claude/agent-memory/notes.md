# Notes

Running observations useful to future agents/sessions. Remove entries that no longer apply.

## Training convergence — run-1/run-2 DIAGNOSED (do NOT just train longer)

The post-rewire training runs (`run-1` = 300 iters/3 seeds, the CHANGELOG rewire
follow-up; `run-2` = 600 iters/3 seeds, narrowed `full`) **do not converge**: no seed
ever reaches `promoteAt` (0.8) on `full`, and the deterministic policy *degrades* on the
hard stages. Full write-up + source citations: `docs/observations.md`
→ **`SUICIDE1_NONCONVERGENCE`**. Three coupled root causes (all verified):
1. `reward.shapingAnneal: linear` zeroes the dense reward over `totalIters`
   (`loop.py:44-45`), so `glide`/`full` (reached late) run terminal-only sparse →
   `policyLoss ≈ 0.001` on `full`.
2. With the reward gradient gone, the constant `entCoef=0.02` term (`ppo.py:69`) inflates
   the free `logStd` (`mlp.py:57`) → σ ~1.0→~3.0 (entropy 2.84→5.0+). Critic is fine
   (`explainedVariance` 0.83–0.92) — failure is policy-side.
3. Promotion fires on a **single** noisy 40-ep eval ≥ 0.8 (`curriculum.py:119`, no
   hysteresis) → stages promote on spikes (drop @0.95 vs mean 0.32), carrying
   under-trained policies forward.
Budget is NOT the lever (2× steps, zero improvement; seed0 stuck in `glide` both runs).
Proposed fixes (floor/per-stage shaping, anneal/lower `entCoef` or cap `logStd`,
N-consecutive-eval promotion) are in the observations entry. CSVs are gitignored.

## Concurrent work — directory agent (live convergence + run-N layout)

**Active (2026-06-22).** The live-convergence / per-run-checkpoint feature is implemented per
`docs/superpowers/specs/2026-06-22-live-convergence-and-run-checkpoints-design.md` (the rewire
spec's §3/§10 out-of-scope counterpart). Ownership / collision map:

- **New files (no collision):** `src/metrics/live.py`, `scripts/live_convergence.py`, `tests/test_live.py`.
- **Shared file `scripts/train.py`:** the directory agent owns its **artifact-path construction** —
  run-number resolution + `checkpoints/run-N/{seed<seed>.pt,best.pt}`, metrics
  `stdout/logs/run-N/seed<seed>.csv`, plot `stdout/convergence-plots/run-N.png` — and the
  live-refresher subprocess spawn/teardown. The suicide-burn rewire owns removing the
  `--model`/`runtime.model` axis. The two are compatible (paths become run-numbered AND
  world-axis-free); sequence edits, don't clobber. `--model`/`--env` survive here only as cosmetic
  plot-title labels (no longer path/config inputs).
- **Artifacts** `checkpoints/run-*/`, `stdout/logs/run-*/`, `stdout/convergence-plots/*.png` are
  gitignored; `.gitkeep` holds the dir skeleton.

## Environment / how to run

- Venv: `.env.local/` (gitignored), Python **3.14.5**. Interpreter:
  `.env.local\Scripts\python.exe`.
- `src.` is a namespace-style package, **not pip-installed**. Import only with the repo root on the
  path: run as `python -m scripts.<name>` from repo root, or set `PYTHONPATH=<repo root>`.
  Running a script by absolute path puts the script's dir (not repo root) on `sys.path` → `ModuleNotFoundError: src`.
- Quick import smoke test (from repo root, with `PYTHONPATH` = repo root): import each of
  `src.env.* src.agents.* src.config.loader src.train.* src.runtime.* src.metrics.*`. All 22 import clean.
- All 7 `__init__.py` are **empty** (no re-exports) → importers must use fully-qualified paths
  (`from src.config.loader import loadConfig`, not `from src.config import loadConfig`).

## Project status

- **Runnable + tested.** `src/` + the scaffold (`scripts/`, `config.yaml`, `tests/`, `docs/`, root
  `README.md`) are present; `python -m pytest -q` is green (150 passed). The single suicide-burn rewire
  is DONE (see below). The original `project-vigil-redux-2d.zip` (gitignored, at repo root) remains the
  source of any not-yet-recovered upstream file. There is no `configs/` dir and no `models/` dir
  (checkpoints live under `checkpoints/run-N/`).

## Suicide-burn rewire — DONE (what changed)

The repo is now a SINGLE binary suicide-burn world. Implemented (see `decisions.md` 2026-06-22):
- `src/env/episode.py` — success gained `isCutOff` (engine commanded off at first toe contact, latched
  from `prevState.engineCommandedOn`); `info` exposes `engineOnAtTouchdown` + `engineTransitions`.
- `src/env/physics.py` — analog branch removed; the binary engine is unconditional.
- `src/config/loader.py` + `config.yaml` — `engineMode`/`minThrottle`/`throttleCutoff`/`runtime.model`
  removed; `PHYSICS_MODEL_VERSION='suicide-1'`; `configs/` deleted.
- `src/agents/scripted.py` — `PdPilot` reworked into a binary single-burn baseline (reads `obs[9]`;
  `IGNITE_LEAD=0.32`, `CUT_SPEED=2.0`; lands ~0.40 touchdown / ~0.20 hop). Gains tuned vs `tests/test_scripted.py`.
- `scripts/{watch,evaluate}.py` — load from `checkpoints/run-N/` via `--run` (no `--model`/`--env`).
- `rewards.py` and `spaces.py` (10-D obs / 2-D action) UNCHANGED.

Carry-forward items (from the per-task reviews):
- `tests/test_evaluate.py` `meanImpactSpeed` margin is thin (1.85 vs 2.0) — flakiness watch if spawns/physics change.
- `tests/test_loop.py::test_evaluateSuccessRateWithPdPilot` asserts only `0<=rate<=1` (its tiny config's
  default 'full' stage is unwinnable for the weak baseline) — weaker coverage, accepted.
- Future direction (NOT current scope): fuel-optimality / tightened-precision reward terms; a stronger
  bang-bang baseline; richer eval (ignition count, fuel-at-touchdown, terminal |x|/|θ|).

## Gotchas confirmed in source

- **Render vs physics flame threshold:** `render.draw` paints a flame at `action[0] > 0.01`, but
  physics treats `> 0.5` as engine-ON in suicide-burn — the rendered flame can mislead for actions in
  (0.01, 0.5]. Align if using the flame to judge ignition timing.
- **Dormant-but-hashed world fields:** `settleTime`, `settleStepCap`, `maxLandingTilt`, `maxLandingOmega`
  are defined in `WorldConfig` but not read by the env (settling is physical now). They still enter the
  world hash, so removing them invalidates existing models.
- **`MLPPolicy.load` uses `torch.load(weights_only=False)`** — fine for local checkpoints; a trust
  concern only if checkpoints ever come from untrusted sources.
- **`plotConvergence` callers:** `scripts/train.py` (final authoritative frame) and, live during
  training, `scripts/live_convergence.py` via `src/metrics/live.py` (re-render from per-seed CSVs).

## Resolved rewire decisions (were open questions)

- "Perfect slam" = safe single-burn landing (upright / on-pad / gentle) + engine cut before touchdown;
  NO fuel-optimality or tightened-precision terms (user choice). Ignition economy is the existing hard
  env cap (≤2 transitions), not a reward penalty. No world tolerances changed beyond removing analog
  fields; `PHYSICS_MODEL_VERSION` bumped to `suicide-1` (no checkpoints existed). Entrypoints recovered
  from the zip. Obs stayed 10-D (no enrichment). `PdPilot` was minimally adapted to binary (not a full rewrite).
