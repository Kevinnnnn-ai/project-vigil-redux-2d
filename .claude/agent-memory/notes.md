# Notes

Running observations useful to future agents/sessions. Remove entries that no longer apply.

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

## Foundation status

- `src/` is a faithful, importable **library** but **not runnable end-to-end**: `scripts/`,
  `config.yaml`, `configs/`, `tests/`, `models/` do not exist yet. `loadConfig()` defaults to
  reading `config.yaml` from cwd → `FileNotFoundError` until one is supplied. Source for those files,
  if needed, is the original `project-vigil-redux-2d.zip` (gitignored, kept at repo root).

## Hover-slam rewire targets (ranked, with why)

1. `src/env/rewards.py` — only place reward math lives; ZERO hover-slam terms today. Add terminal
   velocity-≈0 bonus, ignition-economy penalty (reads `engineTransitions`), and/or fuel-remaining bonus.
2. `src/env/episode.py` — owns success predicate (`isUpright AND isOnPad AND isGentle`) and the
   `info` dict (only `outcome`, `impactSpeed`). Tighten predicate; extend `info` with
   ignition count / fuel / terminal |x| / |θ| so eval+metrics can grade burn quality.
3. `src/config/loader.py` — single place `engineMode` is declared/validated and the only home for new
   reward weights / any `maxIgnitions` field / tightened landing tolerances. Any **world** field added
   bumps `computeWorldHash` and invalidates existing suicide-burn checkpoints.
4. `src/env/physics.py` — owns the suicide-burn ignition state machine (`engineTransitions` cap 2,
   `engineCommandedOn` latch, `SUICIDE_ON_THRESHOLD=0.5`, spool/fuel). Engine logic in `BoosterSim.step`
   and `stepPhysics` must stay byte-identical.
5. `src/agents/scripted.py` — `PdPilot` baseline emits continuous analog throttle and never reads
   obs[9]; structurally cannot honor binary firing / 2-ignition budgeting. A credible hover-slam
   baseline needs new logic (or a new sibling class) — gains tuned against `tests/test_scripted.py`.
6. `src/runtime/evaluate.py` — `runEvaluation` reports only successRate/outcomes/meanImpactSpeed/meanSteps;
   add ignition count, fuel-at-touchdown, terminal centering/uprightness for burn-quality tables.
7. `src/metrics/logger.py` — `CsvLogger` freezes its header on the FIRST record; any new metric must be
   in the first stats dict emitted by `train/loop.py` & `train/curriculum.py` or `writerow` raises mid-run.
8. `src/env/spaces.py` — touch LAST, deliberately. Changing the 10-D obs layout/`VEL_REF`/`OMEGA_REF`
   invalidates ALL models and the world-hash guard will NOT catch it.

## Gotchas confirmed in source

- **Render vs physics flame threshold:** `render.draw` paints a flame at `action[0] > 0.01`, but
  physics treats `> 0.5` as engine-ON in suicide-burn — the rendered flame can mislead for actions in
  (0.01, 0.5]. Align if using the flame to judge ignition timing.
- **Dormant-but-hashed world fields:** `settleTime`, `settleStepCap`, `maxLandingTilt`, `maxLandingOmega`
  are defined in `WorldConfig` but not read by the env (settling is physical now). They still enter the
  world hash, so removing them invalidates existing models.
- **`MLPPolicy.load` uses `torch.load(weights_only=False)`** — fine for local checkpoints; a trust
  concern only if checkpoints ever come from untrusted sources.
- **`plotConvergence` has no in-tree caller** — wired to be invoked by an (absent) external training script.

## Open questions for the rewire (resolve before coding)

1. Numeric definition of a "perfect hover slam" success: which terminal-velocity threshold (on |v|, vy,
   or both) replaces/augments `isGentle`? Bound pad-height overshoot too, or only velocity at contact?
2. Ignition-economy as a hard env-enforced **constraint** (like the existing `transitions>=2` lock) or a
   soft **reward** penalty? Target exactly 1 burn ("true" single-burn) or up-to-2 (current cap)?
3. Will any **world** field change (tolerances, a `maxIgnitions` field)? Each bumps `computeWorldHash`
   (and maybe `PHYSICS_MODEL_VERSION`) → clean-slate retrain. Accepted?
4. Where do the missing entrypoints come from — recover from the zip, regenerate, or author fresh?
5. Does the objective need obs enrichment beyond obs[9] (time-to-ground, fuel-burn integral, predicted
   impact velocity)? If yes, the frozen 10-D obs changes (high blast radius). Confirm 10-D is insufficient first.
6. Rewrite `PdPilot` to binary-firing, or add a NEW hover-slam baseline alongside it (keep analog-baseline tests intact)?
7. Velocity-arrest shaping term in `computePotential`, or rely on `shapingScaleFor` anneal-to-sparse-terminal?
