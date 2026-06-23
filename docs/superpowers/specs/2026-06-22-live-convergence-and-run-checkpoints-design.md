# Live Convergence Plotting & Per-Run Checkpoints — Design Spec

- **Date:** 2026-06-22
- **Status:** Design (awaiting review) → implementation plan
- **Author:** agent (brainstormed with user)

## 1. Summary

Make the training-convergence plot update **live while the model trains**, and reorganize
run artifacts so every training **session (run)** is self-contained:

- Per-run checkpoints in `checkpoints/run-N/` (`seed<seed>.pt` + `best.pt`).
- Per-run convergence plot in `stdout/convergence-plots/run-N.png`, refreshed throughout the
  run and finalized at the end.
- Per-run live metrics CSVs in `stdout/logs/run-N/seed<seed>.csv` (the live plot's data source).

The existing renderer `src/metrics/plot.py:plotConvergence(histories, outPath, rolloutSteps,
numEnvs, title)` is **reused unchanged** — its tested contract (`tests/test_plot.py`) is preserved.
A new module builds `histories` from the on-disk CSVs and calls it. The "auto background
refresher" is implemented as a **subprocess** that re-renders the PNG on an interval, reusing a
standalone CLI tool — so matplotlib never runs in a thread and the same code serves both the
auto path and manual use.

This is the directory-layout / live-plotting counterpart that the suicide-burn rewire spec
(`2026-06-22-single-burn-suicide-burn-rewire-design.md`, §3 and §10) declares out of scope and
delegates to "the directory agent." This spec **defines those paths and the live mechanism**.

## 2. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Run unit | A **run = one training session** (one `scripts.train` invocation, possibly many seeds). All its seeds overlay in one `run-N.png`, matching `plotConvergence`'s existing per-seed-overlay. |
| 2 | Live mechanism | **Auto background refresher**, implemented as a **subprocess** (`scripts/live_convergence.py`) spawned by `train.py` and terminated in a `finally`. Process-isolated (no matplotlib-in-thread), single render implementation, also runnable manually. |
| 3 | Entrypoint | Already recovered from the zip (`scripts/train.py`); this spec modifies it. No new training driver. |
| 4 | Run number | **Auto-increment**: next `run-N` = max existing `checkpoints/run-*` index + 1, created at session start. `--run N` overrides (resume/overwrite). |
| 5 | Refresh interval | **5 s** default, `--interval` configurable. |
| 6 | Artifact paths | checkpoints → `checkpoints/run-N/{seed<seed>.pt,best.pt}`; metrics → `stdout/logs/run-N/seed<seed>.csv`; plot → `stdout/convergence-plots/run-N.png`. |
| 7 | `plot.py` | **Untouched.** The live path constructs `histories` from CSVs, then calls the existing `plotConvergence`. |
| 8 | `--model`/`--env` flags | Kept only as **plot-title / log labels**, removed from path construction (paths are run-numbered). Consistent with the rewire agent's planned axis-collapse. |

## 3. Scope & coordination boundary

This work is the **directory-agent** half of the concurrent two-agent effort. The suicide-burn
rewire agent owns reward/world/config/contract changes and the removal of the `--model`/`--env`
naming axis from *behavior*; it explicitly **defers run-artifact paths, live plotting, and run
logging to this spec** (rewire §3, §4.6, §10).

**Shared file:** `scripts/train.py`. Both agents edit it. Coordination is via
`.claude/agent-memory/notes.md` (rewire §3): before editing `scripts/train.py`, record the claim
and sequence edits to integrate rather than clobber. This spec owns `train.py`'s **path
construction, run-number resolution, and refresher spawn/teardown**; the rewire owns its
`--model`/`runtime.model` removal. The two are compatible: paths become run-numbered (this spec),
and the world axis disappears from them (rewire) — same destination.

**Untouched by this spec** (zero collision surface): `src/metrics/plot.py`, the per-seed training
math (`src/train/loop.py`, `src/train/curriculum.py`), `src/metrics/logger.py`, and the
obs/action/reward/world contracts.

## 4. Architecture & data flow

```
scripts/train.py (parent process)
  ├─ resolveNextRun('checkpoints') -> N ; mkdir run dirs
  ├─ SeedTask.csvPath  = stdout/logs/run-N/seed<seed>.csv   (per seed)
  ├─ SeedTask.savePath = checkpoints/run-N/seed<seed>.pt
  ├─ spawn: python -m scripts.live_convergence --run N --config <cfg> --interval 5
  │         (subprocess; loops: read CSVs -> render run-N.png)
  ├─ runSeeds(...)   ── child procs write+flush seed<seed>.csv every iteration ──┐
  │                                                                              │
  ├─ finally: terminate + wait the live subprocess                              ◄┘
  ├─ copy best -> checkpoints/run-N/best.pt
  └─ final authoritative render: plotConvergence(histories, run-N.png)  (in-memory, complete)
```

- **Source of truth = the CSVs.** `CsvLogger.log` already flushes every iteration, so a separate
  reader sees fresh rows mid-run. The live subprocess only reads; the seed processes only write.
- **No race on the final PNG:** the subprocess is terminated and joined *before* the final
  `plotConvergence` writes the authoritative frame.
- **Serial path (`--serial`, `seedWorkers=1`) works identically** — CSVs are written to disk
  regardless of in-process vs pool execution.

## 5. Components

### 5.1 New — `src/metrics/live.py`
Owns run-artifact path conventions and the CSV→histories→plot bridge. Pure/testable; no CLI.

Constants (SCREAMING_SNAKE):
- `CHECKPOINTS_ROOT = 'checkpoints'`
- `LOGS_ROOT = os.path.join('stdout', 'logs')`
- `PLOTS_DIR = os.path.join('stdout', 'convergence-plots')`
- `RUN_PREFIX = 'run-'`

Functions (camelCase, verb-first):
- `resolveNextRun(checkpointsRoot=CHECKPOINTS_ROOT)` → `int`. Scans for `run-<int>` subdirs;
  returns `max + 1`, or `1` if none/absent. Ignores non-matching names.
- `runCheckpointDir(run, checkpointsRoot=CHECKPOINTS_ROOT)` → `checkpoints/run-N`.
- `runLogsDir(run, logsRoot=LOGS_ROOT)` → `stdout/logs/run-N`.
- `runPlotPath(run, plotsDir=PLOTS_DIR)` → `stdout/convergence-plots/run-N.png`.
- `seedCheckpointPath(run, seed)` → `checkpoints/run-N/seed<seed>.pt`.
- `seedCsvPath(run, seed)` → `stdout/logs/run-N/seed<seed>.csv`.
- `readSeedHistories(logsDir)` → `{seed: [{'iter': int, 'successRate': float}, ...]}`. Globs
  `seed*.csv`, parses the seed index via `seed(\d+)`, reads with `csv.DictReader`, coerces
  `iter`→`int` and `successRate`→`float`. **Robust:** skips rows missing those keys or failing
  coercion (tolerates a half-written final line and partial files); skips files with no usable
  rows. Returns `{}` if `logsDir` is absent/empty.
- `renderConvergence(logsDir, outPath, rolloutSteps, numEnvs, title=None)` → list of plotted
  seeds. Calls `readSeedHistories` then the existing `plotConvergence`; ensures `outPath`'s
  directory exists. Returns `[]` (and still writes an axes-only PNG) when no seed yet has ≥2 eval
  points — `plotConvergence` already enforces the ≥2-point rule and the `successRate >= 0.0`
  sentinel filter, so curriculum `-1.0` rows need no special handling here.

### 5.2 New — `scripts/live_convergence.py`
Thin CLI around `renderConvergence`; the auto-refresher subprocess and a manual tool.

Args:
- `--run N` (required) — selects `stdout/logs/run-N/` and `stdout/convergence-plots/run-N.png`.
- `--config config.yaml` — loaded for `cfg.training.{rolloutSteps,numEnvs}` (the x-axis
  step factors; same values `train.py` uses, honoring `plot.py`'s CRITICAL x-axis guardrail).
- `--interval 5.0` — seconds between re-renders.
- `--once` — render a single frame and exit.
- `--out PATH` — override the default `runPlotPath(run)`.

Loop mirrors the prior iteration's `tools/liveConvergence.py`: render in try/except (a glitch or
"no data yet" prints a notice and continues), `print(..., flush=True)`, `sleep(interval)` until
`--once` or terminated. Headless Agg is inherited from `plot.py`.

### 5.3 Modified — `scripts/train.py`
- Import `src.metrics.live`. Replace `METRICS_DIR`/`modelsDir` construction with run resolution:
  `run = args.run or resolveNextRun()`, create `runCheckpointDir(run)` and `runLogsDir(run)`.
- `SeedTask.savePath = seedCheckpointPath(run, seed)`; `SeedTask.csvPath = seedCsvPath(run, seed)`.
- New `--run` arg (int, default `None` → auto). Keep `--stage`, `--serial`, `--config`. Keep
  `--model`/`--env` only to label the plot title / console lines (no longer in paths).
- Wrap the training body in `try/finally`: in `try`, `subprocess.Popen([sys.executable, '-m',
  'scripts.live_convergence', '--run', str(run), '--config', args.config, '--interval', '5'])`
  then `runSeeds(...)`; in `finally`, `proc.terminate()` + `proc.wait(timeout=5)`, falling back to
  `proc.kill()` if it does not exit — all guarded in try/except so refresher cleanup never masks a
  real training error or `KeyboardInterrupt`.
- After seeds finish and the subprocess is stopped: copy best → `checkpoints/run-N/best.pt`; render
  the authoritative final frame with the existing `plotConvergence(histories, runPlotPath(run),
  …)` (unchanged best-effort try/except — a plot glitch never discards a finished run).
- Console: print the resolved run number and the three artifact roots up front.

### 5.4 New — `tests/test_live.py`
- `resolveNextRun`: empty/absent root → `1`; with `run-1`,`run-3` present → `4`; ignores junk dirs.
- `readSeedHistories`: write two `seed*.csv` files (CsvLogger-shaped, incl. a `-1.0` sentinel row
  and an intentionally truncated final line) → correct `{seed: [...]}`, sentinels retained as
  `-1.0`, truncated row skipped, seed indices parsed.
- `renderConvergence`: from those CSVs writes a non-empty PNG; returns the seeds with ≥2 eval
  points; empty `logsDir` still writes an axes-only PNG and returns `[]`.
- Path helpers return the exact expected strings (use `os.path.join`/`os.sep`-tolerant asserts).

### 5.5 `.gitignore` + `.gitkeep`
- Add ignores for run artifacts: `checkpoints/run-*/`, `stdout/logs/run-*/`,
  `stdout/convergence-plots/*.png`.
- Add tracked `.gitkeep` to `checkpoints/`, `stdout/logs/`, `stdout/convergence-plots/` so the
  skeleton exists on a fresh clone without committing run output.

### 5.6 Docs & agent-memory
- `.claude/agent-memory/notes.md`: record the directory-agent coordination claim on
  `scripts/train.py` path construction + the new files (so the rewire agent sequences edits).
- `.claude/agent-memory/decisions.md`: append the run-N layout + live-refresher-subprocess
  decisions and rationale.
- `.claude/agent-memory/context.md`: update current-state to note runnable training now writes
  run-numbered artifacts with a live convergence plot.

## 6. Edge cases & failure modes
- **Partial / truncated CSV line** mid-write → row skipped by `readSeedHistories` (per-row
  try/except); next refresh picks it up once complete.
- **< 2 eval points early in a run** → `plotConvergence` skips the seed; an axes-only PNG is still
  written (a valid "warming up" frame).
- **Curriculum `-1.0` sentinels** → retained on read, filtered by `plotConvergence`'s existing
  `>= 0.0` guard. No special-casing in the live path.
- **Very fast runs (tests/smoke)** → the subprocess may not render before completion; the final
  in-memory `plotConvergence` frame is authoritative, so output is correct regardless.
- **Windows subprocess teardown** → `terminate()` + bounded `wait()` in `finally`; guarded so a
  teardown hiccup never masks a real training error.
- **`--run` re-use** → overwrites that run's artifacts intentionally (resume/redo).
- **Run-number resolution is parent-only** (single process at session start) → no concurrent
  mkdir race within a session.

## 7. Risks & considerations
- **Two writers of `run-N.png`** (subprocess refreshes + final frame). Mitigated by terminating
  the subprocess before the final write — strict ordering, no overlap.
- **CSV column drift:** the live reader depends only on `iter` + `successRate`, which both trainers
  always emit (curriculum emits the `-1.0` sentinel). New metric columns are ignored, not fatal.
- **Concurrent `scripts/train.py` edits** with the rewire agent — sequenced via agent-memory
  (§3). Path construction here is additive to their axis removal; final integration is a small
  merge of compatible edits on one file.
- **Subprocess start cost** (~one Python import of matplotlib) is negligible against a training
  run; first live frame appears a few seconds in.

## 8. Testing & acceptance
- TDD for `src/metrics/live.py`: write `tests/test_live.py` first, then implement to green.
- `python -m pytest -q` fully green (existing 148 + new `test_live.py`; `test_plot.py` unchanged).
- Smoke: a short run (e.g. `--stage touchdown`, few `totalIters`, `--serial`) writes
  `checkpoints/run-N/`, `stdout/logs/run-N/seed*.csv`, and a `stdout/convergence-plots/run-N.png`
  that visibly updates while training and is finalized at the end. Test artifacts cleaned up
  afterward per AGENTS.md (kept under `stdout/`, no collision with pre-existing data).

## 9. Out of scope
- Reward / world / success-criterion / obs-action changes (suicide-burn rewire agent).
- Removal of `--model`/`runtime.model` *behavior* (rewire agent; this spec only drops them from
  path construction).
- Interactive (GUI window) live plotting — the deliverable is a continuously-refreshed PNG file,
  viewable in any auto-reloading image viewer (matches `plot.py`'s headless Agg).
- Metric channels beyond the existing success-rate-vs-steps curve (e.g. reward overlay).
- Checkpoint retention policy / pruning of old `run-N/` dirs.
