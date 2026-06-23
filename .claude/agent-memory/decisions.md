# Decision Log

Append-only log of choices (agentic and human) and their rationale. Newest at the bottom.

## 2026-06-22 — Lay the `src/` foundation by replicating the zip

- **Replicated `src/` verbatim** from `project-vigil-redux-2d.zip` into the repo root (28 files,
  **0 SHA-256 mismatches**). *Why:* the user wants the repository's foundation laid by replicating
  the zip's `src/` before any rewiring toward the hover-slam goal. Copying byte-for-byte preserves
  the upstream contracts (obs/action, world hash, PBRS) intact as the baseline.
- **Populated `requirements.txt`** from the zip (it was previously empty: `pyyaml/pytest/numpy/torch/
  pygame-ce/pymunk/matplotlib`). *Why:* none of `src/` imports without these; the empty file could
  not support import verification, and the foundation must be verifiable.
- **Scope held to `src/` only.** Deliberately did **not** bring `docs/`, `config.yaml`, `configs/`,
  `scripts/`, `tests/`, or `models/`. *Why:* the user explicitly scoped foundations to "the src/ files"
  and deferred "rewiring the repository" to a later phase. Those files are rewire-phase decisions
  (esp. config/reward/eval, which the hover-slam goal will change).
- **Created `.env.local/` venv on Python 3.14.5** and installed deps. *Why:* `CLAUDE.md` mandates a
  local virtual environment; 3.14 matches the upstream requirement. `.env.local` is gitignored.
- **Verified the foundation** rather than asserting it: `py_compile` all files (ALL OK) + imported
  every module from repo root (22/22 OK). *Why:* `verification-before-completion` — evidence before claims.
- **Ran a 7-agent mapping workflow** over the replicated `src/` to (a) adversarially confirm the
  replication is internally consistent/faithful and (b) pre-stage the rewire. Result: `foundationReady=true`,
  no cross-subsystem contradictions, and a ranked list of hover-slam rewire targets (now in `context.md`/`notes.md`).
  *Why:* ultracode mandate to be exhaustive, and to enter the rewire phase with a grounded map.
- **Did NOT start the rewire** (no goal/behavior changes yet). *Why:* the user said "Once done (with
  related processes and results committed), we can move onto rewiring" — the rewire is a separate,
  upcoming phase that should begin with brainstorming the hover-slam objective.

## 2026-06-22 — Fill the empty `AGENTS.md` doc sections (human-authorized rule override)

- **Filled the three empty top sections of `.claude/AGENTS.md`** — the `# Project Vigil Redux 2D`
  title blurb (1-2 sentence project track), `## Quickstart and Commands` (setup + import smoke-test
  bash block plus a notes cheat sheet), and `## Project Structure` (annotated `src/` tree + the
  "not yet present" gap list). *Why:* the user asked for exactly these three sections.
- **Human override of the `CLAUDE.md` Hard Rule "Never edit `AGENTS.md`".** Those three headers live
  only in `AGENTS.md` (there is no `README.md`), so the request literally targets a protected file.
  Surfaced the conflict via an explicit question; the user chose "Edit AGENTS.md anyway", authorizing
  the override for this request only. *Why log it:* the standing rule still holds by default — future
  agents must NOT treat this as blanket permission to edit `AGENTS.md`.
- **Grounded every structure annotation in source, not memory.** Ran a 6-agent mapping workflow
  (one per `src/` subpackage) that read each module and returned a one-line, source-true purpose +
  public symbols. *Why:* ultracode mandate + the conventions' "do not guess behavior" rule.
- **Wrote the quickstart honestly as a library, not a runnable app.** The block stops at venv + deps +
  import smoke test; no fabricated `train`/`play` commands, and a note states `loadConfig()` raises
  `FileNotFoundError` until a `config.yaml` exists. *Why:* no `scripts/`/`config.yaml`/`tests/` exist
  yet — see `context.md` "Current state".

## 2026-06-22 — Live convergence plot + per-run checkpoint layout

- **Live-updating convergence plot.** `scripts/train.py` spawns `scripts/live_convergence.py` as a
  background **subprocess** that re-renders `stdout/convergence-plots/run-N.png` from the per-seed
  metrics CSVs every `LIVE_REFRESH_SECONDS` (5 s) while training runs; it is torn down in a `finally`,
  after which the run's authoritative final frame is drawn from the COMPLETE in-memory histories.
  *Why a subprocess, not a thread:* pyplot is not thread-safe, seeds train in child processes (so the
  live data only exists on disk as the flushed CSVs), and a subprocess reuses ONE render implementation
  that is also runnable by hand — mirroring `../project-vigil-redux/tools/liveConvergence.py`.
- **`src/metrics/plot.py` reused unchanged.** New `src/metrics/live.py` reads the CSVs into the
  `{seed: history}` shape `plotConvergence` already draws, preserving its tested contract
  (`tests/test_plot.py`). *Why:* the renderer was correct; only a live data source + run layout were missing.
- **Per-run artifact layout.** A run = one `scripts.train` session (may train several seeds, overlaid
  in one plot). Checkpoints → `checkpoints/run-N/{seed<seed>.pt,best.pt}`, metrics →
  `stdout/logs/run-N/seed<seed>.csv`, plot → `stdout/convergence-plots/run-N.png`. Run number
  auto-increments (`resolveNextRun` = max existing `run-*` + 1) with a `--run` override. *Why:* the
  user wants each session self-contained and numbered; replaces the upstream `models/<model>/<env>/`.
  Artifacts gitignored; `.gitkeep` holds the skeleton.
- **`--model`/`--env` demoted to cosmetic plot-title labels** (no longer path/config inputs) —
  forward-compatible with the suicide-burn rewire's planned removal of that axis. Shared file
  `scripts/train.py` coordinated via `notes.md` per the rewire spec §3.
- **TDD'd `src/metrics/live.py`** (`tests/test_live.py`: run-number resolution, CSV→history parsing
  with sentinel retention + truncated-row tolerance, render-to-PNG). Verified end-to-end with a smoke
  train — 4 live `(1 seed(s))` re-renders during training + clean teardown; full suite 160 green.
  Spec: `docs/superpowers/specs/2026-06-22-live-convergence-and-run-checkpoints-design.md`.
