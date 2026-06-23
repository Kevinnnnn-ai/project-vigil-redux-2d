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
