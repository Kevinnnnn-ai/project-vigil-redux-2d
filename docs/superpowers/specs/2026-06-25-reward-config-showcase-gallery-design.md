# Reward-Config Showcase & Gallery — Design

- **Date:** 2026-06-25
- **Status:** Approved (design) — pending spec review, then implementation plan
- **Topic:** Retrain past reward/curriculum milestones in the *current fixed world* and view them all in one viewer.

---

## 1. Goal

Reconstruct each past *documented* training milestone (from `docs/REWARD_LOG.md` + git
history) as a standalone config under `tmp/configs/`, **pinned to today's `world:` block**, so
that:

1. Every retrained model shares the current world hash (`f5c82b420d2a6ebc`) and is therefore
   loadable in the current viewer (`scripts.watch` / `scripts.evaluate`).
2. All milestones can be trained ("train them all") and viewed ("view them all") through small
   helper tooling.

Fidelity to each *documented system* is the objective. **Convergence/landing performance is
explicitly NOT a goal** — a milestone that faithfully fails (e.g., a curriculum that cannot sample
success) is a correct result.

## 2. Non-goals

- No new reward terms (fuel-optimality, tightened precision) — out of scope, noted as a future
  direction in `agent-memory`.
- No reviving the deleted `oobPenalty` reward term (the only genuinely unreproducible past system;
  it needs the removed no-walls world). The user declined this option.
- No change to the core loader, the checkpoint format, the world hash, or `config.yaml` as the
  single control panel.

## 3. Background — load-bearing facts (verified, with citations)

- **The world hash excludes reward/training/curriculum.** `Config.computeWorldHash()`
  (`src/config/loader.py:186-202`) hashes only `asdict(self.world)` (the 25 `WorldConfig` fields,
  `loader.py:64-111`) plus `PHYSICS_MODEL_VERSION='suicide-1'` (`loader.py:59`). Tests confirm
  reward/curriculum edits leave the hash identical (`tests/test_config_loader.py:108-117`).
- **`--config` already exists everywhere.** `loadConfig(path='config.yaml')` opens the path arg
  (`loader.py:315-317`); every entry point forwards a `--config` flag
  (`scripts/train.py:85`, `watch.py:34`, `evaluate.py:36`, `play.py:52`, `live_convergence.py:25`;
  parallel workers via `task.configPath`, `src/train/parallel.py:96`). So
  `--config tmp/configs/X.yaml` works with **zero core code changes**.
- **Checkpoint load rebuilds the net from its own stored architecture.** `MLPPolicy.load`
  (`src/agents/mlp.py:134-141`) reads `obsDim/actDim/hidden` from the checkpoint and rebuilds, then
  loads weights — so differing `training.hidden` across configs still loads fine. The *only* gate is
  the world-hash check in `loadCheckpoint` (`src/agents/checkpoints.py:47-57`), which raises
  `ValueError` if `meta['worldHash'] != cfg.computeWorldHash()` (the *live* config's hash at view
  time).
- **Per-run, non-overwriting checkpoints.** `checkpoints/run-N/{seed<seed>.pt,best.pt}`; `N`
  auto-increments via `resolveNextRun` (`src/metrics/live.py:42-54`) or is forced with `--run`.
  Passing `--run N` at an existing `N` overwrites (`os.makedirs(exist_ok=True)`, `train.py:97`).
- **Provenance gap.** A checkpoint stores only `worldHash` + `stageName` (`mlp.py:122-132`) — NOT
  the reward/curriculum config. Nothing in the path or metadata records *which* reward system
  trained it. `reward.preset` is an inert label, read nowhere (`loader.py:116`; `rewards.py` has no
  preset dispatch).
- **The reward arithmetic never changed.** Every `REWARD_LOG.md` entry is `preset: baseline`. Across
  the whole project the reward *knobs* take only two values: **Variant A** (`shapingAnneal: linear`)
  and **Variant B** (`shapingAnneal: none`, the current `config.yaml`, `8d04e96`). All entries share
  `terminalSuccess 1.0 / terminalCrash -1.0 / gentlenessBonus 0.5 / centeringBonus 0.5 /
  shapingCoef 1.0 / controlCost 0.01 / gamma 0.99`. Historical *model* diversity came from the
  **world physics and curriculum**, not reward.
- **Sources for reconstruction:** the upstream zip is absent (gitignored, file gone) and
  `../project-vigil-redux/` has no project config yaml. Therefore reconstruction draws on (a) this
  repo's `config.yaml` git history (exact) and (b) `REWARD_LOG.md` prose (approximate).

## 4. Approach (chosen: A)

**A — Generated standalone YAMLs + a small `tmp/showcase/` kit; zero core code change.** Each
config is a complete yaml whose `world:` block is produced verbatim from `config.yaml` (⇒ identical
hash ⇒ co-viewable). A generator guarantees world identity; a guard test enforces it.

Rejected: **B** (loader overlay/merge — edits load-bearing `loader.py`, breaks the single-control-
panel invariant); **C** (in-place swap + registry only — not reproducible, contradicts the explicit
`tmp/configs/` request).

## 5. The config set (6 milestones)

All configs = today's `world:` block verbatim + today's `runtime:`/`mode:` + the deltas below.
Base = current `config.yaml` (HEAD). "Base" values for reference: reward Variant B
(`shapingAnneal: none`); `training.entCoef 0.02`, `totalIters 600`, `numEnvs 16`,
`rolloutSteps 2048`, `evalSeeds [0,1,2]`, `hidden [64,64]`; curriculum 5 stages
(touchdown/hop/drop/glide/full), `full` = altitude `[52,52]`, xOffset `[-5,5]`.

| File (`tmp/configs/`) | Reward | `entCoef` | `totalIters` | Curriculum stages | `full` spawn | Fidelity / source |
|---|---|---|---|---|---|---|
| `m1-original-shaping.yaml` | A `linear` | 0.01 | 220 | hop, drop, full (**no touchdown, no glide**) | alt `[40,52]`, x `[-14,14]` | approx (REWARD_LOG 06-12 original / rev1, oobPenalty + no-walls excluded). Expected to *not* sample success — faithful. |
| `m2-walls-touchdown.yaml` | A `linear` | 0.01 | 220 | touchdown, hop, drop, full (**no glide**) | alt `[40,52]`, x `[-14,14]` | approx (REWARD_LOG 06-12 rev2 full-curriculum acceptance) |
| `m3-m5-glide.yaml` | A `linear` | 0.02 | 260 | touchdown, hop, drop, glide, full | alt `[40,52]`, x `[-14,14]` | approx (REWARD_LOG 06-13 M5; its glide-rung + entCoef bump are today's baseline ⇒ ≈ `m4` minus 40 iters — documented near-duplicate) |
| `m4-suicide-run1.yaml` | A `linear` | 0.02 | 300 | touchdown, hop, drop, glide, full | alt `[40,52]`, x `[-14,14]` | **exact** (git `08fcc4d`) — also represents 06-15 terminal-timing & 06-16 pymunk (their changes were world/gate, already current) |
| `m5-run2.yaml` | A `linear` | 0.02 | 600 | touchdown, hop, drop, glide, full | alt `[52,52]`, x `[-5,5]` | **exact** (git `36d58ce`) |
| `m6-anneal-none.yaml` | B `none` | 0.02 | 600 | touchdown, hop, drop, glide, full | alt `[52,52]`, x `[-5,5]` | **exact** (= current `config.yaml`) |

Notes:
- Non-`full` curriculum stages (touchdown/hop/drop/glide), where present, keep the values from
  `config.yaml` verbatim (they were stable across the in-repo history).
- `numEnvs 16` / `rolloutSteps 2048` / `hidden [64,64]` / `evalSeeds`/`evalEpisodes`/`evalEvery`
  stay at base for all (the short diagnostic-run values in the log, e.g. rev2's `numEnvs 8`, were
  one-off debugging, not the milestone's canonical setup).
- `m6` is effectively a copy of `config.yaml`; included for a complete timeline.

## 6. World-identity guarantee

- `gen_configs.py` parses `config.yaml` with PyYAML, copies the parsed `world:` mapping **verbatim**
  into every emitted config, then overlays the per-milestone reward/training/curriculum deltas, and
  dumps a complete valid config. Re-dumping drops YAML comments and reorders keys — **cosmetic only**;
  `computeWorldHash` hashes `asdict(world)` with `sort_keys=True`, so values (not formatting/order)
  determine the hash. Each emitted file gets a provenance header comment (milestone, source,
  "generated — do not hand-edit; re-run gen_configs").
- **Guard test** (`tests/test_showcase_configs.py`): for every milestone file,
  `loadConfig(file).computeWorldHash() == loadConfig('config.yaml').computeWorldHash()`. This fails
  loudly if `config.yaml`'s world is ever changed without regenerating — the exact drift we must
  catch. Plus: each file `loadConfig`s without error, and a spot-check that deltas applied
  (e.g. `m1` has 3 stages and no `touchdown`; `m6` has `shapingAnneal == 'none'`; `m4` `full`
  altitude is `[40,52]`).

## 7. Directory layout & git tracking

```
tmp/
  configs/                 # generated, git-tracked
    m1-original-shaping.yaml
    m2-walls-touchdown.yaml
    m3-m5-glide.yaml
    m4-suicide-run1.yaml
    m5-run2.yaml
    m6-anneal-none.yaml
  showcase/                # the kit, git-tracked
    milestones.py          # SINGLE source of truth: name, file, runNumber, reward variant, deltas, source, fidelity, note
    gen_configs.py         # emit tmp/configs/*.yaml from config.yaml world + milestones deltas; [--fast]
    train_all.py           # subprocess-train each milestone at its reserved --run; write registry; [--serial] [--only NAME]
    gallery.py             # read milestones+registry; launch watch per pick; [--milestone NAME] [--all]
    registry.json          # canonical run-status (written by train_all): name -> {run, trainedAt, bestSuccess, worldHashOk}
    REGISTRY.md            # human-readable table regenerated from registry.json
```

- **Git-track** `tmp/configs/*.yaml` and all of `tmp/showcase/` (durable, reproducible artifacts;
  aligns with the REWARD_LOG reproducibility hard rule). Run artifacts (`checkpoints/run-*`,
  `stdout/logs/run-*`, plots) stay gitignored (already are; `tmp/` is **not** in `.gitignore`, so
  these files are tracked by default — intended).
- `tmp/configs/` is deliberately separate from the (deleted) `configs/` dir: `config.yaml` remains
  the single canonical control panel; `tmp/configs/` is clearly a side experiment.

## 8. Tooling behavior

- **Run as modules from repo root** (per `agent-memory/notes.md` import gotcha): e.g.
  `python -m tmp.showcase.gen_configs`. PEP 420 namespace packages make `tmp.showcase` importable
  with repo root on `sys.path` (the `-m` invocation provides it); the scripts `from src... import`.
  Fallback if namespace import is flaky: a 2-line `sys.path` bootstrap at the top of each script.
- **`gen_configs.py [--fast]`** — default emits each milestone's documented values. `--fast` caps
  `totalIters` (e.g. 60) and sets `evalSeeds: [0]` (one seed) on every emitted config, for quick,
  watchable-but-unconverged models. `--fast` is a *generation-time* concern, so `scripts.train`
  stays untouched.
- **`train_all.py [--serial] [--only NAME]`** — for each milestone (or just `--only`), run
  `python -m scripts.train --config tmp/configs/<file> --run <runNumber>` as a subprocess
  (forwarding `--serial`). Reserved run band: **`run-7001 … run-7006`** (`runNumber` is fixed per
  milestone in `milestones.py`). After each run, update `registry.json` + regenerate `REGISTRY.md`.
  *Caveat:* a later organic `scripts.train` with no `--run` auto-increments to `max+1` (jumps above
  the band) — harmless; the registry disambiguates. The band also makes re-running `train_all`
  deterministically overwrite the same showcase runs (intended: regenerate the gallery in place).
- **`gallery.py [--milestone NAME] [--all]`** — read `milestones.py` (name↔run↔config) and
  `registry.json` (trained?). Default: print the table and prompt for a pick. `--milestone NAME`
  launches that one; `--all` cycles all trained milestones. Launch:
  `python -m scripts.watch --config tmp/configs/<file> --run <runNumber>` — watching each model with
  **its own config** ⇒ correct world hash (matches) AND the milestone's own `full` spawn stage.
  Warn (don't crash) if a run dir is missing (not yet trained).

## 9. Provenance — registry format

`registry.json` (canonical) — status fields use `worldHashOk` (matching the layout in §7), e.g.:
```json
{
  "m4-suicide-run1": {"run": 7004, "config": "tmp/configs/m4-suicide-run1.yaml",
                       "trainedAt": "2026-06-25T12:00:00", "bestSuccess": 0.0, "worldHashOk": true}
}
```
`REGISTRY.md` is a generated table: milestone | run | config | reward | fidelity | trained? | bestSuccess.
`milestones.py` holds the static mapping (name, file, runNumber, reward, deltas, source, fidelity,
note) and is imported by all three scripts so the name↔run↔config mapping has one source of truth.

## 10. Compute

- **Full fidelity (default):** 6 runs, each up to `totalIters` (220–600) × `evalSeeds` (3) — hours
  of training. `train_all.py` runs them sequentially (each `scripts.train` already parallelizes
  seeds via `seedWorkers`).
- **Showcase-speed:** `gen_configs.py --fast` → all 6 train in minutes (1 seed, capped iters), for a
  quick visual gallery. Documented values remain the default.

## 11. Testing plan (TDD)

1. `tests/test_showcase_configs.py` (write first, red): world-hash identity for all 6; each
   `loadConfig`s clean; delta spot-checks (m1 stage count/names, m6 anneal, m4 full altitude).
2. `gen_configs.py` smoke: generating into a temp dir produces 6 parseable configs that pass the
   guard (use the scratchpad temp dir, not `stdout/`; clean up).
3. `train_all.py` / `gallery.py` smoke: a `--fast` + `--only m6` end-to-end into a sentinel run band
   (e.g. `run-9002`), assert a checkpoint + registry entry appear, then **clean up** the sentinel
   `checkpoints/run-9002/`, `stdout/logs/run-9002/`, plot, and registry test entry (per the
   `stdout/` cleanup rule).
4. Full suite stays green (`python -m pytest -q`).

## 12. Documentation updates

- `docs/CHANGELOG.md` — entry for the showcase tooling + `tmp/configs/` set (format-compliant).
- `docs/REWARD_LOG.md` — one entry documenting the reproduction effort: the two reproducible reward
  variants (A linear / B none), the milestone→run mapping reference (registry), and that results are
  "failure-OK" reproductions, not new reward designs.
- `.claude/agent-memory/context.md` — add the showcase feature to current state + scripts list.
- `.claude/agent-memory/decisions.md` — append the decision (reward log yields 2 reproducible reward
  variants; diversity was world/curriculum; chosen Approach A; reserved run band).
- `.claude/agent-memory/notes.md` — how-to (run-as-module, gen→train_all→gallery, drift→regenerate).

## 13. Risks & caveats

- **World drift kills the gallery at once.** Any `config.yaml` `world:` edit changes the hash and
  invalidates every showcase model. Mitigation: guard test fails on drift; recovery is
  `gen_configs` (re-sync) + `train_all` (retrain).
- **Run-band vs auto-increment** interaction (§8) — harmless, documented.
- **`m3 ≈ m4`** near-duplicate (differ by 40 iters) — kept for timeline completeness, flagged.
- **PyYAML re-dump** drops comments / reorders keys in generated files — cosmetic; hash unaffected.
- **`torch.load(weights_only=False)`** in `MLPPolicy.load` — pre-existing; fine for local
  checkpoints only.
- **tmp/ import path** — must run as `python -m tmp.showcase.*` from repo root, else
  `ModuleNotFoundError: src`.

## 14. Out of scope

New reward terms; oobPenalty revival; loader overlay/merge; checkpoint-format changes; modifying
`config.yaml`, the world, or core `scripts/`.

## 15. Open questions

None — all design decisions resolved and approved.
