# Single-Burn Suicide-Burn Rewire — Design Spec

- **Date:** 2026-06-22
- **Status:** Design (awaiting review) → implementation plan
- **Author:** agent (brainstormed with user)

## 1. Summary

Re-aim `project-vigil-redux-2d` at a single objective: train a PPO model to land the
booster with a **true suicide-burn thrust profile — ignite once, cut once, touch down**.
Remove the analog engine world entirely, recover the upstream scaffold from
`project-vigil-redux-2d.zip`, and strip it to a single world / single config build. Add
exactly one behavioral requirement to the success criterion: the engine must be **cut before
touchdown**. The reward function and the 10-D observation/2-D action contract are unchanged.

The thrust profile already exists: `src/env/physics.py` implements a binary suicide-burn engine
with a hard 2-state-change cap (ignite → cut → lock), which is precisely "1 activation, 1
deactivation." This rewire makes that the *only* mode and focuses the repo's config, success
definition, tests, and docs on it.

## 2. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Repo scope | Recover the full scaffold (`scripts/`, `config.yaml`, `tests/`, `docs/`) from the zip, then strip to a single suicide-burn world. |
| 2 | Analog world | **Remove entirely** — delete `engineMode`, the analog physics branch, the analog-only config knobs, and the `lux`/`solis` naming scheme. |
| 3 | Cutoff | **Require a clean ignite→cut→touchdown** — landing with the engine still commanded on is a crash. |
| 4 | Optimization target | **Safe single-burn landing only** — keep the current success gate (upright, on-pad, gentle); add **no** fuel-economy or precision reward terms. |
| 5 | World/model machinery | **Full collapse** — one `config.yaml`, one world, no `--model`/`runtime.model`/`configs/<world>/<env>/` axis. |
| 6 | Scripted baseline | **Keep `PdPilot` as-is** — the env thresholds its continuous throttle at 0.5 and the 2-transition cap bounds it to a single burn. |

## 3. Scope & coordination boundary

**Directory management is owned by a concurrent agent** and is OUT OF SCOPE here. That agent is
rewiring run-artifact layout to:

- a top-level `checkpoints/` directory with **numbered run subdirectories** (replacing
  `models/<model>/<env>/`),
- `stdout/convergence-plots/` for **live-updating** convergence plots (mirroring
  `../project-vigil-redux/`),
- `stdout/logs/` for run logging.

This spec **treats that layout as an interface**: the suicide-burn rewire removes the
`--model`/world naming axis and writes through whatever checkpoint/plot/log paths the directory
agent establishes. It does **not** define those paths. Before editing any shared entry-point file
(especially `scripts/train.py`, which writes checkpoints + plots + metrics), check
`.claude/agent-memory/notes.md` for the directory agent's current status and sequence edits to
integrate rather than clobber.

## 4. Component changes

### 4.1 Engine — `src/env/physics.py`
- Make the suicide-burn path unconditional. Delete the `else: # analog mode` branch
  (`physics.py:373-380`) and the `if world.engineMode == 'suicideBurn':` guard
  (`physics.py:360`).
- Keep verbatim: binary firing (`rawThrottle > SUICIDE_ON_THRESHOLD = 0.5`), the 2-transition
  cap (`engineTransitions >= 2` → lock; `physics.py:364-371`), spool lag, fuel burn,
  `engineCommandedOn` latch.
- `stepPhysics` (test shim) and `BoosterSim.step` must remain byte-identical in engine logic
  (existing invariant).

### 4.2 Success criterion — `src/env/episode.py` (the one behavioral addition)
- At first toe contact, latch the engine command state, mirroring the impact-speed latch
  (`episode.py:208-210`):
  `self._engineOnAtTouchdown = state.engineCommandedOn`.
- New gate at `episode.py:221`:
  `success = isUpright AND isOnPad AND isGentle AND isCutOff`, where
  `isCutOff = not self._engineOnAtTouchdown`.
- Engine-on-at-touchdown therefore classifies as `crash` and pays the existing `terminalCrash`
  (no reward change). `success`/`crash`/`timeout` remain the only outcomes; `terminated` /
  `truncated` stay mutually exclusive.
- Extend `_info` (`episode.py:173-180`) with `engineOnAtTouchdown` and `engineTransitions` so
  evaluation and the HUD can report burn quality.
- Initialize `self._engineOnAtTouchdown = False` in `__init__` and `reset`.

**Why this is low-risk:** in binary suicide-burn mode the engine fires at full or not at all, so a
booster that rides full thrust into the pad gets relaunched and cannot satisfy `isGentle`. The
cutoff gate therefore mostly *formalizes* what the physics already enforces, which is why no
dedicated shaping/terminal reward term is needed (decision #4).

### 4.3 Reward — `src/env/rewards.py`
- **Unchanged.** `computeReward` already pays `terminalCrash` on any non-success terminal, so the
  cutoff requirement rides the outcome classifier. PBRS shaping, control cost, and the single-gamma
  contract are untouched.

### 4.4 Config — `config.yaml` + `src/config/loader.py`
- One `config.yaml` is the suicide-burn control panel. Remove from `world:`: `engineMode`,
  `minThrottle`, `throttleCutoff` (the latter two are analog-only — used solely in the deleted
  `physics.py` branch).
- `loader.py`: drop those three `WorldConfig` fields, their `validateConfig` checks, and the
  `engineMode` enum validation. `computeWorldHash` updates automatically (fewer fields).
- Bump `PHYSICS_MODEL_VERSION` `'pymunk-2' → 'suicide-1'` for a clean break. No `models/`
  checkpoints exist in this repo, so nothing is invalidated.
- Remove `runtime.model`. Remove the **entire `configs/` directory** (both `lux/` and `solis/`);
  the single root `config.yaml` is the only config.

### 4.5 Observation / action contract — `src/env/spaces.py`
- **Unchanged.** 10-D obs, 2-D action. `obs[9] = (2 − engineTransitions)/2` stays meaningful
  (1.0 fresh → 0.5 after ignite → 0.0 after cut/lock). `VEL_REF`/`OMEGA_REF`/`OBS_DIM`/`ACTION_DIM`
  frozen.

### 4.6 Entry points — `scripts/{train,watch,play,evaluate}.py`
- Remove **both naming axes** — the `--model` "thrust profile" and `--env` subdir arguments — and
  all `runtime.model` reads. `--config` defaults to `config.yaml`. Keep `--stage` and `--serial`.
- Checkpoint, convergence-plot, and log destinations follow the directory agent's scheme
  (`checkpoints/<run>/`, `stdout/convergence-plots/`, `stdout/logs/`) — see §3. This spec removes
  the world axis from path construction; it does not set the new paths.

### 4.7 Checkpoints — `src/agents/checkpoints.py`
- Simplify `resolveModelPath` to drop both the `<model>` and `<env>` dimensions, consistent with
  §4.6/§3. The world-hash guard in `loadCheckpoint` is retained unchanged (it now guards the single
  suicide-burn world hash).

### 4.8 Scripted baseline — `src/agents/scripted.py`
- **Keep `PdPilot` unchanged.** Document that in suicide-burn mode its continuous throttle is
  thresholded at 0.5 and bounded to one ignite+cut by the cap, so it serves as a weak but honest
  baseline. No new bang-bang controller (decision #6).

### 4.9 Curriculum — `config.yaml:curriculum`
- Keep the curriculum mechanism and the `touchdown → hop → drop → glide → full` ladder. Spawn
  ranges may need empirical retuning for clean single-burn descents; that is post-implementation
  tuning, not part of this structural rewire.

## 5. Tests — `tests/`
- Update analog-coupled tests:
  - `test_config_loader.py` — remove `engineMode`/`minThrottle`/`throttleCutoff` validation tests;
    update world-hash expectations for the reduced field set + bumped `PHYSICS_MODEL_VERSION`.
  - `test_physics.py` — delete analog-branch tests; keep/expand suicide-burn engine tests; add a
    test asserting the engine performs exactly one ignite + one cut and then locks.
  - `test_checkpoints.py`, `test_spaces.py`, `test_parallel.py` — adjust for the single world / no
    `--model` axis.
- **Add** to `test_episode.py`: engine-on-at-touchdown → `crash`; clean ignite→cut→touchdown
  (upright/on-pad/gentle) → `success`; `_info` exposes `engineOnAtTouchdown`/`engineTransitions`.
- Acceptance: `python -m pytest -q` fully green.

## 6. Docs & agent-memory
- Recover `docs/` from the zip, then rewrite the dual-world narrative (README, `docs/CODE_MAP.md`,
  `docs/GLOSSARY.md`, `docs/ROADMAP.md`) to the single suicide-burn objective.
- Log the analog removal + the cutoff success gate + the `PHYSICS_MODEL_VERSION` bump in
  `docs/CHANGELOG.md` and (for the gate) `docs/REWARD_LOG.md`.
- Update `.claude/agent-memory/{context,decisions,notes}.md` to reflect the single-world design and
  the cutoff gate.

## 7. World-hash / retrain
- No existing `models/` checkpoints in this repo → no checkpoints invalidated. The
  `PHYSICS_MODEL_VERSION` bump and the removed `world:` fields change the hash cleanly; training
  starts fresh.

## 8. Risks & open considerations
- **Credit assignment for the cut:** the only signal for "cut before touchdown" is the terminal
  success/crash, with no dedicated shaping term (per decision #4). Mitigated by binary-mode physics
  (gentle ≈ engine-off). If training fails to learn clean cuts, the fallback — flagged, not done
  now — is a small terminal or PBRS term keyed on `engineOnAtTouchdown`.
- **PdPilot fidelity:** it is a thresholded continuous controller, not a designed bang-bang pilot;
  accepted as a weak baseline.
- **Concurrent directory work:** shared edits to `scripts/train.py` must be sequenced with the
  directory agent (§3) to avoid clobbering its checkpoint/plot/log layout.
- **Curriculum tuning:** existing spawn ranges were tuned for the analog/general task; single-burn
  descents may need retuned rungs (post-implementation).

## 9. Testing & acceptance strategy
- TDD for the behavioral change: write the failing `isCutOff` success tests first, then implement
  the `episode.py` change.
- Update analog-removal tests alongside the `loader.py`/`physics.py`/`config.yaml` edits.
- Gate: `python -m pytest -q` green, then a short smoke train (e.g. `--stage touchdown`, few iters)
  and an `evaluate` run to confirm end-to-end single-burn landings, writing artifacts through the
  directory agent's scheme.

## 10. Out of scope
- Run-artifact directory layout, live convergence plotting, run logging (concurrent agent — §3).
- Fuel-optimality / precision reward terms (decision #4).
- A purpose-built bang-bang baseline controller (decision #6).
- Curriculum spawn-range retuning (post-implementation tuning).
- 3D dynamics, stage separation, gymnasium registration (never in scope for this 2D project).
