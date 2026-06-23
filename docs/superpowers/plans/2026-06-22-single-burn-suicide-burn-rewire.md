# Single-Burn Suicide-Burn Rewire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-aim the repo at one task — land the booster with a true suicide-burn thrust profile (ignite once, cut once) — by adding a "cut-before-touchdown" success gate and removing the analog engine world entirely.

**Architecture:** The binary suicide-burn engine already exists in `src/env/physics.py` (fires full/off, 2-state-change cap). This rewire (1) adds a success requirement in `src/env/episode.py` that the engine be cut before first ground contact, (2) deletes the analog engine branch and its config fields, collapsing to a single `config.yaml`/one world, and (3) removes the `--model`/`--env` naming axis from the scripts. Reward (`src/env/rewards.py`) and the 10-D obs / 2-D action contract (`src/env/spaces.py`) are unchanged.

**Tech Stack:** Python 3.14, PyTorch, Pymunk (Chipmunk2D), NumPy, pygame-ce, PyYAML, pytest.

## Global Constraints

- **Run everything from the repo root** in the activated `.env.local` venv. In commands below, `python` = `C:\Users\Admin\files\projects\project-vigil-redux-2d\.env.local\Scripts\python.exe`. `src.` is a namespace package (not pip-installed); `python -m pytest` / `python -m scripts.*` from root puts cwd on `sys.path`.
- **Code conventions** (`.claude/AGENTS.md`): camelCase variables/functions (verb-first), PascalCase classes, SCREAMING_SNAKE constants, single quotes only, `#` comments only, `_`-prefixed helpers. **Match the surrounding code** — it already follows these.
- **Commits:** small and frequent; **never push**; commit on `main`. **Stage only your own files** with explicit `git add -- <paths>` (other agents are working in this tree). End every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Concurrent directory agent owns run-artifact layout** — `checkpoints/<run>/` (replacing `models/<model>/<env>/`), `stdout/convergence-plots/` (live plots), `stdout/logs/`. Do **not** define these paths. Before editing `scripts/train.py` or `src/train/parallel.py`, **re-read them** (the directory agent may have changed them) and check `.claude/agent-memory/notes.md`.
- **Frozen contract:** do not change `src/env/spaces.py` (`OBS_DIM=10`, `ACTION_DIM=2`, `VEL_REF`, `OMEGA_REF`); `obs[9] = (2 − engineTransitions)/2` stays meaningful.
- **No `models/` checkpoints exist**, so the `PHYSICS_MODEL_VERSION` bump invalidates nothing.
- **Acceptance for every task:** `python -m pytest -q` is green from repo root before the task's commit.

---

### Task 1: Add the cut-before-touchdown success gate (`episode.py`)

The one behavioral change. A success now also requires the engine to have been **cut before first ground contact**. This is mode-independent (it keys off `engineCommandedOn`, which exists in both engine modes), so it lands cleanly before the analog removal.

**Files:**
- Modify: `src/env/episode.py` (`LandingEnv.__init__`, `reset`, `step` contact-latch + classify, `_info`)
- Test: `tests/test_episode.py`

**Interfaces:**
- Produces: `info['engineOnAtTouchdown']` (bool) and `info['engineTransitions']` (int) on the env step's info dict; success classification gains an `isCutOff` term. Consumed later by Task 5 (eval) and the directory agent's HUD/logging.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_episode.py`:

```python
def test_engineFiringAtTouchdownIsCrash(cfg):
    # A true suicide burn must CUT the engine before contact. A booster still
    # commanded-on as it touches down is not a valid suicide burn -> crash, even
    # when it is otherwise upright, on-pad, and gentle.
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(
        x=0.0, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.6, theta=0.0, omega=0.0,
        fuel=0.5, spool=0.0, engineTransitions=1, engineCommandedOn=True,
    )
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['outcome'] == 'crash'
    assert info['engineOnAtTouchdown'] is True


def test_engineCutBeforeTouchdownIsSuccess(cfg):
    # Engine already cut (commanded off) on approach -> eligible for success when
    # upright, on-pad, and gentle. Exposes the new info fields.
    env = LandingEnv(cfg)
    env.reset(np.random.default_rng(0))
    env.state = BoosterState(
        x=0.0, y=cfg.world.legDrop + 0.02, vx=0.0, vy=-0.6, theta=0.0, omega=0.0,
        fuel=0.5, spool=0.0, engineTransitions=2, engineCommandedOn=False,
    )
    terminated = truncated = False
    info = {}
    for _ in range(cfg.world.settleStepCap + 5):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info['outcome'] == 'success'
    assert info['engineOnAtTouchdown'] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_episode.py::test_engineFiringAtTouchdownIsCrash tests/test_episode.py::test_engineCutBeforeTouchdownIsSuccess -q`
Expected: FAIL — `test_engineFiringAtTouchdownIsCrash` asserts `crash` but current code returns `success` (no cut gate); both fail on `KeyError: 'engineOnAtTouchdown'`.

- [ ] **Step 3: Initialize the latch in `__init__` and `reset`**

In `src/env/episode.py`, in `__init__`, after the line `self._hasTouchedDown = False` (the `@TAG[impact-tracking]` block, ~line 124), add:

```python
        # @TAG[cut-gate]: engine command state latched at first toe contact. A true
        # suicide burn must already be cut (engineCommandedOn False) when it touches.
        self._engineOnAtTouchdown = False
```

In `reset`, after the line `self._hasTouchedDown = False` (~line 160), add:

```python
        self._engineOnAtTouchdown = False
```

- [ ] **Step 4: Latch at first contact and add the `isCutOff` gate**

In `step`, in the `@TAG[impact-tracking]` block, change:

```python
        if not self._hasTouchedDown and self._hasToeContact(state):
            self._hasTouchedDown = True
            self._impactSpeed = math.hypot(prevState.vx, prevState.vy)
```

to:

```python
        if not self._hasTouchedDown and self._hasToeContact(state):
            self._hasTouchedDown = True
            self._impactSpeed = math.hypot(prevState.vx, prevState.vy)
            # @TAG[cut-gate]: capture the engine command as the booster ENTERED the
            # contact step (prevState), mirroring impactSpeed's approach-velocity read.
            self._engineOnAtTouchdown = prevState.engineCommandedOn
```

In the `@TAG[outcome-classify]` block, change:

```python
            isGentle = self._impactSpeed <= world.maxLandingSpeed
            outcome = 'success' if (isUpright and isOnPad and isGentle) else 'crash'
```

to:

```python
            isGentle = self._impactSpeed <= world.maxLandingSpeed
            isCutOff = not self._engineOnAtTouchdown
            outcome = 'success' if (isUpright and isOnPad and isGentle and isCutOff) else 'crash'
```

- [ ] **Step 5: Expose the new fields in `_info`**

Change `_info` to:

```python
    def _info(self, state, outcome, impactSpeed):
        return {
            'outcome': outcome,
            'impactSpeed': impactSpeed,
            'x': state.x,
            'y': state.y,
            'fuel': state.fuel,
            'engineOnAtTouchdown': self._engineOnAtTouchdown,
            'engineTransitions': state.engineTransitions,
        }
```

- [ ] **Step 6: Run the new tests, then the full episode suite**

Run: `python -m pytest tests/test_episode.py -q`
Expected: PASS (all episode tests, including the two new ones).

- [ ] **Step 7: Commit**

```bash
git add -- src/env/episode.py tests/test_episode.py
git commit -m "feat: require engine cut before touchdown for landing success

A true suicide burn must ignite once and CUT before contact. LandingEnv now
latches engineCommandedOn from the approach (prevState) at first toe contact and
requires it to be off (isCutOff) for a 'success'; burning into the pad is a crash.
Exposes engineOnAtTouchdown and engineTransitions in info. Reward is unchanged
(the gate rides the existing crash payout).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Remove the analog engine mode (collapse to one world)

Atomic change: the binary suicide-burn engine becomes unconditional; the analog-only config fields and their validations go; `config.yaml` becomes the single suicide-burn control panel; `configs/` is deleted; the world hash is rebumped. Source and tests change together so the suite stays green.

**Files:**
- Modify: `src/env/physics.py` (engine dispatch ~line 359-380; guardrail comment line 59)
- Modify: `src/config/loader.py` (`WorldConfig` fields; `RuntimeConfig.model`; `validateConfig`; `PHYSICS_MODEL_VERSION`)
- Modify: `config.yaml` (remove `engineMode`/`minThrottle`/`throttleCutoff`/`runtime.model`)
- Delete: `configs/` (entire directory)
- Modify: `scripts/watch.py` (HUD `engineMode` reference, line ~88-89)
- Modify: `tests/test_physics.py`, `tests/test_config_loader.py`, `tests/test_episode.py`

**Interfaces:**
- Produces: `WorldConfig` no longer has `engineMode`/`minThrottle`/`throttleCutoff`; `RuntimeConfig` no longer has `model`; `PHYSICS_MODEL_VERSION == 'suicide-1'`. The engine is always binary suicide burn.

- [ ] **Step 1: Update the analog-coupled tests (they must fail first, then pass after the source edits)**

In `tests/test_physics.py`:
- **Delete** these four tests (they assert analog-only behavior or reference removed fields): `test_spoolAsymptotesToCommand`, `test_minThrottleFloorOnceLit`, `test_emptyBoosterCanStillHover`, `test_analogStepPreservesEngineTransitions`.
- **Change** the `burnWorld` fixture from:

```python
@pytest.fixture
def burnWorld():
    return dataclasses.replace(loadConfig('config.yaml').world, engineMode='suicideBurn')
```

to:

```python
@pytest.fixture
def burnWorld():
    # config.yaml is now the single binary suicide-burn world.
    return loadConfig('config.yaml').world
```

In `tests/test_config_loader.py`:
- **Delete** these tests: `test_minThrottleRangeValidation`, `test_runtimeModelDefaultsToLux`, `test_runtimeModelDoesNotAffectWorldHash`, `test_luxAndSolisShipWithDifferentWorldHash`, `test_engineModeDefaultsToAnalog`, `test_engineModeChangesWorldHash`, `test_invalidEngineModeRaises`.
- In `test_worldHasMassAndSpoolFields`, **delete** the line `assert 0.0 < w.minThrottle < 1.0`.
- **Add** this test:

```python
def test_engineModeFieldRemoved(tmp_path):
    # The analog world is gone — the engine is always the binary suicide burn.
    # engineMode is no longer a field, so a stray key fails fast (unknown kwarg).
    with pytest.raises(TypeError):
        loadConfig(_writeConfig(tmp_path, 'world: {engineMode: suicideBurn}'))
```

In `tests/test_episode.py`, replace `test_suicideBurnEpisodeRunsEndToEnd` (keep its `<agent_context>`/`<agent_guardrail>` comments) with:

```python
def test_suicideBurnEpisodeRunsEndToEnd():
    cfg = loadConfig('config.yaml')
    env = LandingEnv(cfg)
    obs = env.reset(np.random.default_rng(0))
    assert obs.shape == (10,)
    terminated = truncated = False
    steps = 0
    # fire once, then cut, then coast — exercise the toggle path to a terminal
    while not (terminated or truncated) and steps < cfg.world.maxSteps + 1:
        engineCmd = 1.0 if steps < 30 else 0.0
        obs, reward, terminated, truncated, info = env.step([engineCmd, 0.0])
        steps += 1
    assert terminated or truncated
    assert env.state.engineTransitions <= 2
    assert obs.shape == (10,)
```

(Also remove the now-unused `import dataclasses` that was local to the old body.)

- [ ] **Step 2: Run the suite to confirm the expected failures**

Run: `python -m pytest tests/test_physics.py tests/test_config_loader.py tests/test_episode.py -q`
Expected: FAIL — `test_engineModeFieldRemoved` fails (field still exists, no TypeError), and the still-present `engineMode`/`minThrottle` references are about to be removed. This pins the source edits.

- [ ] **Step 3: Make the suicide-burn engine unconditional in `physics.py`**

In `src/env/physics.py`, replace the engine dispatch block (currently `if world.engineMode == 'suicideBurn':` … through the `else: # analog mode` branch, ~lines 359-380) with:

```python
        # @TAG[engine-logic]: binary suicide-burn engine — the ONLY engine mode.
        # Fires at FULL or not at all; at most two state-changes (off->on ignite,
        # on->off cut) then the engine locks. engineCommandedOn is the latched
        # intent so it survives spool decay.
        currentlyOn = self._engineCommandedOn
        desiredOn = rawThrottle > SUICIDE_ON_THRESHOLD
        transitions = self._engineTransitions
        if transitions >= 2:
            engineOn = currentlyOn
        elif desiredOn != currentlyOn and hasFuel:
            engineOn = desiredOn
            transitions += 1
        else:
            engineOn = currentlyOn
        engineCommandedOn = engineOn
        effectiveCmd = 1.0 if (engineOn and hasFuel) else 0.0
```

In the `<agent_guardrail>` header (line ~59), change `Any edit to the spool, minThrottle, throttleCutoff, suicideBurn,` to `Any edit to the spool, suicide-burn engine,`.

- [ ] **Step 4: Remove the analog fields + validations + `runtime.model` in `loader.py`; bump the model version**

In `src/config/loader.py`:
- Change `PHYSICS_MODEL_VERSION = 'pymunk-2'` to `PHYSICS_MODEL_VERSION = 'suicide-1'` and update its comment's trailing note to add: `'suicide-1' = the analog engine removed; the world is exclusively the binary suicide burn.`
- In `WorldConfig`, **delete** the two lines `minThrottle: float = 0.3` and `throttleCutoff: float = 0.05`, and **delete** the line `engineMode: str = 'analog'    # analog (continuous throttle) | suicideBurn`.
- In `RuntimeConfig`, **delete** the line `model: str = 'lux'           # thrust profile -> models/<model>/<env>/ (selector, not hashed)`.
- In `validateConfig`, **delete** these blocks:
  - `if not 0.0 < world.minThrottle < 1.0:` … `raise ValueError(... minThrottle ...)` (2 lines)
  - `if not 0.0 <= world.throttleCutoff < world.minThrottle:` … `raise ValueError('world.throttleCutoff must be in [0, minThrottle)')` (2 lines)
  - `if world.engineMode not in ('analog', 'suicideBurn'):` … its `raise ValueError(...)` (the full 4-line block at ~263-266)

- [ ] **Step 5: Make `config.yaml` the single suicide-burn config**

In `config.yaml`:
- In `world:`, **delete** the `minThrottle:`, `throttleCutoff:`, and `engineMode:` lines.
- Add a one-line header note under the top comment block: `# Single world: the binary suicide-burn engine (ignite once, cut once).`
- In `runtime:`, **delete** the `model:` line if present (the master `config.yaml` has none; verify).

- [ ] **Step 6: Delete the `configs/` directory and fix the watch HUD**

```bash
git rm -r configs
```

In `scripts/watch.py`, in the `hud` closure, replace the ignition line (currently gated on `cfg.world.engineMode == 'suicideBurn'`, ~lines 88-90) with an always-on readout, and delete the stale `[GOTCHA]` note about `engineMode`:

```python
            f"fuel {state.fuel * 100:3.0f}%  engine {state.spool * 100:3.0f}%  "
            f"ign {2 - state.engineTransitions}  "
            f"vx {state.vx:+5.1f}  vy {state.vy:+5.1f} m/s  tilt {math.degrees(state.theta):+5.1f} deg",
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (the directory agent's files, if any, are untouched). If a residual analog reference fails, fix it where reported.

- [ ] **Step 8: Commit**

```bash
git add -- src/env/physics.py src/config/loader.py config.yaml scripts/watch.py tests/test_physics.py tests/test_config_loader.py tests/test_episode.py
git commit -m "feat: remove the analog engine world; one binary suicide-burn world

Delete the analog engine branch (physics), the analog-only world fields
(engineMode/minThrottle/throttleCutoff) and their validations, and runtime.model.
config.yaml is now the single suicide-burn control panel; configs/ is removed.
Bump PHYSICS_MODEL_VERSION pymunk-2 -> suicide-1 (no checkpoints exist to
invalidate). Update analog-coupled tests; the engine always fires binary.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(`git rm -r configs` from Step 6 is already staged; it is included in this commit.)

---

### Task 3: Collapse the `--model`/`--env` naming axis (scripts) — COORDINATED

Remove the world/model naming axis from the entry points so there is one config and one world. **This task overlaps the concurrent directory agent's files** (`scripts/train.py` and possibly `src/train/parallel.py`). **Before starting:** re-read `scripts/train.py`, `src/train/parallel.py`, and `.claude/agent-memory/notes.md`. If the directory agent has already removed `--model`/`--env` as part of its `checkpoints/<run>/` collapse, skip the overlapping edits and only do what remains. Target the directory agent's `checkpoints/<run>/` scheme — never re-introduce `models/<model>/<env>/`.

**Files:**
- Modify: `scripts/train.py`, `scripts/watch.py`, `scripts/evaluate.py`
- Modify (if still present): `src/train/parallel.py` (`SeedTask.modelName`/`envName`), `tests/test_parallel.py`
- Test: `python -m pytest -q`

**Interfaces:**
- Consumes: the directory agent's checkpoint output directory (e.g. `checkpoints/<run>/`).
- Produces: scripts with no `--model`/`--env` args and no `cfg.runtime.model` reads.

- [ ] **Step 1: Re-read the shared files and confirm current state**

Run: `python -m pytest -q` (record the green baseline). Open `scripts/train.py` and `src/train/parallel.py` and confirm whether `--model`/`--env`/`modelName`/`envName` still exist. If the directory agent already removed them, note which steps below are already done and skip them.

- [ ] **Step 2: Remove `--model`/`--env` from `scripts/watch.py` and `scripts/evaluate.py`**

In **both** files, delete the `--model` and `--env` `add_argument` lines and their `<agent_context>` comment block, and replace the model-dir construction. In `scripts/watch.py` change:

```python
        modelsDir = os.path.join('models', args.model or cfg.runtime.model, args.env)
```

to (target the directory agent's checkpoint root — confirm the exact path with them; `checkpoints` is the agreed root):

```python
        modelsDir = 'checkpoints'
```

Apply the identical change to the `modelsDir = os.path.join('models', args.model or cfg.runtime.model, args.env)` line in `scripts/evaluate.py`.

- [ ] **Step 3: Remove the naming axis from `scripts/train.py`**

Coordinating with the directory agent's path scheme, remove the `--model`/`--env` args, the `modelName = args.model or cfg.runtime.model` / `modelsDir = os.path.join('models', modelName, args.env)` construction, and the `modelName`/`envName` fields passed to `SeedTask`. Write checkpoints under the directory agent's `checkpoints/<run>/`. **If the directory agent already owns `train.py`'s path construction, do not duplicate it — only strip the `--model`/`--env`/`runtime.model` reads and let their path code stand.**

- [ ] **Step 4: Drop `modelName`/`envName` from `SeedTask` if still present**

If `src/train/parallel.py` still has `modelName`/`envName` on `SeedTask` (and they are only used for the removed path construction), remove those fields and any reference to them, and update `tests/test_parallel.py` accordingly (remove `modelName=`/`envName=` from `SeedTask(...)` constructions).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS. (`tests/test_checkpoints.py` is unaffected — `resolveModelPath` is path-agnostic and its tests construct their own dirs.)

- [ ] **Step 6: Commit**

```bash
git add -- scripts/train.py scripts/watch.py scripts/evaluate.py src/train/parallel.py tests/test_parallel.py
git commit -m "refactor: drop the --model/--env naming axis; single config/one world

Remove the thrust-profile/env selection from train/watch/evaluate and the
SeedTask model/env fields; checkpoints resolve against the single run directory.
Coordinated with the directory-management rewrite (checkpoints/<run>/).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(Only `git add` the files this task actually changed; omit any the directory agent owns.)

---

### Task 4: Recover and rewrite docs to the single-world objective; update agent-memory

**Files:**
- Recover then rewrite: `docs/` (from `project-vigil-redux-2d.zip`), `README.md`
- Modify (careful append/in-place): `.claude/agent-memory/{context,decisions,notes}.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Recover the upstream `docs/` and `README.md` verbatim (without clobbering this plan/spec)**

The zip's `docs/` does not contain this plan or spec (different filenames), so a non-overwriting copy is safe. From repo root:

```bash
# (PowerShell) copy any docs/* and README.md from the extracted zip that don't already exist
```

Use the extracted zip at `…/scratchpad/unzipped/project-vigil-redux-2d/`. Copy `README.md` and the upstream `docs/*.md` (CODE_MAP, GLOSSARY, ROADMAP, CONVENTIONS, CHANGELOG, REWARD_LOG, WORKFLOWS, AGENTS) into place, skipping `docs/superpowers/` (already present).

- [ ] **Step 2: Rewrite the dual-world narrative to one world**

Edit `README.md`, `docs/CODE_MAP.md`, `docs/GLOSSARY.md`, `docs/ROADMAP.md`: remove the lux/solis "two worlds" sections and the `--model`/`--env` usage; describe the single binary suicide-burn world and the ignite-once/cut-once success rule. Update the obs table note for `obs[9]` (always meaningful now). Update command examples to the collapsed CLI.

- [ ] **Step 3: Log the change in CHANGELOG and REWARD_LOG**

Add a dated `docs/CHANGELOG.md` entry: analog world removed, `PHYSICS_MODEL_VERSION` bumped to `suicide-1`, `--model`/`--env` axis dropped. Add a `docs/REWARD_LOG.md` note: success now also requires `isCutOff` (engine off at touchdown); reward weights unchanged.

- [ ] **Step 4: Update agent-memory (re-read first — other agents edit these)**

Re-read `.claude/agent-memory/context.md` and `notes.md` (they may have changed). In `context.md`, replace the "Two worlds" table and the "Stated goal/direction" + "Known gap" sections with the single suicide-burn world and the implemented cut-before-touchdown gate. **Append** (do not overwrite) a `decisions.md` entry summarizing this rewire. Update `notes.md` rewire-targets to reflect completion.

- [ ] **Step 5: Commit**

```bash
git add -- README.md docs .claude/agent-memory/context.md .claude/agent-memory/decisions.md .claude/agent-memory/notes.md
git commit -m "docs: rewrite for the single suicide-burn world; log the rewire

Recover docs/ + README and rewrite the dual-world narrative to one binary
suicide-burn world with the ignite-once/cut-once success rule. Log the analog
removal, PHYSICS_MODEL_VERSION bump, and the cut-before-touchdown gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full verification + smoke train/eval

**Files:** none modified (verification only). Smoke artifacts go to the directory agent's `stdout/`/`checkpoints/` — do not commit them.

- [ ] **Step 1: Full suite green**

Run: `python -m pytest -q`
Expected: PASS, with no analog/engineMode references remaining. Cross-check: `python -m pytest -q | tail -1` shows all passed.

- [ ] **Step 2: Confirm no stale references remain**

Run a search for `engineMode`, `minThrottle`, `throttleCutoff`, `'analog'`, `runtime.model`, `--model`, `--env` across `src/`, `scripts/`, `config.yaml`, `tests/`. Expected: only legitimate suicide-burn references (e.g. `engineTransitions`, `engineCommandedOn`, `SUICIDE_ON_THRESHOLD`). Fix any stragglers and re-run the suite.

- [ ] **Step 3: Smoke train (short)**

Run a minimal training run to confirm the end-to-end path works under the single world (coordinate the exact invocation with the directory agent's entrypoint; e.g. `python -m scripts.train --stage touchdown` with a tiny `totalIters` override). Confirm it completes and writes a checkpoint + plot under the directory agent's scheme. Expected: a non-zero success rate is not required at this depth — only that it runs cleanly and the artifacts land.

- [ ] **Step 4: Smoke evaluate**

Run `python -m scripts.evaluate --episodes 20` against the smoke checkpoint. Confirm the outcome breakdown prints and that `info`-derived metrics (including the cut gate) are exercised without error.

- [ ] **Step 5: Final commit (if any verification fixes were made)**

```bash
git add -- <only files you changed>
git commit -m "test: finalize single suicide-burn world verification

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** §4.1 engine → Task 2; §4.2 success gate → Task 1; §4.3 reward unchanged → respected (no reward task); §4.4 config/loader → Task 2; §4.5 obs unchanged → respected; §4.6 scripts collapse → Task 3; §4.7 checkpoints → Task 3 (resolveModelPath is path-agnostic, no change needed); §4.8 PdPilot kept → respected; §5 tests → Tasks 1-3; §6 docs/agent-memory → Task 4; §7 world-hash/retrain → Task 2 (PHYSICS_MODEL_VERSION bump); §8 risks → noted; §9 acceptance → Task 5. All covered.
- **Placeholder scan:** every code step shows the exact code; test deletions name exact functions; Task 3's coordination is explicit (re-read before editing), not a placeholder.
- **Type consistency:** `engineOnAtTouchdown` (bool) and `engineTransitions` (int) names are consistent across Task 1's `_info`, the gate, and Task 1/5 assertions. `PHYSICS_MODEL_VERSION = 'suicide-1'` used consistently.
- **Known coordination risk:** Task 3 edits files the directory agent owns; mitigated by the re-read-first protocol and surgical `git add`.
