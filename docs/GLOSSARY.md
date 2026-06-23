# Glossary

Project-specific terms an agent will otherwise misread. Definitions are how *this repo* uses
the word, not the general meaning, with a `file:symbol` reference where useful. See
[AGENTS.md](AGENTS.md) for the docs overview.

## Compatibility & contracts

- **World hash** -- a 16-hex SHA-256 digest over the `world:` config fields **plus**
  `PHYSICS_MODEL_VERSION`, computed by `src/config/loader.py:Config.computeWorldHash`. A model
  is loadable iff its stored hash matches the live config's (`checkpoints.py:loadCheckpoint`
  enforces it). Excludes reward/training/curriculum/runtime. See [CONVENTIONS.md](CONVENTIONS.md) S4.
- **Obs contract** -- the fixed 10-D normalized observation in `src/env/spaces.py`:
  `[x/(width/2), y/ceiling, vx/VEL_REF, vy/VEL_REF, sin theta, cos theta, omega/OMEGA_REF, fuel, spool,
  (2 - engineTransitions)/2]`. theta enters **only** as `(sin, cos)` -- there is no raw-theta and no
  boolean-contact channel.
- **`VEL_REF` / `OMEGA_REF`** -- frozen code constants in `spaces.py` (`20.0 m/s` / `3.0 rad/s`),
  the velocity / angular-velocity normalization divisors. Part of the obs contract: changing
  them invalidates models **without** changing the world hash.
- **`PHYSICS_MODEL_VERSION`** -- code constant `'suicide-1'` in `loader.py`, folded into the world
  hash so a change to the simulation *model* (not just a config field) invalidates old
  checkpoints. Previous values: `'pymunk-2'` (Pymunk with sub-stepping), `'pymunk-1'` (Pymunk
  without sub-stepping). The tag was bumped to `'suicide-1'` when the analog engine world was
  removed and the single suicide-burn world became the only world.
- **Checkpoint** -- a saved `.pt` (weights + arch + `worldHash` + `stageName`) under
  `checkpoints/run-N/`. **`seed<seed>.pt`** is one per training seed; **`best.pt`** is the
  best across seeds for that run. Loaded by `checkpoints.py:loadCheckpoint` behind the hash guard.

## World & engine

- **Binary suicide burn** -- the only engine mode in this repo. Any action throttle command
  > `SUICIDE_ON_THRESHOLD` (0.5) fires the engine at **full**; below that it is OFF. At most
  **one ignition** (off->on) followed by **one cutoff** (on->off) are allowed -- 2 state-changes
  total -- then the engine **locks permanently**. Spool-up lag and fuel burn still apply.
  Implemented in `src/env/physics.py:BoosterSim` (`@TAG[engine-logic]`).
- **`engineCommandedOn`** -- `BoosterState` field: the latched ON/OFF *intent*. A cutoff sets
  this False and it stays False even while the spool decays to zero -- prevents the engine from
  relighting during spool decay. Never derive "engine on" from spool > 0 in suicide-burn mode.
- **`engineTransitions`** -- `BoosterState` field: count of on/off state-changes used (0, 1, or
  2). When it reaches 2 the engine is permanently locked to its current state.
- **`ignitionsRemaining`** -- `obs[9]` = `(2 - engineTransitions) / 2`. Values: `1.0` fresh
  (no transitions), `0.5` burning (one transition -- ignited), `0.0` locked (both transitions used).
- **`SUICIDE_ON_THRESHOLD`** -- `physics.py` constant `0.5`. Action throttle above this fires
  the engine at full; at or below is the OFF command.

## Physics & dynamics

- **BoosterSim** -- the persistent `pymunk.Space` in `src/env/physics.py`, owned by `LandingEnv`.
  Holds the dynamic booster body (hull + 2 legs, one rigid body) and the static ground / walls /
  ceiling. Engine/spool/fuel bookkeeping live here; Pymunk resolves collision.
- **Pymunk rigid-body** -- the 2D physics engine (a Chipmunk2D wrapper). Sub-stepped
  `_SUBSTEPS = 4` times per env step (`dt/4`) because Pymunk has no continuous collision
  detection -- without sub-stepping a fast toe tunnels through the ground (`PYMUNK_NO_CCD_SUBSTEP`).
- **`legToes`** -- `physics.py` function returning the two toe-tip world positions from a
  `BoosterState`. The **single** source of toe geometry: both contact detection (`episode.py`)
  and the renderer (`render.py`) read it, so the drawn legs match the collidable shapes exactly.
- **`legDrop`** -- world field (default `0.9 m`): how far a toe sits *below* the base when
  upright. Used by `legToes` and by the topple threshold `atan2(legSpan, bodyHalfLen + legDrop)`
  (approximately 0.32 rad).
- **Spool-up lag** -- first-order throttle response in `BoosterSim`: actual force is
  `spool * maxThrustForce`, and `spool` ramps toward the command at `throttleResponse` per
  second (~0.25 s for a full sweep). The policy commands throttle; the engine spools to it.
- **Gimbal** -- action dimension 1: nozzle deflection in `[-1, 1]`, scaled by `world.maxGimbal`
  (approximately 0.35 rad), applied as both a lateral force and an explicit torque.
- **`maxLandingSpeed`** -- world field (2.0 m/s): the gentle-impact threshold. A faster touchdown
  classifies as crash; the gentleness bonus scales inversely with impact speed under it.

## Episode & outcome

- **Rest-verdict** -- `episode.py`: the outcome is classified by **reading the settled physical
  state**, not predicted. The episode detects first ground contact, latches impact speed from
  the *approach* velocity (`prevState`), then waits for rest (`|v| < REST_SPEED`,
  `|omega| < REST_OMEGA`) before checking the full success predicate.
- **Cut-before-touchdown gate** -- the `isCutOff` success predicate in `episode.py`
  (`@TAG[cut-gate]`): a landing is a success **only if** `engineCommandedOn` was `False` at
  the moment of first toe contact (read from `prevState`, matching the impact-speed convention).
  If the engine was still on when the booster entered the contact step, the outcome is crash
  regardless of attitude or impact speed. `info['engineOnAtTouchdown']` exposes this per-episode.
- **Success predicate** -- all four must hold after settling: (1) upright (`|theta| < standTilt`),
  (2) on-pad (`|x| <= padWidth/2`), (3) gentle impact (`impactSpeed <= maxLandingSpeed`), and
  (4) engine cut before touchdown (`isCutOff`).
- **Impact speed** -- `hypot(prevState.vx, prevState.vy)` -- the approach velocity, read
  pre-contact because the solver arrests post-contact velocity (`FLOOR_CLAMP_EATS_IMPACT_SPEED`).
- **PdPilot** -- the hand-tuned binary single-burn PD controller in `src/agents/scripted.py`.
  It coasts, ignites once, and cuts before touchdown. Under the cut-before-touchdown constraint
  it is a **weak baseline** (~40% success on the easy touchdown stage). The forever-baseline
  that RL must beat.
- **Run** -- a numbered training session. `scripts/train.py` auto-increments the run counter.
  Artifacts for run N: `checkpoints/run-N/seed<seed>.pt`, `checkpoints/run-N/best.pt`,
  `stdout/logs/run-N/seed<seed>.csv`, `stdout/convergence-plots/run-N.png`.

## Reward

- **PBRS / potential-based shaping** -- the shaping term in `rewards.py:computeReward`:
  `coef * shapingScale * gamma * Phi(s') * (1 - done) - Phi(s)`. Adds a dense signal without changing the
  optimal policy (Ng et al. 1999).
- **Phi (potential)** -- `computePotential(state, world)` = `-(dist(pad)/ceiling + speed/VEL_REF +
  |theta|/pi)`. Higher (less negative) means closer to the pad, slower, more upright.
- **`(1 - done)` invariance factor** -- the multiplier on `Phi(s')` that zeroes the potential at
  terminal states. **Required** for policy invariance; must not be removed (guarded by
  `tests/test_rewards.py`). See [CONVENTIONS.md](CONVENTIONS.md) S5.
- **Shaping anneal / `shapingScale`** -- gradual weakening of the PBRS coefficient over training
  (`reward.shapingAnneal` in {`linear`, `none`}); the `[0,1]` factor is computed by
  `shapingScaleFor` in `train/loop.py` and applied per iteration.

## Training

- **GAE** -- Generalized Advantage Estimation, `train/rollout.py:computeGae` -- advantages and
  returns from a trajectory using `gamma` and `lam`. Computed per env (`computeBatchAdvantages`
  loops it over each parallel column).
- **Clipped surrogate** -- the PPO policy loss in `train/ppo.py`:
  `-min(ratio*A, clamp(ratio, 1-clipEps, 1+clipEps)*A)`, `ratio = exp(newLogp - oldLogp)`.
- **Rollout** -- `rolloutSteps` transitions collected across parallel envs with the learner
  sampling stochastically (`collectRollout`). Stores obs, pre-squash action `u`, log-prob, value,
  reward, dones, + `lastValues` for the GAE bootstrap.
- **`explainedVariance`** -- scale-invariant critic-quality metric (`1 - Var(returns-values)/
  Var(returns)`); ~1 perfect, <0 worse than predicting the mean.
- **Curriculum rung / ladder** -- a stage (`CurriculumStage`: name + spawn ranges) and the
  ordered sequence of them (`curriculum.stages`, e.g. `touchdown -> hop -> drop -> glide -> full`).
  `trainCurriculum` climbs them.
- **`promoteAt`** -- the eval-success-rate threshold (default 0.8) at which `trainCurriculum`
  advances to the next rung, carrying the same policy / optimizer / anneal clock.

## Tooling & annotation

- **`@TAG[id]`** -- in-code landmark comment for fine-grained navigation
  (`@ENTRY`, `@ANCHOR`, `@DEP`, `@CONFIG`, `@SIDEFX`, `@INVARIANT`, `@RISK`, `@TODO`). Defined by
  the `code-annotation` skill. Present on the most-edited modules; backfill is ongoing.
- **Header block** -- the `<agent_context>` / `<agent_guardrail>` comment at the top of most
  `src/` files (`[ARCH]`, `[API]`, `[GOTCHA]`, `[CRITICAL]`, `[VALIDATION]`) giving module-scale
  orientation.
- **Subagent roster** -- the five `.claude/agents/` specialists, each owning one cluster:
  `env-physics-engineer` (env/physics/spaces), `reward-shaper` (rewards + REWARD_LOG),
  `ppo-trainer` (train/ + hyperparameters), `evaluator-visualizer` (runtime/ + rendering),
  `rl-reviewer` (read-only RL-correctness review). Route to the owner of the touched module.
