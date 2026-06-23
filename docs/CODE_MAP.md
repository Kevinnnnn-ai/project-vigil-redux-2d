# Code Map

A navigation map of the repository: where things live, how they connect, and the
architectural invariants everything else follows. Update this whenever a module or directory
is added, moved, or renamed. See [AGENTS.md](AGENTS.md) for the docs system overview, and the
`task -> file -> symbol` table in `CLAUDE.md` for the finest-grained index.

## What this project is

A 2D **booster-landing** sandbox: train a **hand-written PPO** agent (no stable-baselines3,
no gymnasium registration) to land a single-stage gimbaled rocket booster soft, centered, and
upright using a **binary suicide-burn engine** (ignite once, cut once, locked after). The world
is a **Pymunk (Chipmunk2D) rigid-body** sim -- the hull + two legs are one body that physically
collides with a static ground; landing, settling, and tip-over **emerge from the solver**. A
scripted PD pilot (`PdPilot`) is the forever-baseline (a weak single-burn baseline: coasts,
ignites once, cuts before touchdown); a pygame viewer lets you watch a policy land or fly it
yourself.

## Directory layout

```
.
|-- config.yaml                 # Single control panel: world / reward / training / curriculum / runtime
|-- scripts/                    # Entry points (run as `python -m scripts.<name>` from repo root)
|   |-- train.py                # PPO trainer: single-stage or full curriculum, concurrent seeds, run-numbered artifacts
|   |-- watch.py                # Deterministic policy rollout in the pygame window
|   |-- play.py                 # Human pilot (keyboard) -- no model loaded
|   |-- evaluate.py             # Headless eval: success rate / outcome breakdown / impact speed vs PdPilot
|   `-- live_convergence.py     # Background subprocess that re-renders the convergence PNG during training
|-- src/                        # Import root `src.` (namespace package; NOT pip-installed)
|   |-- env/                    # The environment: contract, physics, episode loop, reward
|   |   |-- spaces.py           # OBS_DIM=10, ACTION_DIM=2 contract; frozen VEL_REF / OMEGA_REF; encodeObs / toEnvAction
|   |   |-- physics.py          # BoosterSim (persistent pymunk.Space); binary suicide-burn engine; spool / fuel / gimbal; legToes
|   |   |-- episode.py          # LandingEnv: gym-style reset/step; outcome classification (rest-verdict + cut-gate)
|   |   `-- rewards.py          # computeReward (terminal + PBRS shaping + control cost); computePotential (Phi)
|   |-- agents/                 # Policies + checkpoint I/O
|   |   |-- policy.py           # Policy protocol (act: obs -> env-space action)
|   |   |-- mlp.py              # MLPPolicy: actor-critic, tanh-squashed Gaussian
|   |   |-- scripted.py         # PdPilot: binary single-burn PD landing controller (weak baseline)
|   |   `-- checkpoints.py      # resolveModelPath, loadCheckpoint (worldHash compatibility guard)
|   |-- config/
|   |   `-- loader.py           # loadConfig (YAML -> frozen dataclasses), Config.computeWorldHash, PHYSICS_MODEL_VERSION
|   |-- train/                  # The PPO training stack
|   |   |-- loop.py             # trainLanding (single stage: collect->GAE->update->eval->save), evaluateSuccessRate
|   |   |-- curriculum.py       # trainCurriculum (climb stages; promote on eval success rate)
|   |   |-- rollout.py          # collectRollout (vectorized sampling), computeGae, computeBatchAdvantages
|   |   |-- vec_env.py          # VecLandingEnv (parallel independent envs; auto-reset at step time)
|   |   |-- ppo.py              # ppoUpdate (clipped surrogate + value loss + entropy), explainedVariance
|   |   |-- device.py           # resolveDevice (GPU-primary, CPU fallback -- CPU is the benchmarked default)
|   |   `-- parallel.py         # runSeeds (one OS process per seed via ProcessPoolExecutor)
|   |-- runtime/                # Visualization + playback
|   |   |-- loop.py             # runEpisodeLoop (watch + play): pause / step / reset, dwell-frame hold
|   |   |-- render.py           # Renderer (pygame window, HUD); worldToScreen, keysToControls, FPS
|   |   `-- evaluate.py         # runEvaluation (deterministic eval + outcome breakdown)
|   `-- metrics/
|       |-- logger.py           # CsvLogger: dict -> CSV (per-iteration training stats)
|       |-- plot.py             # plotConvergence: success-rate-vs-steps curves
|       `-- live.py             # resolveNextRun, runCheckpointDir, runLogsDir, runPlotPath, seedCheckpointPath, seedCsvPath
|-- tests/                      # ~19 modules, pure where possible; run `python -m pytest -q` from repo root
|-- checkpoints/                # Run-numbered checkpoints (gitignored): run-N/seed<seed>.pt + best.pt
|   `-- run-N/
|       |-- seed<seed>.pt       # Per-seed checkpoint for run N
|       `-- best.pt             # Best across seeds for run N
|-- stdout/                     # Run artifacts (gitignored except .gitkeep)
|   |-- logs/run-N/             # Per-seed metrics CSVs (seed<seed>.csv)
|   `-- convergence-plots/      # Per-run convergence PNG (run-N.png; live-updated during training)
|-- docs/                       # <- agentic documentation (this tree) + superpowers/
|-- .claude/
|   |-- agents/                 # env-physics-engineer, reward-shaper, ppo-trainer, evaluator-visualizer, rl-reviewer
|   `-- skills/                 # code-annotation, rl-debugging
|-- CLAUDE.md                   # Agent-first project spec: navigation index + routing tables + contracts
|-- pytest.ini                  # testpaths=tests, python_files=test_*.py
`-- requirements.txt            # torch, numpy, pyyaml, pygame, pymunk
```

> Run everything **from the repo root**: `src` is a namespace package (`import src.env.episode`),
> not pip-installed. Relative paths (`checkpoints/...`, `stdout/logs/...`) resolve from the cwd.
> See [WORKFLOWS.md](WORKFLOWS.md).

## The core architectural rules (load-bearing invariants)

Unlike a single split, this project is held together by a small set of non-negotiable
contracts. Breaking any of them silently invalidates models or corrupts training.

1. **Frozen obs/action contract** -- `src/env/spaces.py`. `OBS_DIM = 10`, `ACTION_DIM = 2`,
   `VEL_REF = 20 m/s`, `OMEGA_REF = 3 rad/s` are **frozen code constants**. theta enters *only* as
   `(sin, cos)` (decode with `atan2(obs[4], obs[5])`) -- there is no raw-theta and no boolean-contact
   channel. Every policy, encoder, and the PD pilot import these from `spaces.py`. Changing
   `VEL_REF`/`OMEGA_REF` invalidates models **without** changing the world hash -- see rule 2.

2. **World-hash checkpoint guard** -- `src/config/loader.py` (`computeWorldHash`),
   `src/agents/checkpoints.py` (`loadCheckpoint`). A model is loadable **iff** its stored world
   hash equals the live config's. The hash folds in the `world:` fields **plus**
   `PHYSICS_MODEL_VERSION` (`'suicide-1'`). Editing `reward`/`training`/`curriculum`/`runtime`
   keeps old models loadable; editing `world`, the frozen obs refs, or the physics *model*
   itself invalidates them.

3. **Single gamma** -- `config.yaml:training.gamma` is the **only** discount factor. It is read
   by both GAE/PPO (`src/train/rollout.py:computeGae`) **and** reward shaping
   (`src/env/rewards.py:computeReward`). There is no second reward gamma to keep in sync.

4. **PBRS `(1 - done)` potential invariance** -- `src/env/rewards.py`. The shaping term is
   `coef * shapingScale * gamma * Phi(s') * (1 - done) - Phi(s)`, where `Phi = computePotential`. The
   `(1 - done)` factor zeroes Phi at the terminal state and is **required** for policy invariance
   (Ng et al. 1999). Removing it breaks the guarantee -- guarded by
   `tests/test_rewards.py:test_shapingTelescopesToInitialPotential`.

5. **Config as the single control panel** -- `config.yaml`, parsed to **frozen** dataclasses by
   `src/config/loader.py`. There is no `configs/` directory. Every runtime knob (world geometry,
   reward weights, PPO hyperparameters, curriculum stages, device) lives in `config.yaml`. The
   only "magic numbers" in source are the frozen obs refs and the physics-model tag.

6. **Outcome emerges, the episode only observes** -- `src/env/physics.py` (`BoosterSim`) runs
   flight *and* ground contact in one Pymunk solver, sub-stepped `_SUBSTEPS = 4` times per env
   step for rigid impacts. There is no scripted touchdown verdict: `LandingEnv.step`
   classifies success/crash/timeout by **reading the settled physical state**. The additional
   **cut-before-touchdown gate** (`isCutOff`) requires the engine be commanded OFF at the
   first contact step -- a true suicide burn cuts before landing. Impact speed is read from
   the *approach* velocity (`prevState`) because the solver arrests post-contact velocity.

7. **Reward arithmetic lives in one place** -- `src/env/rewards.py:computeReward` is the only
   place reward is assembled. `LandingEnv.step` contains zero reward logic, which keeps reward
   auditing, testing, and tuning atomic.

8. **Run-numbered artifacts** -- each training session increments a run counter.
   `src/metrics/live.py` is the single source for path helpers (`runCheckpointDir`,
   `runLogsDir`, `runPlotPath`, `seedCheckpointPath`, `seedCsvPath`). Never hard-code run
   paths in scripts.

## Per-module symbol tables

### `src/env/spaces.py` -- the obs/action contract (single source of truth)
| Symbol | Kind | Role |
|--------|------|------|
| `OBS_DIM` / `ACTION_DIM` | const | `10` / `2`. The fixed contract dimensions. |
| `VEL_REF` / `OMEGA_REF` | const | `20.0 m/s` / `3.0 rad/s`. Frozen normalization divisors (part of the contract). |
| `encodeObs(state, world)` | function | Builds the 10-D float32 observation from a `BoosterState`. |
| `toEnvAction(a)` | function | Maps net tanh-space `(-1,1)^2` -> env action `[throttle 0..1, gimbal -1..1]`. |

### `src/env/physics.py` -- Pymunk rigid-body simulator
| Symbol | Kind | Role |
|--------|------|------|
| `BoosterState` | dataclass | `(x, y, vx, vy, theta, omega, fuel, spool, engineTransitions, engineCommandedOn)`. |
| `BoosterSim` | class | Persistent `pymunk.Space`; booster body + legs + static ground/walls/ceiling. Binary suicide-burn engine; spool/fuel bookkeeping; Pymunk handles collision. |
| `legToes(state, world)` | function | World-frame positions of the two toe tips -- the **single** source of toe geometry (contact detection AND renderer read it). |
| `stepPhysics(state, action, world)` | function | Pure test shim: builds a transient sim, steps once. Not the runtime path. |
| `_SUBSTEPS`, `_SOLVER_ITERATIONS` | const | `4`, `20`. Sub-ticks per env step (CCD substitute) and contact stiffness. |
| `SUICIDE_ON_THRESHOLD` | const | `0.5`. Engine fires at full when command > this; OFF otherwise. |

### `src/env/episode.py` -- the gym-style env
| Symbol | Kind | Role |
|--------|------|------|
| `LandingEnv` | class | `reset(rng) -> obs`, `step(action) -> obs/reward/terminated/truncated/info`. Owns the `BoosterSim`. |
| `LandingEnv.setStage(stage)` | method | Curriculum stage selector (takes effect on the **next** reset). |
| `REST_SPEED` / `REST_OMEGA` | const | `0.5 m/s` / `0.3 rad/s`. Settled-detection thresholds. |
| `_standTilt(world)` | function | Upright tilt threshold from leg geometry (`atan2(legSpan, bodyHalfLen + legDrop)`). |
| `_engineOnAtTouchdown` | field | Latched at first toe contact from `prevState.engineCommandedOn`; `True` = engine was still on when the booster entered the contact step -- crash. |

### `src/env/rewards.py` -- config-driven reward
| Symbol | Kind | Role |
|--------|------|------|
| `computeReward(cfg, prevState, state, action, outcome, impactSpeed, shapingScale)` | function | Assembles the per-step scalar: terminal payout + PBRS shaping + control cost. |
| `computePotential(state, world)` | function | `Phi = -(dist(pad)/ceiling + speed/VEL_REF + |theta|/pi)`. Higher (less negative) = closer/slower/uprighter. |

### `src/config/loader.py` -- config + compatibility
| Symbol | Kind | Role |
|--------|------|------|
| `loadConfig(path)` | function | YAML -> frozen `Config`; validates (positive physics, `maxThrust > gravity`, enums, stage ranges). |
| `Config` / `WorldConfig` / `RewardConfig` / `TrainingConfig` / `CurriculumStage` / `CurriculumConfig` / `RuntimeConfig` | dataclass (frozen) | The parsed config sections. |
| `Config.computeWorldHash()` | method | SHA-256 over `world:` fields + `PHYSICS_MODEL_VERSION` -> 16-hex digest (the compatibility gate). |
| `PHYSICS_MODEL_VERSION` | const | `'suicide-1'`. Bump on any physics-MODEL change to invalidate old checkpoints. |

### `src/agents/` -- policies + checkpoints
| Symbol | File | Kind | Role |
|--------|------|------|------|
| `Policy` | `policy.py` | protocol | `act(obs) -> env action`; `reset()` hook for stateful policies. |
| `MLPPolicy` | `mlp.py` | class | Actor-critic. `.act()` = deterministic squashed mean (eval/watch); `.sample()` = stochastic (rollout); `.evaluateActions()` = logp+entropy+value (update). `_squashedLogProb` carries the tanh-Jacobian correction. |
| `PdPilot` | `scripted.py` | class | Binary single-burn PD landing controller (coasts, ignites once, cuts before touchdown). The weak forever-baseline (~40% on touchdown stage). |
| `resolveModelPath` / `loadCheckpoint` | `checkpoints.py` | function | Resolve `best`/`seed<N>`/path; load with the world-hash guard (raises `ValueError` on mismatch). |

### `src/train/` -- PPO stack
| Symbol | File | Kind | Role |
|--------|------|------|------|
| `trainLanding` / `evaluateSuccessRate` / `shapingScaleFor` | `loop.py` | function | Single-stage PPO driver; deterministic eval; shaping anneal factor. |
| `trainCurriculum` | `curriculum.py` | function | Climbs stages; promotes when `successRate >= curriculum.promoteAt` (same policy/optimizer carry over). |
| `collectRollout` / `computeGae` / `computeBatchAdvantages` | `rollout.py` | function | Vectorized collection; pure-function GAE per env; batched advantages. |
| `VecLandingEnv` | `vec_env.py` | class | Batch of `LandingEnv`s stepped together; auto-resets finished envs; fans out `setStage`/`setShapingScale`. |
| `ppoUpdate` / `explainedVariance` | `ppo.py` | function | Clipped surrogate + value loss + entropy; scale-invariant critic metric. |
| `resolveDevice` | `device.py` | function | `'auto'` -> cuda-if-available else cpu (CPU is the default -- see [OBSERVATIONS.md](OBSERVATIONS.md) `CPU_BEATS_GPU_FOR_THIS_PPO`). |
| `runSeeds` / `resolveSeedWorkers` / `stageByName` | `parallel.py` | function | Concurrent per-seed training (one process per seed); worker-count resolution; stage lookup. |

### `src/runtime/` -- viewer + eval
| Symbol | File | Kind | Role |
|--------|------|------|------|
| `runEpisodeLoop` | `loop.py` | function | Drives watch/play: polls input intents, renders, holds a dwell frame at episode end. |
| `Renderer` / `Intents` / `worldToScreen` / `keysToControls` / `FPS` | `render.py` | class/function | Pygame window + HUD (pure presentation); world->pixel mapping; key->control mapping. |
| `runEvaluation` | `evaluate.py` | function | Rolls N deterministic episodes -> `{successRate, outcomes, meanImpactSpeed, meanSteps}`. |

### `src/metrics/`
| Symbol | File | Kind | Role |
|--------|------|------|------|
| `CsvLogger` | `logger.py` | class | dict -> CSV (header on first record, one row per `log`). |
| `plotConvergence` | `plot.py` | function | Matplotlib success-rate-vs-steps curves. |
| `resolveNextRun` / `runCheckpointDir` / `runLogsDir` / `runPlotPath` / `seedCheckpointPath` / `seedCsvPath` | `live.py` | function | Single source for all run-numbered path resolution. |

## Where new code goes

| You are adding... | File | Owner subagent | Don't forget |
|----------------|------|----------------|--------------|
| A **reward term / weight** | `src/env/rewards.py`, `config.yaml:reward` | `reward-shaper` | Log it in [REWARD_LOG.md](REWARD_LOG.md) (hard rule); keep PBRS potential-based with the `(1-done)` factor; `tests/test_rewards.py`. |
| An **env / physics / obs change** | `src/env/{physics,episode,spaces}.py` | `env-physics-engineer` | World-geometry edits re-hash and invalidate checkpoints; a new physics MODEL -> bump `PHYSICS_MODEL_VERSION`; regress `test_physics`/`test_spaces`/`test_episode`/`test_scripted`. |
| A **training knob / anneal** | `config.yaml:training`, `src/train/{loop,ppo}.py` | `ppo-trainer` | Never add it to the world hash; `tests/test_loop.py`/`test_curriculum.py`; note rationale in [CHANGELOG.md](CHANGELOG.md). |
| A **curriculum stage** | `config.yaml:curriculum.stages` | `ppo-trainer` | Name is the only string field; ranges are `(lo,hi)` floats; promotion is automatic; log why the rung bridges the gap. |
| An **eval metric** | `src/runtime/evaluate.py` (offline) or `src/train/loop.py` (training-time) | `evaluator-visualizer` | Add the key to the history dict so `CsvLogger` auto-columns it; `tests/test_evaluate.py`. |
