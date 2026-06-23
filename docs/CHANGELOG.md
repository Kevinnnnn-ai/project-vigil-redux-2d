# Changelog

Chronological record of what changed and why. Add a dated entry as part of every change. Newest at top. Reward changes also get a [REWARD_LOG.md](REWARD_LOG.md) entry.

Format:
```text
## <ACTION> | YYYY-MM-DD HH:MM UTC (ACTION ∈ ADD/FEATURE/FIX/REFACTOR/DOCS)
Summary
Reason
Files
Changes
Validation
Impact
Follow-up
Status
```

Document **intent**, not just implementation; reference files by path and code by `file.py:line`. Once written, an entry is immutable.

---

## FEATURE | 2026-06-23 13:32 UTC

Summary:
Reworked the convergence-plot presentation: legend moved outside the axes, axis labels renamed to "Environment Steps" / "Eval Success Rate", x-axis scientific offset pinned to 1e6 on every run, title fixed to "Training Convergence (run-N)", seed legend entries Title-cased ("Seed 1"), and per-seed curves drawn as shape-preserving smooth lines instead of blocky straight segments.

Reason:
User requested a cleaner, more legible convergence figure. The legend overlapped the curves; axis labels and title were inconsistently cased; the x-axis offset auto-upgraded to 1e7/1e8 on longer runs (wanted fixed at 1e6); and the sparse eval points connected by straight lines looked faceted.

Files:
- src/metrics/plot.py — added _monotoneSmooth (pure-numpy PCHIP / Fritsch–Carlson monotone cubic) + _endpointSlope/_sign helpers (plot.py:50); plotConvergence now draws the smoothed curve plus true-point markers (plot.py:131), labels "Environment Steps"/"Eval Success Rate" (plot.py:145), pins the x offset via ax.ticklabel_format(style='sci', scilimits=(6,6)) (plot.py:160), parks the legend at bbox_to_anchor=(1.02, 0.5) and saves with bbox_inches='tight' + bbox_extra_artists (plot.py:165, :176), Title-cases the seed labels "Seed {seed}" (plot.py:138).
- scripts/train.py — convergence title fixed to f'Training Convergence (run-{run})' (train.py:152); --model/--env no longer affect the title (now fully inert cosmetic flags).
- scripts/live_convergence.py — live title cased to f'Training Convergence (run-{args.run})' (live_convergence.py:44).

Changes:
- Smoothing is monotone (PCHIP) ON PURPOSE: no overshoot outside the data, so curves never render points < 0 or > 1 that were not measured; the curve passes through every eval point and < 3 points fall back to the raw polyline.
- scipy is NOT a dependency — the monotone cubic is implemented in numpy only.
- x-axis exponent fix uses the public scilimits low==high path (no deprecated orderOfMagnitude attribute; matplotlib 3.11-safe).
- True eval points are over-plotted as markers (label '_nolegend_') so smoothing never hides where measurements landed.

Validation:
- python -m pytest tests/test_plot.py tests/test_live.py -q -> 13 passed (return values + PNG writing unchanged).
- Empirical checks: ticklabel_format(scilimits=(6,6)) yields offset '1e6' for step maxima 8e5 / 3e6 / 5e7 with no deprecation warning; _monotoneSmooth stays within [0,1], hits endpoints exactly, and does not overshoot on non-monotone input.
- Rendered a 3-seed sample to the scratchpad and visually confirmed legend-outside, labels, 1e6 offset, title, "Seed N" casing, and smooth curves.

Impact:
- Cosmetic/presentation only; no change to logged data, checkpoints, or the plotted series. Both the live (during-training) and final frames pick up the new look automatically.

Follow-up: None. (Markers at true eval points were added for data honesty; trivially removable if undesired.)

Status: Done.

---

## REFACTOR | 2026-06-23 13:18 UTC

Summary:
Silenced the per-frame "live convergence -> ..." status line that scripts/live_convergence.py printed on every re-render. The renderer now prints only on failure; successful frames are silent.

Reason:
The live-convergence subprocess re-renders every `--interval` seconds (default 5s) and printed a success line each pass. Because scripts/train.py spawns it with inherited stdout (`_startLiveRefresher`, scripts/train.py:55), those lines interleaved into the training terminal every few seconds for the whole run — pure noise. User asked for it gone.

Files:
- scripts/live_convergence.py — removed the success `print` (and its now-unused `plotted =` capture) inside the render loop (scripts/live_convergence.py:49); kept the `except` branch that surfaces real failures.

Changes:
- Per-frame success line removed; the PNG still re-renders on the same interval — only the console echo is gone.
- Error path unchanged: `live convergence skipped (no data yet?): ...` still prints if a render raises.
- Applies everywhere (chosen over silencing only the train.py-spawned child) — the standalone `python -m scripts.live_convergence` is also quiet on success now.

Validation:
- tests/test_live.py exercises `renderConvergence` directly (test_live.py:84, :98), not the script's stdout — no test asserts on the removed line; suite unaffected.

Impact:
- Training terminal is clean during a run; the live + final convergence PNGs are unchanged.

Follow-up: None.

Status: Done.

---

## REFACTOR | 2026-06-22 00:00 UTC

Summary:
Rewired the project from a dual-world (analog "lux" + suicide-burn "solis") design to a SINGLE binary suicide-burn world. The analog engine and the lux/solis world names are gone. All user-facing docs (README, CODE_MAP, GLOSSARY, ROADMAP) were rewritten to describe the single world; CHANGELOG and REWARD_LOG received dated entries for the rewire.

Reason:
The dual-world approach (engineMode: analog | suicideBurn, configs/lux/, configs/solis/) was superseded: the project now focuses exclusively on the harder, more realistic binary suicide-burn challenge. Removing the analog world eliminates the lux/solis naming axis, the configs/ directory, and the --model/--env checkpoint selection axis. Run-numbered checkpoints (checkpoints/run-N/) replace the per-model/per-env dirs.

Files:
- src/env/physics.py — engineMode / minThrottle / throttleCutoff removed from WorldConfig; BoosterSim always uses the binary suicide-burn branch.
- src/config/loader.py — PHYSICS_MODEL_VERSION bumped 'pymunk-2' -> 'suicide-1'; engineMode / minThrottle / throttleCutoff fields removed from WorldConfig.
- config.yaml — single control panel; configs/ directory deleted.
- scripts/train.py — --model / --env flags demoted to optional cosmetic plot-title labels; checkpoints written to checkpoints/run-N/ via src/metrics/live.py helpers.
- scripts/watch.py, scripts/evaluate.py — --model / --env removed as routing axes; --run N selects the run number.
- src/env/episode.py — cut-before-touchdown gate added: isCutOff = not _engineOnAtTouchdown; success predicate now requires engine off at contact step.
- src/agents/scripted.py — PdPilot reworked as a binary single-burn baseline (coasts, ignites once, cuts before touchdown).
- src/metrics/live.py — new module: single source for all run-numbered path helpers (resolveNextRun, runCheckpointDir, runLogsDir, runPlotPath, seedCheckpointPath, seedCsvPath).
- scripts/live_convergence.py — new: background subprocess that re-renders the convergence PNG live during training.
- docs/ — README + CODE_MAP + GLOSSARY + ROADMAP rewritten to single-world; this CHANGELOG entry and the REWARD_LOG entry appended.

Changes:
- ONE world only: binary suicide-burn engine. engineMode, minThrottle, throttleCutoff removed from WorldConfig; their validation and hash contributions removed.
- PHYSICS_MODEL_VERSION: 'pymunk-2' -> 'suicide-1'. All prior checkpoints invalidated (none committed; .gitignore covers checkpoints/**/*.pt).
- configs/ directory deleted. config.yaml is the only config file.
- Checkpoint layout: checkpoints/run-N/ replaces models/<model>/<env>/. The --model and --env CLI flags on train/watch/evaluate are gone as routing axes (train keeps them as optional cosmetic plot-title labels only).
- Cut-before-touchdown success gate: success = upright AND on-pad AND gentle-impact AND engine-cut-before-touchdown. info now exposes engineOnAtTouchdown and engineTransitions per episode.
- PdPilot is now a weak binary single-burn baseline (~40% on the easy touchdown stage under the cut-gate).
- Run-numbered metrics: stdout/logs/run-N/seed<seed>.csv; convergence PNG: stdout/convergence-plots/run-N.png (live-updating during training).

Validation:
- Full pytest suite green (no code was broken; docs-only portion of this commit does not affect tests).
- Grep of rewritten user-facing docs (README, CODE_MAP, GLOSSARY, ROADMAP) clean: no stale lux / solis / analog / engineMode / --model / --env / configs/ references.
- docs/superpowers/ intact (specs + plan untouched).

Impact:
- ALL prior checkpoints invalidated by the PHYSICS_MODEL_VERSION bump; retrain required. No checkpoints are committed (.gitignore covers checkpoints/**/*.pt).
- The docs system now describes exactly one world (single suicide burn); the lux/ solis naming axis is gone from all user-facing surfaces.

Follow-up:
- Run a full scripts.train (300 iters x 3 seeds) under the new single-burn config and record a trained result in REWARD_LOG.md.

Status:
- Complete

---

## DOCS | 2026-06-15 00:00 UTC

Summary:
Rewrote README.md: slimmed it from 422 lines to a tight quick-start AND reframed the project as a growing 2D RL testbed whose first environment is `baseline` (landing), rather than a single-purpose landing repo. Added a "Project direction" section explaining that `--env` is the per-environment extension point and that new environments are a config-and-assets change, not a code change.

Reason:
The old README was 422 lines and duplicated reference material that already lives in CLAUDE.md and the source. Separately, the maintainer plans to add environments beyond landing — the README now communicates that intent and explains why the current world is named `baseline`.

Files:
- README.md

Changes:
- Removed the TOC, Personal Results table, Command & API module listing, obs/action contract table, and Tech Stack table (all duplicated CLAUDE.md / source). Net length ~150 lines.
- Reframed the title/intro as "a 2D RL testbed — starting with landing"; added a "Project direction" section (landing is the first env; `--model` = thrust profile vs `--env` = environment; `models/<model>/<env>/` isolation; planned wind/disturbance env motivating a recurrent policy).
- Corrected the checkpoint-compatibility note: `computeWorldHash` hashes the `world:` fields PLUS `PHYSICS_MODEL_VERSION` (old text said "only the `world:` fields", contradicting CLAUDE.md and the README's own limits section).
- Completed two sentences the maintainer had left truncated mid-edit (intro and "What it is").
- Verified the extension-point framing against scripts/{train,watch,evaluate}.py (`--env` defaults to `baseline`, resolves `models/<model>/<env>/`) and the configs/ tree (only `baseline.yaml` exists per profile today).

Validation:
- Documentation reviewed
- Cross-checked claims against scripts/*.py and configs/ layout

Impact:
- README is a fast quick-start that sets expectations for an expanding env list; canonical reference stays in CLAUDE.md/source.

Status:
- Complete

## CONFIG | 2026-06-15 00:00 UTC

Summary:
Gave Solis the same `full`-stage drop positioning Lux received in commit 1183307 — fixed drop height and narrowed horizontal spawn — which had never been mirrored to Solis.

Reason:
Solis's final/real-task curriculum stage still spawned from a variable height (altitude [40, 52]) across the full arena width (xOffset [-14, 14]), while Lux had been changed to a fixed 52 m drop within a tight ±6 m band near the pad. The two worlds were meant to share spawn geometry; Solis was left behind.

Files:
- configs/solis/baseline.yaml

Changes:
- `full` stage altitude: [40.0, 52.0] -> [52.0, 52.0] (fixed drop height)
- `full` stage xOffset: [-14.0, 14.0] -> [-6.0, 6.0] (narrowed near pad)

Validation:
- Diffed against configs/lux/baseline.yaml — `full` stage now matches.
- Curriculum-only change: does NOT touch `world:`, so existing Solis checkpoints remain compatible (world hash unchanged).

Impact:
- Solis training/eval now drops from the same geometry as Lux for the real task. Existing Solis models stay loadable, but a retrain is advisable for them to specialize on the narrower, fixed-height spawn.

Follow-up:
- Consider whether the suicide-burn dynamics warrant Solis-specific spawn values rather than exact Lux parity; for now they match.

Status:
- Complete

## FEATURE | 2026-06-16 02:00 UTC

Summary:
Replaced the entire hand-written physics with the Pymunk (Chipmunk2D) 2D rigid-body engine, so the booster's legs are REAL collidable shapes that physically contact the ground. The booster hull + two legs are one rigid body; the ground, side walls, and ceiling are static segments. Landing, settling, and tip-over now EMERGE from the solver's contact forces / friction / torque — there is no scripted "pivot about a planted toe" verdict anymore. Hard impacts are made rigid (no deep penetration) by advancing the solver in sub-ticks per env step.

Reason:
The previous model (see the 2026-06-15 entry below) was a scripted approximation:
touchdown fired on a coordinate crossing and a hand-written `stepPivot` rotated the booster about a pinned toe; the legs had no hitbox and never collided. The user wanted genuine leg-ground collision physics — "that's why we have it in Pygame". Pymunk is the standard 2D rigid-body engine for the Pygame ecosystem and installs from a prebuilt wheel (Box2D, the alternative with native continuous collision, would not build in this environment).

Files:
- requirements.txt — pymunk>=7.0
- src/env/physics.py — NEW BoosterSim class (persistent pymunk.Space; booster body + hull Poly + 2 leg Segments; static ground/walls/ceiling); engine spool/fuel/analog-vs-suicideBurn logic preserved EXACTLY; thrust applied as a force at the CoM + an explicit gimbal torque; sub-stepping (_SUBSTEPS=4) with forces re-applied per sub-tick; stepPhysics kept as a pure-function test shim. Deleted stepPivot and the legacy contact/contactX BoosterState fields.
- src/env/episode.py — LandingEnv owns a persistent BoosterSim. Outcome is classified from the physical state: gentle (approach speed <= maxLandingSpeed) AND upright (|theta| < atan2(legSpan, bodyHalfLen+legDrop)) AND on-pad -> success; toppled / off-pad / fast -> crash; else timeout at maxSteps. The scripted pivot/verdict/contact-phase machinery is gone.
- src/config/loader.py — PHYSICS_MODEL_VERSION ('pymunk-2') folded into computeWorldHash so a SIMULATION-MODEL change invalidates old checkpoints even though the world: fields are unchanged. settleStepCap is now DORMANT.
- src/runtime/{render,loop}.py, scripts/{watch,play}.py — _bodyPolygon reads world.bodyHalfLen (no drawn-vs-collidable desync); end-of-episode dwell so the touchdown/settle is actually visible; `watch --pilot pd` flies the scripted PD pilot with no checkpoint.
- tests/test_{physics,episode,config_loader}.py — rewritten to assert the Pymunk reality (physical confinement, rest-on-legs, true topple) + a physics-version hash guard.

Changes:
- Engine/spool/fuel dynamics IDENTICAL to the prior model (the engine block is byte-for-byte preserved); only the MOTION integration moved to Pymunk.
- Sub-stepping (4 sub-ticks of dt/4) fixes the "rubbery" feel of hard crashes: a 15 m/s slam penetrated the ground ~0.76 m at 1 substep and oozed back out; at 4 substeps it penetrates ~0.000 m and stops rigidly. Forces are re-applied each sub-tick (Pymunk clears body.force after every space.step) and drag is recomputed from the live velocity; spool/fuel update once per env step.
- World re-hashed (PHYSICS_MODEL_VERSION + the prior leg fields): lux 5597d7fefc2a6d02, solis 57716d5a62445ce3. (The 'pymunk-1' string was the same model before sub-stepping.)

Validation:
- Full suite GREEN: 148 passed (torch training suites included).
- RL-correctness review (rl-reviewer): SAFE TO MERGE. Verified single-process AND CROSS-PROCESS determinism — two fresh spawn processes produce BIT-IDENTICAL trajectories, so the parallel-seed-training invariant holds. PBRS/(1-done), single gamma, terminated/truncated exclusivity, GAE bootstrap, impact-speed from prevState, hash guard, and sub-step force-scaling (0.09% from analytical) all PASS. Obs contract (OBS_DIM=10, VEL_REF/OMEGA_REF, sin/cos theta) unchanged.
- Render-geometry audit: drawn leg toes and hull match the collidable Pymunk shapes to 0.000 mm at theta=0 AND theta=0.3 (toes single-sourced via legToes).
- PdPilot lands ~100% on touchdown and full; deterministic.

Impact:
- ALL existing checkpoints invalidated (physics model changed); retrain required. No checkpoints are committed (.gitignore covers models/**/*.pt).
- New dependency: pymunk (prebuilt wheel; pytest fails on a fresh clone until `pip install -r requirements.txt`).
- Known limitation: Pymunk has no continuous collision detection, so >~20 m/s can tunnel through a static segment for a sub-tick. Sub-stepping bounds this; in practice spawn velocities cap at -12 m/s, well below the threshold.

Follow-up (NOT done — reported, not actioned):
- A full ppo-trainer run on the new dynamics is still needed. PPO is tuned for the OLD dynamics; the PdPilot landing != PPO will. Likely needs curriculum / reward re-tuning (watch for the M5_CURRICULUM_GAP signature).
- Judgment-call cleanups deferred: removing the stepPhysics/boosterCoM test-only helpers and the dormant settleStepCap/maxLandingOmega hashed fields.

Status:
- Complete (physics + reviews + cleanup + docs). Training re-tune deferred.

## FEATURE | 2026-06-15 23:30 UTC

Summary:
The landing legs are now PHYSICAL contact points. An episode has two phases:
FLIGHT (unchanged free integration) ends when the lowest leg TOE reaches the ground (not the body base); SETTLING (new) pins that toe and pivots the booster about it under gravity + carried spin until it rests on both legs (success) or topples (crash). The stand/topple verdict now EMERGES from simulated rigid-body dynamics — it REPLACES the one-shot `tipOverAtTouchdown` prediction. New hashed world fields `legDrop` and `settleStepCap`.

Reason:
Reported defect: the legs never physically contacted the ground — touchdown fired on the body base point and a one-shot inequality (`|theta + omega*settleTime| >= atan2(legSpan, bodyHalfLen)`) PREDICTED toppling at that instant. Tracking tilt against a static inequality at one instant carried no physical meaning: the booster never stood on its legs and then stayed up or fell over, and the renderer even drew the toes 0.9 m BELOW the resting base. This makes the legs real: the booster settles on them and the outcome is physical.

Files:
- src/config/loader.py — WorldConfig.legDrop (0.9) + settleStepCap (120), both hashed; validation
- config.yaml, configs/lux/baseline.yaml, configs/solis/baseline.yaml — new world keys
- src/env/physics.py — legToes() (@TAG[leg-geometry]), boosterCoM() (@TAG[com]), stepPivot() pivot integrator; BoosterState.contact / contactX
- src/env/episode.py — two-phase step(): _flightContact (toe-based touchdown), impact-speed gate, settling loop, settleVerdict (@TAG[rest-verdict]); lowestToe; removed tipOverAtTouchdown and _classify
- src/runtime/render.py — toes drawn from the shared legToes() at world.legDrop (the LEG_DROP constant removed); drawn toes are now the physical toes
- tests/test_{config_loader,physics,episode,render}.py — new coverage

Changes:
- Contact triggers on the LOWEST toe (legToes), so the booster contacts ~legDrop higher than before and the body base no longer drives touchdown.
- Impact-speed gate: a toe-plant with approach speed > maxLandingSpeed (read from prevState, preserving the FLOOR_CLAMP_EATS_IMPACT_SPEED guard) OR off-pad is an IMMEDIATE crash — no settling. A gentle, on-pad contact enters settling.
- stepPivot: rigid rotation about the pinned toe; gravity torque = m*g*leverX about the toe, parallel-axis inertia, semi-implicit Euler; position is RECONSTRUCTED from the pin constraint (toe stays at (contactX, 0)), NOT Euler-integrated. Engine is OFF during settling (spool 0, action ignored).
- settleVerdict resolves in order: (1) TOPPLE crash if the CoM passes outboard of the pinned toe; (2) STAND success if the OTHER (non-pinned) toe reaches the ground (rocked onto both legs — the primary success path); (3) residual REST success (|omega| < REST_OMEGA over the footprint); else still settling. A settleStepCap (120) fallback resolves any case that never damps.
- Verified physical fact: a booster pinned on one toe SELF-RIGHTS for small tilt and only topples past atan2(legSpan, bodyHalfLen + legDrop) ~= 0.322 rad — note this uses bodyHalfLen + legDrop (toe is legDrop below the base), so the pivot is STRICTER on static tilt than the retired prediction's atan2(legSpan, bodyHalfLen) ~= 0.46 rad, and more faithful.

Validation:
- Full suite GREEN: 150 passed (torch suites included — curriculum/ppo/rollout/ vec_env/mlp/loop/evaluate/checkpoints all ran).
- New tests cover: toe geometry + CoM; pin-constraint invariance; self-right below threshold / topple above; settling success/crash; spin-induced topple; fast & off-pad immediate crash; toe-not-base contact; second-leg stand; impact-speed from approach; settling determinism across the phase boundary.
- Regression caught & fixed mid-implementation: a near-upright booster on one toe rotated its free toe THROUGH the ground and wrapped to a crash (broke test_pdPilotLandsThroughTheLoop). Fixed by the second-toe STAND verdict. The scripted PdPilot now lands 30/30 on both touchdown and full stages.
- Headless render smoke: a booster at y=legDrop rests with BOTH toes at y=0 (visual/physics mismatch resolved); resting + settling frames draw cleanly.

Impact:
- world: hash CHANGES (legDrop, settleStepCap hashed): lux 5fe29b437768b405 -> 6578a7cb3d13d4c9, solis -> 1ed307a9f15300c5. Existing checkpoints invalidated (none committed; .gitignore covers models/**/*.pt) and must be retrained; the loadCheckpoint guard rejects stale ones.
- OBS_DIM unchanged at 10 — contact/phase is NOT observed (the policy has no authority once a toe plants and the engine is off; deliberate YAGNI).
- world.settleTime is now a DORMANT hashed field (no longer read by the verdict); left in place to avoid a second re-hash.

Follow-up (NOT done here — reported, not actioned, per plan):
- A short reduced curriculum run (40 iters/1 seed) STALLED on the hop stage at 0.00 with decaying entropy — the M5_CURRICULUM_GAP_VS_DYNAMICS_DIFFICULTY signature (the harder settling dynamics likely opened a curriculum gap; the run was also under-resourced). The env IS solvable (PdPilot 30/30; touchdown stage trains to 0.80). A full ppo-trainer pass on the shipping config is needed; if hop still plateaus, add an intermediate rung and raise entCoef to >= 0.02 (per the M5 observation). This is a ppo-trainer / curriculum tuning decision.

Status:
- Complete (feature + tests). Training re-tune deferred to ppo-trainer (reported).

Spec/plan: docs/superpowers/specs/2026-06-15-physical-leg-ground-contact-design.md, docs/superpowers/plans/2026-06-15-physical-leg-ground-contact.md

## FEATURE | 2026-06-15 21:00 UTC

Summary:
Parallelized per-seed training. The seeds in `training.evalSeeds` now train CONCURRENTLY, one OS process per seed, instead of sequentially. On a multi-core CPU this collapses a 3-seed run from ~3x wall-time toward ~1x. Parallel is the default; `--serial` (or `seedWorkers: 1`) restores the old sequential behavior.

Reason:
Within a seed the 16 envs were already vectorized, but the N full seed runs in `scripts/train.py` ran one after another — wasted parallelism on the now-default CPU training (see the CPU-default entry below). Seeds are independent (each owns its env/net/optimizer/RNG), so they fan out across processes cleanly. Threads would not help (GIL serializes the Python-heavy env sim); multiprocessing does.

Files:
- src/train/parallel.py (new — SeedTask/SeedResult, _runSeed worker, runSeeds dispatcher, resolveSeedWorkers, stageByName moved here from scripts/train.py)
- scripts/train.py (builds SeedTask list, calls runSeeds; new --serial flag; fails fast in parent on bad --stage)
- src/config/loader.py (new training.seedWorkers field: 'auto' | int>=1; validateConfig rejects bool/0/negatives/non-int)
- config.yaml, configs/lux/baseline.yaml, configs/solis/baseline.yaml (seedWorkers: auto)
- tests/test_parallel.py (new)
- docs/superpowers/specs/2026-06-15-parallel-seed-training-design.md (new)
- docs/personal/commands.md (seedWorkers / --serial notes)

Changes:
- One ProcessPoolExecutor process per seed, capped at min(len(evalSeeds), cpu_count) under 'auto'; an int pins/clamps the cap; maxWorkers<=1 runs in-process (no pool) for the serial/test path.
- Each child caps torch.set_num_threads(1) so N processes don't each spawn cpu_count BLAS threads and oversubscribe.
- Worker passes only primitives (config PATH + ints/strings) and reloads the config in the child — required for Windows 'spawn' pickling.
- Results returned SORTED BY SEED so best.pt selection, the summary, and the convergence plot stay order-stable regardless of completion order.
- Per-seed bestRate logic (single-stage max vs curriculum final-stage best) moved into parallel.py and is now shared by the worker and script.

Validation:
- Unit tests passed — `python -m pytest tests/test_parallel.py -v` (7 tests, incl. a real ProcessPoolExecutor parallel-vs-serial parity test).
- Full suite passed — `python -m pytest -q` (132 tests).
- End-to-end smoke on a tiny config: parallel (3 workers) and --serial both produce all seed<N>.pt + best.pt + convergence.png; serial and parallel emitted IDENTICAL per-seed EV/rollout values (determinism invariant holds).

Impact:
- Multi-seed training runs are ~Nx faster on an N-core CPU (N = number of seeds, capped by cores). No change to results, checkpoints, metrics CSVs, or best-across-seeds selection — only console line ORDERING interleaves under a real pool.

Follow-up:
- None. GPU multi-process scheduling intentionally out of scope (stack runs on CPU; `auto` device still resolves per child).

Status:
- Complete

## CONFIG | 2026-06-15 18:30 UTC

Summary:
Defaulted training to CPU (`training.device: cpu`) in both worlds after benchmarking showed CPU is ~2.8x FASTER than GPU for this stack. The GPU wiring added earlier today is kept intact as a dormant, tested fallback — only the config default flips. No code or checkpoints affected.

Reason:
The 64x64 MLP workload is launch-bound, not compute-bound. VecLandingEnv steps its sub-envs in a Python loop over numpy LandingEnvs, so each of the 2048 rollout ticks does a tiny GPU inference + host<->device round-trip; that overhead starves the GPU. Measured (lux/baseline, numEnvs 16, rolloutSteps 2048, 15 timed iters): CPU 10.6 s/iter vs CUDA 29.3 s/iter (2.77x). For a full 300-iter x 3-seed run that is ~2.6h on CPU vs ~7.3h on GPU.

Files:
- configs/lux/baseline.yaml
- configs/solis/baseline.yaml
- docs/observations.md (CPU_BEATS_GPU_FOR_THIS_PPO)
- docs/commands.md

Changes:
- Set `training.device: cpu` under the training block of both baseline configs, with an inline comment explaining the why and how to re-enable GPU ('auto').
- Logged the benchmark + root cause in observations (CPU_BEATS_GPU_FOR_THIS_PPO).
- Noted the device knob in the command cheat-sheet.

Validation:
- Benchmarked with a throwaway scripts/bench_device.py timing the real collect->GAE->update cycle on each device (deleted after use).
- Device wiring already covered by tests/test_device.py (unchanged, still passing).

Impact:
- Default training runs now use CPU and finish ~2.8x sooner. GPU remains one config edit away (device: auto). Checkpoints are device-independent (device is not in computeWorldHash), so this does not invalidate any model.

Follow-up:
- A genuinely batched/vectorized env (one tensor step over all sub-envs, no Python per-env loop) is the only path that would make GPU competitive; not worth it at this net size. See observations entry.

Status:
- Complete

## FEATURE | 2026-06-15 18:00 UTC

Summary:
Wired GPU-primary / CPU-fallback device handling into the training stack (the §2 code change spec'd in the since-removed docs/personal/gpu-setup.md, previously unimplemented). A run now uses CUDA when present and falls back to CPU automatically; pin to CPU with `training.device: cpu`. (Superseded same day by the CONFIG entry above: CPU is the benchmarked default — GPU is ~2.8x slower.)

Reason:
The stack had ZERO device handling, so a CUDA torch wheel alone still trained on CPU even on a GPU box. "GPU-primary with CPU fallback" required this code change in addition to the install.

Files:
- src/train/device.py (new — resolveDevice helper, single source of truth)
- src/config/loader.py (TrainingConfig.device: 'auto' | 'cpu' + validation)
- src/train/loop.py, src/train/curriculum.py (resolve device, .to(device), optimizer built after, device-threaded rollout + update tensors)
- src/train/rollout.py (collectRollout(device=...); .cpu().numpy() at every inference boundary)
- src/agents/mlp.py (act() builds input on the model's own device, .cpu().numpy())
- tests/test_device.py (new)

Changes:
- `resolveDevice('auto')`: cuda if torch.cuda.is_available() else cpu; 'cpu' forces fallback.
- `training.device` is a TRAINING field — NOT hashed, so checkpoints stay portable between the CPU laptop and the GPU PC. load() keeps map_location='cpu'.
- Every boundary `.numpy()` is now `.cpu().numpy()` (a no-op on CPU), guarding the CUDA-tensor `.numpy()` raise.

Validation:
- Unit tests passed: full suite 125 passed (was 121; +4 in test_device.py), headless (SDL_VIDEODRIVER=dummy), on BOTH a CPU-only box and the CUDA box.
- CPU-fallback smoke: `scripts.train` single-stage AND curriculum run end-to-end; prints `device: cpu`, trains, evals, saves a checkpoint.
- CUDA verified on real hardware (RTX 5060 / Ryzen 5600, torch 2.11.0+cu128): `cuda True`, smoke train printed `device: cuda`, exit 0, and a GPU-saved checkpoint loaded back fine (map_location='cpu', portable).

Impact:
- On the GPU PC, the PPO update phase runs on cuda automatically; the NumPy-bound rollout stays CPU-bound (modest net speedup for a 64×64 MLP).
- No behavior change on CPU-only machines (fallback path is identical).

Note:
- The RTX 5060 (Blackwell, sm_120) needs the cu128 wheel specifically; earlier CUDA indexes lack sm_120 kernels. Newest cu128 build is torch 2.11.0+cu128.

Status:
- Complete (CPU-fallback + CUDA paths both validated)

## FEATURE | 2026-06-15 12:00 UTC

Summary:
Gave the booster a physical ground (it now rests on the surface instead of sinking through it), landing legs (drawn, and their stance sets the stability footprint), and a leg-footprint tip-over verdict that REPLACES the old flat maxLandingTilt / maxLandingOmega success caps. New hashed world fields: `legSpan`, `bodyHalfLen`, `settleTime`.

Reason:
Requested: a physical ground so the booster does not pass through it; a tilt check for whether it falls over after landing; and landing legs to complement that check. The legs are not decoration — their toe span (legSpan) is the footprint the tip-over test resolves, so a wider stance buys real tilt tolerance.

Files:
- src/env/physics.py — @TAG[ground-floor]: clamp y >= 0 and zero downward vy on contact
- src/env/episode.py — tipOverAtTouchdown() (@TAG[tip-over]); _classify takes impactSpeed; step() derives impactSpeed from the pre-clamp APPROACH velocity (prevState)
- src/config/loader.py — WorldConfig.legSpan/bodyHalfLen/settleTime + validation
- config.yaml, configs/lux/baseline.yaml, configs/solis/baseline.yaml — new world keys
- src/runtime/render.py — splayed landing legs (_legSegments) + crisp ground surface line
- scripts/play.py — HUD now shows the tip-over angle, not the retired flat tilt cap
- tests/test_physics.py — floor clamp tests; tests/test_episode.py — tip-over / spin / impact-speed tests

Changes:
- Physical floor: stepPhysics clamps the base to y >= 0 and kills downward velocity (inelastic, mirrors the existing wall/ceiling clamp). Upward velocity is left intact.
- Tip-over verdict (instant, at the contact step — no post-landing settle phase is simulated): topples iff |theta + omega*settleTime| >= atan2(legSpan, bodyHalfLen). The omega*settleTime lookahead catches a booster rotating toward a toe.
- Success bar change: maxLandingTilt / maxLandingOmega no longer gate success; the footprint test does. Default legSpan 0.9 / bodyHalfLen 1.8 -> tip angle ~0.46 rad (~27 deg), LOOSER than the retired 0.15 rad flat cap. Those two keys survive only as HUD reference text in play.py.
- Coupling fix: because the floor clamp zeroes post-step vy, impact speed is now read from prevState (the approach velocity) so a fast descent still classifies as a crash.
- Legs are drawn in the booster body frame so they rotate with theta; their toe span equals world.legSpan (the physics footprint).

Validation:
- Unit tests passed: 87 tests across physics/episode/config/render/rewards/spaces/ scripted/runtime_loop (torch-dependent suites — curriculum/checkpoints/loop/ppo/ rollout/vec_env/mlp/evaluate — not run locally; no torch installed). New tests cover the floor clamp, static-tilt and spin-induced tip-over, within-footprint success, and the impact-speed-from-approach regression.
- Rendered resting (tilted, on legs) and airborne (legs + flame) frames headlessly and visually confirmed legs, ground line, and resting-on-surface behavior.

Impact:
- world: hash CHANGES (three new world fields), so any existing checkpoint is invalidated and must be retrained — acceptable because no checkpoints exist in the repo yet.
- The success bar is now LOOSER on attitude (footprint ~0.46 rad vs old 0.15 rad). Models trained against this env will face an easier tilt criterion; the PdPilot still aims for near-upright, so its success rate is unaffected.

Follow-up:
- If a stricter (more realistic) tip-over angle is wanted, narrow legSpan; the validator and tests parameterize off the world fields, so no code change is needed.
- Tip-over is an instant verdict; a future option is to simulate the post-contact settle (pivot about a leg toe) — that would be a larger physics change and re-hash the world.

Status:
- Complete

## FEATURE | 2026-06-15 00:00 UTC

Summary:
Training sessions now auto-generate a convergence plot (eval success rate vs. cumulative environment steps, one overlaid line per seed) and store it as `convergence.png` alongside the checkpoints in `models/<model>/<env>/`.

Reason:
Requested: after a training session, graph the convergence and store it with the model files in their respective directories. Note the y-series is `successRate` (the eval landing rate / promotion signal), NOT episode reward — the metrics CSV / history logs no reward channel, so success rate is the available convergence signal. The x-axis converts PPO `iter` to env steps via `iter * rolloutSteps * numEnvs` (2048 * 16 = 32768 steps/iter for the lux/solis baselines).

Files:
- src/metrics/plot.py (new) — `plotConvergence(histories, outPath, rolloutSteps, numEnvs, title)`
- scripts/train.py — collect `{seed: history}`; after the seed loop, write the plot
- tests/test_plot.py (new) — PNG written / non-empty; stub seed (<2 rows) skipped

Changes:
- New headless ('Agg' backend) plot module; pure function of in-memory histories.
- train.py writes `models/<model>/<env>/convergence.png` in a warn-and-continue try/except so a plotting glitch never discards a finished run.
- Seeds with <2 recorded iterations (stub/aborted runs) are skipped automatically.

Validation:
- Unit tests passed (tests/test_plot.py, 2 tests).
- Smoke-rendered from real runs/seed{1,2}_metrics.csv: seed0 stub skipped, x-axis spans ~9.8M env steps (299 iters x 32768). `import scripts.train` clean.
- tests/test_loop.py + tests/test_curriculum.py still pass.

Impact:
- Every `python -m scripts.train` run leaves a convergence figure next to best.pt. `.gitignore` ignores only `models/**/*.pt`, so the PNG (unlike the checkpoints) IS tracked and committed.

Follow-up:
- If true episode reward is wanted on the y-axis, the training loop must first log a `meanEpisodeReward` per iteration (not currently captured).

Status:
- Complete

## CHANGE | 2026-06-14 00:00 UTC

Summary:
The lux/baseline `full` curriculum stage now spawns the booster from a FIXED drop height (altitude [52.0, 52.0]) with a narrowed horizontal range (xOffset [-6.0, 6.0]). Every reset drops from y=52.0 (top of screen, just under ceiling 60) at a randomized x within [-6, 6]. Earlier stages (touchdown/hop/drop/glide) are unchanged.

Reason:
Requested behavior: drop from a fixed top-of-screen height each time, at a changing horizontal position near the pad. The "fixed height" is achieved by collapsing the altitude uniform range to a single point (uniform(52,52) is deterministic) — no env code change. The horizontal range was narrowed from the wide [-14,14] to [-6,6]. Note [-6,6] extends ~2 m past each pad edge (pad spans [-4,4]); the strict "within the pad" constraint was intentionally relaxed (user-confirmed) to avoid editing world.padWidth, which would change the world hash and invalidate existing checkpoints.

Files:
- configs/lux/baseline.yaml (full stage: altitude, xOffset)
- docs/superpowers/specs/2026-06-14-fixed-height-pad-confined-spawn-design.md (design)

Changes:
- full stage altitude [40.0, 52.0] -> [52.0, 52.0]
- full stage xOffset [-14.0, 14.0] -> [-6.0, 6.0]

Validation:
- Unit tests passed (tests/test_config_loader.py: 23 passed)
- Verified across 2000 resets: y == 52.0 exactly every time; x within [-6, 6]
- World hash unchanged (5fe29b437768b405 before and after) — lux/baseline checkpoints remain loadable

Impact:
- The `full` stage (what watch/play/evaluate use) is now EASIER than before: fixed altitude and a much narrower horizontal spread. Evaluation comparisons against older runs are no longer apples-to-apples.

Follow-up:
- None.

Status:
- Complete

## ADD | 2026-06-15 03:04 UTC

Summary:
Nested training environments one level deeper under each thrust profile: models now live at models/<model>/<env>/ and configs at configs/<model>/<env>.yaml, selected by a new --env flag (default 'baseline') on train/watch/evaluate. Existing lux/solis checkpoints and configs were migrated into the 'baseline' env.

Reason:
To train and store models from different environments side by side without the lux/solis checkpoint dirs colliding, while keeping lux and solis as the two thrust profiles. The <env> level is purely organizational; checkpoint compatibility stays enforced solely by the worldHash guard (so a physics-differing env is still rejected on cross-load, and a reward/training-only env that shares world: stays loadable).

Files:
- scripts/train.py, scripts/watch.py, scripts/evaluate.py
- src/agents/checkpoints.py (comment only)
- tests/test_checkpoints.py
- configs/lux/baseline.yaml (moved from configs/lux.yaml)
- configs/solis/baseline.yaml (moved from configs/solis.yaml)
- models/lux/baseline/, models/solis/baseline/ (.pt files moved in; .gitkeep added)
- docs/commands.md, docs/observations.md, CLAUDE.md

Changes:
- Added --env to train/watch/evaluate; modelsDir = os.path.join('models', <model>, args.env)
- No logic change to checkpoints.py — the unchanged resolver is handed the deeper path
- git mv of the two configs into configs/<model>/baseline.yaml; on-disk move of the git-ignored .pt files into models/<model>/baseline/
- Added a regression test covering nested models/<model>/<env>/ resolution
- Updated the command cheat-sheet, CLAUDE.md path references/entry_points, and added an observation that the <env> level is organizational, not a hash axis

Validation:
- Full pytest suite green
- evaluate --model lux --env baseline --config configs/lux/baseline.yaml resolves models/lux/baseline/best.pt and loads under the worldHash guard; omitting --env resolves identically (default 'baseline')
- Same verified for solis

Impact:
- New environments can be added per profile (e.g. configs/lux/highwind.yaml -> models/lux/highwind/) with no further code change

Follow-up:
- No new environment configs created yet — 'baseline' is the only env established by this change

Status:
- Complete

Spec/plan: docs/superpowers/specs/2026-06-14-per-environment-model-subdirs-design.md, docs/superpowers/plans/2026-06-14-per-environment-model-subdirs.md

## ADD | 2026-06-13 12:59 UTC

Summary:
Added a hashed world.engineMode field (analog|suicideBurn). suicideBurn gives Solis a true suicide-burn engine: binary full-throttle, at most two state-changes (one ignition, one cutoff) then the engine locks permanently; intent latched in BoosterState.engineCommandedOn so a cutoff is never undone by spool decay. Obs grew 9->10 (index 9 = ignitions-remaining). Lux stays analog. Foundations only — Solis reward/curriculum tuning is follow-up.

Reason:
Give the two models genuinely different thrust profiles (Lux dynamic throttle vs Solis fixed full-throttle suicide burn) as distinct, hash-incompatible worlds.

Files:
- src/config/loader.py (engineMode field + validation)
- src/env/physics.py (BoosterState.engineTransitions + engineCommandedOn; suicideBurn branch + SUICIDE_ON_THRESHOLD)
- src/env/spaces.py (OBS_DIM 10, index 9)
- configs/solis.yaml (suicideBurn)
- configs/lux.yaml + config.yaml (analog)
- scripts/watch.py (HUD ignitions)
- tests/test_config_loader.py, tests/test_physics.py, tests/test_spaces.py, tests/test_episode.py

Changes:
- engineMode enum ('analog'|'suicideBurn') is included in computeWorldHash(), making Lux and Solis hash-incompatible
- suicideBurn branch: engine is binary (full throttle or off), threshold SUICIDE_ON_THRESHOLD=0.5
- At most 2 state-transitions (one ignition + one cutoff); engine locks permanently after the second transition
- Cutoff intent latched in BoosterState.engineCommandedOn — a commanded-off engine stays off even while spool decays toward zero
- A transition is counted at command time (not at spool threshold); minThrottle/throttleCutoff are bypassed in suicideBurn
- Obs index 9 = ignitionsRemaining = (2-engineTransitions)/2 (0.0, 0.5, or 1.0)
- lux/solis world-hash guard test FLIPPED to assert the two hashes DIFFER (was checking equality)
- watch.py HUD shows `ign N` in suicideBurn mode

Validation:
- 112 unit tests passed (.venv/Scripts/python -m pytest -q)
- suicideBurn cutoff bug (engine relighting during spool decay) found in code review, fixed by latching intent in engineCommandedOn, guarded by two regression tests (cutoff-from-full and no-relight-during-decay)

Impact:
- ALL prior checkpoints invalidated: OBS_DIM grew 9->10 AND engineMode is now part of the world hash — both Lux and Solis must be retrained from scratch
- models/lux/best.pt trained under OBS_DIM=9 no longer loads under the new world hash

Follow-up:
- Tune Solis reward/curriculum to make the suicide burn the efficient optimum (log in docs/reward-log.md when done)
- Implement a suicide-burn PD pilot baseline

Status:
- Complete

## ADD | 2026-06-13 11:32 UTC

Summary:
Named-model-dirs reorganization: models/ is now split into per-model subdirs (models/lux/, models/solis/); the current checkpoint moved to models/lux/. Scripts gained --model <name> and --checkpoint <sel> flags; runtime.model config field added; configs/lux.yaml and configs/solis.yaml share one world hash.

Reason:
Adopt the tag-simulation models/<name>/ convention to host multiple named models representing two landing styles (lux and solis). The two configs share an identical world: block so checkpoints stay cross-compatible (M0 contract).

Files:
- src/config/loader.py (runtime.model field)
- scripts/train.py, scripts/watch.py, scripts/evaluate.py
- configs/lux.yaml, configs/solis.yaml
- models/lux/.gitkeep, models/solis/.gitkeep (current checkpoints moved to models/lux/)
- tests/test_config_loader.py, tests/test_checkpoints.py
- docs/change-log.md, docs/observations.md, README.md, src/agents/checkpoints.py (comment)

Changes:
- Added runtime.model config field (default 'lux') to src/config/loader.py
- Moved existing checkpoints into models/lux/; scaffolded empty models/solis/ with .gitkeep
- scripts/train.py now takes --model <name> (default cfg.runtime.model) and writes to models/<name>/
- scripts/watch.py and scripts/evaluate.py take --model <name> (subdir) and --checkpoint <sel> (best/seed<N>/path); old --models flag removed
- Added configs/lux.yaml and configs/solis.yaml (identical world: block; differ only in runtime.model; solis is a clone of lux for now)
- Added resolveModelPath test for nested model dir in tests/test_checkpoints.py

Validation:
- 98 unit tests passed
- Manual e2e load confirmed: models/lux/best.pt resolves correctly under configs/lux.yaml's world hash (stage full)

Impact:
- Scripts are now invoked with --model lux (or --model solis); --checkpoint replaces the old within-dir selector; the old --models flag no longer exists

Follow-up:
- Differentiate solis's reward/curriculum (its own landing style) — a separate change, logged in docs/reward-log.md when done.

Status:
- Complete

## ADD | 2026-06-13 01:50 UTC

Summary:
Milestone 5 — variable mass + thrust dynamics. Thrust is now a force (accel = force/mass, mass shrinks as fuel burns), with first-order engine spool-up, a minimum-throttle floor, and mass-scaled rotational inertia. The trained net lands the full difficulty at 100% under the new dynamics.

Reason:
First elevation of the simulation toward real-rocket behavior (spec docs/superpowers/specs/2026-06-13-variable-mass-thrust-design.md), foundation for later wind / mission-profile / recurrent-AI milestones.

Files:
- src/env/physics.py (force/mass/spool model; BoosterState.spool)
- src/env/spaces.py (OBS_DIM 8 -> 9; spool at index 8)
- src/config/loader.py + config.yaml (force/mass/spool world knobs; validation; glide curriculum rung; entCoef 0.02)
- src/agents/scripted.py (mass- and spool-aware PD pilot)
- scripts/{watch,play}.py (engine-spool HUD readout)
- tests/test_{physics,spaces,config_loader,episode}.py
- docs/{observations,reward-log}.md

Changes:
- Thrust force / mass dynamics; spool lag (throttleResponse) + minThrottle floor (0.3, tuned so a near-empty booster can still hover — NOT a hoverslam)
- spool added to state + observation; world hash changed (old models rejected with a clear error — verified)
- Two tuning invariants locked by tests: full-mass accel > g (liftoff) AND empty min-throttle accel < g (hover possible)
- Curriculum gained a 'glide' rung (30-40m) and entCoef 0.01->0.02 after the first run stalled (see below)

Validation:
- 95 unit tests passed under the force/mass/spool model
- PD pilot lands 100% on all 5 stages (solvability gate holds)
- First curriculum run STALLED on full at 0.17 (drop->full gap too large under the harder dynamics; entropy collapsed). Diagnosed and fixed with the glide rung + entCoef bump (docs/observations.md M5_CURRICULUM_GAP).
- Re-run: climbed all 5 stages (promotions iters 5/10/45/50), full -> 1.00 on entry and held. Held-out eval (seed 999, 200 eps, full): 100% success, mean impact 0.98 m/s, 262-step episodes. best.pt loads under shipped config.

Impact:
- The booster now behaves like a real (lightening, lagging) rocket and the AI masters it at full difficulty. Old M4 checkpoints are invalidated (retrain).

Follow-up:
- Next elevation (own milestone): wind / disturbance forces — the keystone that motivates the recurrent-policy work.

Status:
- Complete

## ADD | 2026-06-13 00:05 UTC

Summary:
Milestone 4 curriculum loop complete and validated end-to-end. A single training run now climbs touchdown -> hop -> drop -> full autonomously and produces a full-difficulty checkpoint that lands 100% of episodes. Fixed an entropy collapse on the hard stage by enabling an entropy bonus.

Reason:
Final spec milestone (5-of-5). The entCoef change was forced by evidence: with entCoef=0.0 the policy std collapsed 1.0 -> 0.14 on `full`, stalling at 0.17 then degrading to 0.00 (see docs/observations.md M4_ENTROPY_COLLAPSE).

Files:
- src/train/curriculum.py (trainCurriculum stage-promotion loop)
- scripts/train.py (curriculum by default; --stage for single-stage)
- config.yaml (entCoef 0.0 -> 0.01)
- README.md, docs/commands.md
- tests/test_curriculum.py
- docs/{observations,reward-log}.md

Changes:
- Promotion at eval iters when successRate >= curriculum.promoteAt; same policy/optimizer/anneal clock carried across stages (skill transfer)
- Saves best-by-success on the FINAL stage (or last promotion snapshot if the ladder never reaches it)
- evaluateFn injectable so promotion logic is unit-tested with scripted rates

Validation:
- 96 unit tests passed
- Acceptance run (220 iters, seed 0): promoted at iters 5/30/60, full -> 1.00
- Independent eval (seed 999, 200 episodes, full stage): 100% success, mean impact 0.38 m/s, 212-step episodes vs PdPilot 0.49 m/s / 317 steps
- best.pt worldHash matches shipped config (scripts.watch loads it)

Impact:
- The project is feature-complete: playable game + a trained net that lands full-difficulty drops at 100%.

Follow-up:
- Multi-seed run to confirm stability (acceptance used a single seed)
- Optional: tighten world knobs if the user wants a harder challenge

Status:
- Complete

## ADD | 2026-06-12 23:00 UTC

Summary:
Milestone 3 runtime: pygame-ce renderer (booster body, gimbal-deflected flame, pad, HUD), mode-free episode loop, hash-guarded checkpoint loading, and the play / watch / evaluate entry scripts.

Reason:
Spec milestone 4-of-5 — the human-playable game and the windows into trained behavior.

Files:
- src/runtime/{render,loop,evaluate}.py
- src/agents/checkpoints.py
- scripts/{play,watch,evaluate}.py
- tests/test_{render,runtime_loop,checkpoints,evaluate}.py

Changes:
- render.py presentation-only; pure worldToScreen/keysToControls unit-tested under SDL dummy driver; controls: W/S throttle ramp, A/D rotate nose (D -> gimbal -1 so "press right = rotate right")
- runEpisodeLoop: single action source, pause/step/speed/reset, autoReset
- loadCheckpoint enforces the M0 worldHash contract with a clear error

Validation:
- Unit tests passed (86 total)
- Headless smoke: play/watch/evaluate all ran under SDL_VIDEODRIVER=dummy; evaluate on hop: trained net 100% success, mean impact 0.46 m/s, 88-step episodes vs PdPilot 100% / 0.49 m/s / 201 steps (net lands 2.3x faster)

Impact:
- The game is playable; trained models are observable and scoreable.

Follow-up:
- M4 curriculum loop, then a full-difficulty training run.

Status:
- Complete

## ADD | 2026-06-12 22:20 UTC

Summary:
Milestone 2 training stack: MLPPolicy (tanh-squashed Gaussian actor-critic), VecLandingEnv, GAE rollouts, PPO update, CSV metrics, scripts/train.py. Two anti-exploit corrections found by systematic debugging: physics walls replace the oob terminal, and the curriculum gains a 'touchdown' first rung.

Reason:
Spec milestone 3-of-5. The corrections were forced by evidence: PPO discounted away every penalty by procrastinating (oob drift, then hover-to-fuel-death) — see docs/reward-log.md "baseline (rev 2)" and docs/observations.md M2_DISCOUNT_PROCRASTINATION_EXPLOITS.

Files:
- src/agents/mlp.py, src/train/{vec_env,rollout,ppo,loop}.py
- src/metrics/logger.py, scripts/train.py
- src/env/physics.py (walls), src/env/{episode,rewards}.py (oob removed)
- src/env/spaces.py (toEnvAction), config.yaml (touchdown stage)
- docs/superpowers/specs/... (amendments section)
- tests/test_{mlp,vec_env,rollout,ppo,loop}.py + updated env/reward tests

Changes:
- Net acts in tanh space; spaces.toEnvAction maps throttle affinely at the env boundary; PPO ratios live entirely in pre-squash space
- Both terminated and truncated are true MDP terminals (GAE bootstraps 0)
- trainLanding(stage=, policy=) supports curriculum chaining; saves best by eval success rate; linear shaping anneal via vecEnv.setShapingScale

Validation:
- 67 unit tests passed
- Ladder run: touchdown stage 0 -> 1.00 success by iter 10; continuing the same policy on hop: 0.75 -> 1.00 by iter 10 (15 iters each, seed 0/1)

Impact:
- Training is proven end-to-end; M4 curriculum loop can chain stages.

Follow-up:
- M3 runtime (render/play/watch/evaluate)
- M4 formalizes the stage-promotion loop validated by hand here

Status:
- Complete

## ADD | 2026-06-12 21:10 UTC

Summary:
Milestone 1 environment: pure gimbaled-booster physics, 8-D obs / 2-D action contract, LandingEnv with success/crash/oob/timeout classification, isolated config-driven rewards, Policy ABC, and a PD-pilot baseline that lands 100%/100%/99% on the hop/drop/full stages.

Reason:
Core simulation layer per the spec; the PD pilot is the solvability proof required before any RL training (M2).

Files:
- src/env/{physics,spaces,rewards,episode}.py
- src/agents/{policy,scripted}.py
- tests/test_{physics,spaces,rewards,episode,scripted}.py
- docs/reward-log.md (baseline preset entry)
- docs/superpowers/plans/2026-06-12-booster-milestone-1-environment.md

Changes:
- Semi-implicit Euler; thrust gimbal with throttle-scaled torque (sign: nozzle toward +x rotates nose toward -x); fuel gates thrust before burning
- Obs: [x, y, vx, vy, sin/cos theta, omega, fuel] with VEL_REF=20, OMEGA_REF=3 (code constants — part of the obs contract, NOT in the world hash)
- Reward: graded terminal + potential-based shaping (telescoping-tested) + control cost; timeout pays full terminalCrash (anti-hover exploit)
- Outcome precedence: touchdown beats oob beats timeout

Validation:
- Unit tests passed (38 passed total)
- PD pilot success-rate sweep recorded in docs/observations.md

Impact:
- M2 can train against LandingEnv; PdPilot is the eval baseline.

Follow-up:
- Default difficulty is generous (see M1_PD_PILOT_NEAR_PERFECT observation); revisit world knobs after first human play / trained-net watch.

Status:
- Complete

## ADD | 2026-06-12 20:30 UTC

Summary:
Milestone 0 scaffolding: project skeleton, config.yaml control panel, typed config loader with world-compatibility hash and fail-fast validation, docs system mirrored from tag-simulation.

Reason:
Foundation for the booster-landing project per the approved design spec (docs/superpowers/specs/2026-06-12-booster-landing-design.md). The config loader is the only module that reads YAML; the world hash is the compatibility boundary between saved models and the current physics.

Files:
- config.yaml
- src/config/loader.py
- tests/test_config_loader.py
- CLAUDE.md, pytest.ini, requirements.txt, .gitignore
- docs/{change-log,observations,reward-log,coding-conventions}.md
- docs/superpowers/plans/2026-06-12-booster-milestone-0-scaffolding.md

Changes:
- Frozen dataclasses (World/Reward/Training/CurriculumStage/Curriculum/Runtime/Config)
- computeWorldHash() over world fields only
- validateConfig(): positive physics values, maxThrust > gravity, mode enum, shapingAnneal enum, curriculum stage range sanity
- Curriculum spawn ranges live in curriculum.stages, NOT world (not hashed)

Validation:
- Unit tests passed (15 passed: parsing, defaults, tuples, stages, hash, validation)

Impact:
- All later milestones import their knobs from src/config/loader.py

Follow-up:
- M1 environment (physics/spaces/episode/rewards + scripted PD pilot)

Status:
- Complete
