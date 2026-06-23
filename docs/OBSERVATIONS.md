# Observations

Non-obvious findings, gotchas, and open discrepancies discovered while working in this repo —
the things that aren't visible from a quick read of the code and would otherwise be re-learned
the hard way. A shared memory layer: read it before non-trivial analysis or debugging, and
record anything surprising as you find it. Mark an entry **RESOLVED** (don't delete) when it no
longer holds. Newest at top; entries are immutable except to correct factual errors. See
[AGENTS.md](AGENTS.md) for the docs overview.

Format: `## <CONTEXT_ID> | YYYY-MM-DD HH:MM UTC` followed by `Observation` / `Context` /
`Evidence` (optional) / `Implication` / `Recommendation` (optional). Log only what has future
reuse value — not generic statements, one-off trivia, or speculation without evidence.

---

## PYMUNK_RIGIDBODY_PHYSICS | 2026-06-16 02:00 UTC

Observation:
The physics is now the Pymunk (Chipmunk2D) rigid-body engine, NOT a hand-written
integrator. src/env/physics.py BoosterSim owns a persistent pymunk.Space; the
booster hull + 2 legs are ONE rigid body that physically collides with a static
ground segment (+ walls/ceiling). Landing/settling/tip-over EMERGE from the
solver's contact forces — stepPivot/settleVerdict/tipOverAtTouchdown and the whole
scripted-pivot model are DELETED (this supersedes PHYSICAL_LEG_SETTLING_REPLACED_
INSTANT_VERDICT below, which described that now-removed model). episode.py only
classifies the outcome from the physical resting state.

Context:
2026-06-16 FEATURE. User wanted real leg-ground collision ("that's why we have it
in Pygame"). Box2D (the alternative, has native CCD) would NOT build on this
Python 3.14 box (no wheel; needs MSVC + SWIG; failed every way). Pymunk installs
from a prebuilt wheel.

Evidence / reusable facts:
- CROSS-PROCESS DETERMINISM HOLDS: two fresh 'spawn' processes running the same
  seed+actions produce BIT-IDENTICAL trajectories (verified in the rl-reviewer
  pass). So the parallel-seed-training invariant (serial==parallel per-seed
  values) survives the engine swap. Do NOT assume a C physics engine breaks
  determinism here — it was checked and it doesn't.
- engine/spool/fuel logic is preserved byte-for-byte from the old stepPhysics;
  only MOTION integration moved to Pymunk. Force scaling verified within 0.09%
  of the analytical F/m*dt (not 4x / not 1/4x despite sub-stepping).
- coordinate map: pymunk body.angle = -theta, omega_repo = -body.angular_velocity,
  body.position = CoM = base + body_up*bodyHalfLen. Documented @TAG[angle-map].
- render fidelity: drawn leg toes + hull match the collidable Pymunk shapes to
  0.000 mm at theta=0 AND under tilt, because the toe is single-sourced via
  legToes and _bodyPolygon now reads world.bodyHalfLen (no hardcoded literal).

Implication:
- pymunk is a hard dependency (requirements.txt). pytest fails on a fresh clone
  until `pip install -r requirements.txt`.
- computeWorldHash folds in PHYSICS_MODEL_VERSION ('pymunk-2') so a physics-MODEL
  change invalidates checkpoints even with unchanged world: fields. Bump it on any
  future dynamics-altering model change.
- settleStepCap / maxLandingTilt / maxLandingOmega / settleTime are now DORMANT
  hashed fields (kept to avoid re-hashing again).

Recommendation:
- See PYMUNK_NO_CCD_SUBSTEP below for the tunneling/rigidity gotcha.
- A full ppo-trainer run on the new dynamics is still OUTSTANDING — PPO is tuned
  for the old physics; PdPilot landing != PPO will. Watch for M5_CURRICULUM_GAP.

## PYMUNK_NO_CCD_SUBSTEP | 2026-06-16 02:00 UTC

Observation:
Pymunk has NO continuous collision detection (CCD), so a fast-moving toe sinks
deep into the ground in a single dt before the contact solver reacts, then gets
shoved back out over several frames — the visible "rubbery / not-rigid" feel on
hard crashes. FIX: advance the solver in _SUBSTEPS (=4) sub-ticks of dt/_SUBSTEPS
per env step (forces RE-APPLIED each sub-tick because Pymunk zeroes body.force
after every space.step; drag recomputed from live velocity; spool/fuel updated
once per env step). Also stiffened: space.iterations=20, collision_slop=0.01.

Context:
The user reported buggy "not-rigid" hard crashes in play-testing. MEASURED: a
15 m/s slam penetrated the ground ~0.76 m at 1 substep; at 4 substeps it
penetrates ~0.000 m and stops rigidly. The body was already rigid (hull+legs on
one Body) — the issue was penetration, not flex.

Implication:
- Sub-stepping CHANGED the dynamics (bumped PHYSICS_MODEL_VERSION pymunk-1 ->
  pymunk-2). The stiffer contact also damps spin more, so the omega needed to
  topple a gentle contact rose (~5 -> ~8 rad/s in the tests).
- Residual limit: above ~20 m/s a toe can still tunnel for a sub-tick. In practice
  spawn velocities cap at -12 m/s, well below. If you raise spawn speeds or dt,
  re-check tunneling (raise _SUBSTEPS) — do NOT expect Pymunk to catch fast
  contacts without sub-stepping.

Recommendation:
- To tune crash feel, adjust _SUBSTEPS / _SOLVER_ITERATIONS / _COLLISION_SLOP in
  src/env/physics.py (all named constants). Do NOT switch engines for tunneling
  alone — sub-stepping closes most of the gap and Box2D won't install here anyway.

## PHYSICAL_LEG_SETTLING_REPLACED_INSTANT_VERDICT | 2026-06-15 23:30 UTC

Observation:
Touchdown is now a TWO-PHASE process and the stand/topple outcome is SIMULATED,
not predicted. Flight ends when the LOWEST leg toe (legToes, src/env/physics.py)
reaches y<=0 — NOT the body base. A gentle, on-pad contact then enters a SETTLING
phase: stepPivot pins that toe and rotates the booster about it (gravity torque +
carried spin, engine off, position reconstructed from the pin constraint) until
settleVerdict resolves. This REPLACED tipOverAtTouchdown and the
|theta+omega*settleTime| inequality (both removed). settleTime is now a dormant
hashed field. Supersedes LEG_FOOTPRINT_REPLACED_FLAT_TILT_CAP (that footprint
PREDICTION no longer exists).

Context:
2026-06-15 FEATURE (physical leg-ground contact). New hashed world fields legDrop
(0.9) + settleStepCap (120); world re-hashed (lux 5fe29b437768b405 ->
6578a7cb3d13d4c9). src/env/physics.py stepPivot + legToes + boosterCoM;
src/env/episode.py settleVerdict + lowestToe + two-phase step().

Evidence:
- A booster pinned on ONE toe SELF-RIGHTS for small tilt and only topples past
  atan2(legSpan, bodyHalfLen + legDrop) ~= 0.322 rad. Note the lever uses
  bodyHalfLen + legDrop (the toe sits legDrop BELOW the base), so this is
  STRICTER than the retired prediction's atan2(legSpan, bodyHalfLen) ~= 0.46 rad.
- settleVerdict order is load-bearing: TOPPLE (CoM outboard of the pinned toe) is
  checked BEFORE the second-toe STAND, because an outward topple LIFTS the other
  toe (otherToe.y > 0), so it can never be misread as a second-toe success.
- REGRESSION found in implementation: a near-upright booster on one toe rotated
  its FREE toe straight through the ground (otherToe.y going negative and
  sinking) and wrapped to theta ~= -2.9 -> crash, breaking
  test_pdPilotLandsThroughTheLoop. ROOT CAUSE: the single-toe pivot had no
  second-toe contact, so nothing stopped the inward rotation. FIX: settleVerdict
  returns 'success' the moment the non-pinned toe reaches y<=0 (rocked onto both
  legs). PdPilot then lands 30/30 on touchdown and full.

Implication:
The settling success basin is "low horizontal speed + low spin + small tilt at
contact" — exactly the soft/centered/upright landing the reward already targets.
The model pins only ONE toe and lets angular drag + the second-toe-contact stop
it; it does NOT do full two-contact rigid-body restitution (no bounce). The
settleStepCap (120 steps) is a safety net for a booster that never damps, NOT the
primary path. impactSpeed is still read from prevState (the approach velocity) —
the FLOOR_CLAMP_EATS_IMPACT_SPEED coupling still holds.

Recommendation:
To make landings tippier, NARROW legSpan or RAISE legDrop/bodyHalfLen (the topple
threshold is atan2(legSpan, bodyHalfLen + legDrop)). Do NOT reintroduce
tipOverAtTouchdown. If you ever want the policy to fight a tip with thrust during
settling, you must (a) keep the engine live in the settling branch and (b) add a
contact channel to the obs (currently OBS_DIM stays 10 — contact is NOT observed
because the engine is off and the policy has no authority once settling).

## LEG_SETTLING_CURRICULUM_GAP_UNVERIFIED | 2026-06-15 23:30 UTC

Observation:
After the physical-settling change, a SHORT reduced curriculum run (40 iters, 1
seed, numEnvs 8, rolloutSteps 1024) STALLED on the hop stage: 0.00 success for
~35 iters post-promotion while entropy decayed monotonically 2.84 -> 1.59 and EV
stayed high (0.90+). Probing the resulting policy: 40/40 crashes and 0 episodes
reached the legs (no toe contact) — it failed the flight APPROACH, not settling.
This is the M5_CURRICULUM_GAP_VS_DYNAMICS_DIFFICULTY signature.

Context:
Task 7 validation of the leg-contact feature. The run was deliberately
under-resourced (shipping config is 300 iters / 3 seeds / 16 envs / 2048 steps),
so a curriculum gap and undertraining are CONFOUNDED — this run alone cannot tell
them apart. The env IS solvable: PdPilot lands 30/30 on touchdown and full, and
the touchdown stage trains to 0.80 immediately.

Implication:
A full-length ppo-trainer pass on the shipping config is required before claiming
RL trainability under the new dynamics. NOT YET DONE — the implementation agent
reported this rather than re-tuning (curriculum/training is a ppo-trainer
decision). The harder settling bar plausibly opened a real gap.

Recommendation:
Run scripts.train on configs/lux/baseline.yaml (300x3) and watch the
post-promotion success/entropy trend. If hop (or any stage) plateaus at 0.00 with
decaying entropy, add an intermediate curriculum rung before that stage AND raise
entCoef to >= 0.02 (the documented M5 fix) — do NOT just add iterations to a
collapsed policy. Re-run before updating docs/reward-log.md with a result.

## CPU_BEATS_GPU_FOR_THIS_PPO | 2026-06-15 18:00 UTC

Observation:
CPU training is ~2.8x FASTER than GPU for this stack, not slower-but-close — MEASURED, not
assumed. Benchmark (configs/lux/baseline.yaml, numEnvs 16, rolloutSteps 2048, hidden 64x64,
15 timed iters after a warm-up, on a machine with torch 2.11.0+cu128 and CUDA available):
CPU 10.6 s/iter vs CUDA 29.3 s/iter (2.77x). Root cause: the workload is launch-bound, not
compute-bound. The 64x64 MLP is trivial arithmetic, and VecLandingEnv.step (src/train/vec_env.py)
steps its sub-envs in a plain Python for-loop over numpy LandingEnvs — so each of the 2048
rollout ticks does tiny GPU inference -> .cpu().numpy() -> CPU physics -> back to GPU. The
per-launch / host<->device transfer overhead dominates and starves the GPU.

Context:
User asked "is GPU slower than CPU here?". Confirmed with a throwaway scripts/bench_device.py
(deleted after use) timing the real collect->GAE->update cycle on each device. Resolution:
set training.device: cpu in configs/{lux,solis}/baseline.yaml. The GPU wiring (src/train/device.py,
resolveDevice) is KEPT as a dormant, tested fallback — set device back to 'auto' to re-enable
GPU without code changes.

Implication:
Do NOT re-benchmark or "try GPU" expecting a speedup unless the architecture changes in a way
that makes it compute-bound: a much larger net (wide/deep), much larger batches, OR a genuinely
batched/vectorized env that steps all sub-envs in one tensor op (removing the Python step loop
and the per-tick CPU<->GPU round-trips). Under the current Python-loop vec_env, GPU will keep
losing regardless of GPU model. Device never enters computeWorldHash, so CPU- and GPU-trained
checkpoints are interchangeable — switching device does not invalidate models.

Recommendation:
Train on CPU (the configs now default to it). If someone proposes GPU, point them here first.

## FLOOR_CLAMP_EATS_IMPACT_SPEED | 2026-06-15 12:00 UTC

Observation:
The physical-ground floor clamp (stepPhysics @TAG[ground-floor]: y>=0, downward vy->0 at
contact) DESTROYS the touchdown velocity signal if the success/crash verdict reads speed off
the post-step state. After the clamp, state.vy at any touchdown is ~0, so EVERY landing looks
"gentle" and even a 12 m/s slam classifies as success. The fix: read impact speed from the
APPROACH velocity (prevState, the velocity the instant before contact), not the clamped
resting state. episode.step() now computes impactSpeed = hypot(prevState.vx, prevState.vy)
and passes it into _classify(); tilt/omega for the tip-over test are NOT clamped so they are
still read off the post-step state.

Context:
Adding a physical ground + leg tip-over check (2026-06-15 change-log FEATURE). Caught by
test_fastTouchdownIsCrash flipping to 'success' the moment the clamp landed in stepPhysics.

Implication:
Any inelastic position clamp that zeroes a velocity component and any downstream logic that
reads that component are COUPLED through ordering. If a future change adds restitution/bounce
or moves the clamp, re-verify impact-speed classification. The regression guard is
test_fastTouchdownStillCrashDespiteFloorClampingVelocity (tests/test_episode.py).

Recommendation:
Treat the floor clamp as a rendering/resting guarantee only; never derive an impact metric
from a post-clamp velocity. Keep impact speed sourced from prevState.

## LEG_FOOTPRINT_REPLACED_FLAT_TILT_CAP | 2026-06-15 12:00 UTC

Observation:
The leg-footprint tip-over test (tipOverAtTouchdown) REPLACED the flat maxLandingTilt /
maxLandingOmega success gates — it is now the SINGLE attitude/spin stability criterion.
Topples iff |theta + omega*settleTime| >= atan2(legSpan, bodyHalfLen). With defaults
(legSpan 0.9, bodyHalfLen 1.8) the tip angle is ~0.46 rad (~27 deg), LOOSER than the retired
0.15 rad flat cap. maxLandingTilt / maxLandingOmega are NOT removed (removing them would
re-hash the world again) — they remain as HUD reference strings in scripts/play.py only and
no longer affect any outcome.

Context:
2026-06-15 FEATURE. legSpan/bodyHalfLen/settleTime were added as HASHED world fields
(geometry == physics), so the lux/solis world hashes changed and all checkpoints (none exist
yet) would need retraining.

Implication:
Keeping both the flat caps AND the footprint test as gates makes the footprint test INERT at
default legs, because 0.15 rad is far stricter than 0.46 rad — the stricter gate always wins.
The design choice was to let the legs be the real criterion, so the flat caps were dropped
from _classify. If a future agent wants a stricter tip angle, NARROW legSpan rather than
re-adding maxLandingTilt to the gate.

Recommendation:
The tip-over angle is a pure function of legSpan/bodyHalfLen; tune it there. Tests parameterize
off the world fields (tests/test_episode.py _thetaCrit), so changing legSpan does not break
them. Do not reintroduce the flat-cap gate unless you also intend the footprint test to be
inert.

## FIXED_SPAWN_VIA_DEGENERATE_RANGE | 2026-06-14 00:00 UTC

Observation:
A deterministic ("fixed") spawn value needs no env code change — collapse the curriculum
stage's uniform range to a single point (e.g. altitude [52.0, 52.0]). LandingEnv.reset()
draws every field via rng.uniform(*range), and uniform(a, a) always returns a. The config
validator permits equal endpoints (its inversion check is lo > hi, not lo >= hi).

Context:
configs/lux/baseline.yaml `full` stage was set to a fixed drop height and a narrowed
xOffset (2026-06-14 change-log entry). src/env/episode.py:59-72, src/config/loader.py:223.

Implication:
Per-field fixed spawns are a pure config lever. Do NOT special-case reset() with hardcoded
constants — it breaks the curriculum-stage abstraction. Editing spawn ranges (curriculum)
does NOT touch the world hash, so checkpoints stay loadable; only world: edits invalidate.

Recommendation:
For "always drop from X" requests, prefer a degenerate range over code. If a horizontal
range must stay strictly on the pad, keep |x| <= world.padWidth/2; widening padWidth to fit
a wider range changes the world hash and invalidates existing models.

## MODEL_ENV_SUBDIR_IS_ORGANIZATIONAL | 2026-06-15 03:04 UTC

Observation:
The models/<model>/<env>/ subdir level (added 2026-06-15) is ORGANIZATIONAL, not a
compatibility axis. Two envs that share an identical world: block share a worldHash and
their checkpoints remain mutually loadable; envs that differ in world: get distinct hashes
and the existing loadCheckpoint guard rejects cross-loading. Do NOT add hash logic keyed on
the <env> path — computeWorldHash() already covers compatibility.

Context:
The per-environment-model-subdirs change nested models one level deeper
(models/<model>/<env>/, configs/<model>/<env>.yaml) behind a new --env flag (default
'baseline') on train/watch/evaluate. src/agents/checkpoints.py was unchanged — the resolver
is simply handed the deeper path.

Evidence:
configs/lux/baseline.yaml and configs/solis/baseline.yaml now hash to DIFFERENT worldHashes
(lux 5fe29b437768b405, solis 416d12b7bc318784) because engineMode is a hashed world field —
this supersedes the older MODEL_DIRS_LUX_SOLIS observation that claimed an identical world:
block. evaluate --model lux --env baseline loads models/lux/baseline/best.pt cleanly;
omitting --env resolves identically (argparse default 'baseline').

Implication:
The "mixed" case is intended: some envs differ only in reward/training (same hash, mutually
loadable), others differ in physics (distinct hash, guard rejects cross-load). The directory
level carries no compatibility meaning on its own.

Recommendation:
When adding an environment, create configs/<model>/<env>.yaml and train into
models/<model>/<env>/; always pair --model/--env with the matching --config. Never rely on
the <env> path for compatibility — let the worldHash guard do that.

## SUICIDE_BURN_WORLD | 2026-06-13 12:59 UTC

Observation:
engineMode is a HASHED world field, so analog (Lux) and suicideBurn (Solis) are distinct
worlds — checkpoints are NOT interchangeable (M0 guard enforces this). OBS_DIM grew 9->10,
so EVERY prior checkpoint (including the original models/lux/best.pt) is invalid and must
be retrained.

Context:
The suicide-burn foundations work: world.engineMode added to configs; suicideBurn branch
added to stepPhysics; BoosterState gained engineTransitions and engineCommandedOn; obs index
9 (ignitionsRemaining) added to spaces.py.

Evidence:
lux and solis world hashes differ (test_luxAndSolisShipWithDifferentWorldHash asserts they
are NOT equal, having been flipped from the prior equality assertion); a cutoff-from-full
regression test confirms the engine reaches spool 0 and stays there; 112 tests pass.

Implication:
The suicideBurn engine state is a LATCHED INTENT (BoosterState.engineCommandedOn), NOT
spool-derived. Deriving "on" from spool>0 caused a cutoff to relight during spool decay —
this bug was found in code review and fixed by latching intent. A transition is counted at
COMMAND time; the engine locks permanently after 2 transitions; minThrottle/throttleCutoff
are bypassed in suicideBurn.

Recommendation:
When extending suicideBurn, never re-derive engine on/off from spool; always use the latched
engineCommandedOn field. Keep engineMode in computeWorldHash() so the two models can never
cross-load checkpoints.

## MODEL_DIRS_LUX_SOLIS | 2026-06-13 11:32 UTC

Observation:
lux and solis intentionally share an IDENTICAL world: block, so their checkpoints
are cross-compatible. The two landing styles diverge via reward/curriculum, NOT
physics. Solis is currently a clone of lux; differentiation is deferred to a
separate change.

Context:
The named-model-dirs change (configs/lux.yaml + configs/solis.yaml, models/<name>/)
introduced two named model configs. Both derive from the same world settings as
config.yaml, so computeWorldHash() returns the same value across all three files.

Evidence:
computeWorldHash() is equal across config.yaml, configs/lux.yaml, and
configs/solis.yaml. models/lux/best.pt loads correctly under configs/lux.yaml's
world hash (stage full, verified manually).

Implication:
A future agent must keep the world: blocks identical across model configs, or
checkpoints stop being interchangeable. If the world: block ever diverges between
lux and solis, each model will need its own independently trained checkpoints and
loadCheckpoint's hash guard will enforce this automatically.

Recommendation:
Note the deliberate CLI rename: --model now means the model NAME/subdir (e.g.
lux, solis), and --checkpoint is the within-dir selector (best, seed<N>, or an
explicit path). Do not expect --model best to work anymore — that was the old
interface. When differentiating solis's reward/curriculum, touch only reward:
and curriculum: in configs/solis.yaml; keep world: in sync with configs/lux.yaml.

## M5_CURRICULUM_GAP_VS_DYNAMICS_DIFFICULTY | 2026-06-13 01:40 UTC

Observation:
A curriculum gap that transfers fine under easy dynamics can become untraversable
when the dynamics get harder. The M4 curriculum (drop 20-30m -> full 40-52m,
entCoef 0.01) trained full to 1.00 under constant-mass/instant-throttle physics.
Under the M5 force/spool/variable-mass dynamics the SAME curriculum stalled full
at 0.17: the policy promoted from drop at 93% but scored 0.00 for ~100 iters on
full while entropy decayed +1.86 -> -0.28, then never recovered.

Context:
M5 first vs second validation run (runs/curriculum_acceptance.yaml). Fix:
inserted a 'glide' rung (30-40m) between drop and full, and raised entCoef
0.01 -> 0.02. Result: full hit 1.00 on entry (iter 55) and held it; entropy
stayed +2.3 to +4.1 (never collapsed). Held-out eval (seed 999, 200 eps) on
full: 100% success.

Evidence:
The tell was 100 consecutive iters of 0.00 success with monotonically decaying
entropy — zero landing reward means the policy optimizes only shaping +
control-cost and converges to deterministic crashing. Once it stops landing
even in stochastic rollouts, there is no signal to recover from.

Implication:
Curriculum spacing is coupled to dynamics difficulty, not absolute. When the
physics get harder (this milestone, and future wind/mission-profile ones), the
ladder may need finer rungs AND more exploration, even if the old spacing worked.
The two failure signatures to watch: (1) a stage that scores 0.00 for many iters
right after a promotion = gap too large; (2) decaying entropy = exploration too
weak. They often co-occur and the fixes stack.

Recommendation:
When adding difficulty (wind, harder profiles), re-run the curriculum and watch
for a post-promotion 0.00 plateau. If seen, add an intermediate rung before
reaching for more iterations (more steps on a collapsed policy do nothing — see
M4_ENTROPY_COLLAPSE). Keep entCoef >= 0.02 under the M5 dynamics.

## M5_FORCE_MODEL_TUNING_CONSTRAINT | 2026-06-13 01:00 UTC

Observation:
Under the variable-mass force model, the "hovering stays possible" guarantee is
a tuning constraint between three knobs: minThrottle, maxThrustForce, dryMass.
A near-empty booster's minimum thrust acceleration is
minThrottle*maxThrustForce/dryMass — this must stay BELOW gravity or the booster
is forced upward at min throttle (a hoverslam, which we deliberately did not
want yet). With maxThrustForce 30 and dryMass 1.0, minThrottle 0.4 gives 12.0 >
9.8 (forced hoverslam — wrong); minThrottle 0.3 gives 9.0 < 9.8 (hover possible
— correct). Defaults ship at 0.3.

Context:
Caught by test_emptyBoosterCanStillHover during M5 implementation. The
test_liftoffCapableValidation (full-mass accel > g) and this hover test together
pin the safe region.

Implication:
Any future change to maxThrustForce, dryMass, fuelMass, or minThrottle must keep
both invariants: full-mass accel > g (can lift off) AND empty min-throttle accel
< g (can still hover). The two tests guard this; do not weaken them.

Recommendation:
To make the sim a hoverslam challenge later, deliberately set minThrottle high
enough that empty min-throttle accel EXCEEDS g, then flip the hover test to
assert the opposite and rewrite the PD pilot's strategy (it currently assumes it
can regulate descent rate, which a hoverslam forbids).

## M4_ENTROPY_COLLAPSE_ON_HARD_STAGE | 2026-06-12 23:40 UTC

Observation:
With entCoef=0.0 (inherited from tag-simulation), the curriculum climbed
touchdown->hop->drop autonomously but stalled on the `full` stage at 0.17
success, then DEGRADED to 0.00 while differential entropy fell from +2.84 to
-1.07. Recovered policy std: 1.0 -> 0.142 — the Gaussian collapsed and the
policy went nearly deterministic before it had learned the hardest spawns.

Context:
M4 acceptance run, 120 iters, runs/curriculum_acceptance.yaml seed 0. The
promotion mechanism itself worked perfectly (promoted at iters 5/20/50).

Evidence:
EV stayed high (0.85-0.99) throughout — the CRITIC was fine; the failure was
exploration, not value estimation. success peaked iter ~85 (0.17) then
0.13/0.00/0.03 — classic premature convergence, not undertraining.

Implication:
tag-simulation's entCoef=0.0 relied on self-play to keep policies lively;
a single-agent task has no such pressure. Exploration-hard final stages need
a positive entropy bonus or the std collapses into a local optimum.

Recommendation:
Keep entCoef >= 0.01 in config.yaml. If a stage still stalls, suspect
exploration (watch the entropy trend) BEFORE adding more iterations — more
steps on a collapsed policy do nothing. A high EV with falling success is the
signature.

## M2_DISCOUNT_PROCRASTINATION_EXPLOITS | 2026-06-12 22:10 UTC

Observation:
Under gamma=0.99 with all reachable terminals negative, PPO reliably learns to
DELAY termination rather than to land: first by drifting out of the side
boundary slowly (oob terminal), then — after walls were added — by hovering
until fuel death (~390 steps). A terminal penalty of ANY finite size is
neutralized by gamma^T for large T, so penalty tuning cannot fix this class.

Context:
M2 hop-stage validation runs (15 iters, numEnvs 8, rolloutSteps 1024); behavior
probed by rolling the trained policy deterministically and logging touchdown
state. See docs/reward-log.md 'baseline (rev 2)' for the full evidence trail.

Implication:
(1) The world must be inescapable (walls clamp; no oob outcome) — structural,
not a penalty. (2) The first curriculum stage must be easy enough that random
exploration samples 'success' outcomes; gradients alone never find the success
basin from altitude. Both are now load-bearing for any training run.

Recommendation:
Never reintroduce an out-of-bounds terminal or remove the 'touchdown' rung
without re-running the ladder validation. If training ever flatlines at 0
success again, FIRST probe what the deterministic policy does at episode end
(outcome/impact/steps/fuel table) before touching hyperparameters.

## M1_PD_PILOT_NEAR_PERFECT | 2026-06-12 21:05 UTC

Observation:
The scripted PdPilot lands at 100%/100%/99% success over 100 seeded episodes on
the hop/drop/full curriculum stages respectively (seed 0, default config.yaml).

Context:
Measured after implementing src/agents/scripted.py against src/env/episode.py,
via a 100-episode sweep per stage.

Evidence:
hop {success: 100}, drop {success: 100}, full {success: 99, oob: 1}.

Implication:
The default world tuning (maxThrust 15 vs gravity 9.8, fuelBurnRate 0.08,
padWidth 8) makes the task generously solvable — good for verifying the RL
pipeline, but PPO reaching ~100% on `full` should NOT be read as impressive.
The curriculum may even be unnecessary at this difficulty.

Recommendation:
If the game feels too easy (human play or trained nets), tighten world knobs
(narrower padWidth, lower maxThrust, higher fuelBurnRate, wider spawn ranges)
rather than touching reward. Re-run tests/test_scripted.py after any such edit —
its thresholds (0.8 hop / 0.4 drop) are the solvability contract.
