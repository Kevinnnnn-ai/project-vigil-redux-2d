# Roadmap

Near-term direction and -- just as important -- what is **intentionally not built yet**, so an
agent builds on-plan and does not mistake a deliberate gap for a bug. This is rough direction,
not a dated schedule. See [AGENTS.md](AGENTS.md) for the docs overview. When something here
ships, move the "why" to [CHANGELOG.md](CHANGELOG.md) (and, for rewards, [REWARD_LOG.md](REWARD_LOG.md))
and drop or check it off here.

## Done -- the stack is trained-capable end to end

- **Physics**: Pymunk (Chipmunk2D) rigid-body sim with the hull + two legs as one body that
  physically collides a static ground; sub-stepped 4x for rigid hard impacts; variable-mass
  fuel/spool dynamics; **binary suicide-burn engine** (ignite once, cut once, lock). Outcome
  (land/settle/topple) **emerges from the solver** -- no scripted verdict.
- **Env & reward**: `LandingEnv` with success/crash/timeout classified from the resting state;
  graded terminal payouts + potential-based shaping + control cost (reward math unchanged across
  every physics rev). 10-D obs contract. **Cut-before-touchdown gate** added to the success
  predicate (`isCutOff` in `episode.py`).
- **Training**: hand-written PPO (64x64 tanh-squashed actor-critic) + GAE; CPU-default device
  (GPU ~2.8x slower here); parallel per-seed training; run-numbered checkpoints and metrics;
  live-updating convergence PNG.
- **Curriculum**: a 5-rung ladder (`touchdown -> hop -> drop -> glide -> full`) with promotion on
  eval success rate; the glide rung + `entCoef 0.02` resolved the M5 entropy-collapse stall.
- **Runtime**: pygame viewer (`watch`), human pilot (`play`), headless eval vs `PdPilot`
  (`evaluate`); hash-guarded checkpoint loading; run-numbered checkpoint layout.

## Next -- validate PPO training under the single suicide-burn world (open ITERATE)

The reward design is settled; the **open item is a training/curriculum pass** under the
single-burn constraint. The cut-before-touchdown gate tightens the success basin -- PdPilot
is a weak binary baseline (~40% on touchdown), and it is unknown whether PPO will transfer:

- Run a full `scripts.train` (300 iters x 3 seeds) under the current `config.yaml`. Watch the
  curriculum signatures (`M5_CURRICULUM_GAP_VS_DYNAMICS_DIFFICULTY` in [OBSERVATIONS.md](OBSERVATIONS.md)):
  a stage that plateaus at 0.00 for many iters right after a promotion = gap too large;
  monotonically decaying entropy = exploration too weak.
- If a stage plateaus at 0.00 with decaying entropy: **add a finer rung before that stage AND
  raise `entCoef >= 0.02`** -- do not just add iterations to a collapsed policy.
- Once a full run validates, record the trained result in [REWARD_LOG.md](REWARD_LOG.md) and
  update the ITERATE entries to KEEP.

## Then -- landing-task variants (from `docs/personal/ideas-to-expand.md`)

Build on the validated stack. Roughly increasing scope:

- **Soft landing** -- simply land on the ground (the current task, simplest framing).
- **Targeted soft landing** -- land on a specific area (a reward/curriculum extension: tighten the
  centering term / narrow the pad).
- **Hover slam** -- re-introduce a `minThrottle` field (currently removed) set high enough
  that empty min-throttle accel *exceeds* gravity (`M5_FORCE_MODEL_TUNING_CONSTRAINT`),
  rewrite the PD pilot's descent strategy, and retrain. This is a deliberate future physics
  change (adds a previously-removed field back), not a reward edit.

Further out (not started): wind / disturbance forces (motivates a recurrent policy), higher
difficulty tiers, multi-phase mission profiles.

## Not yet built (deliberate)

- **No full single-burn training result yet** -- the env/reward/physics are shipped and solvable
  (PdPilot passes the cut gate), but a full PPO run under the suicide-burn world with the
  cut-before-touchdown gate has not been recorded (see Next). Do not read PdPilot's ~40% as a
  PPO result.
- **Dormant hashed world fields** -- `settleStepCap`, `maxLandingTilt`, `maxLandingOmega`,
  `settleTime` are still in the hash (to avoid re-hashing) but are no longer read by outcome
  classification; the last two survive only as HUD text in `scripts/play.py`. Leave them unless
  you intend to re-hash and retrain.
- **Dormant test-only / fallback code** -- `stepPhysics` and `boosterCoM` are test shims; the GPU
  wiring (`resolveDevice`, `device: auto`) is a kept-but-unused fallback. Do not delete without a
  reason.
- **No wind, mission profiles, or recurrent policy** -- explicitly future work, not a gap.

If you are tempted to start a variant before the single-burn training pass lands, do the training
re-validation first -- an untrained baseline is a shaky foundation to extend from. Green tests
before either.
