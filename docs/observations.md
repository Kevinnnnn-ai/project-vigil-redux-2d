# Observations

Durable training/behavior **signatures** discovered while running this project — the
"why it behaved that way" log, distinct from the CHANGELOG (what changed) and the
REWARD_LOG (reward design history). Each entry is a named signature: symptom, root
cause (verified in source/data), what is NOT the cause, and the fix (or proposed fix).
CHANGELOG entries may reference a signature by name.

> Re-established 2026-06-23 (the upstream `observations.md` was not carried into the
> single-suicide-burn tree). Historical signatures `M0`–`M5` (e.g.
> `M2_DISCOUNT_PROCRASTINATION_EXPLOITS`, `M5_CURRICULUM_GAP_VS_DYNAMICS_DIFFICULTY`,
> `CPU_BEATS_GPU_FOR_THIS_PPO`) live in `CHANGELOG.md` history.

---

## SUICIDE1_NONCONVERGENCE — the agent never learns `full`, and trains itself *worse*

**Discovered:** 2026-06-23, from `run-1` (300 iters / ~9.8M steps) and `run-2`
(600 iters / ~19.7M steps), 3 seeds each (`stdout/logs/run-{1,2}/seed{0,1,2}.csv`,
gitignored). Diagnosed + adversarially verified (two refutation passes).

### Symptom
- **No seed in either run ever reaches `promoteAt` (0.8) on the `full` stage** (the real task).
- `run-2`: only 2/3 seeds even *reach* `full`; deterministic eval peaks **0.70 (seed1) / 0.625 (seed2)** — single noisy evals, not stable solving — then **decay** (seed1 → 0.225). `seed0` never escapes `glide` in 369 iters.
- The convergence plot's wild swing is **within a single stage**, not curriculum-transition dips: the **deterministic mean** policy (`act` = squashed mean, `src/agents/mlp.py:80-94`) thrashes 0.00↔~0.78. (seed0 `glide`: 17 swings ≥0.3 over 74 evals; seed1 `full`: 14/57.) So it is genuine policy degradation, **not** eval *sampling* noise.

### Root cause — three coupled mechanisms (all source-verified)
1. **Reward goes sparse exactly on the hard stages.** `reward.shapingAnneal: linear`
   anneals shaping `1.0 → ~0` **globally over `training.totalIters`**
   (`shapingScaleFor`, `src/train/loop.py:44-45` → `1.0 - it/totalIters`). Because the
   curriculum reaches `glide`/`full` *late*, those stages always run in the low-shaping
   tail → reward is effectively terminal-only ±1 (the cut-before-touchdown gate just
   rides `terminalCrash`; no dense signal for "cut before touchdown"). Advantages
   collapse: **`policyLoss ≈ 0.001` on `full`**. `Pearson(shapingScale, entropy) = −0.83` in `run-2`.
2. **Unopposed entropy bonus inflates the policy.** `logStd` is a free
   state-independent `nn.Parameter` (`src/agents/mlp.py:57`); the PPO loss is
   `… − entCoef·entropy` (`src/train/ppo.py:69`, `entCoef=0.02`). With the reward
   gradient gone (mech. 1), the entropy term is the only consistent force on `logStd`
   → **σ inflates ~1.0 → ~3.0** (entropy 2.84 → 5.0+; entropy 2.838 ↔ σ=1.0 for the
   2-D Gaussian). The greedy policy drifts/degrades. The **critic stays healthy**
   (`explainedVariance` 0.83–0.92) → the failure is **policy-side, not value-side**.
   This is the *opposite* of the historical `M5` entropy-**collapse**.
3. **Promotion fires on noise.** `src/train/curriculum.py:119` promotes on a **single**
   40-episode eval ≥ `promoteAt`, no averaging/hysteresis. The eval is a point estimate
   over a *fresh* random spawn set each call (`evalRng` advances), so it is a noisy
   estimator. Stages advance on transient spikes (e.g. `run-2` seed1: `drop` promoted
   at **0.950** vs stage mean **0.32**), carrying under-trained policies into harder
   stages and compounding mechs. 1–2.

### What is NOT the cause
- **Training budget.** `run-1 → run-2` doubled steps (9.8M → 19.7M) *and* narrowed
  `full` (alt `[40,52]→[52,52]`, xOff `[-14,14]→[-5,5]`); convergence did not improve
  and `seed0` made **zero** progress (stuck in `glide` in both runs). More steps = more
  time under a dead reward gradient → **do not just train longer**.
- **The critic.** `explainedVariance` is healthy throughout.

### Fix — proposed (not yet implemented; brainstorm/plan before actioning)
- **Keep a dense gradient on hard stages:** floor `shapingScale` (don't anneal to 0),
  and/or anneal **per-stage** rather than globally over `totalIters`; or add explicit
  shaping for engine-cut-before-touchdown so `full` isn't terminal-only.
- **Constrain exploration:** anneal/lower `entCoef` and/or cap `logStd` so σ can't
  inflate to ~3 once the reward gradient weakens.
- **Robust promotion:** require **N consecutive** evals (or a moving average / more
  episodes) ≥ `promoteAt` so the ladder stops advancing on spikes.

### Reproduce the analysis
Per-seed CSVs `stdout/logs/run-{1,2}/seed{0,1,2}.csv` (gitignored). The signature is
re-derivable from `config.yaml` + `src/train/{loop,ppo,curriculum}.py` even without the
artifacts. Columns: `policyLoss,valueLoss,entropy,approxKl,clipFrac,explainedVariance,rolloutSuccess,iter,stage,successRate,promoted` (`successRate=-1.0` off eval iters).

**Status:** diagnosed + verified 2026-06-23. Fix pending.
