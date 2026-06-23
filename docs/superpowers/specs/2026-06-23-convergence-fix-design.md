# Convergence Fix (run-3) — Design Spec

- **Date:** 2026-06-23
- **Status:** Design (awaiting review) → implementation plan
- **Author:** agent (brainstormed with user)

## 1. Summary

The post-rewire training runs do not converge: across `run-1` (300 iters / 3 seeds) and
`run-2` (600 iters / 3 seeds, narrowed `full`), **no seed ever sustains success on the real
`full` stage**, and the deterministic policy *degrades* on the hard stages. Full diagnosis:
`docs/observations.md` → **`SUICIDE1_NONCONVERGENCE`**.

The upstream root cause is **reward sparsity on the hard stages**: `reward.shapingAnneal:
linear` anneals the potential-based (PBRS) shaping `1.0 → ~0` **globally over
`training.totalIters`** (`shapingScaleFor`, `src/train/loop.py:44-45`). Because the
suicide-burn cut-gate slows curriculum progression, `glide`/`full` are reached *late* — in the
near-zero-shaping tail — so they train with only the sparse terminal ±1 signal. Advantages
collapse (`policyLoss ≈ 0.001` on `full`), and the constant `entCoef=0.02` entropy bonus then
inflates the policy's `logStd` (σ ~1.0 → ~3.0), so the mean policy never settles.

This spec applies the **minimal, root-cause, one-variable fix**: stop annealing the PBRS
shaping. It is theoretically free — PBRS is policy-invariant (Ng et al. 1999) and telescopes to
`−Φ(s₀)` (un-hackable; guarded by `test_shapingTelescopesToInitialPotential`), so constant
shaping cannot change the optimum or be farmed; it only restores the dense learning signal.

## 2. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Fix scope | **Minimal root-cause change only** — `shapingAnneal: none`. Do NOT also change `entCoef` or promotion logic in this pass (keeps run-3 a clean one-variable delta from run-2 and matches the REWARD_LOG one-variable discipline). |
| 2 | How to stop annealing | **Config value `none`** (already supported: `shapingScaleFor` returns constant `1.0` for non-`linear`). No code change to the anneal function; the `linear` branch is retained for future use. |
| 3 | Everything else | **Held constant vs run-2**: `totalIters: 600`, narrowed `full` stage, `entCoef: 0.02`, single-eval promotion — unchanged. |
| 4 | In-session depth | **Implement + cheap-validate** (unit suite + a constant-shaping assertion + a short isolated smoke run). The full `run-3` (600 iters × 3 seeds) is launched by the user. |
| 5 | Validation hygiene | Smoke artifacts go to an **isolated `--run`/temp location** and are **cleaned up**; `run-1`/`run-2` artifacts are never touched (`stdout/` rule). |

## 3. The change

- **`config.yaml`** (reward block, currently `config.yaml:41`): `shapingAnneal: linear` →
  **`shapingAnneal: none`**.
- That is the entire behavioral change. Data flow it affects: `scripts/train.py` /
  `src/train/{loop,curriculum}.py` call `shapingScaleFor(cfg, it, totalIters)` each iter and
  pass the result as `shapingScale` into `computeReward` (`src/env/rewards.py:97-102`), where it
  multiplies **only** the PBRS term. With `none`, `shapingScale ≡ 1.0` for every iter and stage.

## 4. Validation plan (in-session, cheap)

1. **Enum check** — `none` is already an accepted `reward.shapingAnneal` value
   (`validateConfig`, `src/config/loader.py:276` allows `linear|none`; dataclass default
   `linear` at `loader.py:121`), so this is config-only with no loader change. Confirm the real
   `config.yaml` still loads after the edit.
2. **Unit suite** — `python -m pytest -q` green. Update any test that pins the shipped config's
   `shapingAnneal == 'linear'` (candidate: `tests/test_config_loader.py`) to `none`. Do not
   weaken `tests/test_loop.py` coverage of the `linear` branch (the function keeps both paths).
3. **Constant-shaping assertion** — under the loaded config, `shapingScaleFor(cfg, it, 600) ==
   1.0` for representative `it` (0, mid, last).
4. **Short smoke** — a tiny config (1 seed, ~8 iters, small `numEnvs`/`evalEpisodes`) trains
   end-to-end cleanly with shaping constant; artifacts to an isolated `--run`/temp dir, removed
   after. Confirms no crash and healthy loss/entropy at small scale.

## 5. Required documentation (hard rules)

- **`docs/REWARD_LOG.md`** — new entry (newest on top): preset `baseline`, tags
  `[shaping] [anneal] [curriculum]`; Hypothesis (constant PBRS restores the dense signal on the
  late-reached hard stages → convergence); Config (`shapingAnneal: none`, all else identical to
  run-2); Result (pending run-3); Verdict `ITERATE`.
- **`docs/CHANGELOG.md`** — new `CONFIG` entry (the change + why, referencing
  `SUICIDE1_NONCONVERGENCE`).
- **`docs/observations.md`** — update `SUICIDE1_NONCONVERGENCE` **Status** → "fix in progress:
  `shapingAnneal: none`, run-3 pending".
- **`.claude/agent-memory/{decisions,notes}.md`** — log the decision + a pointer.

## 6. Acceptance criteria (judged on the user's run-3)

- Most/all 3 seeds **reach `full`** and **sustain** eval success ≥ `promoteAt` (0.8) — judged on
  a last-N-evals mean, **not** a single spike.
- **Entropy stays bounded** on the hard stages (M5 healthy range was ~+2.3..+4.1; no σ inflation
  toward ~3).
- `stdout/convergence-plots/run-3.png` **converges-and-holds** rather than oscillating 0↔~0.8.
- Optional, matching the M5 bar: held-out eval (`seed 999`, 200 eps) on `full` ≥ ~80–90%.

## 7. Contingency — next one-variable levers (only if run-3 still fails)

In order, each its own change + REWARD_LOG/CHANGELOG entry:
1. **Exploration** — anneal/lower `entCoef` and/or cap `logStd` so σ cannot inflate once the
   gradient weakens.
2. **Promotion robustness** — require **N consecutive** evals (or a moving average / more
   episodes) ≥ `promoteAt` to stop promoting on noise (`src/train/curriculum.py:119`).
3. **Last resort** — add dense shaping for the cut-before-touchdown behavior (a genuine reward
   term; larger change, REWARD_LOG-gated).

## 8. Out of scope

- Any `entCoef` / `logStd` / promotion change (deferred to §7 contingencies).
- Reward-math edits (terminal payouts, control cost, the potential function `Φ`).
- World/physics changes (no world-hash impact; checkpoints unaffected).
- Running the full `run-3` in-session (user-launched).
