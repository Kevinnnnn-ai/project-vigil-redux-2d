# Reward Log

Every reward experiment gets one entry, newest on top. Logging each reward version (with its structure and weights) is a hard rule. Each entry maps to a named `reward.preset` in `config.yaml` plus the exact knobs used, so it is reproducible.

## Entry format

```
## YYYY-MM-DD — preset: <name>  [tags e.g. [shaping] [terminal] [curriculum-stage]]
Hypothesis:
<the behavior change you expect>
Config:
<preset name + key reward numbers>
Result:
<landing success rate, MEAN over training.evalSeeds> + <what you saw on watch>
Verdict:
KEEP | REVERT | ITERATE — <next step>
```

---

# Entries

## 2026-06-25 — preset: showcase (m1–m6)  [showcase] [curriculum] [milestone-replay]

Hypothesis:
Past models can be re-shown in a single fixed world by retraining each documented milestone's reward+curriculum configuration on today's world (hash f5c82b420d2a6ebc). Reward changes (shapingAnneal: linear vs none) never invalidate the world hash or checkpoint compatibility — historical model diversity was world/curriculum variation, not reward. The kit reconstructs 6 milestones as standalone configs so any past configuration can be run without editing config.yaml.

Config:
Two reproducible reward variants across 6 milestone configs in tmp/configs/: Variant A — shapingAnneal: linear (m1–m4, early curriculum stages); Variant B — shapingAnneal: none (m5–m6, post-convergence-fix). All configs share world hash f5c82b420d2a6ebc. Run mapping 7001–7006 written to tmp/showcase/registry.json at train time by train_all.py.

Result:
PENDING — training launched by the user via python tmp/showcase/train_all.py. These are "failure-OK" reproductions of past configurations on the current world, not new reward designs; exact success rates TBD.

Verdict:
ITERATE — record per-milestone convergence outcomes (success rates, promoted stages, entropy) in tmp/showcase/REGISTRY.md as each run completes.

## 2026-06-23 — preset: baseline  [shaping] [anneal] [curriculum]

Hypothesis:
Keeping PBRS shaping fully ON for the whole run (shapingAnneal: none -> constant scale 1.0) restores a dense learning signal on the late-reached hard stages (glide/full), fixing the SUICIDE1_NONCONVERGENCE non-convergence WITHOUT changing the optimum — PBRS is policy-invariant (Ng et al. 1999) and telescopes to -Phi(s0), so constant shaping cannot distort the optimal policy or be reward-hacked.
Config:
Reward ARITHMETIC unchanged; the only knob change is reward.shapingAnneal: linear -> none. All else identical to run-2: terminalSuccess 1.0, terminalCrash -1.0, gentlenessBonus 0.5, centeringBonus 0.5, shapingCoef 1.0, controlCost 0.01; training entCoef 0.02, totalIters 600; narrowed full stage.
Result:
PENDING — run-3 (600 iters x 3 seeds) not yet launched. Unit-validated: shapingScaleFor is constant 1.0 under the shipped config (pytest 152 passed). A short isolated end-to-end smoke under shapingAnneal: none is part of the pre-launch validation (plan Task 2).
Verdict:
ITERATE — launch run-3 and record convergence here (seeds reaching full + sustained eval success >= promoteAt, entropy bounded). If full still does not converge, escalate one variable at a time per spec §7 (entCoef/logStd, then promotion hysteresis).

## 2026-06-22 — preset: baseline (UNCHANGED)  [suicide-burn-world] [cut-gate] [no-math-change]

Hypothesis:
The cut-before-touchdown SUCCESS gate (isCutOff = engine commanded OFF at the first contact step) requires no new reward term — it rides the existing crash payout. A landing that fails the cut gate is classified 'crash' and pays terminalCrash, which is already the signal for an unacceptable outcome. The gate is purely a success predicate in episode.py, not a reward edit.
Config:
baseline reward UNCHANGED (src/env/rewards.py untouched; math identical to all prior entries). The gate is isCutOff = not _engineOnAtTouchdown in episode.py. isCutOff is the fourth success condition alongside isUpright, isOnPad, isGentle.
Result:
NOT YET TRAINED to a result under the single-burn world with the cut gate. The env is solvable (PdPilot coasts, ignites once, cuts before contact; passes the gate). Full PPO run outstanding — see ROADMAP.md.
Verdict:
ITERATE — keep the reward math; the open item is a full ppo-trainer run (300 iters x 3 seeds) under the single suicide-burn world. Record the result here when done.

## 2026-06-16 — preset: baseline (UNCHANGED)  [pymunk] [physics-model] [no-math-change]

Hypothesis:
Swapping the simulator to the Pymunk rigid-body engine (real leg-ground collision; outcome emerges from the solver) needs no reward change — the same graded terminal + PBRS + control cost still rewards a soft, centered, upright landing, since "comes to rest upright on the pad after a gentle impact" is exactly what the terms favor.
Config:
baseline reward UNCHANGED (src/env/rewards.py untouched; math identical). What changed is the PHYSICS MODEL behind the outcome, not the reward: success/crash/timeout are now classified from the Pymunk resting state instead of a scripted pivot verdict. impactSpeed still sourced from the pre-contact approach velocity (prevState).
Result:
NOT YET TRAINED to a result under the Pymunk dynamics. PdPilot baseline lands ~100% on touchdown and full, so env + reward remain solvable. The rl-reviewer confirmed PBRS (1-done), single gamma, and GAE bootstrap are intact under the new env.
Verdict:
ITERATE — keep the reward math; the open item is a full ppo-trainer run (curriculum/training, NOT reward) on the new dynamics. Re-run scripts.train and, if a stage plateaus at 0.00 with decaying entropy, add a finer rung + entCoef >= 0.02 before recording a trained result.

## 2026-06-15 — preset: baseline (UNCHANGED)  [terminal-timing] [physical-legs] [no-math-change]

Hypothesis:
Making the legs physical (a settling phase that pivots about the planted toe; outcome emerges from simulated dynamics) does not require any reward-math change — the SAME graded terminal + potential shaping + control cost still rewards a soft, centered, upright landing, because the new success basin (low speed + low spin + small tilt at contact) is exactly what the existing terms favor.
Config:
baseline reward UNCHANGED (terminalSuccess 1.0, terminalCrash -1.0, gentlenessBonus 0.5, centeringBonus 0.5, shapingCoef 1.0, controlCost 0.01). What changed is TIMING, not arithmetic: the terminal outcome (success/crash) is now paid at the RESTING verdict (end of the settling phase) instead of at the base-crossing instant. Per-step shaping + control cost run THROUGH the settling steps (engine off -> control cost ~0; Phi keeps telescoping until the terminal step zeroes it via (1-done)). The immediate-crash gate (fast/off-pad toe-plant) still pays the terminal crash at the contact step.
Result:
NOT YET TRAINED to a result under the new dynamics. A short reduced run stalled on hop (under-resourced + likely a curriculum gap — see observations LEG_SETTLING_CURRICULUM_GAP_UNVERIFIED). PdPilot baseline lands 30/30 on touchdown and full, so the env + reward remain solvable.
Verdict:
ITERATE — keep the reward math; the open item is curriculum/training (ppo-trainer), NOT reward. Re-run scripts.train on the shipping config and, if a stage plateaus at 0.00 with decaying entropy, add a finer rung + entCoef >= 0.02 before recording a trained result here.

## 2026-06-13 — preset: baseline (M5 dynamics)  [variable-mass] [spool] [curriculum]

Hypothesis:
The unchanged reward design (graded terminal + potential shaping + control cost) still produces optimal landings under the harder M5 force/mass/spool dynamics, given a finer curriculum and a bit more exploration. Reward arithmetic is identical; only the world (physics) and curriculum changed, so this entry tracks whether the design survives the dynamics change.
Config:
baseline reward unchanged. World swapped to the force/mass/spool model (maxThrustForce 30, dryMass 1.0, fuelMass 0.6, minThrottle 0.3, throttleResponse 4.0). Curriculum: added a 'glide' rung (30-40m) between drop and full; entCoef 0.01 -> 0.02; totalIters 260.
Result:
First attempt (old 4-stage curriculum, entCoef 0.01) STALLED: full stuck at 0.17, policy scored 0.00 for ~100 iters post-promotion while entropy collapsed +1.86 -> -0.28 (zero landing reward -> deterministic crashing; see observations M5_CURRICULUM_GAP). Second attempt (glide rung + entCoef 0.02): climbed touchdown/hop/drop/glide/full (promotions iters 5/10/45/50), full hit 1.00 ON ENTRY (iter 55) and held it through iter 259; entropy stayed +2.3..+4.1, never collapsed. Held-out eval (seed 999, 200 episodes) on FULL: 100% success, mean impact 0.98 m/s, 262-step episodes (vs PdPilot 0.49 m/s, 323 steps — the net lands faster but harder, both within the 2.0 m/s limit). best.pt is a full-difficulty 100% lander under variable-mass dynamics.
Verdict:
KEEP — the reward design needs no change for harder dynamics; the curriculum spacing does. Lesson: curriculum gap traversability is coupled to dynamics difficulty (a gap fine under easy physics can be untraversable under hard physics); fix with a finer rung + more exploration, not reward edits.

## 2026-06-12 — preset: baseline (rev 2)  [anti-exploit] [world-walls] [curriculum]

Hypothesis:
Two structural changes close the discount-procrastination loopholes PPO found in rev 1, without touching the reward arithmetic itself: (a) physics walls (sides + ceiling clamp, zero into-wall velocity) replace the oob terminal entirely; (b) a 'touchdown' first curriculum rung (y 1-3 m, centered, slow) makes the success basin samplable by exploration.
Config:
baseline unchanged except: oobPenalty knob removed (oob outcome no longer exists). Curriculum gains stage 0 'touchdown'.
Result:
EVIDENCE TRAIL (15-iter runs, numEnvs 8, rolloutSteps 1024, seed 0): rev 1 as designed: policy flew sideways out of bounds (30/30 oob, tilt ~-1.1 rad) — oob -1 at step ~88 discounts to -0.41, beating a ground crash -0.75 at step ~29. Adding oobPenalty -2 only slowed the exit (28/30 oob at step ~165, -2*0.99^165 = -0.38): ANY finite oob penalty is discounted away by leaving slowly enough. With walls: policy hovered ~390 steps to fuel death (crash discounts to -0.02; discounted hover control cost only ~-0.35) — success (+1.5 at ~step 60 ~ +1.1 discounted) dominates but was NEVER SAMPLED from 8-12 m spawns. With the touchdown rung: success 0 -> 1.00 by iter 10, then continuing the same policy on hop: 0.75 -> 1.00 by iter 10. FULL-CURRICULUM ACCEPTANCE (220 iters, entCoef 0.01, seed 0): climbed touchdown->hop->drop->full (promotions iters 5/30/60), full reached 1.00. Held-out eval (seed 999, 200 episodes) on the FULL stage: 100% success, mean impact 0.38 m/s, 212-step episodes — gentler and faster than PdPilot (0.49 m/s, 317 steps). best.pt is a full-difficulty 100% lander.
Verdict:
KEEP — walls + samplable first rung are load-bearing, and the reward design produces optimal landings once exploration is preserved (entCoef 0.01; see observations M4_ENTROPY_COLLAPSE). Lesson: when every reachable terminal is negative, gamma<1 makes procrastination optimal; fix by making success reachable, not by tuning penalties.

## 2026-06-12 — preset: baseline  [terminal] [shaping] [control-cost]

Hypothesis:
Graded terminal payouts (success bonuses for gentleness/centering, crash scaled by impact severity) plus potential-based shaping Phi = -(dist/ceiling + speed/VEL_REF + |theta|/pi) give PPO a dense enough signal to land from the easy curriculum stages without distorting the optimal policy (shaping is policy-invariant).
Config:
baseline — terminalSuccess 1.0, terminalCrash -1.0, gentlenessBonus 0.5, centeringBonus 0.5, shapingCoef 1.0 (linear anneal), controlCost 0.01. DESIGN DECISION: timeout pays the full flat terminalCrash — without it, hovering until the clock runs out strictly dominates a risky landing attempt and the policy stalls.
Result:
(pending — first training runs land in M2)
Verdict:
ITERATE — initial implementation, no training evidence yet.
