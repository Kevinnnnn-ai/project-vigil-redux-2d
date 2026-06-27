<div align="center">

# Project Vigil Redux 2D

A 2D reinforcement-learning sandbox where a from-scratch PPO agent learns to fly and land a single-stage, suicide-burn rocket booster inside a Pymunk rigid-body physics simulation.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A52.1-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pymunk](https://img.shields.io/badge/Pymunk-%E2%89%A57.0-2C8EBB?style=for-the-badge)
![pygame-ce](https://img.shields.io/badge/pygame--ce-%E2%89%A52.5-2A9D8F?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%E2%89%A53.8-11557C?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-local-informational?style=for-the-badge)

The name evokes a watchful waiting—a *vigil*—for the one moment that matters: the booster falls under gravity, and the agent must hold its single engine ignition until the last possible second, the whole descent rendered in two dimensions.

</div>

---

## Ⅰ • Table of Contents

- [Ⅱ • Features](#ⅱ--features)
- [Ⅲ • Demonstration](#ⅲ--demonstration)
- [Ⅳ • Quick Start](#ⅳ--quick-start)
- [Ⅴ • Installation](#ⅴ--installation)
- [Ⅵ • Usage](#ⅵ--usage)
- [Ⅶ • Configuration](#ⅶ--configuration)
- [Ⅷ • Reference](#ⅷ--reference)
- [Ⅸ • License](#ⅸ--license)
- [Ⅹ • Authors](#ⅹ--authors)
- [Ⅺ • Contact](#ⅺ--contact)

<br>

## Ⅱ • Features

- **From-scratch PPO** — a hand-written clipped-surrogate PPO update with hand-rolled Generalized Advantage Estimation, built on only `torch` and `numpy` (no stable-baselines, no gymnasium).
- **Pymunk rigid-body physics** — the booster is one rigid body (a hull plus two legs) in a Chipmunk2D solver; thrust, gimbal torque, fuel-coupled mass, and explicit drag integrate every step.
- **Binary suicide-burn engine** — continuous throttle is thresholded to on/off and latched to at most two transitions (one ignite, one cut); the engine fires at full or not at all.
- **Emergent landing verdict** — touchdown, settle-to-rest, and the four-gate success test (upright, on-pad, gentle, engine-cut-before-touchdown) are read from solver state, never scripted.
- **Automatic curriculum** — five spawn stages (`touchdown → hop → drop → glide → full`) auto-promote once the eval success rate clears `promoteAt`, carrying the policy and optimizer forward across rungs.
- **Potential-based reward shaping** — graded terminal payouts, policy-invariant PBRS shaping (Ng et al. 1999), and a quadratic control cost, all tuned through one `baseline` preset.
- **Parallel multi-seed training** — every eval seed trains in its own OS process and is judged across seeds; device selection is GPU-primary with an automatic CPU fallback.
- **World-hash checkpoints** — a 16-hex SHA-256 of the `world:` block plus a `suicide-1` physics tag stamps each checkpoint, so loading a model against a changed world fails fast.
- **Watch, play, and evaluate** — a pygame-ce renderer to watch a trained model or fly by keyboard, a headless net-versus-`PdPilot` (a deliberately weak PD baseline) harness, and live convergence plots over numbered runs.

<br>

## Ⅲ • Demonstration

Training (`python -m scripts.train`) spawns one process per seed; each worker streams a per-eval line and stage promotions (interleaved across seeds), while the driver prints a startup banner, a per-seed result, and a cross-seed summary:

```text
run-1: training 3 seed(s) [0, 1, 2] with 3 worker(s)
  checkpoints -> checkpoints\run-1\   metrics -> stdout\logs\run-1\   live plot -> stdout\convergence-plots\run-1.png
[curriculum seed 0]      device: cpu
[touchdown seed 1]       iter    0  success 0.07  rollout 0.15  EV -0.00  entropy 2.844
[curriculum seed 1]      iter   30  PROMOTED -> stage hop (rate 0.82)
[drop seed 0]            iter  900  success 0.44  rollout 0.39  EV +0.33  entropy 2.061
[seed 1] best success 0.69 -> checkpoints\run-1\seed1.pt

curriculum->full: success across seeds (0, 1, 2) = 0.57 +/- 0.14  (min 0.38)
best.pt <- checkpoints\run-1\seed1.pt (0.69)  [checkpoints\run-1\]
convergence plot -> stdout\convergence-plots\run-1.png
```

Evaluating (`python -m scripts.evaluate`) prints the trained net beside the scripted `PdPilot` baseline:

```text
evaluate: stage touchdown, 100 episodes, seed 0
  best.pt      success 76.00%   crash: 20  success: 76  timeout: 4
               impact mean 1.18 m/s   episode mean 96 steps
  PdPilot      success 41.00%   crash: 55  success: 41  timeout: 4
               impact mean 1.74 m/s   episode mean 88 steps
```

Each seed also streams metrics to `stdout/logs/run-N/seed<seed>.csv`, and `stdout/convergence-plots/run-N.png` overlays each seed's eval success rate against cumulative environment steps. The figures above are illustrative of the output format—exact values vary by run, seed, and checkpoint, and the shipped defaults are not benchmarked to convergence on the `full` stage.

<br>

## Ⅳ • Quick Start

```powershell
# 1. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the booster (auto-curriculum, parallel seeds) -> checkpoints\run-N\
python -m scripts.train

# 4. Watch the best trained model land
python -m scripts.watch
```

**Note** — `python -m scripts.train` is long-running (default `totalIters: 1400` per stage, across every seed in `evalSeeds`) and auto-creates a fresh `checkpoints\run-N\` each invocation. For a fast sanity check, train a single easy stage instead: `python -m scripts.train --stage touchdown`. Device selection is GPU-primary with an automatic CPU fallback (`training.device: auto`); set `device: cpu` in `config.yaml` to force CPU.

<br>

## Ⅴ • Installation

### Requirements

- **Python 3.14** (developed on 3.14.5)

### Dependencies

Pinned in [requirements.txt](requirements.txt):

| Library | Version | Role |
|---------|---------|------|
| `pyyaml` | `>= 6.0` | Parsing `config.yaml` |
| `pytest` | `>= 8.0` | Test suite |
| `numpy` | `>= 2.1` | Observation and rollout arrays, GAE math |
| `torch` | `>= 2.6` | Tensors, autograd, the from-scratch PPO network and update |
| `pygame-ce` | `>= 2.5` | Rendering for `watch` and `play` |
| `pymunk` | `>= 7.0` | Chipmunk2D rigid-body physics (the booster simulation) |
| `matplotlib` | `>= 3.8` | Convergence plots |

### Steps

```powershell
# 1. Clone the repository and move into it
git clone https://github.com/Kevinnnnn-ai/project-vigil-redux-2d.git
cd project-vigil-redux-2d

# 2. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 3. Install all required dependencies
pip install -r requirements.txt
```

<br>

## Ⅵ • Usage

All commands run from the **repository root** with the project virtual environment active. Behavior is chosen by **which `scripts/*` module you run**—`config.yaml`'s `mode:` key is validated but never dispatches.

### Train

Runs the full auto-curriculum across `evalSeeds`, one process per seed:

```powershell
python -m scripts.train
```

- `--stage hop` trains a single stage (no promotion); `--run N` forces the run number; `--serial` runs seeds sequentially; `--config PATH` selects a config file.

### Watch

Renders a trained model (or the scripted PD pilot) on the `full` stage:

```powershell
python -m scripts.watch                 # best checkpoint of the latest run
python -m scripts.watch --pilot pd      # scripted PdPilot, no checkpoint
python -m scripts.watch --checkpoint seed1 --run 3
```

- In-window keys: `space` pause, `n` step, `r` reset, `-`/`=` speed, `esc`/`q` quit.

### Play

Fly the booster manually:

```powershell
python -m scripts.play                  # full stage
python -m scripts.play --stage hop      # easier spawns
```

- In-window keys: `w`/`s` throttle, `a`/`d` rotate, `space` pause, `r` reset, `esc`/`q` quit.

### Evaluate

Compares the trained net against the `PdPilot` baseline, headless:

```powershell
python -m scripts.evaluate --checkpoint seed2 --stage drop --episodes 200
```

### Run the tests

```powershell
python -m pytest -q
```

<br>

## Ⅶ • Configuration

Every behavior and training knob lives in [config.yaml](config.yaml). Editing any field under `world:` changes the world hash and **invalidates existing checkpoints**; editing `reward:`, `training:`, or `curriculum:` does not.

### `world:` — physics (hashed compatibility group)

| Constant | Default | Meaning |
|----------|---------|---------|
| `width` | `40.0` | Arena width; side walls clamp at ±`width`/2 |
| `ceiling` | `60.0` | Top clamp height |
| `padWidth` | `8.0` | Landing-pad width, centered at `x = 0` |
| `gravity` | `9.8` | Downward acceleration |
| `dryMass` / `fuelMass` | `1.0` / `0.6` | Empty mass and full-tank fuel mass (full = 1.6) |
| `maxThrustForce` | `30.0` | Peak engine force at full spool |
| `maxGimbal` | `0.35` | Max nozzle deflection (rad) |
| `throttleResponse` / `gimbalResponse` | `4.0` / `4.0` | First-order spool and nozzle-slew lag rates |
| `dt` / `maxSteps` | `0.05` / `600` | Physics timestep and 30 s episode cap |
| `maxLandingSpeed` | `2.0` | Gentle-touchdown speed gate (m/s) |
| `legSpan` / `bodyHalfLen` | `0.9` / `1.8` | Leg footprint and body half-length that, with `legDrop`, set the upright/tip-over angle |

### `reward:` — payouts and shaping

| Constant | Default | Meaning |
|----------|---------|---------|
| `preset` | `baseline` | Documentation label (no code branches on it) |
| `terminalSuccess` | `1.0` | Reward for a successful landing |
| `terminalCrash` | `-1.0` | Base penalty for a crash or timeout |
| `gentlenessBonus` | `0.5` | Scaled by the margin under the landing-speed limit |
| `centeringBonus` | `0.5` | Scaled by closeness to pad center |
| `shapingCoef` | `1.0` | PBRS shaping weight |
| `shapingAnneal` | `none` | `none` keeps shaping full-scale; `linear` fades it to ~0 over the run |
| `controlCost` | `0.01` | Quadratic penalty on throttle and gimbal |

### `training:` — PPO and runs

| Constant | Default | Meaning |
|----------|---------|---------|
| `lr` | `3.0e-4` | Adam learning rate |
| `gamma` / `gaeLambda` | `0.99` / `0.95` | Discount and GAE lambda |
| `clipEps` | `0.2` | PPO clip range |
| `epochs` / `minibatchSize` | `10` / `64` | PPO update epochs and minibatch size |
| `rolloutSteps` | `2048` | Steps per env per iteration |
| `entCoef` | `0.02` | Entropy bonus (sustains exploration) |
| `vfCoef` / `maxGradNorm` | `0.5` / `0.5` | Value-loss weight and gradient clip |
| `numEnvs` | `16` | Parallel vectorized envs |
| `evalSeeds` | `[0, 1, 2]` | Seeds trained and judged per run |
| `evalEpisodes` / `evalEvery` | `40` / `5` | Eval episodes and iterations between evals |
| `totalIters` | `1400` | Iterations per stage |
| `hidden` | `[64, 64]` | MLP hidden-layer sizes |
| `seedWorkers` | `auto` | Processes for concurrent per-seed training |
| `device` | `auto`* | GPU when available, else CPU; `cpu` forces the fallback (*dataclass default — not written in `config.yaml`) |

### `curriculum:`, `mode:`, and `runtime:`

| Constant | Default | Meaning |
|----------|---------|---------|
| `promoteAt` | `0.8` | Eval success rate required to advance a stage |
| `stages` | `touchdown → hop → drop → glide → full` | Ascending spawn difficulty; `full` is the real task |
| `mode` | `train` | Validated only—does **not** dispatch; pick a `scripts/*` module instead |
| `watchModel` | `best` | Default checkpoint selector for `watch` and `evaluate` |
| `evaluateEpisodes` | `100` | Episodes for `scripts.evaluate` |

<br>

## Ⅷ • Reference

### Project layout

```text
project-vigil-redux-2d/
├─ config.yaml                  # the control panel: every behavior/training knob
├─ requirements.txt
├─ scripts/                     # entry points (run as: python -m scripts.<name>)
│  ├─ train.py                  # parallel per-seed training driver
│  ├─ watch.py                  # render a trained model (or the PD pilot)
│  ├─ play.py                   # manual keyboard control
│  ├─ evaluate.py               # net vs PdPilot success comparison
│  └─ live_convergence.py       # standalone live convergence plot
├─ src/
│  ├─ config/loader.py          # frozen-dataclass config, validation, world hash
│  ├─ env/
│  │  ├─ physics.py             # Pymunk BoosterSim: rigid-body dynamics, suicide-burn engine
│  │  ├─ episode.py             # LandingEnv: step loop, touchdown/settle/success gates
│  │  ├─ rewards.py             # graded terminal payouts + PBRS shaping + control cost
│  │  └─ spaces.py              # 11-D observation encoder, 2-D action mapping
│  ├─ agents/
│  │  ├─ mlp.py                 # MLPPolicy actor-critic ((64, 64) tanh, tanh-squashed Gaussian)
│  │  ├─ policy.py              # Policy interface
│  │  ├─ scripted.py            # PdPilot weak PD baseline
│  │  └─ checkpoints.py         # save/load, worldHash guard, selector resolution
│  ├─ train/
│  │  ├─ ppo.py                 # hand-written clipped PPO update
│  │  ├─ rollout.py             # GAE advantage computation
│  │  ├─ loop.py                # single-stage trainLanding + shaping-anneal schedule
│  │  ├─ curriculum.py          # auto-advancing stage ladder (trainCurriculum)
│  │  ├─ vec_env.py             # vectorized LandingEnv batch
│  │  ├─ parallel.py            # per-seed ProcessPoolExecutor
│  │  └─ device.py              # torch device selection (auto/cpu)
│  ├─ runtime/
│  │  ├─ loop.py                # runEpisodeLoop (watch/play)
│  │  ├─ render.py              # pygame-ce Renderer + HUD
│  │  └─ evaluate.py            # runEvaluation
│  └─ metrics/
│     ├─ logger.py              # CsvLogger (per-seed metrics CSV)
│     ├─ live.py                # run-number resolution, artifact paths, live render
│     └─ plot.py                # convergence plot (success vs env steps)
├─ checkpoints/                 # run-N/{seed<seed>.pt, best.pt}; run-N/ subdirs gitignored
├─ stdout/
│  ├─ logs/run-N/               # per-seed training metrics CSVs
│  ├─ convergence-plots/        # run-N.png (updates live during training)
│  └─ console-logs/             # manual transcript captures
├─ docs/                        # CHANGELOG, OBSERVATIONS, REWARD_LOG (+ superpowers specs/plans)
├─ tests/                       # pytest suite (173 tests)
└─ tmp/                         # showcase kit: reconstructs documented milestones m1–m6
   ├─ configs/                  # m1..m6 standalone --config snapshots (reserved run band 7001–7006)
   └─ showcase/                 # gen_configs.py, train_all.py, gallery.py, milestones.py
```

### Key entry points

- **`python -m scripts.train`** — resolves the next run number, then trains each seed (`src/train/parallel.py`) through the auto-curriculum (`trainCurriculum`), or a single stage (`trainLanding`) when `--stage` is passed.
- **`python -m scripts.watch` / `play`** — drive `runEpisodeLoop` (`src/runtime/loop.py`) with the `pygame-ce` `Renderer`; the action source is a loaded checkpoint or human input.
- **`python -m scripts.evaluate`** — `runEvaluation` (`src/runtime/evaluate.py`) scores the net against `PdPilot`, headless.
- **`config.yaml`** — `loadConfig` (`src/config/loader.py`) parses it into frozen dataclasses, validates fail-fast, and derives `computeWorldHash()` for checkpoint compatibility.

### How it works at a glance

1. `config.yaml` defines the world, reward, training, and curriculum; the loader validates it and hashes the `world:` block (plus a `suicide-1` physics-model tag) into a 16-hex digest.
2. `scripts.train` launches one process per seed; each runs PPO over 16 vectorized Pymunk environments.
3. The agent sees an 11-D observation and emits a 2-D continuous action `[throttle, gimbal]`; reward = graded terminal payout + PBRS shaping + control cost.
4. The curriculum advances `touchdown → … → full` once eval success ≥ `promoteAt`; the best-by-success checkpoint per seed is saved, and the strongest seed is copied to `best.pt`.
5. `watch`, `play`, and `evaluate` load a checkpoint (guarded by its world hash) on the `full` stage.

### External

- [Pymunk](https://www.pymunk.org/) — the Chipmunk2D Python physics library behind the booster simulation.
- [PyTorch](https://pytorch.org/) — tensors and autograd for the from-scratch PPO.
- [pygame-ce](https://pyga.me/) — the rendering backend for `watch` and `play`.
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — the PPO algorithm implemented here.
- Policy invariance under reward transformations (Ng, Harada & Russell, 1999) — the basis for the potential-based shaping.

<br>

## Ⅸ • License

Released under the [MIT License](LICENSE). You are free to use, modify, and distribute this software, provided the original copyright notice and license text are retained.

<br>

## Ⅹ • Authors

- **Kevinnnnn-ai** — author and maintainer ([github.com/Kevinnnnn-ai](https://github.com/Kevinnnnn-ai))

<br>

## Ⅺ • Contact

- **Repository** — [github.com/Kevinnnnn-ai/project-vigil-redux-2d](https://github.com/Kevinnnnn-ai/project-vigil-redux-2d)
- **Issues** — please open a [GitHub issue](https://github.com/Kevinnnnn-ai/project-vigil-redux-2d/issues) for bugs, questions, or feature requests

<br>

---

*Last Updated: June 26, 2026*
