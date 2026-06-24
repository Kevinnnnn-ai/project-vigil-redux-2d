<div align="center">

# 2d-ppo-booster-landing

### Reinforcement learning that lands a 2D gimbaled booster upright

A hand-written **PPO** agent trained to fly a single-stage rocket booster. Physics is a real rigid-body simulation via [Pymunk](https://www.pymunk.org/) (Chipmunk2D): the booster's legs physically collide with the ground, and landing, settling, and tip-over **emerge from the solver** rather than from a scripted verdict. A curriculum ladder walks the policy from gentle touchdowns up to full free-fall descents.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pymunk](https://img.shields.io/badge/Pymunk-Chipmunk2D-1B998B?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-CE-2EA44F?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge)

</div>
<br>



---



<br>

## Table of Contents

- [About](#about)
    - [How It Works](#how-it-works)
    - [The Physics](#the-physics)
    - [The Engine: Binary Suicide Burn](#the-engine-binary-suicide-burn)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
- [Usage / Quick Start](#usage--quick-start)
    - [Train](#train)
    - [Watch](#watch)
    - [Evaluate](#evaluate)
    - [Play](#play)
- [Configuration](#configuration)
- [Observation & Action Contract](#observation--action-contract)
- [Reward](#reward)
- [Project Layout](#project-layout)
- [Tests](#tests)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
    - [Known Limitations](#known-limitations)
- [License](#license)
- [Authors & Acknowledgments](#authors--acknowledgments)
- [Contact](#contact)

<br>



---



## About

The agent must pilot a gimbaled rocket booster — controlling **main throttle** and **nozzle deflection** — from a given starting point to a desired end position. A touchdown counts as a **success** only if the booster comes to rest **upright**, **on the pad**, and **gently** (slow approach speed), **with the engine cut before the first toe contact**; anything else is a crash. Because the booster is a rigid body whose legs collide with the ground, the difference between standing and toppling is decided by the physics solver, not by a hand-written rule.

The learning algorithm is **Proximal Policy Optimization (PPO)** implemented from scratch in PyTorch — clipped surrogate objective, generalized advantage estimation (GAE), value-function and entropy terms — driving an actor-critic MLP. Training runs across parallel vectorized environments and climbs a **curriculum** of progressively harder spawn conditions.

<br>



### How It Works

1. **Environment** — A custom gym-*style* env ([src/env/episode.py](src/env/episode.py), `LandingEnv`) wraps the Pymunk simulator, encodes a 10-dim observation, maps the 2-dim action onto engine commands, and classifies each episode's outcome.
2. **Policy** — An actor-critic MLP ([src/agents/mlp.py](src/agents/mlp.py), `MLPPolicy`) outputs a tanh-squashed action distribution and a state-value estimate.
3. **Rollout + GAE** — Vectorized envs ([src/train/vec_env.py](src/train/vec_env.py)) collect rollouts; advantages are computed with GAE ([src/train/rollout.py](src/train/rollout.py)).
4. **PPO update** — The clipped-surrogate objective plus value loss and an entropy bonus updates the policy ([src/train/ppo.py](src/train/ppo.py)).
5. **Curriculum** — A ladder of spawn-condition stages (`touchdown → hop → drop → glide → full`) promotes the policy to the next rung once its success rate clears `curriculum.promoteAt` ([src/train/curriculum.py](src/train/curriculum.py)).

A scripted **PD baseline** (`PdPilot`, [src/agents/scripted.py](src/agents/scripted.py)) provides a weak binary single-burn reference (~40% success on the touchdown stage) to score the learned policy against on identical seeds.

<br>



### The Physics

The simulator ([src/env/physics.py](src/env/physics.py), `BoosterSim`) is a persistent Pymunk `Space`. The booster hull and its two legs are a single rigid body that physically collides with a static ground segment. Key consequences:

- **Contact is detected on the lowest leg toe**, not the body base — so at rest the base sits a leg's drop above the ground.
- **Impact speed** is read from the *approach* velocity, because the solver arrests post-contact velocity to ~0.
- The solver is advanced in **sub-steps** per env step so hard impacts resolve rigidly (Chipmunk2D has no continuous collision detection).
- A gentle, upright descent rocks down onto both legs and stands; enough tilt or spin physically **topples** it. There is no scripted "did it land" check — `LandingEnv.step` only *observes* the resulting physical state.

<br>



### The Engine: Binary Suicide Burn

This project uses a **single binary suicide-burn engine** — there is no analog throttle mode. The engine logic:

- Any action throttle command **above `SUICIDE_ON_THRESHOLD` (0.5)** fires the engine at **full thrust**; at or below 0.5 the engine is **OFF**.
- The engine allows **at most one ignition** (off → on) followed by **at most one cutoff** (on → off) — two state-changes total — then **locks permanently** in whatever state it is in.
- Spool-up lag and fuel burn still apply: the commanded state is a latch (`engineCommandedOn`), while actual thrust ramps via `spool`.
- **Cut-before-touchdown is required for success**: the engine must be commanded OFF before the first toe contact (`isCutOff` gate in `episode.py`). A landing with the engine still firing at contact is a crash, regardless of attitude or speed.

`obs[9]` (`ignitionsRemaining`) = `(2 − engineTransitions) / 2`: `1.0` fresh, `0.5` burning (ignited once), `0.0` locked (both transitions used).

<br>



## Installation

```bash
git clone https://github.com/Kevinnnnn-ai/2d-ppo-booster-landing.git
cd 2d-ppo-booster-landing

python -m venv .env.local
# Windows (PowerShell):
.env.local\Scripts\Activate.ps1
# macOS / Linux:
source .env.local/bin/activate

pip install -r requirements.txt
```

<br>



### Prerequisites

- **Python 3.14** (developed and tested on 3.14.5).
- A standard platform (Windows, macOS, or Linux). PyTorch runs on CPU out of the box; a CUDA-capable GPU is optional and used automatically if available.
- The packages in [requirements.txt](requirements.txt) (see [Tech Stack](#tech-stack) for what each does).

<br>



## Usage / Quick Start

> **Run all commands from the repository root.** The package import root is `src.` (e.g. `src.env.episode`); it is not installed, so scripts must run from the project root. Activate `.env.local` first.

There is **no `--model`/`--env` world-selection axis**. Configuration is `config.yaml` only. Training auto-numbers runs; use `--run N` to target a specific run.

<br>



### Train

Train across all configured seeds and save the best checkpoint. By default this runs the full curriculum (`touchdown → full`):

```bash
python -m scripts.train
```

For a single-stage run with no promotion:

```bash
python -m scripts.train --stage hop
```

Force a specific run number (default: auto-increment):

```bash
python -m scripts.train --run 4
```

Train seeds one-at-a-time (for debug/repro):

```bash
python -m scripts.train --serial
```

Per-seed checkpoints go to `checkpoints/run-N/seed<seed>.pt`; the best across seeds is copied to `checkpoints/run-N/best.pt`. Per-iteration metrics CSVs land in `stdout/logs/run-N/`. A live-updating convergence PNG is written to `stdout/convergence-plots/run-N.png`.

<br>



### Watch

Open an interactive pygame window and watch the trained policy fly (uses the latest run by default):

```bash
python -m scripts.watch
python -m scripts.watch --run 3 --checkpoint best
python -m scripts.watch --run 3 --checkpoint seed1 --stage drop
python -m scripts.watch --pilot pd    # scripted PD pilot, no checkpoint needed
```

Controls: `space`=pause, `n`=step, `r`=reset, `-`/`=`=speed, `esc`=quit.

<br>



### Evaluate

Score a checkpoint deterministically over `runtime.evaluateEpisodes` episodes (default 100) against the `PdPilot` baseline on the same seeds — reporting success rate, mean impact speed, outcome breakdown, and mean episode length:

```bash
python -m scripts.evaluate
python -m scripts.evaluate --run 3 --checkpoint seed2 --stage drop --episodes 200
```

<br>



### Play

Fly the booster yourself with no model — useful for building intuition about the dynamics and the success bar:

```bash
python -m scripts.play
python -m scripts.play --stage hop
```

Controls: `w`/`s` (up/down) = throttle, `a`/`d` (left/right) = gimbal, `space`=pause, `n`=step, `r`=reset, `-`/`=`=speed, `esc`/`q`=quit. HUD shows fuel, throttle, spool, velocity, tilt, and the landing record.

<br>



## Configuration

Everything is driven by a single YAML control panel ([config.yaml](config.yaml)). There is no `configs/` directory — `config.yaml` is the only config. It is split into compatibility-isolated groups:

| Group | Purpose | Affects checkpoint compatibility? |
|-------|---------|-----------------------------------|
| `world:`      | Physics constants — gravity, thrust, gimbal, fuel/mass, drag, landing limits, leg geometry, `dt`, `maxSteps` | **Yes** — editing these (or the physics-model version) invalidates existing models |
| `reward:`     | Terminal payouts, shaping coefficient, control cost (see [Reward](#reward)) | No |
| `training:`   | PPO hyperparameters — `lr`, `gamma`, `gaeLambda`, `clipEps`, `epochs`, `rolloutSteps`, `numEnvs`, network `hidden`, eval seeds/episodes | No |
| `curriculum:` | Per-stage spawn ranges and the `promoteAt` threshold | No |
| `runtime:`    | Watch/evaluate settings (`watchModel`, `evaluateEpisodes`) | No |

A model is loadable iff its stored **world hash** matches the live config's. `Config.computeWorldHash` ([src/config/loader.py](src/config/loader.py)) hashes the `world:` fields **plus a physics-model version tag** (`PHYSICS_MODEL_VERSION='suicide-1'`) — so a change to the physics model itself invalidates old checkpoints even when the `world:` numbers are unchanged.

<br>



## Observation & Action Contract

The agent sees an **11-dimensional** float observation and emits a **2-dimensional** action (see [src/env/spaces.py](src/env/spaces.py)).

**Observation (`OBS_DIM = 11`):**

| idx | field | meaning | normalization |
|----:|-------|---------|---------------|
| 0 | x | horizontal offset from pad center | / (width / 2) |
| 1 | y | altitude of base above ground | / ceiling |
| 2 | vx | horizontal velocity | / `VEL_REF` (20 m/s) |
| 3 | vy | vertical velocity | / `VEL_REF` |
| 4 | sin θ | attitude (sin) | already in [-1, 1] |
| 5 | cos θ | attitude (cos) | already in [-1, 1] |
| 6 | ω | angular velocity | / `OMEGA_REF` (3 rad/s) |
| 7 | fuel | remaining tank fraction | already in [0, 1] |
| 8 | spool | actual (spooled) throttle | already in [0, 1] |
| 9 | ignitions remaining | `(2 − engineTransitions) / 2` | `1.0` fresh, `0.5` burning, `0.0` locked |
| 10 | gimbal | actual (lagged) nozzle deflection — slews toward the command at `world.gimbalResponse` | already in [-1, 1] |

**Action (`ACTION_DIM = 2`):**

| idx | action | mapping |
|----:|--------|---------|
| 0 | main throttle | net outputs tanh-space `(-1, 1)` → affine-mapped to env-space `[0, 1]`; above 0.5 fires engine at full, at or below is OFF |
| 1 | gimbal | `[-1, 1]` → scaled by `world.maxGimbal` → nozzle deflection (lateral force + torque); **slew-rate limited** by `world.gimbalResponse` (~0.5 s full sweep, no instant side-to-side flip) — `obs[10]` reports the actual lagged angle |

> `VEL_REF` / `OMEGA_REF` are frozen code constants and part of the obs contract; attitude enters only as `(sin, cos)` — decode with `atan2(obs[4], obs[5])`.

<br>



## Reward

Per-step reward = terminal payout (on outcome) + potential-based shaping + control cost ([src/env/rewards.py](src/env/rewards.py); every version logged in [docs/REWARD_LOG.md](docs/REWARD_LOG.md)).

| Kind | Term | Form |
|------|------|------|
| terminal | land success | flat `terminalSuccess`, plus the two bonuses below |
| terminal | gentleness bonus | `gentlenessBonus` scaled by margin under the landing limits |
| terminal | centering bonus | `centeringBonus` scaled by closeness to pad center |
| terminal | crash | `terminalCrash` (timeout also pays the full crash penalty, as an anti-stall measure) |
| shaping | PBRS | `+coef·γ·Φ(s′) − Φ(s)`, zeroed at the terminal state |
| cost | control effort | `−controlCost · effort` (actions clipped first) |

> The shaping potential `Φ = −( dist(pad)/ceiling + speed/VEL_REF + |θ|/π )`. The `(1 − done)` factor that zeroes `Φ` at the terminal state is **required** for policy invariance (Ng et al. 1999) — do not remove it.

<br>



## Project Layout

| Path | What lives there |
|------|------------------|
| [config.yaml](config.yaml)          | Single control panel: world / reward / training / curriculum / runtime |
| [src/env/](src/env/)               | Physics sim, gym-style env, obs/action spaces, reward |
| [src/agents/](src/agents/)         | Actor-critic MLP, scripted PD baseline, checkpoint I/O |
| [src/train/](src/train/)           | PPO update, rollout/GAE, vectorized envs, curriculum, single-stage driver |
| [src/runtime/](src/runtime/)       | Per-frame loop, pygame renderer, deterministic evaluation |
| [src/config/](src/config/)         | Config loader and world-hash compatibility guard |
| [src/metrics/](src/metrics/)       | CSV metrics logger, convergence plotter, run-numbered path helpers |
| [scripts/](scripts/)               | `train`, `watch`, `evaluate`, `play` entry points |
| [checkpoints/run-N/](checkpoints/) | Per-run checkpoints (gitignored): `seed<seed>.pt` + `best.pt` |
| [stdout/logs/run-N/](stdout/)      | Per-seed metrics CSVs (gitignored) |
| [stdout/convergence-plots/](stdout/) | Per-run convergence PNG `run-N.png` (live-updated during training) |
| [docs/](docs/)                     | Agentic docs: see [docs/AGENTS.md](docs/AGENTS.md) (code map, conventions, workflows, glossary, changelog, observations, reward log, roadmap) |

<br>



## Tests

A `pytest` suite under [tests/](tests/) covers the config loader and world hash, episode reset/step, checkpoint I/O, the MLP, the PPO loop, curriculum, parallel envs, evaluation, and device selection. Run it from the repository root:

```bash
python -m pytest -q
```

<br>



## Tech Stack

| Tool | Role |
|------|------|
| [Python 3.14](https://www.python.org/) | Implementation language |
| [PyTorch](https://pytorch.org/) | Actor-critic network, autograd, the PPO update |
| [NumPy](https://numpy.org/) | Observations, rollouts, vectorized env state |
| [Pymunk](https://www.pymunk.org/) | Rigid-body physics (Chipmunk2D) — booster, legs, ground contact |
| [Pygame CE](https://pyga.me/) | Interactive watch window and human play mode |
| [PyYAML](https://pyyaml.org/) | Config loading |
| [Matplotlib](https://matplotlib.org/) | Training convergence plots |

<br>



## Roadmap

- [ ] Full PPO training run under the single suicide-burn world to validate trainability end-to-end.
- [ ] Recurrent or stacked-frame policies to handle the engine spool-up lag.
- [ ] Additional curriculum stages and automated stage-tuning.
- [ ] Domain randomization over `world:` parameters for robustness.
- [ ] Export trained policies to a standalone, dependency-light inference path.

<br>



### Known Limitations

- **2D, single-stage only.** No 3D dynamics and no stage separation.
- **Single physics world.** A checkpoint is loadable only if its stored world hash matches the live `config.yaml`; editing `world:` or bumping `PHYSICS_MODEL_VERSION` invalidates existing checkpoints.
- **First training run pays setup cost** while parallel env workers spin up.
- **No `gymnasium` registration.** The env is gym-*style* but custom; external RL libraries will not see an `env_id`.

<br>



## License

The source code in this repository is released under the [MIT License](LICENSE).

<br>



## Authors & Acknowledgments

- **Kevin Jie** — author and maintainer.
- Physics powered by **Pymunk** / **Chipmunk2D**; the learning stack is a from-scratch PPO implementation in **PyTorch**.

<br>



## Contact

Questions, suggestions, or issues? Open an issue on the repository, or reach out at **kevinwjie@gmail.com**.

<br>



---



<div align="center">

*Last updated: 2026-06-22*

</div>
