# Project Vigil Redux 2D

A 2D reinforcement-learning sandbox in which a from-scratch PPO agent learns to fly and land a single-stage, gimbaled rocket booster inside a Pymunk (Chipmunk2D) rigid-body simulation—landing, settling, and tip-over emerge from the physics solver, not from a scripted verdict. The project's track is to re-aim the whole stack at one objective: training models to perfectly execute the ideal suicide burn (a 'true hover slam')—a fuel-optimal, (near-)single-burn descent that arrests velocity to ~0 exactly at the pad, upright and centered.

## Quickstart and Commands

```bash
# Run everything from the repo ROOT (the `src.` package is not pip-installed).

# 1. Create the local virtual environment (Python 3.14)
python -m venv .env.local

# 2. Activate it
source .env.local/Scripts/activate          # Git Bash on Windows
#   PowerShell:  .\.env.local\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Smoke-test the foundation: every src/ module must import from repo root
PYTHONPATH=. python -c "import src.config.loader, src.env.episode, src.env.physics, src.env.rewards, src.env.spaces, src.agents.checkpoints, src.agents.mlp, src.agents.policy, src.agents.scripted, src.train.curriculum, src.train.device, src.train.loop, src.train.parallel, src.train.ppo, src.train.rollout, src.train.vec_env, src.runtime.evaluate, src.runtime.loop, src.runtime.render, src.metrics.logger, src.metrics.plot; print('imports OK')"
```

Notes cheat sheet:

- Always work inside the local venv `.env.local/` (gitignored, Python 3.14); deps are pinned in `requirements.txt` (`pyyaml`, `pytest`, `numpy`, `torch`, `pygame-ce`, `pymunk`, `matplotlib`).
- Run from the repo root with the root on the path—`python -m <pkg.module>` or `PYTHONPATH=.`. Invoking a file by absolute path puts the script's directory (not the root) on `sys.path`, giving `ModuleNotFoundError: src`.
- Every `__init__.py` is empty (no re-exports), so import fully-qualified: `from src.config.loader import loadConfig`, not `from src.config import loadConfig`.
- The foundation is an importable library, **not yet runnable end-to-end**: there is no `scripts/`, `config.yaml`, `configs/`, `tests/`, or `models/` yet. `loadConfig()` defaults to reading `config.yaml` from the cwd, so it raises `FileNotFoundError` until a config is supplied (deferred to the hover-slam rewire).
- One world switch, `WorldConfig.engineMode`, picks the dynamics: `analog` (lux—continuous throttle in `[0, 1]`) vs `suicideBurn` (solis—binary full-thrust firing, capped at 2 ignition transitions). It is a `world:` field, so it enters `computeWorldHash`; analog and suicide-burn checkpoints are mutually incompatible.
- Put test files under `tests/`; write test output only into `stdout/` without clobbering pre-existing files, and clean up afterward.

## Project Structure

```text
project-vigil-redux-2d/
├─ src/                        # importable library; run from repo root (python -m … or PYTHONPATH=.)
│  ├─ __init__.py              # empty package marker
│  ├─ config/
│  │  ├─ __init__.py
│  │  └─ loader.py             # config single-source-of-truth: config.yaml -> frozen dataclasses; world-only compatibility hash
│  ├─ env/
│  │  ├─ __init__.py
│  │  ├─ spaces.py             # frozen 10-D obs / 2-D action layout, encode/decode + normalization constants (OBS_DIM, VEL_REF, …)
│  │  ├─ physics.py            # Pymunk rigid-body booster sim: persistent space, engine/spool/fuel logic, geometry helpers
│  │  ├─ rewards.py            # config-driven reward: terminal payouts + potential-based shaping + control-effort cost
│  │  └─ episode.py            # LandingEnv: composes sim+obs+reward, steps physics, classifies success/crash/timeout
│  ├─ agents/
│  │  ├─ __init__.py
│  │  ├─ policy.py             # abstract Policy interface: (10,) obs -> (2,) [throttle, gimbal]
│  │  ├─ mlp.py                # MLPPolicy actor-critic (tanh-squashed diagonal Gaussian); inference, training, checkpoint I/O
│  │  ├─ scripted.py           # PdPilot: hand-tuned PD landing controller, the RL baseline
│  │  └─ checkpoints.py        # resolve checkpoint paths; load with worldHash compatibility guard
│  ├─ train/
│  │  ├─ __init__.py
│  │  ├─ device.py             # torch device selection (auto -> cuda/cpu, or forced cpu)
│  │  ├─ ppo.py                # hand-written PPO update (clipped surrogate + value loss + entropy) + explained-variance
│  │  ├─ rollout.py            # vectorized rollout collection + per-env GAE advantages/returns
│  │  ├─ vec_env.py            # VecLandingEnv: batch of LandingEnvs, auto-reset, seeded RNG, stage/shaping fan-out
│  │  ├─ loop.py               # single-stage PPO driver: collect -> GAE -> update -> eval -> save-best, with shaping anneal
│  │  ├─ curriculum.py         # spawn-difficulty stage ladder; promote when eval success >= promoteAt
│  │  └─ parallel.py           # per-seed PPO across OS processes (ProcessPoolExecutor); gather sorted SeedResults
│  ├─ runtime/
│  │  ├─ __init__.py
│  │  ├─ evaluate.py           # deterministic eval episodes -> success rate, per-outcome counts, mean impact speed, mean steps
│  │  ├─ loop.py               # per-frame episode loop: renderer + action source, pause/step/speed/reset, end-of-episode dwell
│  │  └─ render.py             # pygame-ce Renderer: draws world + HUD, polls input into an Intents struct
│  └─ metrics/
│     ├─ __init__.py
│     ├─ logger.py             # CsvLogger: dict -> CSV, header frozen from the first record
│     └─ plot.py               # plotConvergence: per-seed success-rate curves vs cumulative env steps (headless Agg)
├─ requirements.txt            # pinned deps: pyyaml, pytest, numpy, torch, pygame-ce, pymunk, matplotlib
├─ .gitignore                  # ignores .env.local/, project-vigil-redux-2d.zip, __pycache__/
├─ .env.local/                 # local virtual environment (Python 3.14), gitignored
├─ project-vigil-redux-2d.zip  # upstream source archive (gitignored); origin of the replicated src/
└─ .claude/                    # agent config: CLAUDE.md, AGENTS.md, agent-memory/, skills/, output-styles/
```

Not yet present (deferred to the hover-slam rewire): `scripts/{train,watch,play,evaluate}.py`, a top-level `config.yaml`, `configs/{lux,solis}/*.yaml`, `tests/`, `models/`, and `docs/`. `src/` was replicated verbatim from `project-vigil-redux-2d.zip` and is verified importable, but it is a library—not an end-to-end runnable app—until those entrypoints and a config exist.

## Code Conventions

Keep all code short but descriptive. Try to not abbreviate unless it's common in the codebase.

- **Variables** — Use camel case. Local iteration values should always be denoted with `i` or `j`. Local, short-lived, positional values should be denoted with letters from the alphabet (e.g., `a`, `b`, etc.), with the order they appear in corresponding to the order of the alphabet. Never use underscore prefixes.
    - **Booleans** — Always start the variable with a boolean prefix (e.g., `can`, `has`, `is`, etc.).
    - **Strings** — Never use `"`, only use `'`.
    - **Constants** — Use screaming snake case.
- **Functions** — Use camel case. Always start the function with a verb (e.g., `run`, `calc`, etc.). Prefix helper functions with an underscore (e.g., `_runOX`).
- **Classes** — Use pascal case. Prefix helper classes with an underscore.
- **Whitespace** — When separating sections (e.g., `imports`, `variables`, `functions`, etc.), use exactly 3 empty newlines. Otherwise, only deploy single newlines—and even then, use them sparingly. However, with nested sections—unless lengthy—avoid applying this rule (e.g., `functions` in `classes`).
- **Comments** — Never use `'''` for multi-line, in-line, explanation comments. Always use `#`.
- **Multi-line Brackets** — When opening brackets or parentheses across multiple lines, open them up all the way, ensuring each hierarchy of bracket recieves its own line.

## Testing

- Do not guess behavior.
- Verify assumptions by reading source, fixtures, and tests.
- Place test scripts and related files in the directory `tests/` or subdirectories under it.
- Note important findings in local `agent-memory/` for debugging and operational troubleshooting.
- Output testing data to `stdout/`, but ensure it doesn't conflict with pre-existing data.
- Always clean up after testing.

## Security Rules

- Commit work as often as possible, usually committing small portions of work to keep local copy preserved.
- Look to keep commits small and organized.
- Look to always attach descriptions to commits, covering what was changed and why.
