# Conventions

The rules every change in this repo follows. The naming/formatting rules are spelled out in §1
(with worked examples); the annotation rules live in the `.claude/skills/code-annotation` skill;
the RL-correctness rules are enforced by the `rl-reviewer` subagent. This page consolidates them
so an agent finds them from the docs index.
When a rule and a skill disagree, the skill is authoritative — fix this page. See
[AGENTS.md](AGENTS.md) for the docs overview and [CODE_MAP.md](CODE_MAP.md) for why the
architecture is shaped this way.

## 1. Naming & formatting (camelCase repo)

- **`camelCase`** for variables, parameters, and functions (`userName`, `runTask`,
  `computeReward`) — **never** `snake_case`. ⚠️ This is the opposite of PEP 8; consistency with
  the repo beats the external style guide.
- **`PascalCase`** for classes (`LandingEnv`, `BoosterSim`, `MLPPolicy`, `PdPilot`).
- **`SCREAMING_SNAKE_CASE`** for module-level constants (`OBS_DIM`, `VEL_REF`,
  `PHYSICS_MODEL_VERSION`, `REST_SPEED`).
- **Boolean variables** take an explicit prefix: `is` / `has` / `can` / `should` / `no`
  (`isReady`, `hasFuel`, `canIgnite`, `shouldPromote`, `noResults`).
- **Functions begin with a verb** where practical (`computePotential`, `resolveDevice`,
  `loadCheckpoint`, `encodeObs`).
- **Single quotes** for strings (`'success'`, `'pymunk-2'`) — exceptions are triple-quoted
  strings and cases where single quotes would force excessive escaping (`"It's ready."`).
- **Multi-line collections / arg lists**: opening and closing brackets on their own lines, and
  a **trailing comma** on the last element.

```python
reward = {
    'terminalSuccess': 1.0,
    'controlCost': 0.01,
}
```

Style priority when rules conflict: (1) existing file style, (2) repo-specific conventions,
(3) the naming rules above, (4) external guides (PEP 8, formatter defaults). Consistency
**within a file** outranks any single rule.

<details>
<summary>Worked examples</summary>

```python
# datatypes / functions: camelCase
userName = 'John'
totalCost = 25
def runTask(): ...
def calcTotal(): ...

# booleans: explicit prefix
isReady = True
hasPermission = False
canExecute = True
shouldRetry = False
noResults = True

# constants: SCREAMING_SNAKE_CASE
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 30

# classes: PascalCase
class UserManager: ...

# multi-line collections: brackets on own lines, trailing comma
userData = {
    'name': 'John',
    'age': 30,
}
runCommand(
    argOne,
    argTwo,
)
```

</details>

## 2. Every file is annotated (two layers)

Apply BOTH on every code write/edit (full spec: `code-annotation` skill):

- **File-header block** at the top: `<agent_context>` (`[ARCH]`, `[API]`, `[GOTCHA]`) and
  `<agent_guardrail>` (`[CRITICAL]`, `[VALIDATION]`). Minimum: `[ARCH]` + `[VALIDATION]`.
  This block is present on most `src/` files — keep and update it in place; don't rewrite it
  wholesale.
- **`@TAG[id]` landmarks**: `@ENTRY` on entry points, `@ANCHOR` on key sections/functions,
  `@DEP[→id]` on cross-module dependencies, `@CONFIG[key]` on config reads, `@SIDEFX` on
  external mutations (file writes, RNG, checkpoint saves), `@INVARIANT` on guarded assumptions,
  `@RISK` on hazards, `@TODO[id]` on stubs. Ids are stable, kebab-case, unique per file; the
  apply procedure is **idempotent** (match by id, update stale text in place, never duplicate,
  remove orphans).

> Migration status (CLAUDE.md NOTE): the header block is on most `src/` files; the `@TAG[id]`
> layer is **partially backfilled** — present on the most-edited modules (e.g. `@TAG[rest-verdict]`,
> `@TAG[angle-map]`), absent on the rest. When you edit a file, complete *that file's* layer as
> part of the change. A repo-wide backfill is a separate, explicit task.

## 3. Config is the single control panel

Every tunable — world geometry, reward weights, PPO hyperparameters, curriculum stages,
runtime device — lives in `config.yaml` (or a `configs/{lux,solis}/<env>.yaml` override) and is
parsed to a **frozen** dataclass by `src/config/loader.py`. Do **not** hardcode a knob in
source. The only constants that legitimately live in code are the frozen obs refs
(`VEL_REF`, `OMEGA_REF`) and the physics-model tag (`PHYSICS_MODEL_VERSION`) — both are part of
the compatibility contract, not tuning. See [CODE_MAP.md](CODE_MAP.md) rule 5 and
[GLOSSARY.md](GLOSSARY.md).

## 4. The world hash is the compatibility boundary

`Config.computeWorldHash()` hashes the `world:` fields **plus** `PHYSICS_MODEL_VERSION`. A
checkpoint loads iff its stored hash matches. Therefore:

- Editing `reward` / `training` / `curriculum` / `runtime` keeps existing models loadable —
  these must **never** block `watch`/`play`/`evaluate`.
- Editing `world` geometry, the frozen obs refs, or the physics *model* itself invalidates all
  prior checkpoints. If you change the physics model (not just a tuning field), **bump
  `PHYSICS_MODEL_VERSION`** so the guard catches stale checkpoints even when `world:` is
  unchanged.
- ALWAYS pair `--model <m> --env <e>` with `--config configs/<m>/<e>.yaml` — the dir picks the
  checkpoint folder; the **config** supplies the world hash. `lux` (analog) and `solis`
  (suicide-burn) are not interchangeable. See [WORKFLOWS.md](WORKFLOWS.md) and the
  `MODEL_ENV_SUBDIR_IS_ORGANIZATIONAL` / `SUICIDE_BURN_WORLD` notes in [OBSERVATIONS.md](OBSERVATIONS.md).

## 5. RL-correctness invariants (the `rl-reviewer` checklist)

These are the bugs this stack is uniquely prone to. The read-only `rl-reviewer` gates them
before merge; don't break them:

- **Single gamma** — `cfg.training.gamma` is the only discount; GAE and reward shaping both read
  it. No second reward gamma.
- **PBRS `(1 − done)`** — the shaping term zeroes Φ at terminal states (Ng et al. 1999). Removing
  the factor breaks policy invariance. Guarded by `tests/test_rewards.py`.
- **tanh-squash log-prob Jacobian** — `MLPPolicy._squashedLogProb` must carry the
  `log(1 − tanh²)` correction, or the policy-gradient ratio is wrong.
- **Clip-then-cost** — actions are clipped (`throttle [0,1]`, `gimbal [-1,1]`) **before** the
  control-cost term reads them; don't double-clip.
- **Deterministic eval** — evaluation uses `policy.act()` (the squashed **mean**, no sampling).
  Sampling during eval inflates variance and breaks promotion gates.
- **Impact speed is pre-contact** — read `impactSpeed` from the approach velocity (`prevState`);
  the solver arrests post-contact velocity, so reading it post-step reports ~0. See
  `FLOOR_CLAMP_EATS_IMPACT_SPEED` in [OBSERVATIONS.md](OBSERVATIONS.md).
- **Timeout pays the full crash penalty** — deliberate anti-stall. Without it, hovering to the
  clock dominates a risky landing. See `M2_DISCOUNT_PROCRASTINATION_EXPLOITS`.

## 6. Everything testable is tested (from the repo root)

Run `python -m pytest -q` from the repo root (`src` is a namespace package — see
[WORKFLOWS.md](WORKFLOWS.md)). Add/extend a `tests/test_<module>.py` for any behavior you
change. The load-bearing guards — `test_shapingTelescopesToInitialPotential` (PBRS),
`test_scripted.py` thresholds (solvability), the worldHash guard tests — must stay green.

## 7. Keep the docs current as part of the change

This is a **hard rule** in `CLAUDE.md`, not a separate chore:

- Code/behavior change → [CHANGELOG.md](CHANGELOG.md) entry (newest at top).
- **Reward** change → [REWARD_LOG.md](REWARD_LOG.md) entry — every reward version is logged with
  its structure and weights. Never change rewards silently.
- New/moved module → [CODE_MAP.md](CODE_MAP.md).
- Surprising finding or resolved discrepancy → [OBSERVATIONS.md](OBSERVATIONS.md).
