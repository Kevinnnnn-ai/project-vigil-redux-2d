# Convergence Fix (run-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the PBRS shaping from annealing to zero so the late-reached hard curriculum stages keep a dense learning signal, fixing the `SUICIDE1_NONCONVERGENCE` non-convergence.

**Architecture:** One config value flips `reward.shapingAnneal` from `linear` to `none`, which makes `shapingScaleFor` return a constant `1.0` for every iteration. The shaping term it scales is potential-based (PBRS), so constant shaping is policy-invariant and un-hackable — it only restores dense guidance. Two regression tests lock the behavior; a short isolated smoke confirms end-to-end health; the mandatory reward/change logs are updated.

**Tech Stack:** Python 3.14.5 (`.env.local` venv), PyTorch, Pymunk, pytest, PyYAML. Run from repo root; `src` is not pip-installed.

## Global Constraints

- **One-variable change only:** the sole behavioral edit is `config.yaml` `reward.shapingAnneal: linear -> none`. Do NOT change `training.entCoef` (0.02), `training.totalIters` (600), the promotion logic (`src/train/curriculum.py:119`), or the `full`-stage spawn. (Those are deferred contingencies — see the spec §7.)
- **Do NOT touch reward arithmetic:** `src/env/rewards.py` is unchanged; in particular never remove the `(1 - done)` factor (PBRS policy-invariance, guarded by `test_shapingTelescopesToInitialPotential`).
- **No world / obs / action / world-hash change:** checkpoints are unaffected.
- **Code conventions (`.claude/AGENTS.md`):** camelCase vars/functions, `_` prefix for helpers, SCREAMING_SNAKE constants, single-quoted strings, `#` comments only.
- **Hard rule — logging:** a `reward.*` change MUST get a `docs/REWARD_LOG.md` entry and a `docs/CHANGELOG.md` entry.
- **Run discipline:** from repo root with `.env.local\Scripts\python.exe`; `src` imports need the repo root on `PYTHONPATH` (use `PYTHONPATH=.` or `python -m`). Spec: `docs/superpowers/specs/2026-06-23-convergence-fix-design.md`.
- **Never push.** Commit only, on the current branch.
- **Smoke isolation:** smoke artifacts go ONLY to the sentinel `run-9001` dirs (gitignored) and are deleted after; never read/write `run-1`/`run-2`/`run-3` artifacts.

---

### Task 1: Disable the shaping anneal (config + tests + mandatory logs)

**Files:**
- Modify: `config.yaml:41` (`reward.shapingAnneal`)
- Test: `tests/test_config_loader.py` (add `test_repoConfigDisablesShapingAnneal`)
- Test: `tests/test_loop.py` (add `test_shapingScaleForNoneIsConstant`)
- Modify: `docs/REWARD_LOG.md` (new entry, newest under `# Entries`)
- Modify: `docs/CHANGELOG.md` (new `CONFIG` entry, newest under the first `---`)

**Interfaces:**
- Consumes: `loadConfig(path) -> Config` (`src/config/loader.py`); `shapingScaleFor(cfg, it, totalIters) -> float` (`src/train/loop.py:41`); `trainCurriculum(cfg, seed, savePath, csvPath=None, evaluateFn=None) -> list[dict]` (`src/train/curriculum.py:44`).
- Produces: a shipped `config.yaml` whose `reward.shapingAnneal == 'none'`; no new public symbols.

- [ ] **Step 1: Write the failing shipped-config test**

Add to `tests/test_config_loader.py` (end of file):

```python
def test_repoConfigDisablesShapingAnneal():
    # Fix for SUICIDE1_NONCONVERGENCE (docs/observations.md): the shipped config
    # keeps PBRS shaping ON for the whole run (no anneal), so the late-reached
    # hard stages keep a dense gradient. PBRS is policy-invariant, so this does
    # not change the optimum. See docs/REWARD_LOG.md.
    cfg = loadConfig('config.yaml')
    assert cfg.reward.shapingAnneal == 'none'
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.env.local/Scripts/python.exe -m pytest tests/test_config_loader.py::test_repoConfigDisablesShapingAnneal -q`
Expected: FAIL — `assert 'linear' == 'none'` (the shipped config still anneals).

- [ ] **Step 3: Flip the config value**

In `config.yaml`, the reward block line (currently `config.yaml:41`):

```yaml
  shapingAnneal: linear
```

becomes:

```yaml
  shapingAnneal: none      # PBRS stays full-scale all run (fix: SUICIDE1_NONCONVERGENCE)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./.env.local/Scripts/python.exe -m pytest tests/test_config_loader.py::test_repoConfigDisablesShapingAnneal -q`
Expected: PASS.

- [ ] **Step 5: Add coverage for the `none` branch of `shapingScaleFor`**

Add to `tests/test_loop.py` (after `test_shapingScaleSchedules`):

```python
def test_shapingScaleForNoneIsConstant(tmp_path):
    # shapingAnneal: none -> the anneal factor is a constant 1.0 at every iter,
    # so the PBRS shaping signal stays fully on through the late curriculum
    # stages (the linear branch decays it to ~0 over totalIters).
    path = tmp_path / 'config.yaml'
    path.write_text(textwrap.dedent('''
        reward:
          shapingAnneal: none
    '''), encoding='utf-8')
    cfg = loadConfig(str(path))
    assert cfg.reward.shapingAnneal == 'none'
    for it in (0, 5, 300, 599):
        assert shapingScaleFor(cfg, it, 600) == pytest.approx(1.0)
```

- [ ] **Step 6: Run the new loop test to verify it passes**

Run: `./.env.local/Scripts/python.exe -m pytest tests/test_loop.py::test_shapingScaleForNoneIsConstant -q`
Expected: PASS.

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `./.env.local/Scripts/python.exe -m pytest -q`
Expected: PASS — previous green count + 2 (no test pins `config.yaml`'s `shapingAnneal`, and `test_shapingScaleSchedules` uses its own `linear` fixture, so nothing breaks).

- [ ] **Step 8: Get the UTC timestamp for the log entries**

Run: `date -u +"%Y-%m-%d %H:%M UTC"`
Use the printed value as `<UTC>` in Steps 9–10.

- [ ] **Step 9: Add the REWARD_LOG entry**

In `docs/REWARD_LOG.md`, insert directly under the `# Entries` heading (newest on top):

```markdown
## 2026-06-23 — preset: baseline  [shaping] [anneal] [curriculum]

Hypothesis:
Keeping PBRS shaping fully ON for the whole run (shapingAnneal: none -> constant scale 1.0) restores a dense learning signal on the late-reached hard stages (glide/full), fixing the SUICIDE1_NONCONVERGENCE non-convergence WITHOUT changing the optimum — PBRS is policy-invariant (Ng et al. 1999) and telescopes to -Phi(s0), so constant shaping cannot distort the optimal policy or be reward-hacked.
Config:
Reward ARITHMETIC unchanged; the only knob change is reward.shapingAnneal: linear -> none. All else identical to run-2: terminalSuccess 1.0, terminalCrash -1.0, gentlenessBonus 0.5, centeringBonus 0.5, shapingCoef 1.0, controlCost 0.01; training entCoef 0.02, totalIters 600; narrowed full stage.
Result:
PENDING — run-3 (600 iters x 3 seeds) not yet launched. Validated: shapingScaleFor is constant 1.0 under the shipped config (unit), and a short isolated smoke trains end-to-end with finite losses under shapingAnneal: none.
Verdict:
ITERATE — launch run-3 and record convergence here (seeds reaching full + sustained eval success >= promoteAt, entropy bounded). If full still does not converge, escalate one variable at a time per spec §7 (entCoef/logStd, then promotion hysteresis).
```

- [ ] **Step 10: Add the CHANGELOG entry**

In `docs/CHANGELOG.md`, insert directly under the first `---` (newest on top), replacing `<UTC>` with the Step 8 value:

```markdown
## CONFIG | <UTC>

Summary:
Disabled the PBRS shaping anneal: reward.shapingAnneal linear -> none in config.yaml, so the potential-based shaping stays at full scale (1.0) for the entire run instead of decaying to ~0 over totalIters.

Reason:
Diagnosed non-convergence (SUICIDE1_NONCONVERGENCE, docs/observations.md): the global linear anneal (shapingScaleFor, src/train/loop.py:44-45) zeroed the dense reward before the curriculum reached glide/full, starving the hard stages of gradient (policyLoss ~0.001 on full) and letting the constant entCoef bonus inflate the policy (sigma ~1->3). Restoring constant shaping is policy-invariant (Ng et al. 1999; telescopes to -Phi(s0), un-hackable), so it adds dense guidance without changing the optimum.

Files:
- config.yaml — reward.shapingAnneal: linear -> none.
- tests/test_config_loader.py — test_repoConfigDisablesShapingAnneal asserts the shipped config disables the anneal.
- tests/test_loop.py — test_shapingScaleForNoneIsConstant covers the previously-untested none branch (constant 1.0).
- docs/REWARD_LOG.md — new entry for this reward-config change.

Changes:
- shapingScaleFor returns constant 1.0 under the shipped config (was 1.0 - it/totalIters).
- No reward arithmetic, world, obs/action, or world-hash change; checkpoints unaffected.

Validation:
- python -m pytest -q green (added 2 tests).
- Isolated smoke (tiny config, 1 seed, 4 iters, shapingAnneal none): trains end-to-end, finite losses, shaping constant 1.0; artifacts written to the gitignored run-9001 dirs and deleted.

Impact:
- Training only. run-3 keeps dense PBRS guidance on glide/full. No checkpoint invalidation.

Follow-up:
- User launches run-3 (600 iters x 3 seeds); record convergence in REWARD_LOG. If still non-convergent, escalate per spec §7 (entCoef/logStd, then promotion hysteresis).

Status: Done (change); run-3 pending.
```

- [ ] **Step 11: Commit**

```bash
git add config.yaml tests/test_config_loader.py tests/test_loop.py docs/REWARD_LOG.md docs/CHANGELOG.md
git commit -m "fix(reward): disable PBRS shaping anneal (shapingAnneal: none)" \
  -m "Fixes SUICIDE1_NONCONVERGENCE: the global linear anneal starved the late-reached hard stages of dense gradient. PBRS is policy-invariant, so constant shaping restores guidance without changing the optimum. One-variable delta from run-2." \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Cheap end-to-end smoke validation (no commit)

**Files:**
- Create (scratchpad, NOT the repo): `<scratchpad>/smoke_shaping.py`
- Touches + cleans: `stdout/logs/run-9001/`, `checkpoints/run-9001/` (gitignored sentinel dirs)

**Interfaces:**
- Consumes: `loadConfig`, `shapingScaleFor`, `trainCurriculum` (signatures in Task 1).
- Produces: console line `SMOKE OK: ...` — no committed artifacts.

- [ ] **Step 1: Write the smoke script**

Create `<scratchpad>/smoke_shaping.py` (use the session scratchpad dir):

```python
import math, os, shutil, textwrap

from src.config.loader import loadConfig
from src.train.curriculum import trainCurriculum
from src.train.loop import shapingScaleFor

RUN = 'run-9001'
logsDir = os.path.join('stdout', 'logs', RUN)
ckptDir = os.path.join('checkpoints', RUN)
os.makedirs(logsDir, exist_ok=True)
os.makedirs(ckptDir, exist_ok=True)
try:
    cfgPath = os.path.join(logsDir, 'smoke_config.yaml')
    with open(cfgPath, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent('''
            training:
              numEnvs: 2
              rolloutSteps: 64
              epochs: 2
              minibatchSize: 32
              evalEpisodes: 4
              evalEvery: 2
              totalIters: 4
              hidden: [16]
              evalSeeds: [0]
            reward:
              shapingAnneal: none
        '''))
    cfg = loadConfig(cfgPath)
    assert cfg.reward.shapingAnneal == 'none'
    for it in range(cfg.training.totalIters):
        assert shapingScaleFor(cfg, it, cfg.training.totalIters) == 1.0
    history = trainCurriculum(
        cfg, seed=0,
        savePath=os.path.join(ckptDir, 'seed0.pt'),
        csvPath=os.path.join(logsDir, 'seed0.csv'),
    )
    assert len(history) == cfg.training.totalIters, len(history)
    for h in history:
        for k in ('policyLoss', 'valueLoss', 'entropy'):
            assert math.isfinite(h[k]), (k, h[k])
    print('SMOKE OK: shapingAnneal=none trains end-to-end; shaping constant 1.0; losses finite')
finally:
    shutil.rmtree(logsDir, ignore_errors=True)
    shutil.rmtree(ckptDir, ignore_errors=True)
```

- [ ] **Step 2: Run the smoke from the repo root**

Run (repo root, so `stdout`/`checkpoints` resolve and `src` imports via `PYTHONPATH=.`):

`PYTHONPATH=. ./.env.local/Scripts/python.exe "<scratchpad>/smoke_shaping.py"`

Expected: prints `SMOKE OK: shapingAnneal=none trains end-to-end; shaping constant 1.0; losses finite` and exits 0.

- [ ] **Step 3: Confirm no leftover sentinel artifacts**

Run: `ls stdout/logs/ checkpoints/`
Expected: NO `run-9001` directory under either (the `finally` block removed them). `run-1`/`run-2` untouched. If `run-9001` remains, run `rm -rf stdout/logs/run-9001 checkpoints/run-9001`.

---

### Task 3: Update the diagnosis signature + decision log (commit)

**Files:**
- Modify: `docs/observations.md` (`SUICIDE1_NONCONVERGENCE` Status line)
- Modify: `.claude/agent-memory/decisions.md` (append a dated entry)
- Modify: `.claude/agent-memory/notes.md` (status of the "Training convergence" section)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the observations Status line**

In `docs/observations.md`, replace the final line of the `SUICIDE1_NONCONVERGENCE` entry:

```markdown
**Status:** diagnosed + verified 2026-06-23. Fix pending.
```

with:

```markdown
**Status:** diagnosed + verified 2026-06-23. **Fix applied 2026-06-23** — `reward.shapingAnneal: none` (mechanism 1); run-3 pending. See `docs/REWARD_LOG.md`, `docs/CHANGELOG.md`, and spec `docs/superpowers/specs/2026-06-23-convergence-fix-design.md`. Mechanisms 2–3 (entCoef/logStd, promotion hysteresis) remain as contingencies if run-3 still fails.
```

- [ ] **Step 2: Append the decision log entry**

Append to `.claude/agent-memory/decisions.md` (newest at the bottom):

```markdown
## 2026-06-23 — Fix run-2 non-convergence: disable the PBRS shaping anneal

- **`reward.shapingAnneal: linear -> none`** (`config.yaml`). *Why:* diagnosed
  `SUICIDE1_NONCONVERGENCE` (`docs/observations.md`) — the global linear anneal
  (`shapingScaleFor`, `loop.py:44-45`) zeroed the dense PBRS signal before the curriculum
  reached `glide`/`full`, starving the hard stages (`policyLoss ~0.001` on `full`); the
  constant `entCoef=0.02` then inflated the policy (sigma ~1->3, entropy 2.84->5.0+). PBRS is
  policy-invariant and telescopes to `-Phi(s0)`, so constant shaping restores dense guidance
  with NO change to the optimum and no reward-hacking risk.
- **Minimal one-variable change (user choice).** `entCoef`/`logStd` and promotion hysteresis
  are deferred to contingencies (spec `2026-06-23-convergence-fix-design.md` §7), so run-3 is a
  clean delta from run-2.
- **Validation:** unit (shipped config `none` + `none`-branch coverage) + an isolated smoke
  (tiny config, 1 seed, 4 iters); the full run-3 is launched by the user.
```

- [ ] **Step 3: Update the notes status**

In `.claude/agent-memory/notes.md`, at the END of the "Training convergence — run-1/run-2 DIAGNOSED" section (just before the next `##` heading), add:

```markdown
**Fix applied 2026-06-23:** `reward.shapingAnneal: none` (mechanism 1 — restore dense PBRS on
the hard stages). run-3 not yet launched. Spec
`docs/superpowers/specs/2026-06-23-convergence-fix-design.md`; contingencies (entCoef/logStd,
promotion hysteresis) untouched.
```

- [ ] **Step 4: Commit**

```bash
git add docs/observations.md .claude/agent-memory/decisions.md .claude/agent-memory/notes.md
git commit -m "docs: mark SUICIDE1_NONCONVERGENCE fix applied (shapingAnneal: none)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (spec → task):**
- §3 the change (`shapingAnneal: none`) → Task 1 Step 3. ✓
- §4.1 enum check → confirmed pre-plan (`loader.py:276` allows `none`); no loader change. ✓
- §4.2 unit suite + fix tests pinning `linear` → Task 1 Step 7 (none pin the shipped value; the `linear` fixture test is untouched). ✓
- §4.3 constant-shaping assertion → Task 1 Step 5 + Task 2 Step 1. ✓
- §4.4 short isolated smoke + cleanup → Task 2. ✓
- §5 REWARD_LOG + CHANGELOG + observations + agent-memory → Task 1 Steps 9–10, Task 3. ✓
- §6 acceptance criteria → judged on the user's run-3 (out of plan scope, noted in logs). ✓
- §7 contingencies → explicitly deferred (Global Constraints + Task 3 notes). ✓

**Placeholder scan:** No TBD/TODO. `<UTC>` and `<scratchpad>` are explicit fill-ins with the exact command/dir to source them (Task 1 Step 8; session scratchpad). "Result: PENDING" is intentional (run-3 not yet run).

**Type/name consistency:** `shapingScaleFor(cfg, it, totalIters)`, `trainCurriculum(cfg, seed, savePath, csvPath=, evaluateFn=)`, `loadConfig(path)`, and `cfg.reward.shapingAnneal` are used identically across all tasks and match source (`src/train/loop.py:41`, `src/train/curriculum.py:44`, `src/config/loader.py:121,276`).
