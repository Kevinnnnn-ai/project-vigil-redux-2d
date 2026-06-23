# Agentic Documentation Index

This `docs/` tree is the **agent-facing knowledge base** for this repository — a 2D
single-stage **gimbaled booster-landing** sandbox where a hand-written **PPO** agent learns
to land a Pymunk rigid-body rocket soft, centered, and upright. It exists so an agent can
orient itself in seconds without re-reading the whole tree. Read this file first.

> This is documentation *for agents working on the repo*. The repo-root `README` (embedded in
> `CLAUDE.md`) is the dense machine-navigable project spec — start there for the task→file→symbol
> index, the obs/action/reward contracts, and the subagent/skill routing tables. This tree
> complements it with the cross-file map, conventions, workflows, history, and findings.

## The documents

| File | What it answers | Read it when |
|------|-----------------|--------------|
| [CODE_MAP.md](CODE_MAP.md) | Where everything lives, how env/agent/train/runtime fit, and the core architectural rule. | You need to find code or understand structure before editing. |
| [CONVENTIONS.md](CONVENTIONS.md) | The rules every change follows (naming, two-layer annotation, config-as-control-panel, doc-keeping, RL-correctness). | Before writing or editing any code. |
| [WORKFLOWS.md](WORKFLOWS.md) | Exact commands: setup, train, watch, evaluate, play, run tests, add a reward term end-to-end. | Before running anything — folds in the command gotchas. |
| [GLOSSARY.md](GLOSSARY.md) | What project-specific terms mean here (world hash, PBRS, lux/solis, spool, legToes, rest-verdict, …). | When a term is ambiguous or unfamiliar. |
| [CHANGELOG.md](CHANGELOG.md) | What changed and why, in reverse-chronological order. | You need history/intent behind a piece of code, or you just shipped a change. |
| [OBSERVATIONS.md](OBSERVATIONS.md) | Non-obvious findings, gotchas, open discrepancies, and "watch out for" notes. | Before touching anything subtle; whenever you discover something surprising. |
| [REWARD_LOG.md](REWARD_LOG.md) | Every reward experiment: hypothesis, config, result, verdict. The evidence trail for reward tuning. | Before any reward change; whenever you ship one (a hard rule). |
| [ROADMAP.md](ROADMAP.md) | What's planned, what's open, and what's intentionally not built yet. | Before starting new work, so you build on-plan. |

`docs/personal/` holds the author's scratch (a command cheat-sheet, brainstorm notes, the
project bootstrapper); `docs/superpowers/` holds the historical per-change plans and specs.
Neither is part of this index — they are not maintained as agent-facing reference.

## How to keep these current (do this as part of your change, not after)

- **Made a code change?** Add a dated entry to [CHANGELOG.md](CHANGELOG.md) (newest at top).
- **Changed or tuned a reward?** Add an entry to [REWARD_LOG.md](REWARD_LOG.md) — this is a
  **hard rule** (CLAUDE.md): every reward version is logged with its structure and weights.
- **Added/moved/renamed a module or directory?** Update [CODE_MAP.md](CODE_MAP.md).
- **Discovered a trap, a surprising behavior, or resolved an open discrepancy?**
  Record or update it in [OBSERVATIONS.md](OBSERVATIONS.md).
- Keep entries terse and factual. Link to `file.py:line` so references stay clickable.
- These docs complement the in-code annotation layers (`<agent_context>` header blocks and
  `@TAG[id]` landmarks — see the `code-annotation` skill). Docs give the cross-file map;
  annotations give the in-file pins. Don't duplicate one in the other.

## Related agent infrastructure (not docs, but worth knowing)

- **Subagents** (`.claude/agents/`): `env-physics-engineer` (env dynamics / obs-action),
  `reward-shaper` (reward design + owns [REWARD_LOG.md](REWARD_LOG.md)), `ppo-trainer`
  (training / hyperparameters / curriculum), `evaluator-visualizer` (eval metrics / rollout
  rendering), `rl-reviewer` (read-only RL-correctness review). Route work to the agent that
  owns the module — see the routing table in `CLAUDE.md`.
- **Skills** (`.claude/skills/`): `code-annotation` (the two-layer `@TAG[id]` convention),
  `rl-debugging` (ordered diagnostic checklist when training/env misbehaves).
