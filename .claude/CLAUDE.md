# Agent Rules

## While Working in `.`

- Always read `./.claude/AGENTS.md` for quickstart context, project structure, code conventions, testing rules, and security guidelines. (Project specific entry point)
- Always read `./.claude/agent-memory/*` for a decision log, list of concurrent project facts, and list of important notes left by other agents.
- Always read `./docs/CHANGELOG.md` for a chronological list of what changed and why.
- Always read `./docs/OBSERVATIONS.md` for a list of running important observations made by other session agents.
- Always read `./docs/REWARD_LOG.md` for a list of changes made to the reward system.

## While Working With `./.claude/agent-memory/context.md`

- Always keep and update a running list of durable facts about the project, including noted future goals and the stated direction of the project.
- It should not be a history or log of changing facts; only keep concurrent information.

## While Working With `./.claude/agent-memory/decisions.md`

- Always append logs with each decision.
- This should be a log of both agentic and human choices made and why.

## While Working With `./.claude/agent-memory/notes.md`

- Always put information in here that would be beneficial for future agents and sessions to know.
- Always remove irrelevant entries that no longer apply to this repository.
- This should be a list of running observations and notes.

## While Working in `./stdout/*`

- Never interfere with any pre-existing data in here.
- Always ensure test data generates in here only.
- Always ensure that test data will not interfere with any pre-existing files.
- Always clean up any testing data.

## While Working in `./docs/CHANGELOG.md`

- Never interfere with the instructions laid out.
- Always follow entry format.
- Always maintain this changelog with each major change made to the repository.

## While Working in `./docs/REWARD_LOG.md`

- Never interfere with the instructions laid out.
- Always follow entry format.
- Always maintain this reward log with each major change made to the reward system.

## General Rules

- Always understand the codebase and its conventions—the codebase is the source of truth.
- Always prioritize reading the codebase to understand the project.
- Always write performant code.
- Always use `gh` CLI tool for fetching data from `github.com`.
- This is a Windows system, not a macOS or Linux Bash environment.

# Hard Rules

## Always Do

- Always work out of a local virtual environment—usually `.env.local/`.
- Always try to commit changes in small amounts, following security rules listed in `AGENTS.md`.
- Always update documentation.
- Always follow code conventions (`AGENTS.md`).
- Always follow security rules (`AGENTS.md`).
- Always follow testing guidelines (`AGENTS.md`).

## Never Do

- Never push commits.
- Never edit this file—the `CLAUDE.md` or `CLAUDE.local.md`.
- Never edit `AGENTS.md`.
- Never edit `settings.json` or `settings.local.json`.