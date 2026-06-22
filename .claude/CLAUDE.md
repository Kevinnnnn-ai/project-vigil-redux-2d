# Agent Rules

## While Working in `.`

- Read `./.claude/AGENTS.md` for quickstart context, project structure, code conventions, testing rules, and security guidelines. (Project specific entry point)
- Read `./.claude/agent-memory/*` for a decision log, list of concurrent project facts, and list of important notes left by other agents.

### While Working With `./.claude/agent-memory/context.md`

- Keep and update a running list of durable facts about the project, including noted future goals and the stated direction of the project.
- It should not be a history or log of changing facts; only keep concurrent information.

### While Working With `./.claude/agent-memory/decisions.md`

- This should be a log of both agentic and human choices made and why.
- Append logs with each decision.

### While Working With `./.claude/agent-memory/notes.md`

- This should be a list of running observations and notes.
- Put information in here that would be beneficial for future agents and sessions to know.
- Remove irrelevant entries that no longer apply to this repository.

## While Working in `./stdout/*`

- Do not interfere with any pre-existing data in here.
- Ensure test data generates in here only.
- Ensure that test data will not interfere with any pre-existing files.
- Clean up any testing data.

## General Rules

- Understand the codebase and its conventions—the codebase is the source of truth.
- Always prioritize reading the codebase to understand the project.
- Write performant code.
- Use `gh` CLI tool for fetching data from `github.com`.
- This is a Windows system, not a macOS or Linux Bash environment.

# Hard Rules

## Always Do

- Always work out of a local virtual environment—usually `.env.local/`.
- Always try to commit changes in small amounts, following security rules listed in `AGENTS.md`.
- Always update documentation.
- Always follow code conventions (`AGENTS.md`).
- Always follow security rules (`AGENTS.md`).
- Alwats follow testing guidelines (`AGENTS.md`).

## Never Do

- Never push commits.
- Never edit this file—the `CLAUDE.md` or `CLAUDE.local.md`.
- Never edit `AGENTS.md`.
- Never edit `settings.json` or `settings.local.json`.