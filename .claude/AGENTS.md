# Project Vigil Redux 2D



## Quickstart and Commands



## Project Structure



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
