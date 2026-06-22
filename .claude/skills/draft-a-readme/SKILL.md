---
name: draft-a-readme
description: This skill can only be triggered through the slash command call—`/draft-a-readme`. This skill writes a README file from scratch in the root directory of a project given a project's context, skill instructions, and a followable workflow.
---

# Task

Draft a concise, primarily bulleted, straightforward, and accurate one-shot `README.md` that is easy to understand and follow—even for someone who is new to the project.

# Input

- The only input is a prompt asking the user if they would like the current `README.md` to be overwritten—given there is already one.
- Trigger upon a `/draft-a-readme` call.

# Output

Output a `README.md` in the project's root directory, only overwriting an existing `README.md` if the user confirms that the current one should be overwritten.

- **Main Header** — Aligned center; project title (H1); description of the project and its contextualized purpose (1-2 brief sentences); `https://img.shields.io/badge` badges with `?style=` (ordered: language version coverage, listed requirement libraries, license, and build status); `---` to separate the header from the `README.md` body.
    - Past the header, separate each section only with "`<br>`" for better readability.
- **Table of Contents** — Titled "Ⅰ • Table of Contents"; list and link every header (`Ⅱ` onward) in the `README.md` structure, each as a GitHub anchor (lowercase, spaces and `•` collapsed to `-`).
- **Features** — Titled "Ⅱ • Features"; lists each notable capability of the project as a bullet, leading with a bold name followed by a one-line explanation grounded in the actual source.
- **Demonstration** — Titled "Ⅲ • Demonstration"; shows what running the project looks like—representative console output, generated artifacts, or sample results—in fenced code blocks, with brief prose tying each block to what produced it.
- **Quick Start** — Titled "Ⅳ • Quick Start"; the shortest path from clone to a working run, as a single annotated code block; call out any destructive or long-running step in a **`Note`**.
- **Installation** — Titled "Ⅴ • Installation"; subsections for **`Requirements`** (runtime/language version), **`Dependencies`** (a table of library, version, and role drawn from the project's manifest), and **`Steps`** (clone, environment, install; as a code block).
- **Usage** — Titled "Ⅵ • Usage"; one subsection per common task, each with the exact command or code snippet and a sentence on what it does; state where commands are run from.
- **Configuration** — Titled "Ⅶ • Configuration"; tables of the real tunable constants/settings (name, default, meaning) with a link to the file each lives in.
- **Reference** — Titled "Ⅷ • Reference"; the project layout as an annotated tree, key entry points, an at-a-glance description of how the project works, and any external resources it depends on.
- **License** — Titled "Ⅸ • License"; state the license from a `LICENSE` file if present, otherwise say none is distributed and that rights are reserved.
- **Authors** — Titled "Ⅹ • Authors"; list authors/maintainers with a link to their profile, derived from git history or the manifest.
- **Contact** — Titled "Ⅺ • Contact"; repository link and where to report issues.
- **Footer** — Separated from the `README.md` body via a `---`; it should say, "*Last Updated: `<current date>`*".

# Instructions

1. Get the current `README.md` from the project root.
    - If it is empty or does not exist, proceed to overwrite it.
    - If it has content, prompt the user to view it and decide whether to overwrite it. Stop if they decline.
2. Read [references/reference-README.md](references/reference-README.md) to internalize the exact format to reproduce.
3. Read the local repository to understand the project: its manifest/dependency file, entry points, source modules, configuration constants, and any existing docs. Prefer reading source over guessing.
4. Draft the `README.md` from scratch in the project root, following the proposed structure and the format of the reference.

# Never Do

- Never make up content.
- Never abbreviate a term without first stating what it stands for.
- Never copy facts, names, versions, or paths from the reference example into the output unless it's truth.
- Never overwrite a non-empty `README.md` without explicit user confirmation.
- Never put spaces between em-dashes (e.g., "This is an em-dash—a piece of punctuation.") unless it's used in place of colon to denote listings.