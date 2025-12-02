<!-- Auto-generated guidance for AI coding assistants working in this repository -->
# Copilot / AI assistant instructions (concise)

Repository snapshot
- Single top-level source file: `main.py` (currently empty).

Goal for an AI contributor
- Be minimal and explicit: ask before making structural changes beyond adding small modules.
- Prefer to produce small, testable edits and a short justification for the change.

What to inspect first
- Open `main.py` and report whether it contains runnable code, a CLI entrypoint, or TODOs.
- If `main.py` is empty (as currently), ask the repo owner whether to scaffold a package, add a script, or implement a specific feature.

How to modify code here (concrete rules)
- Ask a clarifying question before creating new top-level packages or multiple files.
- When adding code, include a one-line summary at the top of the change describing intent (e.g., "Add minimal CLI to load config").
- Keep changes focused to a single responsibility (one feature or fix per PR/patch).

Commands and workflows (what an assistant can run locally)
- Run the script (if implemented): `python3 main.py` from the repository root.
- Use a virtualenv if installing deps: `python3 -m venv .venv && source .venv/bin/activate`.

Testing and verification
- This repo currently has no test files. If you add tests, put them under a `tests/` directory and use `pytest`.
- Before asking the user to run tests, provide the exact commands and brief rationale for added tests.

Conventions & preferences (repo-specific)
- Minimal, explicit commits: include a short subject and one-line body explaining the why.
- Avoid making assumptions about runtime or dependencies; ask if you need to add dependencies or a `requirements.txt`.

Integration points & external dependencies
- No external integrations or dependencies are discoverable in the repository root. If you add integrations, document them in a short `README.md` or in the PR body.

When to escalate to the human maintainer
- Any change that creates or modifies persistent state (DBs, remote services, new credentials).
- Adding or bumping third-party dependencies.
- Large refactors or introducing an application structure (packages, entrypoints) — ask before proceeding.

Examples (what to do next)
- If asked to implement a basic script: propose the exact file layout, a one-paragraph plan, and a minimal runnable proof-of-concept in `main.py`.
- If asked to scaffold a package: propose `package/`, `__main__.py`, and a `tests/` folder; wait for approval before applying.

If anything here is unclear, ask one targeted question rather than making a large change.
