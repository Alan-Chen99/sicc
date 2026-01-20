# Agent Instructions

This file is for AI coding agents working in this repository.

## Project

- **Name**: Stationeers IC10 Compiler (SICC)
- **Language**: Python (requires **3.13+**; see `pyproject.toml`)
- **Package**: `sicc/`
- **CLI**: `sicc` (entrypoint: `sicc._cli:main`)

## Setup

- Default to non-uv commands.
- You are typically already in a activated venv with the project installed editable. Default to running without `uv run` prefix.
  - Reason: tooling that uses global caches (like `uv`) may touch paths outside the workspace.

## Common Commands

- Run sicc: `sicc tests/<file>.py -p <program>`
- Run sicc with verbose: `sicc tests/<file>.py -p <program> -v`
- Tests: `pytest`
- Coverage: `pytest --cov`
- Format: `black .` and `isort .`
- Type check: `basedpyright`

## Testing

- Run `basedpyright` before tests when possible (errors do not block running tests).
- Agents should not introduce any new `basedpyright` errors and must fix them before returning results.
- Many tests write compiler output snapshots to `tests/outputs/*.out`.
- These outputs are tracked for review, but **changes are manually verified** (there are **no additional automated checks/assertions** on the output text).
- If output changes, treat the resulting diff in `tests/outputs/` as part of the review surface.
  - Use "readable output" from `sicc tests/<file>.py -p <program>` to check programs rather than raw output written to files
- Unless otherwise specified, the only forms of tests you write are programs that produces program output

## Conventions

- Keep changes small and focused; avoid drive-by refactors.
- Changes must pass `basedpyright` (strict type checking).
- `# pyright: ignore[...]` comments are allowed when needed; prefer narrow, line-level ignores over file-wide disables.
- Use `assert` for internal invariants/bugs; use `raise TypeError` for incorrect calls from users/callers.
- Underscore-prefixed functions are private; default to `_name` unless the function is expected to be used from another module.
- Add/update tests in `tests/` for behavior changes.
- Update `README.md` and/or `examples/` when changing public APIs or CLI behavior.
- Agents should not run formatters (`black` / `isort`) unless explicitly asked.
