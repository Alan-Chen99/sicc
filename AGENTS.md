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
- Regenerate `sicc/devices.py` from `scripts/game_data/stationeers_logic.json`: `python scripts/build_types.py`
- Re-dump `stationeers_logic.json` from the game after a Stationeers update: see `scripts/game_data/dump.sh`

## Testing

- Run `basedpyright` before tests when possible (errors do not block running tests).
- Agents should not introduce any new `basedpyright` errors and must fix them before returning results.
- Running `pytest` writes compiler output snapshots to `tests/outputs/*.out`.
- These outputs are tracked for review, but **changes are manually verified** (there are **no additional automated checks/assertions** on the output text).
- **`tests/outputs/*.out` contains outputs of the current code, not reference outputs.**
- Any output different than those on `origin/master` should be manually reviewed.
  - Outputs should always be compared against ones on `origin/master` rather than any local branch.
  - After making changes, regenerate `tests/outputs/*` with `pytest`.
  - When there are differences, use "readable output" from `sicc tests/<file>.py -p <program>` to check each changed program manually.
  - If there are local changes relevant to a test, do not assume identical outputs to `origin/master` imply correctness.
- For "verify outputs" requests, assume the user has already run `pytest` and committed any regenerated `tests/outputs/*.out` files. In this case:
  - Run `pytest` or sometimes just `pytest tests/<file>.py` to confirm outputs are up to date.
  - Diff against `origin/master` to identify differences (e.g. `git diff origin/master -- tests/outputs`).
  - If there are differences, use "readable output" (`sicc tests/test_transpile.py -p <program>`) to re-check each changed program.
  - Diagnostics and warnings emitted by `sicc` and what a user get when they call `print` (all of which are produced by `sicc ... -p ...` on default verbose setting) must also be checked for any test case whose output differs.
  - Any test case with different output than `origin/master` must be completely reverified.
- Unless otherwise specified, the only forms of tests you write are programs that produce program output.

## Conventions

- Keep changes small and focused; avoid drive-by refactors.
- Keep files unrelated to the requested operation as-is (avoid incidental edits/additions).
- Changes must pass `basedpyright` (strict type checking).
- `# pyright: ignore[...]` comments are allowed when needed; prefer narrow, line-level ignores over file-wide disables.
- Use `assert` for internal invariants/bugs; use `raise TypeError` for incorrect calls from users/callers.
- Underscore-prefixed functions are private; default to `_name` unless the function is expected to be used from another module.
- Add/update tests in `tests/` for behavior changes.
- Update `README.md` and/or `examples/` when changing public APIs or CLI behavior.
- Agents should not run formatters (`black` / `isort`) unless explicitly asked.
