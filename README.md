# Stationeers IC10 Compiler (SICC)

[![CI](https://github.com/Alan-Chen99/sicc/actions/workflows/ci.yml/badge.svg)](https://github.com/Alan-Chen99/sicc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/alan-chen99/sicc/graph/badge.svg?token=X8IE0XQ7NW)](https://codecov.io/github/alan-chen99/sicc)

SICC is yet another compiler for Stationeers IC10. It compiles to minimize code size.

This works by writing python code which gets **traced** and then compiled into IC10.
See [examples/explained.py](examples/explained.py) for how this works.

**This is a WIP; currently output may not be correct or even valid. API is subject to change without notice.**

## Install (more options will be supported later)

Install [uv](https://github.com/astral-sh/uv) and run

```bash
git clone https://github.com/Alan-Chen99/sicc-template.git
cd sicc-template
uv sync
# optional: upgrade to latest git version
# uv lock --upgrade
source ./.venv/bin/activate
python main.py
# or: sicc main.py
```

## Project Goals

- free to write high-level constructs with bloat, and compile those away
- be able to utilize the whole instruction set, including conditional function calls
- efficient function calls without push and pop, by using another register as ra if ra is not available
- reorder instructions and blocks to save lines

## Internals

This compiler uses Static single-assignment form (SSA), but it does not have phi instructions; instead it has mutable variables (MVar) representing varaibles that opts out of SSA invariant. MVar must be first converted from/to Var before used; The register allocator (not created yet) will tries its best to allocate read/write mvar operands to the same register. MVar are printed with an underline in the current main.py output.

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies; so it is recommended to use uv to install dependencies locked in `uv.lock`

This codebase is statically typed; this works with pyright only.
