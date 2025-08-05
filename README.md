# Stationeers IC10 Compiler (SICC)

SICC is yet another compiler for Stationeers IC10. It compiles to minimize code size.

**This is a WIP and does not yet work; see main.py, and run python main.py to see what it currently does**.

This works by writing python code which gets **traced** and then compiled into IC10.

The python code you will write will approximately look like:

```python
@sicc.wrap_main
def main():
    # sicc will run your function once;

    for x in ["door1", "door2"]:  # python control flow will execute at "compile time"
        # let sicc know you want an operation by *staging* it (calling it once)
        GlassDoors(x).On = True

        # spedcial control flow to condition for stuff at runtime
        with if_(GlassDoors(x).Lock):
            GlassDoors(x).On = False

    # the above will get *traced* as:

    # GlassDoors("door1").On = True
    # if GlassDoors("door1").Lock:
    #     GlassDoors("door1").On = False
    # GlassDoors("door2").On = True
    # if GlassDoors("door2").Lock:
    #     GlassDoors("door2").On = False

    # and then the compiler will compile this into assembly
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
