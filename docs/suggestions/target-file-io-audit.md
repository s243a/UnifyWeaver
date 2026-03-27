# Target File I/O Audit

## Problem
All transitive closure templates use stdin for reading input facts.
This doesn't work in browser environments (SciREPL), notebooks, or
any context without a terminal.

## Current State

| Target | Stdin | File Read | File Write | Status |
|--------|-------|-----------|------------|--------|
| R | ✅ readLines(stdin) | ❌ | ❌ | stdin only |
| TypR | ❌ (uses @{ }@ seeds) | ❌ | ❌ | seeds embedded |
| Lua | ✅ io.lines() | ❌ | ❌ | stdin only |
| Python | ✅ sys.stdin | ❌ | ❌ | stdin only |
| Bash | ❌ (uses template_system) | ❌ | ❌ | different approach |
| C | ✅ scanf | ❌ | ❌ | stdin only |
| C++ | ✅ cin | ❌ | ❌ | stdin only |
| Rust | ✅ stdin().lines() | ❌ | ❌ | stdin only |
| Go | ✅ bufio.Scanner(stdin) | ❌ | ❌ | stdin only |
| Ruby | ✅ STDIN.each_line | ❌ | ❌ | stdin only |
| Perl | ✅ <STDIN> | ❌ | ❌ | stdin only |
| TypeScript | ✅ readline | ❌ | ❌ | stdin only |
| PowerShell | ✅ Read-Host | ❌ | ❌ | stdin only |
| Kotlin | ❌ | ❌ | ❌ | no I/O |
| Scala | ❌ | ❌ | ❌ | no I/O |
| Clojure | ❌ | ❌ | ❌ | no I/O |
| Jython | ✅ sys.stdin | ❌ | ❌ | stdin only |
| Elixir | ❌ | ❌ | ❌ | no I/O |
| F# | ❌ | ❌ | ❌ | no I/O |
| Haskell | ❌ | ❌ | ❌ | no I/O |

## Required Fix

Each target should support three input modes:

1. **stdin** (current) — for CLI usage
2. **file** — read facts from a file path
3. **embedded** — facts hardcoded in the generated code (like TypR's seeds)

### Template option: `input_mode`
```prolog
compile_recursive(ancestor/2, [target(lua), input_mode(file), input_file("facts.txt")], Code).
compile_recursive(ancestor/2, [target(lua), input_mode(embedded)], Code).
compile_recursive(ancestor/2, [target(lua), input_mode(stdin)], Code).  % default
```

### For SciREPL/notebook use:
- `input_mode(embedded)` generates seed statements from asserted facts
- `input_mode(file)` reads from SharedVFS or NotebookVFS path
- `input_mode(vfs)` reads from a named cell: `nb.read("family_tree", ".code")`

### Priority
High — stdin doesn't work in:
- Browser (SciREPL, any WASM runtime)
- Notebooks (Jupyter, SciREPL)
- Serverless/cloud functions
- Many testing frameworks

The TypR target already uses embedded seeds — this pattern should be
generalized to all targets.

## Implementation Plan
1. Add `input_mode` option to `compile_recursive`
2. Add `base_seed_code` (like TypR has) to all targets
3. Make `embedded` the default for notebook/WASM contexts
4. Keep `stdin` as default for CLI contexts
5. Add `file` mode with target-specific file reading
