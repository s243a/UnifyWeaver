# TypR Target — Suggested Improvements

## For Codex / Future Work

### 1. Raw-R blocks could be reduced
The current TypR target wraps most logic in `@{ ... }@` raw-R escape blocks.
This defeats the purpose of TypR's type system since the raw R code isn't
type-checked. As TypR matures (especially variadic function support and
environment/hash types), these blocks should be progressively replaced with
native TypR constructs.

**Example:** `parent_graph <- @{ new.env(hash = TRUE, parent = emptyenv()) }@;`
could become native when TypR supports environment types.

### 2. CLI entry point generation
The R target generates a CLI section (`if (!interactive()) { ... }`) but the
TypR target doesn't. Add an optional `cli(true)` flag that generates the
stdin-reading entry point in TypR syntax.

### 3. Test coverage
Add regression tests for:
- `compile_recursive(ancestor/2, [target(typr)], Code)` — transitive closure
- `compile_recursive(factorial/2, [target(typr), memo(false)], Code)` — linear recursion
- Compare TypR output vs R output for semantic equivalence

### 4. TypR upstream issues to track
- **Variadic functions**: `cat("x =", x)` doesn't work, need `cat(paste("x =", x))`
- **Braceless function bodies**: `function(x) x * x` drops the body, must use braces
- **`io.ty` not loadable**: New .ty files aren't picked up by the std generator parser
- **`default.ty` skipped**: Contains syntax the parser rejects (`{}` record types, `let` definitions mixed with `@` declarations)

### 5. Workbook generation
The "Prolog generates TypR" workbook pattern works well for demos:
1. Prolog cell: define predicates + assert facts
2. Prolog cell: `compile_recursive(pred/N, [target(typr)], Code), nb_write(...)`
3. TypR cell: execute the generated code

Consider auto-generating this workbook pattern as a SciREPL package.

### 6. Cross-cell variable persistence for TypR kernel
Currently each TypR cell runs independently — variables don't persist across
cells. This is because `executeRaw()` creates a fresh `captureR` context each
time. To fix this, the TypR kernel should:

1. Run generated R code in webR's **global environment** (not a shelter)
2. Accumulate the TypR std library preamble only on first execution
3. Skip re-emitting `Integer()`, `Character()`, etc. on subsequent cells
4. Track which definitions have been loaded to avoid redefinition errors

This would allow natural notebook workflows:
```
Cell 1 (TypR): let x <- 42;
Cell 2 (TypR): cat(x);   # ← currently fails, x not in scope
```

**Workaround:** Put all code in a single cell. The Prolog compiler cells
append queries to the generated code for this reason.
