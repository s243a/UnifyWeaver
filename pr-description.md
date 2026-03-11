# Add R Language Target

## Summary

- Add a complete R language target with 58 bindings across 7 categories, recursive pattern compilation, data.table integration, and fixpoint generator pipelines
- Refactor advanced recursion compilers from hard-coded bash to target-polymorphic dispatch using multifile predicates, enabling any target to register its own code generators
- Fix several R code generation bugs discovered during Rscript testing (invalid identifiers, fold direction, early loop return, memo key crashes, coercion warnings)

## What's New

### R Target (`r_target.pl`)
- Full `compile_predicate_to_r/3` with single-clause and multi-clause (if/else if chain) support
- `compile_facts_to_r/3` for fact-only predicates (generates `get_all_`, `stream_`, `contains_` helpers)
- `compile_r_pipeline/3` with sequential and fixpoint generator evaluation modes
- Tree recursion R generators registered via `tree_recursion:compile_tree_pattern/6` multifile delegation

### R Bindings (`r_bindings.pl`) — 58 bindings, 7 categories
| Category | Count | Examples |
|----------|-------|---------|
| Built-ins | 4 | print, cat, paste, nchar |
| Math | 15 | abs, sqrt, floor, ceiling, round, log, exp, sin, cos, tan |
| String | 12 | gsub, sub, toupper, tolower, trimws, substr, strsplit, sprintf, grep, grepl |
| Type Conversion | 7 | as.numeric, as.integer, as.character, as.logical, is.numeric, is.character, is.logical |
| Vector/List | 9 | c, rev, sort, unique, which, seq, rep, head, tail |
| File I/O | 7 | file.exists, file.path, dirname, basename, readLines, writeLines, normalizePath |
| DataFrame | 4 | data.table, setDT, merge, dcast |

### Recursion Pattern Support (all 6 patterns)
| Pattern | Bash | R |
|---------|------|---|
| Tail recursion | Existing | New |
| Linear recursion | Existing | New |
| Tree recursion | Refactored (multifile) | New |
| Mutual recursion | Existing | New |
| Multicall linear | Existing | New |
| Direct multicall | Existing | New |

### Architecture: Multifile Target Delegation
`tree_recursion.pl` now uses `compile_tree_pattern/6` as a multifile predicate. Targets register their own code generators instead of the core module containing if/else branches for each language. This pattern can be extended to the other 5 recursion modules.

## Bug Fixes
- **Invalid R identifiers**: Prolog internal variables (`_21562`) prefixed with `v` to produce valid R names
- **Wrong fold direction**: `Reduce(..., right=TRUE)` replaced with left fold for accumulator patterns
- **Empty list memo key**: `paste(c(), collapse=",")` produces `""` which crashes R environments; now falls back to `"__empty__"`
- **Early loop return**: `return()` inside `for` loop in tail recursion exited after first element
- **Coercion warnings**: Wrapped `as.numeric()` in `suppressWarnings` with NA check
- **Mutual recursion dispatch**: `commandArgs(TRUE)` returns strings; now wrapped in `as.numeric()` for arithmetic
- **Singleton variable**: Fixed `init_component(Name, Config)` warning in `data_table_component.pl`

## Test Plan
- [x] `test_r_bindings/0` — 58 bindings registered, all categories pass
- [x] `test_r_pipeline/0` — sequential and generator pipelines compile
- [x] `test_tree_recursion/0` — bash and R tree_sum generated via multifile dispatch
- [x] `test_mutual_recursion/0` — R even/odd generated
- [x] `test_multicall_linear/0` — R fibonacci generated
- [x] `Rscript tree_sum.R` → 6
- [x] `Rscript even_odd.R is_even 4` → TRUE
- [x] `Rscript fib_multicall.R 10` → 55
- [x] `Rscript factorial.R` → 720
- [x] `Rscript sum_list.R` → 15
- [x] `Rscript count_items.R` → 3
- [x] `Rscript list_length.R` → 5
- [x] Bash targets verified no regressions

## Files Changed (11 files, +1948 / -52)
- `src/unifyweaver/targets/r_target.pl` — new R target module
- `src/unifyweaver/bindings/r_bindings.pl` — new R bindings
- `src/unifyweaver/targets/r_runtime/data_table_component.pl` — data.table component
- `src/unifyweaver/core/target_registry.pl` — register R target
- `src/unifyweaver/core/advanced/tree_recursion.pl` — multifile refactor
- `src/unifyweaver/core/advanced/tail_recursion.pl` — R support + bug fixes
- `src/unifyweaver/core/advanced/linear_recursion.pl` — R support + bug fixes
- `src/unifyweaver/core/advanced/mutual_recursion.pl` — R support
- `src/unifyweaver/core/advanced/multicall_linear_recursion.pl` — R support
- `src/unifyweaver/core/advanced/direct_multi_call_recursion.pl` — R support
- `docs/BINDING_MATRIX.md` — updated R coverage
