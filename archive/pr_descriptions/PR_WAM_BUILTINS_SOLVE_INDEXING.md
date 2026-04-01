# PR: feat: built-in predicates, solve_wam, first-arg indexing, and clause_body_analysis refactor

**Title:** `feat: WAM builtins, solve_wam, indexing, and clause_body_analysis refactor`

## Summary

- Add `builtin_call` instruction for arithmetic (`is/2`, `+`, `-`, `*`, `/`, `//`, `mod`, `abs`), comparison (`>/2`, `</2`, `>=/2`, `=</2`, `=:=/2`, `=\=/2`), equality (`==/2`, `\==/2`, `\=/2`), type checks (`atom/1`, `number/1`, `integer/1`, `float/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1`), and control (`true/0`, `fail/0`, `!/0`, `\+/1`)
- Compiler recognizes builtins and emits `builtin_call` + `proceed` instead of `call`/`execute`
- Arithmetic evaluation supports heap-ref dereferencing for `put_structure`-built expressions
- Add `solve_wam/4` API: executes a query and extracts variable bindings as `Name=Value` pairs from the final register state — supports Prolog variables in queries
- Add first-argument indexing via `switch_on_constant`: compiler generates index tables for multi-clause predicates with all-atomic first arguments; runtime dispatches directly to matching clause labels
- Refactor `is_builtin_pred/2` to delegate to `clause_body_analysis:is_guard_goal/2` for guard detection instead of maintaining a hardcoded list — automatically picks up any predicate the shared analysis classifies as a guard (comparisons, type checks)

## Test plan

- [x] All 7 WAM target compiler tests pass
- [x] All 12 E2E tests pass — including new tests for `>/2`, `is/2`, `atom/1`, `solve_wam`, and `switch_on_constant`
- [x] Verified `e2e_double(3, 6)` succeeds via `is/2` with heap-ref arithmetic
- [x] Verified `solve_wam` returns `['City'=paris]` for `e2e_capital(france, City)`
- [x] Verified `switch_on_constant` appears in compiled output and execution still works
- [x] Verified `e2e_is_atom(hello)` succeeds via delegated type check builtin
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Companion PR

- Education repo: `docs: add builtin_call, switch_on_constant to Book 17 ISA` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
