# PR: feat: recursive E2E, nested compound unification, and write mode

**Title:** `feat: recursive E2E, nested compound unification, and write mode`

## Summary

- Add E2E test for recursive `ancestor/2` — exercises multi-clause backtracking, recursive calls, and Yi registers through the full compile-then-execute pipeline
- Add E2E and compiler tests for compound head unification (`get_structure` + `unify_constant`)
- Compiler now emits recursive `get_structure` + `unify_*` sequences for nested compound sub-arguments in head matching (previously emitted bare `unify_variable` placeholder)
- Runtime `get_structure` now operates in dual mode: **read mode** (matches existing compound terms) and **write mode** (constructs structures on the heap when the register is unbound)
- `unify_variable`, `unify_value`, `unify_constant` all handle both read mode (match from `unify_ctx`) and write mode (push to heap via `write_ctx`)
- Fix `backtrack` to preserve choice points on the stack — `trust_me`/`retry_me_else` handle popping, not backtrack itself (was causing nested backtracking across predicates to fail)
- Fix `parse_arg` to convert numeric strings to numbers so `unify_constant 255` correctly matches integer `255`
- Archive PR descriptions for PRs #1105, #1111, #1113

## Test plan

- [x] All 6 WAM target compiler tests pass (facts, single clause, recursion, put_structure, compound head, module)
- [x] All 5 E2E tests pass (fact execution, backtracking, grandparent rule, recursive ancestor, compound head)
- [x] Verified `ancestor(alice, charlie)` succeeds via recursive path: alice→bob (parent), bob→charlie (parent)
- [x] Verified `color(rgb(255,0,0), red)` succeeds via compound head matching with `get_structure rgb/3` + `unify_constant`
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Companion PR

- Education repo: `docs: document read/write mode for get_structure and unify_* in Book 17` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
