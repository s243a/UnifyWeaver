# PR: feat: list builtins, findall_wam, and structure/term indexing

**Title:** `feat: list builtins, findall_wam, and structure/term indexing`

## Summary

- Add list builtins `member/2`, `append/3`, `length/2` as native runtime operations that delegate to Prolog's list predicates; compiler recognizes them via `is_builtin_pred` and emits `builtin_call`
- Add `findall_wam/4` API that collects all solutions by iteratively running `run_loop`, extracting bindings, then backtracking via remaining choice points until exhausted
- Add `switch_on_structure` indexing for predicates with all-compound first arguments — indexes on functor/arity to jump directly to matching clauses
- Add `switch_on_term` for predicates with mixed constant/structure first arguments — type-based dispatch to the appropriate constant or structure index table
- Compiler `classify_first_args` analyzes first arguments across clauses and selects the optimal indexing strategy (constant, structure, or term-based)
- Archive PR descriptions for PR #1124

## Test plan

- [x] All 7 WAM target compiler tests pass
- [x] All 15 E2E tests pass — including new tests for `findall_wam` (2 solutions), `member/2`, and `switch_on_structure`
- [x] Verified `findall_wam` returns 2 binding sets for `e2e_capital/2` (france/paris + germany/berlin)
- [x] Verified `member(b, [a,b,c])` succeeds via list builtin
- [x] Verified `switch_on_structure` appears in compiled output for `e2e_shape_area/2` with compound heads `circle/1` and `rect/2`
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Companion PR

- Education repo: `docs: add switch_on_structure and switch_on_term to Book 17 ISA` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
