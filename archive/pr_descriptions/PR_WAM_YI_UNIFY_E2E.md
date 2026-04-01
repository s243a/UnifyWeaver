# PR: feat: Yi permanent variables, unify_* runtime, and E2E rule test

**Title:** `feat: Yi permanent variables, unify_* runtime, and E2E rule test`

## Summary

- Implement automatic permanent variable (Yi) allocation in the WAM compiler — variables that survive across `call` boundaries are detected and assigned Yi registers stored in the environment frame, while temporaries remain in Xi registers
- Emit `allocate` before head instructions so `get_variable Yi, Ai` can write directly to the environment frame; place `deallocate` after `put_value Yi` reads but before `execute` for correct TCO
- Add runtime `step_wam` clauses for `get_structure`, `unify_variable`, `unify_value`, and `unify_constant` enabling compound head term matching via a `unify_ctx` stack mechanism
- Add Yi register support in the runtime via `get_reg`/`put_reg` helpers that dispatch to the environment frame for Y-prefixed registers
- Extend `get_constant` and `get_value` to perform unification with unbound variables (not just strict equality), enabling rule execution with variable arguments
- Add E2E test that compiles `parent/2` facts + `grandparent/2` rule into WAM and executes `grandparent(alice, charlie)` through the full pipeline

## Test plan

- [x] All 5 WAM target compiler tests pass (facts, single clause, recursion, put_structure, module)
- [x] All 3 E2E tests pass — including new `test_wam_rule_execution` for grandparent
- [x] Verified compiled output shows correct Yi allocation: `get_variable Y2, A2` in head, `put_variable Y1, A2` / `put_value Y1, A1` / `put_value Y2, A2` in body
- [x] Verified `deallocate` appears after Yi reads, before `execute`
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Companion PR

- Education repo: `docs: update Book 17 for Yi permanent variables and correct WAM output` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
