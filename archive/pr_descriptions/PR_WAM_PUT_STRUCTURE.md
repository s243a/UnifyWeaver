# PR: feat: add put_structure compound term support and document CP safety

**Title:** `feat: add put_structure compound term support and CP safety docs`

## Summary

- Implement `put_structure`/`set_variable`/`set_value`/`set_constant` in the WAM compiler and runtime, enabling compound term construction in body goals (e.g., `foo(pair(X, done))`)
- Add `compile_set_arguments/4` helper to emit `set_*` sequences for compound sub-arguments
- Document the CP (Continuation Pointer) safety invariant: `call` safely overwrites CP because `allocate` always precedes it in multi-goal bodies, and single-goal bodies use `execute` instead
- Add `test_wam_put_structure` test covering compound body argument compilation

## Test plan

- [x] All 5 existing WAM target tests pass
- [x] Both E2E tests pass (fact execution + backtracking)
- [x] New `test_wam_put_structure` verifies `put_structure pair/2` appears in compiled output for `test_wrap/1`
- [x] Verified compiled output manually: `put_structure pair/2, A1` / `set_value X1` / `set_constant done` sequence is correct

## Companion PR

- Education repo: `docs: document put_structure and set_* instructions in Book 17` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
