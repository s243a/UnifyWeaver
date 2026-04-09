# PR: feat: get_list/put_list, second-arg indexing, and peephole optimizer

**Title:** `feat: get_list/put_list, second-arg indexing, and peephole optimizer`

## Summary

- Add `get_list`/`put_list` instructions as list-specific sugar for `get_structure`/`put_structure` with functor `'./2'` — compiler emits these for `[H|T]` patterns in head and body arguments, runtime decomposes Prolog lists in read mode or constructs on heap in write mode
- Add second-argument indexing (`switch_on_constant_a2`) — when first-argument indexing fails (all variable first args) but all second arguments are atomic, the compiler indexes on A2 instead
- Add peephole optimizer post-pass (`peephole_optimize/2`) that eliminates redundant instruction sequences: `put_value Xn, Ai` + `get_variable Xn, Ai` identity pairs, duplicate consecutive `put_*` instructions, and `get_variable Xn, Ai` + `put_value Xn, Ai` pass-throughs
- Archive PR descriptions for PR #1132

## Test plan

- [x] All 7 WAM target compiler tests pass
- [x] All 18 E2E tests pass — including new tests for `get_list` decomposition with execution, peephole optimization, and `switch_on_constant_a2` emission
- [x] Verified `get_list` appears in compiled output for `e2e_head/2` and execution succeeds for `e2e_head([a,b,c], a)`
- [x] Verified `switch_on_constant_a2` emitted for `e2e_greet/2` (variable first arg, constant second arg)
- [x] Peephole optimizer runs without breaking any existing tests
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Companion PR

- Education repo: `docs: add get_list, put_list, switch_on_constant_a2 to Book 17 ISA` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
