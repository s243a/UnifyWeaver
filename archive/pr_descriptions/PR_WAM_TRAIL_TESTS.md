# PR: feat: add trail unwinding on backtrack and harden test harness

**Title:** `feat: add WAM trail unwinding and CI-compatible test harness`

## Summary

- Implement trail-based register unwinding in the WAM runtime so backtracking correctly restores register state instead of carrying stale bindings
- Register-mutating instructions (`get_variable`, `put_variable`, `put_value`, `put_constant`) now record `trail(Key, OldValue)` entries before mutation
- `backtrack/2` diffs the current trail against the saved trail from the choice point and undoes bindings in reverse order
- Harden both `test_wam_target.pl` and `test_wam_e2e.pl` to exit with `halt(1)` on any test failure, making them CI-compatible (previously printed `[FAIL]` but exited 0)

## Test plan

- [x] All 5 WAM target compiler tests pass
- [x] Both E2E tests pass (fact execution + backtracking to second fact)
- [x] Verified exit code is 0 on success
- [x] Verified `halt(1)` path is wired correctly via `test_failed/0` dynamic predicate
- [x] No warnings (renamed `del_assoc/4` to `remove_assoc_key/4` to avoid assoc library conflict)

## Companion PR

- Education repo: `docs: document unify_* instructions in Book 17 ISA` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
