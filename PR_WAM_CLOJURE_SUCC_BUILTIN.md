# PR Title

feat(wam-clojure): add succ/2 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `succ/2`.
- Direct-lowers `succ/2` to a runtime helper instead of routing through the interpreted builtin path.
- Supports conservative integer forward and backward modes.
- Fails cleanly for negative predecessors, non-positive successor back mode, and unbound/unbound mode.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`succ/2` succeeds when the second argument is the successor of the first argument.

The implementation supports:

- `succ(+N, ?M)` when `N` is a nonnegative integer.
- `succ(?N, +M)` when `M` is a positive integer.

Unsupported or invalid modes fail cleanly rather than raising ISO-style errors.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
