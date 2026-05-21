# PR Title

feat(wam-clojure): render numeric sub_atom text

# PR Description

## Summary

- Extend the Clojure WAM `sub_atom/5` runtime helper to render numeric source values through `atom-number-text`.
- Extend bound `Sub` matching to accept numeric runtime values through the same text-rendering path.
- Add smoke coverage for numeric source and numeric sub arguments without widening unsupported source-unbound behavior.

## Behavior

This brings Clojure WAM `sub_atom/5` closer to the C++ WAM runtime behavior for rendered atomic text values:

- `sub_atom(A, 1, 3, 1, 234)` succeeds for runtime argument `A = 12345`.
- `sub_atom(12345, 1, 2, _, S)` succeeds for runtime argument `S = 23`.

Existing atom/string `sub_atom/5` modes remain unchanged, including conservative failure for unbound source values.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
