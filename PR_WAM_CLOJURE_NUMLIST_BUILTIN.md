# PR Title

feat(wam-clojure): add numlist/3 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `numlist/3`.
- Direct-lowers `numlist/3` to a runtime helper instead of stepping through the interpreted builtin path.
- Supports integer `Low`, integer `High`, and output-list unification.
- Fails cleanly for `Low > High` and unbound/non-integer bounds.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`numlist/3` accepts integer lower and upper bounds and unifies the third argument with the proper list of integers from `Low` through `High`, inclusive.

This initial Clojure WAM implementation is conservative: it supports the common `(+Low, +High, ?List)` mode and fails for unsupported modes or invalid bounds rather than raising ISO-style errors.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
