# PR Title

feat(wam-clojure): lower atomic_list_concat builtins

# PR Description

## Summary

- Adds direct lowered Clojure WAM support for `atomic_list_concat/2` and `atomic_list_concat/3`.
- Implements a runtime helper that handles the bound-list concatenation direction and unifies the joined atom output.
- Adds lowered-emitter assertions and runtime smoke coverage for success, mismatch, improper-list, and unsupported unbound-list cases.

## Scope

This is a conservative parity step for the Clojure lowered WAM tier. It supports deterministic list-bound uses such as `atomic_list_concat([f,o,o], A)` and `atomic_list_concat([f,o], o, A)`. Modes requiring splitting an atom/string back into list parts remain unsupported and fail through the existing backtracking path.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
