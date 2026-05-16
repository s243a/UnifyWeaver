# PR Title

feat(wam-clojure): add select/3 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `select/3`.
- Direct-lowers `select/3` to a runtime helper instead of stepping through the interpreted builtin path.
- Reuses the existing multi-solution result-binding path to enumerate one-removal list candidates.
- Supports conservative proper-list mode and fails cleanly for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`select/3` accepts a proper list in the second argument, removes one element at a time, and unifies the first argument with the removed element and the third argument with the remaining list.

This initial Clojure WAM implementation supports the common `(?Elem, +List, ?Rest)` mode. It intentionally does not generate candidate input lists from an unbound second argument.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
