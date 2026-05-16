# PR Title

feat(wam-clojure): add reverse/2 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `reverse/2`.
- Reuses the existing proper-list conversion and `list->term` runtime helpers.
- Direct-lowers `reverse/2` to a runtime helper instead of stepping through the interpreted builtin path.
- Fails cleanly for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`reverse/2` accepts a proper list in the first argument and unifies the second argument with the reversed list.

This initial Clojure WAM implementation is deterministic and conservative: it supports the common `(+List, ?Reversed)` mode and fails for unbound first arguments rather than generating lists relationally.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
