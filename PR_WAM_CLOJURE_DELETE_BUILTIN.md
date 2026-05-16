# PR Title

feat(wam-clojure): add delete/3 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `delete/3`.
- Uses the existing `term-identical?` helper so removal follows identity semantics, matching the target list-builtin plan.
- Reuses the existing proper-list conversion and `list->term` helpers.
- Direct-lowers `delete/3` to a runtime helper instead of stepping through the interpreted builtin path.
- Fails cleanly for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`delete/3` accepts a proper list in the first argument, removes all elements identical to the second argument, and unifies the third argument with the resulting list.

This initial Clojure WAM implementation is deterministic and conservative: it supports the common `(+List, +Elem, ?Rest)` mode and fails for unbound input lists rather than generating lists relationally.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
