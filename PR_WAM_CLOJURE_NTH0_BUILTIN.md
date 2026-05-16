# PR Title

feat(wam-clojure): add nth0/3 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `nth0/3`.
- Reuses the existing proper-list conversion and unification helpers.
- Direct-lowers `nth0/3` to a runtime helper instead of stepping through the interpreted builtin path.
- Fails cleanly for negative, out-of-range, unbound, or non-integer indexes and for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`nth0/3` accepts a non-negative integer index in the first argument and a proper list in the second argument, then unifies the third argument with the item at that zero-based index.

This initial Clojure WAM implementation is deterministic and conservative: it supports the common `(+Index, +List, ?Elem)` mode and fails for unbound index/list arguments rather than enumerating indexes or generating lists relationally.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
