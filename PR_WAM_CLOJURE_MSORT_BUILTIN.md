# PR Title

feat(wam-clojure): add msort/2 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `msort/2`.
- Reuses the existing standard term-order comparator used by `sort/2`, `compare/3`, and the term-order predicates.
- Preserves duplicate terms while returning the output list in standard term order.
- Fails cleanly for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`msort/2` accepts a proper list in the first argument and unifies the second argument with a list ordered by standard term order. Unlike `sort/2`, duplicate terms are retained.

The implementation is deterministic and does not enumerate permutations. It inspects and sorts existing list items without binding variables in the input list.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
