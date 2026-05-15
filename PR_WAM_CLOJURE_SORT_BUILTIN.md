# PR Title

feat(wam-clojure): add sort/2 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `sort/2`.
- Reuses the existing `term-compare` ordering used by `@</2`, `@=</2`, `@>/2`, `@>=/2`, and `compare/3`.
- Converts proper input lists to runtime items, sorts them, removes duplicate terms, rebuilds a WAM list, and unifies the output.
- Fails cleanly for improper or unbound input lists.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`sort/2` accepts a proper list in the first argument and unifies the second argument with a duplicate-free list ordered by standard term order.

The implementation is deterministic and does not enumerate list permutations. It inspects and sorts existing terms without binding variables in the input list.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
