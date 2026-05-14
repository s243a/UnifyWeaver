# PR Title

feat(wam-clojure): add standard term ordering builtins

# PR Description

## Summary

- Adds Clojure WAM lowered-emitter support for `@</2`, `@=</2`, `@>/2`, and `@>=/2`.
- Adds runtime `term-compare` plus predicate helpers for less-than, less-or-equal, greater-than, and greater-or-equal.
- Wires interpreted runtime dispatch for all four standard term-ordering builtins.
- Adds lowered-emitter tests for each builtin.
- Adds runtime smoke coverage for atoms, numbers, compounds, variables, equality boundaries, and non-binding behavior.

## Semantics

The runtime comparison follows a conservative standard-order policy:

- Variables and unbound placeholders sort before numbers.
- Numbers sort before atoms.
- Atoms sort before compound terms.
- Atoms compare by deinterned atom name instead of intern ID.
- Compound terms compare by arity, then functor name, then recursive argument comparison.

These builtins inspect current terms after dereferencing but do not unify or mutate bindings.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
