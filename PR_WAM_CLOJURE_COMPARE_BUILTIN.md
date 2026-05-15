# PR Title

feat(wam-clojure): add compare/3 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `compare/3`.
- Reuses the existing `term-compare` implementation from the standard term-ordering builtins.
- Adds runtime `apply-compare-solution`, which unifies the first argument with `<`, `=`, or `>`.
- Wires interpreted runtime dispatch for `compare/3`.
- Direct-lowers `call compare/3` in addition to `execute compare/3`.
- Hardens direct-builtin arity matching so parsed arities may be variables, integers, strings, or atoms.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`compare/3` compares the second and third arguments using the same term ordering as `@</2`, `@=</2`, `@>/2`, and `@>=/2`, then unifies the first argument with one of:

- `<`
- `=`
- `>`

The compared terms are dereferenced for inspection but are not unified or mutated.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
