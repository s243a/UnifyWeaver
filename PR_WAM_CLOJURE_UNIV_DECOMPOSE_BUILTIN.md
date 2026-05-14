# PR Title

feat(wam-clojure): add univ decompose builtin

# PR Description

## Summary

- Adds Clojure WAM lowered-emitter recognition for `=../2`.
- Implements runtime `apply-univ-solution` for decompose/read mode.
- Converts bound terms into proper Prolog lists:
  - `f(a,b) =.. [f,a,b]`
  - `a =.. [a]`
  - `42 =.. [42]`
  - `[a,b] =.. ['[|]', a, [b]]`
- Wires interpreted built-in dispatch for `"=../2"`.
- Adds lowered-emitter and runtime smoke coverage for struct, atom, number, list, and mismatch cases.

## Scope

This PR intentionally implements decompose/read mode only: the first argument must already be a bound atom, number, or compound/list term.

Compose mode, such as `T =.. [f,a,b]`, is left for a follow-up PR.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
