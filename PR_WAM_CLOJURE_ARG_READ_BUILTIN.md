# PR Title

feat(wam-clojure): add read-mode arg builtin

# PR Description

## Summary

- Adds Clojure WAM lowered-emitter recognition for `arg/3`.
- Implements runtime `apply-arg-solution` for read-mode `arg(+Index, +Term, ?Arg)`.
- Wires interpreted built-in dispatch for `"arg/3"`.
- Adds lowered-emitter and runtime smoke coverage for compound, list, out-of-bounds, and non-compound cases.

## Scope

This PR implements read-mode argument extraction only. It expects `Index` to be a positive integer and `Term` to be a compound/list structure, then unifies the selected 1-based argument with `Arg`.

It intentionally does not add broader term-construction or enumeration behavior.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
