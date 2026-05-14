# PR Title

feat(wam-clojure): add univ compose builtin

# PR Description

## Summary

- Extends Clojure WAM `=../2` support with compose mode.
- Builds terms from proper univ lists when the first argument is unbound.
- Synthesizes and interns `Name/Arity` functor keys for compound construction.
- Keeps existing decompose mode for bound first arguments.
- Adds runtime smoke coverage for struct, atom, number, list, empty-list failure, and invalid functor-head failure cases.

## Scope

This PR completes the narrow `=../2` bidirectional path for Clojure WAM:

- Decompose mode remains `BoundTerm =.. List`.
- Compose mode adds `UnboundTerm =.. ProperList`.

The compose path requires a non-empty proper list. Multi-element lists require an atom head as the functor name; one-element lists compose to the atomic head value.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
