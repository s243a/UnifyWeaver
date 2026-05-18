# feat(wam-clojure): lower atom_string/2

## Summary

- Adds direct lowered Clojure WAM support for `atom_string/2`.
- Adds a runtime helper for bidirectional atom/text unification in the Clojure WAM scaffold.
- Treats raw EDN strings and symbols as atom-like text in this helper path, matching the existing CLI smoke-test encoding.
- Extends lowered-emitter and runtime smoke coverage for forward, reverse, mismatch, and unbound-pair cases.

## Notes

- The implementation follows the current Clojure WAM representation where atom-like text and strings are collapsed for generated runtime purposes.
- `string_to_atom/2` is intentionally not enabled in this PR because it is not currently registered in the central WAM builtin table. The helper accepts a `string_to_atom/2` predicate key internally so that a later registry PR can wire it without changing the runtime semantics.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
