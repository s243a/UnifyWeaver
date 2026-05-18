# feat(wam-clojure): wire string_to_atom/2

## Summary

- Registers `string_to_atom/2` in the central WAM builtin table.
- Adds direct lowered Clojure WAM recognition for `string_to_atom/2`.
- Wires generated Clojure runtime dispatch through the existing atom/string conversion helper.
- Extends lowered-emitter and runtime smoke coverage for forward, reverse, mismatch, and unbound-pair cases.

## Notes

- `string_to_atom/2` uses the same Clojure runtime helper as `atom_string/2`, with reversed argument roles.
- This follows the current Clojure WAM scaffold convention where atom-like text and strings collapse to the same runtime representation.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
