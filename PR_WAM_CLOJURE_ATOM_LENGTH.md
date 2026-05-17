# PR Title

feat(wam-clojure): add lowered atom length builtin

# PR Description

## Summary

- Adds direct lowered Clojure WAM support for `atom_length/2`.
- Adds `string_length/2` as the same runtime path, matching the current Clojure WAM atom/string text representation.
- Extends runtime smoke coverage for success, mismatch, unbound input failure, and string alias behavior.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
