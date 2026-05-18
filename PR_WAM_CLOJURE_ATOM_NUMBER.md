# PR Title

feat(wam-clojure): wire atom_number/2

# PR Description

## Summary

- Registers `atom_number/2` as a WAM builtin.
- Adds direct lowered Clojure emission for `atom_number/2` through `runtime/apply-atom-number-solution`.
- Implements Clojure runtime support for atom-to-number and number-to-atom conversion.
- Tightens bound/bound text-conversion handling so numeric-looking generated WAM literals compare by normalized text instead of accidental host-term shape.
- Adds lowered-emitter and runtime smoke coverage for forward, reverse, failure, and unbound `atom_number/2` cases.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
