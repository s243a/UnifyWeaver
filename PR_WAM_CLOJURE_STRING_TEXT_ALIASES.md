# PR Title

feat(wam-clojure): add string text conversion aliases

# PR Description

## Summary

- Adds direct lowered Clojure WAM support for `string_codes/2` and `string_chars/2`.
- Routes both aliases through the existing text conversion helper used by atom and number conversions.
- Treats string aliases as atom-style text conversions, matching the current Clojure WAM atom/string representation.
- Extends lowered-emitter and runtime smoke coverage for forward and reverse string code/char conversion.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
