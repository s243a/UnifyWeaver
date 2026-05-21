# PR Title

test(wam-clojure): cover number_chars reverse mode

# PR Description

## Summary

- Adds Clojure WAM smoke coverage for `number_chars/2` reverse conversion from character atoms to a number.
- Adds a negative smoke case for invalid character lists.
- Confirms the existing `apply-text-conversion-solution` runtime path handles these modes without runtime changes.

## Notes

- The reverse fixture builds digit character atoms through `char_code/2` to avoid the existing quoted digit atom encoding issue in generated Clojure source.

## Testing

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
