# feat(wam-clojure): lower char_code/2

## Summary

- Adds direct lowered Clojure WAM support for `char_code/2`.
- Adds a runtime helper for bidirectional single-character atom/code conversion.
- Wires `char_code/2` into the generated Clojure runtime builtin dispatcher.
- Extends lowered-emitter and runtime smoke coverage for forward, reverse, mismatch, bad-character, and unbound-pair cases.

## Notes

- The helper accepts a bound one-character atom/string and unifies the code argument with its integer code point.
- Reverse mode accepts a bound valid integer code point and interns/unifies the corresponding one-character atom.
- Invalid character terms, invalid codes, and unbound/unbound mode fail conservatively.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
