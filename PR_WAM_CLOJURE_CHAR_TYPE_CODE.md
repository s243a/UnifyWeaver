# PR Title

feat(wam-clojure): support char_type code form

# PR Description

## Summary

- Adds Clojure WAM runtime handling for `char_type(Char, code(Code))`.
- Supports both forward validation from a known character to a code and reverse construction from a valid code to a character atom.
- Adds smoke coverage for forward, reverse, and mismatch cases in the generated Clojure runtime fixture.

## Testing

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
