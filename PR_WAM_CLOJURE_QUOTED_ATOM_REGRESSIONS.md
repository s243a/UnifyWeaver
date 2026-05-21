# PR Title

test(wam-clojure): cover quoted atom escaping regressions

# PR Description

## Summary

- Adds Clojure WAM runtime smoke coverage for a quoted atom containing an embedded single quote.
- Adds Clojure WAM runtime smoke coverage for a quoted atom containing a literal backslash.
- Asserts both predicates are directly lowered into generated Clojure functions.

## Why

The previous quoted-atom escaping fix covered numeric quoted atoms through `number_chars/2`. These tests lock down the broader escaping path for non-numeric atom text that requires WAM quoting and Clojure string escaping.

## Testing

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
