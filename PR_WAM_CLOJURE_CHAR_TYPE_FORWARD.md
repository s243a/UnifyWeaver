# PR Title

feat(wam-clojure): lower forward char_type checks

# PR Description

## Summary

- Add `char_type/2` to the Clojure WAM direct builtin set.
- Add a conservative Clojure runtime helper for forward character classification when both arguments are bound.
- Add runtime smoke coverage for alpha, digit/alnum, space/whitespace, and a negative lower-case check.

## Behavior

The Clojure WAM runtime now handles deterministic forward checks such as:

- `char_type(a, alpha)`
- `char_type(C, digit)` after deriving `C` with `char_code(C, 0'5)`
- `char_type(C, space)` and `char_type(C, whitespace)` after deriving `C` with `char_code(C, 32)`
- `char_type('A', lower)` correctly fails

Supported class atoms in this slice:

- `alpha`
- `alnum`
- `digit`
- `upper`
- `lower`
- `space`
- `whitespace`
- `ascii`
- `punct`

Compound `char_type/2` forms such as `upper(Lower)`, `lower(Upper)`, `to_upper/1`, `to_lower/1`, `digit/1`, and `code/1` remain follow-up work.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
