# PR Title

feat(wam-clojure): split atomic_list_concat output

# PR Description

## Summary

- Extends lowered Clojure WAM `atomic_list_concat/3` support with deterministic output splitting.
- Adds a runtime helper that turns split text segments into interned atom terms and unifies them with the list argument.
- Adds smoke coverage for split success, split mismatch, and unsupported unbound-separator fallback.

## Scope

This keeps the implementation conservative. The new supported mode is `atomic_list_concat(List, Separator, Atom)` when `List` is unbound, `Separator` is a bound non-empty atomic value, and `Atom` is bound. Existing bound-list concatenation behavior remains unchanged. Unsupported modes, including unbound separators and `atomic_list_concat/2` with an unbound list, continue to fail through the lowered runtime backtracking path.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
