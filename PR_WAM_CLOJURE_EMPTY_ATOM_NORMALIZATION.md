# PR Title

fix(wam-clojure): normalize empty atom constants

# PR Description

## Summary

- Normalizes WAM atom constant token `''` to empty atom text `""` in the Clojure WAM generator.
- Applies the same normalization to compile-time atom seeds and switch-on-constant entries.
- Updates the lowered emitter literal path so Prolog empty atoms are emitted as empty Clojure strings.
- Adds runtime smoke coverage for `atomic_list_concat(L, o, foo), L = [f,'','']`.

## Why

The Clojure WAM target previously treated serialized Prolog empty atom tokens as the literal two-character atom text `''`. That broke cases where generated code needed to unify against actual empty atom segments, especially the deterministic `atomic_list_concat/3` split path added in the previous PR.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
