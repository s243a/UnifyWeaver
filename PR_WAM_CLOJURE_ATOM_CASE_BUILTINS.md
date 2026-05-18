# PR Title

feat(wam-clojure): lower atom case builtins

# PR Description

## Summary

- Adds direct lowered Clojure support for `upcase_atom/2` and `downcase_atom/2`.
- Implements a shared `runtime/apply-atom-case-solution` helper using existing atom text/interning semantics.
- Adds lowered-emitter coverage for both builtins.
- Extends the Clojure runtime smoke suite with success, mismatch, unbound-source, and generated-code assertions.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
