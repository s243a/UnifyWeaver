# PR Title

feat(wam-clojure): add multi-mode sub_atom builtin

# PR Description

## Summary

- Adds direct lowered Clojure WAM support for `sub_atom/5`.
- Implements multi-mode substring enumeration for variable `Before`, `Length`, `After`, and `SubAtom` against a ground source atom.
- Reuses the existing foreign-solution choice point path so later goals can backtrack into additional substring candidates.
- Extends runtime smoke coverage for extraction, prefix/suffix checks, later-match backtracking, no-match failure, and unbound source failure.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
