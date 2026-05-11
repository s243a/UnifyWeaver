# fix(wam-clojure): prevent build-frame self-aliasing

## Summary

Fixes a Clojure WAM runtime edge case where `put_structure` / `put_list` could unify a freshly built term with the old value of the same argument register, creating self-referential terms in aliasing scenarios.

## What Changed

- Added an `a-slot?` helper for generated Clojure WAM registers.
- Changed build-frame finalization so `A*` argument-register targets are overwritten with the newly built term.
- Preserved the previous `assign-or-unify-reg` behavior for `X*` / `Y*` targets, which is still needed for nested build placeholders.
- Strengthened the `copy_term/2` smoke regression to cover the previously problematic aliased source shape.

## Validation

```sh
git diff --check
swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl
swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl
timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl
```

## Notes

- A blunt overwrite of all build targets regressed nested list construction on backtracking.
- This fix intentionally distinguishes temporary argument registers from local build placeholders.
