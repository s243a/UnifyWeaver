# feat(wam-clojure): add append split mode

## Summary

Extends Clojure WAM `append/3` support from deterministic concat mode to split mode.

## What Changed

- Refactors lowered `append/3` emission to call shared `runtime/apply-append-solution`.
- Keeps concat mode for `append(+ListA, +ListB, ?Out)`.
- Adds split mode for `append(A, B, GroundList)`.
- Enumerates prefix/suffix splits using existing foreign-style choice-point machinery.
- Adds runtime smoke coverage for every split of `[a,b,c]`.

## Semantics

Supported modes after this PR:

- `append(+ListA, +ListB, ?Out)`
- `append(?Prefix, ?Suffix, +GroundList)`

The split-mode path enumerates all valid prefix/suffix pairs and supports backtracking to later splits.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
