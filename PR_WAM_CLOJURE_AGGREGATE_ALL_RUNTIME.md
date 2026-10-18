# PR Title

feat(wam-clojure): support aggregate_all runtime modes

# Description

## Summary

- Extends the Clojure WAM runtime aggregate finalizer beyond `findall`/`collect` to support `aggregate_all/3` modes: `count`, `sum`, `bag`, `set`, `min`, and `max`.
- Uses existing runtime term ordering for `set` sorting/deduplication so generated results match SWI-style standard order for the covered cases.
- Preserves SWI-compatible empty aggregate behavior: `count` and `sum` return `0`, `bag` and `set` return `[]`, and empty `min`/`max` fail.
- Adds generated Clojure smoke coverage for the new aggregate modes, including negative assertions and empty-min failure.

## Scope

- This is sequential/runtime-mediated aggregate support for the existing WAM aggregate frame.
- Direct `bagof/3` and `setof/3` are not changed here; the observed compiler path still emits them as runtime calls rather than Clojure WAM `begin_aggregate` instructions.
- No lowered aggregate path or parallel aggregate execution is introduced in this PR.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
- `swipl -q -s tests/test_wam_clojure_lowered_t4.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_lowered_ite_exec.pl`
- `swipl -q -g run_tests -t halt tests/core/test_clojure_native_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_benchmark_generator.pl`
