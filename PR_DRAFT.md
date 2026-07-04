# PR Title

feat(wam-clojure): default inline bagof/setof compilation

# Description

## Summary

- Makes Clojure WAM project compilation default to `inline_bagof_setof(true)`.
- Preserves explicit opt-out with `inline_bagof_setof(false)`.
- Removes the runtime smoke test's explicit inline option so it validates the target default.
- Adds generator regressions for default inline aggregate emission and opt-out fallback emission.

## Why

- The Clojure WAM runtime now supports inline `bagof/3` and `setof/3`, including witness grouping and backtracking across groups.
- Leaving inline support opt-in would keep normal Clojure WAM project generation on the old `execute bagof/3`/`setof/3` path.
- This change makes the merged runtime capability the default Clojure target behavior while retaining a conservative escape hatch.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
- `swipl -q -s tests/test_wam_clojure_lowered_t4.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_lowered_ite_exec.pl`
- `swipl -q -g run_tests -t halt tests/core/test_clojure_native_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_benchmark_generator.pl`
