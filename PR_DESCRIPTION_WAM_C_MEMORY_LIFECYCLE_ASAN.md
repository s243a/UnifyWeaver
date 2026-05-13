# test: Add WAM C ASAN lifecycle smoke

## Summary

This PR adds an AddressSanitizer-backed WAM-C executable smoke for runtime lifecycle coverage. The smoke recompiles generated WAM-C with `-fsanitize=address` and exercises repeated setup, switch-table replacement, indexed dispatch, FactSource loading, native foreign kernel dispatch, and repeated top-level predicate calls.

## What Changed

- Adds an optional ASAN lifecycle smoke to `tests/test_wam_c_target.pl`.
- Redirects sanitizer executable output to a per-smoke log path on failure.
- Prunes top-level choicepoints after `wam_run_predicate` returns so later top-level calls do not restore stale heap or register snapshots.
- Guards `retry_me_else` when indexed dispatch jumps directly into a later clause without a sequential choicepoint.
- Updates `WAM_C_TARGET_NEXT_STEPS.md` to mark this branch ready and recommend the lowered helper prototype next.

## Validation

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels --repetitions 1`
- `git diff --check`

## Follow-Up

Next recommended branch: `feat/wam-c-lowered-helper-prototype`.
