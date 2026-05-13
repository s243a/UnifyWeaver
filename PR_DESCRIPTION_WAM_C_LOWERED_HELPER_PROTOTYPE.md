# feat: Prototype WAM C lowered fact helpers

## Summary

This PR adds the first opt-in lowered/native helper path for the WAM-C target. Constant fact-only predicates can now be emitted as native C foreign handlers, while the normal predicate entry point remains a generated `call_foreign` trampoline through the existing WAM-C foreign registry.

## What Changed

- Adds `lowered_helpers(true)` support to WAM-C project generation.
- Detects simple fact-only predicates whose head arguments are atoms or integers.
- Emits native C `WamForeignHandler` helpers for those fact rows.
- Registers lowered helpers through `setup_lowered_wam_c_helpers`.
- Keeps detected recursive kernels excluded from this generic lowered-helper path.
- Adds generation and executable smoke coverage for a lowered fact-only predicate.
- Updates `WAM_C_TARGET_NEXT_STEPS.md` to mark this branch active and recommend planner/selection metadata next.

## Validation

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels --repetitions 1`
- `git diff --check`

## Follow-Up

Next recommended branch: `feat/wam-c-lowered-helper-planner`.
