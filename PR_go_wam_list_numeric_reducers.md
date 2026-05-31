# Title

Add Go WAM numeric list reducer builtins

# Description

## Summary

- Adds direct Go WAM lowering for `sum_list/2`, `min_list/2`, and `max_list/2`.
- Implements runtime support for numeric proper lists:
  - `sum_list/2` succeeds on empty lists with `0`.
  - `min_list/2` and `max_list/2` require at least one numeric element.
  - all-integer inputs return integers; mixed float inputs return floats.
- Adds generated Go WAM builtin E2E coverage for integer lists, mixed-float sums, singleton min/max, empty-list behavior, non-numeric failure, and malformed-list failure.
- Updates the Go WAM parity audit to document the C++/LLVM numeric reducer parity surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
