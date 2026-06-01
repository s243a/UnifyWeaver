# Title

Add Go WAM between/3 builtin parity

# Description

## Summary

- Adds direct Go WAM lowering for `between/3`.
- Implements runtime support for deterministic `between(+Low, +High, +Value)` range checks.
- Implements enumerable `between(+Low, +High, -Value)` via a dedicated choicepoint resume state.
- Adds generated Go WAM builtin E2E coverage for range success, range failure, invalid bounds, singleton ranges, and `findall/3` enumeration.
- Updates the Go WAM parity audit to document the R/C++ `between/3` parity surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
