Title: Cover Go WAM append builtin

Description:

## Summary

- register `append/3` in the Go WAM direct-builtin table
- add generated Go WAM E2E coverage for bounded deterministic `append/3`
- cover proper-list concatenation, empty-left and empty-right behavior, bound-output success, mismatch failure, malformed-list failure, and unsupported unbound input modes
- update the Go WAM parity audit for the covered structural builtin surface

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `git diff --check`
