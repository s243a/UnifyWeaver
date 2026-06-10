Title: Add Go WAM environment builtins

Description:

## Summary

- add Go WAM direct builtin lowering for `getenv/2` and `setenv/2`
- implement runtime handling with atom-only arguments, missing-env failure, and empty-value support
- extend the generated Go WAM builtin E2E test and parity audit for the LLVM environment surface

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `git diff --check`
