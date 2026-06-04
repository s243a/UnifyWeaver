Title: Add Go WAM math arithmetic functions

Description:

## Summary

- add Go WAM arithmetic evaluation for `sqrt/1`, `sin/1`, `cos/1`, `tan/1`, `asin/1`, `acos/1`, `atan/1`, `floor/1`, `ceiling/1`, and `round/1`
- extend generated Go WAM E2E coverage for the new math functions
- cover invalid-domain failure for `sqrt(-1)`, `asin(2)`, and `acos(2)`
- update the Go WAM parity audit for the bounded R/LLVM unary math evaluator surface

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `git diff --check`
