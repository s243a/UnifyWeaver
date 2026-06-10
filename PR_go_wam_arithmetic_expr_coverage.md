Title: Cover Go WAM arithmetic expressions

Description:

## Summary

- add generated Go WAM E2E coverage for arithmetic expression evaluation through `is/2`
- cover unary numeric functions, coercion helpers, division/modulo, min/max, exponentiation, bitwise operators, shifts, and evaluator failure cases
- fix Go string escaping for structure functors in both normal WAM literal emission and lowered Go emission, so operators such as `/\` and `\/` compile correctly
- update the Go WAM parity audit against the Rust/Haskell arithmetic evaluator baseline

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `git diff --check`
