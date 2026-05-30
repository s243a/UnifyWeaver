# Title

Add Go WAM permutation/2 builtin parity

# Description

## Summary

- Adds direct Go WAM lowering for `permutation/2`.
- Implements bounded runtime support matching the R target parity surface:
  - `permutation(+List, +List)` succeeds when both lists contain the same terms after term-order sorting.
  - `permutation(+List, -List)` unifies the second argument with the original list as the identity permutation.
- Adds generated Go WAM builtin E2E coverage for success, failure, duplicate, mixed atom/integer, identity, and malformed-list cases.
- Updates the Go WAM parity audit to include `permutation/2` and note that full nondeterministic permutation enumeration remains outside this scaffold.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
