# PR Title

Add Go WAM union/3 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `union/3`
- Implement deterministic proper-list `union(+A, +B, -Union)` runtime handling
- Extend the generated Go WAM builtin E2E test for `A ++ subtract(B, A)` behavior, left-list duplicate preservation, right-list match filtering, empty-list behavior, identical-list results, inclusion-exclusion with `intersection/3`, and malformed-list failure
- Update the Go WAM parity audit to record the new LLVM set-list parity coverage

## Verification

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
git diff --check
```
