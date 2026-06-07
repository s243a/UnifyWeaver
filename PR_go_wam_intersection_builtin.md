# PR Title

Add Go WAM intersection/3 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `intersection/3`
- Implement deterministic proper-list `intersection(+Left, +Right, -Result)` runtime handling
- Extend the generated Go WAM builtin E2E test for right-list match retention, duplicate retention, order preservation, disjoint lists, empty-list behavior, all-kept results, and malformed-list failure
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
