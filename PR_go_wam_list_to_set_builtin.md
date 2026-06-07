# PR Title

Add Go WAM list_to_set/2 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `list_to_set/2`
- Implement deterministic `list_to_set/2` runtime handling that preserves first occurrences
- Extend the generated Go WAM builtin E2E test for duplicate removal, empty and singleton lists, numeric duplicates, and malformed-list failure
- Update the Go WAM parity audit to record the new C++/LLVM list-set parity coverage

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
