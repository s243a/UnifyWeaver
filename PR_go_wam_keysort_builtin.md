# PR Title

Add Go WAM keysort/2 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `keysort/2`
- Implement deterministic proper-list `keysort(+Pairs, -Sorted)` runtime handling for `Key-Value` pairs
- Preserve stable ordering for duplicate keys and support integer, float, mixed numeric, and atom keys
- Extend the generated Go WAM builtin E2E test for empty and singleton lists, duplicate-key stability, malformed-list failure, and non-pair failure
- Update the Go WAM parity audit to record the new LLVM key-value sorting parity coverage

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
