# PR Title

Add Go WAM put_char/1 and put_code/1 builtins

# PR Description

## Summary

- Adds direct Go WAM lowering for `put_char/1` and `put_code/1`.
- Implements runtime stdout dispatch for single-character atom output and integer character-code output.
- Extends the generated Go WAM builtin E2E test to cover successful output plus invalid argument failure.
- Updates the Go WAM parity audit to record the new Python/C++/LLVM single-character output parity slice.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
```
