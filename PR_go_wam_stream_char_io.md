# PR Title

Add Go WAM stream character IO builtins

# PR Description

## Summary

- Adds direct Go WAM lowering for `open/3`, `close/1`, `get_char/2`, `peek_char/2`, `get_code/2`, `put_char/2`, and `put_code/2`.
- Adds file-backed `StreamHandle` runtime support with buffered read handles and write/append file modes.
- Extends generated Go WAM builtin E2E coverage for stream read, peek, EOF code/atom behavior, and output file emission.
- Updates the Go WAM parity audit to record the bounded Python stream-character IO parity slice.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
```
