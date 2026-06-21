# PR Title

Add Go WAM read_line_to_string/2 builtin

# PR Description

## Summary

- Adds direct Go WAM lowering for `read_line_to_string/2`.
- Implements file-backed stream line reading in the Go WAM runtime.
- Extends generated Go WAM builtin E2E coverage for normal lines, blank lines, final lines without trailing newline, and EOF as `end_of_file`.
- Updates the Go WAM parity audit to record the bounded Python line-reader parity slice.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
```
