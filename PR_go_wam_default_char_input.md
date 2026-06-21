# PR Title

Add Go WAM default character input builtins

# PR Description

## Summary

- Adds direct Go WAM lowering for `get_char/1`, `peek_char/1`, and `get_code/1`.
- Adds state-local default stdin reader support for the Go WAM runtime.
- Implements EOF behavior matching the bounded Python default-input surface: `end_of_file` for character reads and `-1` for code reads.
- Extends the generated Go WAM builtin E2E test with deterministic stdin pipe coverage for peek, consume, and EOF behavior.
- Updates the Go WAM parity audit to record the new default character input parity slice.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
```
