# PR Title

Add Go WAM `read_string/5` builtin parity

# PR Description

## Summary

- Registers `read_string/5` as a direct Go WAM builtin.
- Adds Go WAM runtime support for bounded file-backed stream reads, unifying the actual character count and output atom.
- Extends the generated Go builtin E2E test to cover bounded reads, short EOF reads, zero-count EOF reads, and closed-stream failure.
- Updates the Go WAM parity audit to record the Python-aligned bounded chunk-read surface.

## Tests

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

## Push Command

```sh
git push -u origin feat/wam-go-read-string
```
