# PR Title

Add Go WAM print and writeln builtins

# PR Description

## Summary

- Register `print/1` and `writeln/1` as direct Go WAM builtins.
- Implement Go WAM runtime handling for `print/1` as a `write/1` alias and `writeln/1` as term output plus newline.
- Add generated Go WAM E2E coverage for `write/1`, `print/1`, `writeln/1`, and `nl/0`.
- Update the Go WAM parity audit to record the closed simple-output gap against richer sibling targets.

## Validation

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

Existing non-fatal choicepoint warnings still appear in the same Go WAM tests.
