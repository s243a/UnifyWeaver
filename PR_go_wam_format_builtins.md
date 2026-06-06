# PR Title

Add bounded Go WAM format builtins

# PR Description

## Summary

- Register `format/1` and `format/2` as direct Go WAM builtins.
- Add bounded Go WAM runtime formatting for `~w`, `~n`, and `~~`.
- Add generated Go WAM E2E coverage for format output and missing-argument failure.
- Fix Go WAM text parsing so quoted constants with spaces and tildes survive translation into Go instructions.
- Update the Go WAM parity audit for the closed bounded-format output gap.

## Scope

This intentionally keeps `format/3`, stream capture, and broader directives separate from this slice.

## Validation

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

Existing non-fatal choicepoint warnings still appear in the same Go WAM tests.
