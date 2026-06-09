# PR Title

Add Go WAM write_canonical builtin

# PR Description

## Summary

- Register `write_canonical/1` as a direct Go WAM builtin.
- Add recursive canonical rendering for quoted atoms, proper lists, compound terms, numbers, refs, and unbound variables.
- Add generated Go WAM E2E coverage for quoted atoms, lists, and compounds.
- Update the Go WAM parity audit for the closed Python/C++ canonical-output gap.

## Scope

This keeps `format/3`, stream capture, `with_output_to/2`, and stream I/O out of scope.

## Validation

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

Existing non-fatal choicepoint warnings still appear in the same Go WAM tests.
