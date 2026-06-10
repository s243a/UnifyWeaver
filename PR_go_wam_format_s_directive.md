# PR Title

Add Go WAM format string directive

# PR Description

## Summary

- Add bounded `~s` support for Go WAM `format/1` and `format/2`.
- Render atom arguments directly as text for `~s`.
- Render proper integer code lists as string output for `~s`.
- Fail malformed code lists instead of falling back silently.
- Add generated Go WAM E2E coverage and update the Go WAM parity audit.

## Scope

This keeps `format/3`, stream capture, `with_output_to/2`, tab directives, and richer canonical quoting out of scope.

## Validation

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

Existing non-fatal choicepoint warnings still appear in the same Go WAM tests.
