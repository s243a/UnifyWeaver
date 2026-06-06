# PR Title

Broaden Go WAM format directive support

# PR Description

## Summary

- Extend Go WAM `format/1` and `format/2` directive handling beyond `~w`, `~n`, and `~~`.
- Add bounded support for `~a`, `~d`, `~p`, and `~q`.
- Keep `~p` and `~q` aligned with the current Go WAM term rendering path.
- Add generated Go WAM E2E coverage for the expanded directive set.
- Update the Go WAM parity audit for the broadened R/Python/C++ format subset.

## Scope

This does not add `format/3`, stream capture, `with_output_to/2`, `~s`, tab directives, or richer canonical quoting. Those remain separate runtime-surface slices.

## Validation

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`

Existing non-fatal choicepoint warnings still appear in the same Go WAM tests.
