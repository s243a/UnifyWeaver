# PR Title

Close Go WAM choicepoint parity audit

# PR Description

## Summary

- Updates the Go WAM parity audit to mark choice points/backtracking as present for the current resume-state baseline.
- Adds generated-runtime assertions covering Go WAM resume-state support for:
  - normal clause choicepoints
  - indexed alternatives
  - `member/2` builtin retries
  - foreign stream retries
  - indexed atom fact streams
- Confirms this was an audit/test coverage closure rather than a runtime rewrite.

## Verification

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
git diff --check
```

## Follow-Up

- Continue broadening Go WAM E2E coverage for remaining cross-target builtin edge cases.
