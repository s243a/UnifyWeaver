# PR Title

Add Go WAM race-to-true negation control

# PR Description

## Summary

- Adds Go WAM parallel negation support for multi-clause WAM targets.
- Updates `\+/1` to detect multi-clause goal entry points and dispatch through `runNegationParallel`.
- Adds `negationChoiceTargets`, `hasParallelNegationChoices`, `runNegationParallel`, and `raceToTrue`.
- Keeps single-clause and builtin-shaped negation on the existing isolated sequential path.
- Extends generated Go builtin E2E assertions to verify the race-to-true helper is emitted.
- Updates the Go WAM parity audit so control is no longer marked partial for the current baseline.

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
