# PR Title

Add Go WAM ground/1 builtin parity

# PR Description

## Summary

- Adds direct Go WAM lowering for `ground/1`.
- Implements generated Go runtime groundness checks for atoms, numbers, compounds, structures, lists, and empty lists.
- Extends the Go WAM builtin E2E test with positive and negative `ground/1` cases, including nested unbound variables and list placeholders.
- Updates the Go WAM parity audit to record `ground/1` coverage against the current cross-target baseline.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(ground/1,1), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-ground-builtin
```
