# PR Title

Add Go WAM string_code/3 builtin parity

# PR Description

## Summary

- Adds direct Go WAM lowering for `string_code/3`.
- Implements deterministic generated Go runtime support for `string_code(+Index,+String,?Code)` with 1-based indexing.
- Extends the generated Go WAM builtin E2E test with code lookup, code unification, invalid index failure, unbound argument failure, non-text source failure, and bound-code mismatch failure.
- Updates the Go WAM parity audit to record forward-mode `string_code/3` coverage against the R/C++ deterministic text-code lookup surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(string_code/3,3), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-string-code-builtin
```
