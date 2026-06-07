# PR Title

Add Go WAM tab/1 builtin parity

# PR Description

## Summary

- Adds direct Go WAM lowering for `tab/1`.
- Implements generated Go runtime support for `tab(+N)`, writing `N` spaces for nonnegative integer arguments.
- Extends the generated Go WAM builtin E2E test with nonnegative space output, zero-width success, negative integer failure, unbound argument failure, and non-integer failure.
- Updates the Go WAM parity audit I/O row to record `tab/1` coverage against the R/C++ I/O polish surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(tab/1,1), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-tab-builtin
```
