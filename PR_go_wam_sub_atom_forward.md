# PR Title

Add Go WAM forward-mode sub_atom/5 builtin

# PR Description

## Summary

- Adds direct Go WAM lowering for `sub_atom/5`.
- Extends the generated Go WAM runtime builtin dispatcher to read A4/A5 for arity-5 builtins.
- Implements deterministic forward-mode `sub_atom(+Atom,+Before,+Length,?After,?SubAtom)`.
- Adds generated Go WAM E2E coverage for extraction, empty substring extraction, numeric source conversion, mismatch failure, out-of-range failure, unbound source failure, and unsupported enumerable modes.
- Updates the Go WAM parity audit to record forward-mode `sub_atom/5` coverage against the R WAM baseline while leaving full nondeterministic enumeration for a later slice.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(sub_atom/5,5), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-sub-atom-forward
```
