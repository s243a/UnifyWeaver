# PR Title

Add Go WAM atom_concat/3 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_concat/3`.
- Implement deterministic runtime support when both input arguments are atoms.
- Cover output binding, bound-output success, mismatch failure, unbound-input failure, and non-atom input failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record the atom-concat conversion surface alongside the existing atom-number and atom-case builtins.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_concat/3,3), X), writeln(X), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
