# PR Title

Add Go WAM atom case builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `upcase_atom/2` and `downcase_atom/2`.
- Implement deterministic runtime execution for atom case conversion.
- Cover converted-output binding, bound-output success, mismatch failure, unbound-source failure, and non-atom source failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record the atom-case conversion surface alongside `atom_number/2`.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(upcase_atom/2,2), U), wam_go_target:wam_instruction_to_go_literal(call(downcase_atom/2,2), D), writeln(U), writeln(D), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
