# PR Title

Add Go WAM atom_number/2 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_number/2`.
- Implement bidirectional `atom_number/2` runtime behavior for atom-to-number and number-to-atom conversion.
- Cover integer, float, numeric atom, invalid atom, and both-unbound cases in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record the new atom/number conversion surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_number/2,2), X), writeln(X), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
