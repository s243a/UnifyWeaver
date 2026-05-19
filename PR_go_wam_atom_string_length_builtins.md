# PR Title

Add Go WAM atom_length/2 and string_length/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_length/2` and `string_length/2`.
- Implement deterministic runtime length checks over atom text.
- Cover length binding, bound-length success, mismatch failure, unbound-source failure, and invalid-source failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record the text-length surface alongside the existing atom-number, atom-case, and atom-concat builtins.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_length/2,2), A), wam_go_target:wam_instruction_to_go_literal(call(string_length/2,2), S), writeln(A), writeln(S), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
