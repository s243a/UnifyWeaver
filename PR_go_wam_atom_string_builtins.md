# PR Title

Add Go WAM atom_string/2 and string_to_atom/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_string/2` and `string_to_atom/2`.
- Implement bidirectional runtime atom/text unification for the current Go WAM text representation.
- Cover atom-to-text binding, bound-text success, mismatch failure, text-to-atom binding, and both-unbound failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record atom/string conversion alongside the existing atom/text builtins.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_string/2,2), A), wam_go_target:wam_instruction_to_go_literal(call(string_to_atom/2,2), S), writeln(A), writeln(S), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
