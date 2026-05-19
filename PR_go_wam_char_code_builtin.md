# PR Title

Add Go WAM char_code/2 builtin parity

# PR Description

## Summary

- Add direct Go WAM lowering for `char_code/2`.
- Implement bidirectional runtime conversion for one-character atoms and integer character codes.
- Cover char-to-code binding, bound-code success, mismatch failure, code-to-char binding, multi-character atom failure, both-unbound failure, and invalid-code failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record the char-code conversion surface alongside the existing atom/text builtins.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(char_code/2,2), X), writeln(X), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
