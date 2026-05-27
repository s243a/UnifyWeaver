# PR Title

Add Go WAM number_codes/2 and number_chars/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `number_codes/2` and `number_chars/2`.
- Implement runtime number/list conversion using the existing code-list and char-list text helpers plus the `atom_number/2` numeric parsing/formatting path.
- Cover number-to-list binding, bound-list success, mismatch failure, list-to-number binding for integer, negative, and float text, both-unbound failure, and invalid-list failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record number code-list and char-list conversion parity with the Clojure text conversion surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(number_codes/2,2), C), wam_go_target:wam_instruction_to_go_literal(call(number_chars/2,2), H), writeln(C), writeln(H), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
