# PR Title

Add Go WAM string_codes/2 and string_chars/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `string_codes/2` and `string_chars/2`.
- Reuse the existing atom text/list conversion dispatch for the Go WAM string aliases.
- Cover string-to-list binding, bound-list success, mismatch failure, list-to-string binding, empty-string conversion, both-unbound failure, and invalid-list failure in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record string list-conversion alias parity with the Clojure text conversion surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(string_codes/2,2), C), wam_go_target:wam_instruction_to_go_literal(call(string_chars/2,2), H), writeln(C), writeln(H), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
