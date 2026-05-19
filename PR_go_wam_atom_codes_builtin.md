# PR Title

Add Go WAM atom_codes/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_codes/2`.
- Implement runtime atom/code-list conversion for atom-to-codes and codes-to-atom modes.
- Handle bound code-list comparison through decoded text, including lists materialized through heap cons cells inside `\+/1`.
- Cover forward, reverse, empty-atom, mismatch, both-unbound, and invalid-code cases in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record atom/code-list conversion parity with the Clojure text conversion surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_codes/2,2), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
