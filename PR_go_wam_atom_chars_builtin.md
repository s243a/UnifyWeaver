# PR Title

Add Go WAM atom_chars/2 parity

# PR Description

## Summary

- Add direct Go WAM lowering for `atom_chars/2`.
- Implement runtime atom/char-list conversion for atom-to-chars and chars-to-atom modes.
- Share the text-list dispatch path with `atom_codes/2` while preserving char-specific validation for single-character atoms.
- Cover forward, reverse, empty-atom, mismatch, both-unbound, and invalid-char cases in the generated Go WAM builtin E2E test.
- Update the Go WAM parity audit to record atom/char-list conversion parity with the Clojure text conversion surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(atom_chars/2,2), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
