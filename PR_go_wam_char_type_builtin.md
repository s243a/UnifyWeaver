# PR Title

Add Go WAM char_type/2 builtin parity

# PR Description

## Summary

- Adds direct Go WAM lowering for `char_type/2`.
- Implements forward-mode generated Go runtime support for atom category checks.
- Covers `alpha`, `alnum`, `digit`, `space`, `white`, `upper`, `lower`, `punct`, `ascii`, `csym`, `csymf`, and `newline`.
- Extends the generated Go WAM builtin E2E test with success and failure cases, including multi-character atoms, unbound arguments, and unknown categories.
- Updates the Go WAM parity audit to record forward-mode `char_type/2` coverage against the R/Clojure atom-category surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(char_type/2,2), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-char-type-builtin
```
