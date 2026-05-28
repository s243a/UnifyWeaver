# PR Title

Add Go WAM split_string/4 builtin parity

# PR Description

## Summary

- Adds direct Go WAM lowering for `split_string/4`.
- Implements deterministic generated Go runtime support for `split_string(+String,+Separators,+Pad,?SubStrings)`.
- Splits on any separator character and trims leading/trailing pad characters from each segment.
- Extends the generated Go WAM builtin E2E test with separator splitting, empty input, adjacent separators, no-separator padding, separator plus pad trimming, multiple separators, numeric source conversion, unbound input failure, and bound-output mismatch failure.
- Updates the Go WAM parity audit to record forward-mode `split_string/4` coverage against the R/C++ deterministic split-string surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(split_string/4,4), G), writeln(G), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`

## Push

```sh
git push -u origin feat/wam-go-split-string-builtin
```
