# Title

Preserve improper list arguments in WAM codegen

# Description

## Summary

- Fixes WAM argument codegen so `put_list` is used only for proper list literals.
- Lets improper lists such as `[a|b]` compile as ordinary `[|]/2` structures, preserving the non-list tail.
- Restores Go WAM builtin behavior for malformed-list rejection in `select/3`, `delete/3`, `permutation/2`, `sort/2`, and similar list builtins.
- Refreshes the Go WAM builtin test assertion for the current inlined negation lowering, which emits `fail/0` instead of a direct `\+/1` meta-call.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
