# PR Title

Audit Go WAM parity against Rust and Haskell

# PR Description

## Summary

- Add a Go WAM parity audit comparing the Go hybrid WAM target against the Rust and Haskell runtime/builtin baseline.
- Document concrete Go parity gaps for follow-up work, including `member/2` enumeration, term builtins, `copy_term/2`, `is_list/1`, `display/1`, `=/2`, `set` aggregation, and the likely `=</2` dispatch typo.
- Update stale Go generator assertions to match the current atom interning output (`internAtom("...")`) instead of old raw `&Atom{Name: ...}` literals.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
