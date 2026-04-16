# PR Title

`feat(wam-go): auto-detect foreign lowering and clean target hygiene`

# PR Description

## Summary

This PR closes the next major parity gap between the Go hybrid/WAM target and the Rust/Haskell hybrid targets by adding `foreign_lowering(true)` auto-detection to the Go WAM pipeline.

It also cleans the focused warning noise introduced by the Go foreign-lowering path so local verification is easier to read.

## What Changed

- Added Go-side auto-detection for `foreign_lowering(true)` in the WAM Go target.
- Preserved module-qualified predicate handling so detection works for non-`user` predicates, including plunit-defined predicates.
- Synthesized Go foreign-lowering specs from detected recursive kernels instead of requiring explicit `foreign_predicate(...)` declarations.
- Added support for these Go auto-detected kernels:
  - `countdown_sum2`
  - `list_suffix2`
  - `list_suffixes2`
  - `transitive_closure2`
- Reused the existing Go `CallForeign` setup path by generating the same setup ops used by explicit foreign lowering.
- Added a focused test covering auto-detected transitive-closure lowering.
- Cleaned focused warning hygiene in the touched code:
  - declared `wam_line_to_go_literal/4` discontiguous in the Go WAM target
  - removed singleton/local override noise from the Go foreign-lowering test file
  - removed avoidable choicepoints from the edited foreign-lowering tests

## Why

Before this change, the Go hybrid/WAM target only supported foreign lowering when the caller provided an explicit `foreign_predicate(...)` spec.

Rust and Haskell already go further by recognizing known recursive kernel shapes and lowering them automatically. This PR brings Go onto that same path for the kernels the current Go runtime can already execute.

## Verification

Passed locally:

- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`

## Scope Notes

- This PR does not add new Go native kernels beyond those already supported by the runtime.
- This PR does not attempt repo-wide warning cleanup; it only cleans the focused warning noise in the touched Go foreign-lowering path.
- Remaining warning output still comes from older unrelated areas such as `go_target.pl` and legacy Go WAM tests.

## Follow-Ups

- Add Go auto-detection parity for the remaining graph kernels once the Go runtime supports them.
- Do a separate warning-hygiene pass over existing Go target and Go WAM test files.
