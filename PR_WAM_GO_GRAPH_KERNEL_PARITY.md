# PR Title

`feat(wam-go): add graph-kernel foreign parity and warning hygiene`

# PR Description

## Summary

This PR brings the Go hybrid/WAM target much closer to the Rust/Haskell hybrid targets by adding the remaining graph-kernel foreign execution paths, extending `foreign_lowering(true)` auto-detection to those kernels, and cleaning the focused warning noise in the Go WAM verification path.

## What Changed

- Added Go native foreign execution support for:
  - `transitive_distance3`
  - `transitive_parent_distance4`
  - `transitive_step_parent_distance5`
  - `weighted_shortest_path3`
  - `astar_shortest_path4`
- Added weighted-edge runtime storage and registration support in the Go WAM runtime.
- Extended Go `foreign_lowering(true)` auto-detection to synthesize foreign specs for:
  - weighted shortest path
  - A* shortest path
  - the remaining graph-distance kernels
- Added Go-side emission for weighted-edge registration setup.
- Expanded Go foreign-lowering tests to cover:
  - auto-detect code generation for the newly supported kernels
  - end-to-end Go runtime execution for the new graph kernels
- Cleaned focused Go warning/test hygiene:
  - added `discontiguous` declarations for split predicates in `go_target.pl`
  - removed avoidable choicepoints in the touched Go WAM tests
  - removed local directory-helper overrides from the touched Go tests
  - avoided the semantic compiler weak-import warning by excluding `is_semantic_predicate/1` from the import that conflicts with the local definition

## Why

Before this change, Go had hybrid/WAM parity for the basic foreign kernels and the auto-detect path for the simpler recursive shapes, but it still lagged Rust/Haskell on the graph-oriented kernels that matter most for recursive path/search workloads.

This PR closes that gap by giving Go both:

- runtime support for the missing graph kernels
- auto-detect lowering for those same kernels

That means Go can now follow the same pattern as Rust/Haskell for the supported graph/search clause shapes instead of falling back to generic WAM execution.

## Verification

Passed locally:

- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`

The new coverage includes both generation-time auto-detect assertions and end-to-end Go runtime execution for the added graph kernels.

## Scope Notes

- This PR focuses on the Go hybrid/WAM runtime and emitter path only.
- It does not attempt broader repo-wide warning cleanup outside the focused Go target/test path.
- Residual warnings still exist in older unrelated files such as `wam_target.pl` and `go_runtime/custom_go.pl`.

## Follow-Ups

- Add stronger ordering/semantic assertions for weighted/A* multi-result behavior if we want stricter parity guarantees beyond basic execution coverage.
- Continue reducing unrelated warning noise in older Go/runtime infrastructure files.
