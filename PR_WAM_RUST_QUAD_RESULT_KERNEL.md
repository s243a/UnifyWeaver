## PR Title

Add a quad-result foreign kernel to Rust WAM lowering

## Summary

This PR extends the Rust WAM foreign-lowering path with a higher-arity tuple case by adding a new `tuple:4` kernel and validating it end to end.

The new kernel, `tc_step_parent_distance/5`, proves that the current foreign result model scales beyond `tuple:3` without another runtime refactor.

## What Changed

- Added a new recursive kernel family:
  - `transitive_step_parent_distance5`
- Added compiler detection and foreign spec generation for:
  - `tc_step_parent_distance/5`
- Registered the new kernel with:
  - result layout `tuple:4`
  - result mode `stream`
- Added Rust runtime support to emit quadruple result tuples:
  - `(target, first_step, terminal_parent, distance)`
- Added target tests covering:
  - kernel detection
  - declarative spec generation
  - registry enumeration
  - compiler-selected foreign lowering
- Added runtime integration coverage for:
  - enumerating all results
  - exact success case matching
  - failure case matching
- Updated the design retrospective to document the new `tuple:4` path.

## Why

The previous branch established:

- tuple-shaped foreign result layouts
- explicit result delivery modes
- deterministic collection support
- direct foreign-only wrappers

The next useful proof was to show that `tuple:N` is not just a renamed `single/pair/triple` ladder. Adding a real `tuple:4` kernel demonstrates that the current result-shape and delivery-mode split scales to higher arities without reopening the runtime design.

## Validation

- `swipl -q -g run_tests -t halt tests/test_wam_rust_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_rust_runtime.pl`

Both passed locally.

## Scope

This PR is intentionally narrow:

- it adds one higher-arity kernel to prove the current architecture
- it does not introduce a new result encoding model
- it does not broaden general WAM instruction coverage
