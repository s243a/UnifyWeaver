# PR Title

Validate reverse Rust WAM foreign lowering end to end

# PR Description

## Summary

This PR closes the gap between codegen coverage and execution coverage for Rust WAM foreign-lowered transitive closure predicates.

It adds end-to-end generated Rust runtime validation for the reverse closure case and fixes the schema matcher so forward and reverse transitive-closure predicates lower with the correct fact orientation.

## What Changed

- Added runtime integration coverage for both:
  - `tc_ancestor/2`
  - `tc_descendant/2`
- Extended the generated Rust runtime test so reverse closure execution is validated through actual backtracking and bound-variable results.
- Fixed transitive-closure schema detection in the Rust compiler so:
  - forward closure predicates keep `parent -> child` fact orientation
  - reverse closure predicates use `child -> parent` fact orientation
- Updated target/codegen assertions to reflect the correct emitted fact-pair tables.
- Updated the Rust WAM retrospective/design doc to reflect the current foreign-lowering architecture and verified predicate families.

## Why

The previous branch established that reverse closure patterns such as `tc_descendant/2` could be selected for foreign lowering at codegen time, but that path did not yet have execution-level validation.

Adding the runtime test exposed a real bug:

- reverse closures were being matched too permissively during schema detection
- emitted fact pairs could end up in the wrong orientation for the native runtime handler

This PR fixes that and proves the reverse path works end to end in a generated Rust project.

## Validation

Ran locally:

- `swipl -q -g run_tests -t halt tests/test_wam_rust_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_rust_runtime.pl`

Both passed locally.

## Scope

This is a focused correctness-and-coverage follow-up to the foreign-lowering generalization work. It does not expand foreign lowering to new predicate families; it makes the existing transitive-closure family safer by validating both:

- forward closure lowering
- reverse closure lowering
