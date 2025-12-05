# PR Title: Update Roadmap - Mark Go DB & Rust Target as Completed

## Description

This PR updates the project documentation to reflect the current state of the codebase, specifically regarding the Go and Rust targets.

## Changes

-   **`FUTURE_WORK.md`**:
    -   Moved "Go Target Enhancements > Database Integration" to "Completed".
    -   Moved "Rust Target" to "Completed".
    -   Updated priority ranking to reflect these completions.
-   **`docs/GO_TARGET.md`**:
    -   Promoted "Bbolt Database Support" from "Future Enhancements" to "Current Features".

## Motivation

The `FUTURE_WORK.md` document listed `bbolt` integration and the Rust target as future/planned work, but code inspection revealed these features are already implemented. This update aligns the documentation with reality.

## Verification

-   [x] Verified `bbolt` code exists in `src/unifyweaver/targets/go_target.pl`.
-   [x] Verified Rust target code exists in `src/unifyweaver/targets/rust_target.pl`.
