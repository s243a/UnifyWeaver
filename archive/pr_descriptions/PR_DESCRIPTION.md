# Feature: Go Target Advanced Schema Validation

## Summary
This PR adds advanced schema validation capabilities to the Go target. Users can now define constraints on JSON fields, such as range checks, format validation, and optionality. The generated Go code enforces these constraints efficiently during stream processing.

## Changes
-   **`src/unifyweaver/targets/go_target.pl`**:
    -   Updated `json_schema/2` to accept `field(Name, Type, Options)`.
    -   Implemented validation logic generation for:
        -   `min(N)`, `max(N)` (integer/float)
        -   `format(email)` (string)
        -   `optional` (skip validation if missing, don't skip record)
    -   Added dynamic import of `strings` package when needed.

## Usage
```prolog
:- json_schema(user, [
    field(age, integer, [min(18)]),
    field(email, string, [format(email)]),
    field(nickname, string, [optional])
]).
```

## Verification
-   Added `tests/test_go_validation.pl` which verifies:
    -   Valid records are processed.
    -   Records violating `min`, `max`, or `format` are skipped.
    -   Records missing `optional` fields are processed (not skipped).
