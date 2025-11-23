# feat(python): Initial implementation of Python target

## Description
This PR implements the initial Python target for UnifyWeaver (Phase 1), enabling the compilation of Prolog predicates into standalone Python scripts.

## Changes
- **New Module**: `src/unifyweaver/targets/python_target.pl`
    - Implements a generator-based pipeline for memory efficiency.
    - Supports `jsonl` (default) and `nul_json` input/output formats.
    - Translates basic Prolog predicates:
        - `get_dict/3` -> `var = record.get('key')`
        - Comparisons (`>/2`) -> `if not (var > val): continue`
        - Unification (`=/2`) -> `var = {'key': val}`
    - Automatically detects output variables.
- **Tests**: `tests/core/test_python_target.pl`
    - Verifies module exports, code structure, filter logic, and projection logic.

## Verification
- All tests in `tests/core/test_python_target.pl` passed.
- Verified generated Python code structure manually.

## Related Issues
- Addresses "Python Target Implementation" task.
