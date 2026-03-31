# PR: Add WAM Target Compiler and Symbolic Runtime Emulator

## Summary
This PR adds a new code generation target for the Warren Abstract Machine (WAM), serving as a universal low-level fallback hub for complex Prolog predicates. It includes a robust compiler, an enhanced symbolic runtime with backtracking support for verification, and comprehensive testing.

## Changes
- **WAM Target Compiler** (`src/unifyweaver/targets/wam_target.pl`):
    - Compiles Prolog predicates to symbolic WAM instructions.
    - **Backtracking Support**: Emits `try_me_else`, `retry_me_else`, and `trust_me` for multi-clause predicates.
    - **Memory Management**: Implements `allocate`/`deallocate` for rule environments.
    - **Variable Mapping**: Robust X-register allocation for temporary variables, including support for compound arguments in body goals.
    - **Unification**: Full support for `get_constant`, `get_variable`, `get_value`, and `get_structure`.
    - **Term Construction**: Support for `put_constant`, `put_variable`, and `put_value`.
    - **Optimization**: Implements Tail Call Optimization (TCO) via the `execute` instruction.
- **Enhanced Symbolic WAM Runtime** (`src/unifyweaver/runtime/wam_runtime.pl`):
    - A symbolic emulator to execute and verify WAM assembly.
    - **backtracking Engine**: Uses a Choice Point (CP) stack to handle non-deterministic execution.
    - **Instruction Parser**: Robustly parses WAM assembly strings into executable terms.
    - **Architecture**: Implements a dedicated `CP` (Continuation Pointer) register to properly separate return addresses from the environment stack, preventing frame collisions.
- **Template System Integration**:
    - Registered a new `wam_module` template in `template_system.pl`.
    - Added `templates/targets/wam/module.mustache` for structured WAM output.
- **Documentation Updates**:
    - Updated `README.md`, `docs/ARCHITECTURE.md`, and core design docs to integrate WAM as a universal fallback hub.
- **Test Suites**:
    - `tests/test_wam_target.pl`: Unit tests for facts, rules, recursion, and module templates.
    - `tests/test_wam_e2e.pl`: True end-to-end tests verifying Compile -> Parse -> Load -> Execute with backtracking.

## Verification
```bash
# Run target unit tests
swipl -g run_tests -t halt tests/test_wam_target.pl

# Run E2E execution tests (includes backtracking verification)
swipl -g run_tests -t halt tests/test_wam_e2e.pl

# Run internal runtime unit test
swipl -g "use_module('src/unifyweaver/runtime/wam_runtime'), test_wam_runtime, halt"
```

## Related
- Documentation PR in `education` repo: "docs: add Book 17 - WAM Target"
