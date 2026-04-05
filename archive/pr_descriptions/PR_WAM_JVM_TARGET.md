# feat: hybrid WAM-JVM target for Jamaica and Krakatau assembly

## Summary

- Add a single `wam_jvm_target.pl` module that transpiles WAM instructions to JVM bytecode in both Jamaica (symbolic) and Krakatau (numeric) assembly formats, parameterized on `OutputFmt`
- 26 WAM instruction arms covering head unification, body construction, control flow, choice points, indexing, and builtins
- Format dispatch layer abstracting class/method syntax, comment style, invoke format, field declarations, and getfield/putfield between the two JVM assembly formats
- JVM WAM bindings for HashMap, ArrayList, boxing/unboxing, type checks, arithmetic, and string operations
- Mustache templates for WamState class and main entry point in both formats
- 19 tests covering format dispatch, step generation, runtime assembly, both-format parity, choice point bytecode, field/getfield/putfield dispatch, and builtin delegation

### Files added

| File | Lines | Purpose |
|---|---|---|
| `src/unifyweaver/targets/wam_jvm_target.pl` | ~710 | Core transpiler: format dispatch + 26 instruction arms + helpers + predicate wrapper + project generation |
| `src/unifyweaver/bindings/jvm_wam_bindings.pl` | ~180 | Type mappings and binding declarations (HashMap, ArrayList, boxing, arithmetic, strings) |
| `templates/targets/jvm_wam/WamState.jamaica.mustache` | 38 | Jamaica WamState class template |
| `templates/targets/jvm_wam/WamState.krakatau.mustache` | 67 | Krakatau WamState class template |
| `templates/targets/jvm_wam/main.jamaica.mustache` | 25 | Jamaica entry point template |
| `templates/targets/jvm_wam/main.krakatau.mustache` | 30 | Krakatau entry point template |
| `tests/test_wam_jvm_target.pl` | ~230 | 19 tests: format dispatch, parity, choice points, field decls, runtime assembly |

### Architecture

Follows the established WAM transpilation target pattern (`wam_rust_target`, `wam_go_target`, `wam_llvm_target`):
- **Phase 2**: `wam_jvm_case/2` facts collected via `findall` into a step method
- **Phase 3**: Helper methods (run loop, backtrack, unwind trail)
- **Phase 4**: Assembly into complete WamState class
- **Phase 5**: Predicate wrapper + project generation

The module is an internal transpilation layer (not registered in `target_registry.pl`), consistent with how all other WAM transpilation targets are structured.

## Test plan

- [x] Module loads without errors
- [x] 19/19 tests pass covering:
  - Format dispatch (comments, class headers, method headers, field decls, getfield/putfield, invoke format)
  - Step generation for both Jamaica and Krakatau formats
  - Helper method generation for both formats
  - Full runtime assembly for both formats
  - All 26 instruction arms present and exercised in both formats (parity test)
  - Choice point instructions present with correct bytecode content
  - Unification instructions present
  - builtin_call delegates to executeBuiltin
- [x] 2 rounds of external review — all identified issues resolved

🤖 Generated with [Claude Code](https://claude.com/claude-code)
