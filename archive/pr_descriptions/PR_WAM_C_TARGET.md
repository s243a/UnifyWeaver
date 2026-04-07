# feat: WAM-C transpilation target

## Summary

- Add `wam_c_target.pl` module that transpiles WAM instructions to C source code using structs, tagged unions, dynamic arrays, and manual memory management
- 26 WAM instruction arms as C `switch` cases covering head unification, body construction, control flow, choice points, indexing, and builtins — all fully implemented
- 8 helper functions generated: `wam_run` (while-loop), `wam_backtrack`, `wam_unwind_trail`, `wam_unify`, `wam_execute_builtin`, `wam_eval_arith`, `wam_trail_binding`, `wam_resolve_label`
- C WAM bindings for register map ops, dynamic array ops, value constructors, type checks, arithmetic, and string/memory operations
- Runtime header template (`wam_runtime.h`) with all typedefs, enums, structs, and function prototypes
- Makefile template with `gcc -O2 -Wall -std=c99`
- 15 tests covering step generation, all instruction categories, choice point semantics, and C idioms

### Files added

| File | Lines | Purpose |
|---|---|---|
| `src/unifyweaver/targets/wam_c_target.pl` | ~550 | Core transpiler: 26 instruction arms + helpers + predicate wrapper + Makefile generation |
| `src/unifyweaver/bindings/c_wam_bindings.pl` | ~170 | Type mappings and binding declarations (register map, arrays, value constructors, type checks, arithmetic, strings) |
| `templates/targets/c_wam/wam_runtime.h.mustache` | ~130 | Header with WamValue tagged union, WamState struct, instruction enum, function prototypes |
| `templates/targets/c_wam/Makefile.mustache` | ~19 | gcc build template |
| `tests/test_wam_c_target.pl` | ~200 | 15 tests across generation, instruction categories, and C idioms |

### Architecture

Follows the established WAM transpilation target pattern (`wam_rust_target`, `wam_go_target`, `wam_elixir_target`):
- **Phase 2**: `wam_c_case/2` facts collected via `findall` into a `switch (instr->tag)` body inside `wam_step()`
- **Phase 3**: Helper functions generated as C function bodies
- **Assembly**: Combined into complete `.c` file with `#include` directives
- **Predicate wrapper**: Parses WAM text into `wam_emit_*()` calls that populate the instruction array

WAM state uses idiomatic C:
```c
typedef struct {
    int pc, cp;
    char *reg_keys[WAM_MAX_REGS];   // string-keyed register map
    WamValue *reg_vals[WAM_MAX_REGS];
    WamValue **heap;                  // dynamic arrays with size/cap
    TrailEntry *trail;
    ChoicePoint *choice_points;
    WamInstr *code;
    // ...
} WamState;
```

Values as tagged unions (`VAL_ATOM`, `VAL_INT`, `VAL_REF`, `VAL_STR`, `VAL_UNBOUND`, `VAL_LIST`).

Internal transpilation layer — not registered in `target_registry.pl`, consistent with all other WAM transpilation targets.

## Test plan

- [x] Module loads without errors
- [x] 15/15 tests pass covering:
  - `wam_step()` switch generation with instruction tag dispatch
  - All 26 instruction arms present (head unification, body construction, unification, control flow, choice points, indexing, builtins)
  - Choice point code correctly uses `wam_push_choice_point`/`wam_update_choice_point`/`wam_pop_choice_point`
  - `builtin_call` delegates to `wam_execute_builtin`
  - C idioms: pointer access (`state->`), return 1/0, `malloc`/`realloc`/`strdup`, `while` loop run
  - Full runtime assembly includes `#include` directives and header reference
- [x] External review confirmed complete instruction coverage and proper C memory management patterns

🤖 Generated with [Claude Code](https://claude.com/claude-code)
