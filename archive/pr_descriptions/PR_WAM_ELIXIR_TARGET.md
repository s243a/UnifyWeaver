# feat: WAM-Elixir transpilation target

## Summary

- Add `wam_elixir_target.pl` module that transpiles WAM instructions to idiomatic Elixir source code using immutable structs, pattern matching, pipe operators, and recursive control flow
- 26 WAM instruction arms as Elixir `case` expressions covering head unification, body construction, control flow, choice points, indexing, and builtins — all fully implemented (no stubs)
- Functional helper generation: recursive `run/1` loop, `backtrack/1`, `unwind_trail/2`, `unify/3`, `execute_builtin/3`, `eval_arith/2`, plus utility helpers for trail binding, register access, and functor parsing
- Elixir WAM bindings for Map ops, list ops, arithmetic, type checks, and string operations
- Predicate wrapper and Mix project scaffold generation
- 14 tests covering step generation, all instruction categories, choice point state save/restore, Elixir idiom correctness, immutable state updates, and recursive run loop shape

### Files added

| File | Lines | Purpose |
|---|---|---|
| `src/unifyweaver/targets/wam_elixir_target.pl` | ~530 | Core transpiler: 26 instruction arms + helpers + predicate wrapper + Mix project generation |
| `src/unifyweaver/bindings/elixir_wam_bindings.pl` | ~140 | Type mappings and binding declarations (Map, list, arithmetic, type checks, strings) |
| `templates/targets/elixir_wam/wam_runtime.ex.mustache` | 16 | WamRuntime module template with WamState struct |
| `templates/targets/elixir_wam/main.ex.mustache` | 20 | Entry point module template |
| `tests/test_wam_elixir_target.pl` | ~180 | 14 tests across generation, instruction categories, idioms, and correctness |

### Architecture

Follows the established WAM transpilation target pattern (`wam_rust_target`, `wam_go_target`, `wam_jvm_target`):
- **Phase 2**: `wam_elixir_case/2` facts collected via `findall` into a `case instr do` expression inside `def step/2`
- **Phase 3**: Helper functions generated as named `def`/`defp` bodies
- **Assembly**: Combined into `defmodule WamRuntime do ... end`
- **Predicate wrapper**: Parses WAM text into Elixir instruction tuples, wraps in a module with `code/0`, `labels/0`, and `run/1`

WAM state maps naturally to Elixir's functional model:
```elixir
%WamState{pc: 1, cp: :halt, regs: %{}, heap: [], trail: [],
          choice_points: [], stack: [], code: [], labels: %{}}
```

Internal transpilation layer — not registered in `target_registry.pl`, consistent with all other WAM transpilation targets.

## Test plan

- [x] Module loads without errors
- [x] 14/14 tests pass covering:
  - Step/case expression generation
  - All 26 instruction arms present (head unification, body construction, unification, control flow, choice points, indexing, builtins)
  - Compound term instructions fully implemented (get_structure, get_list, put_structure, put_list, unify_variable, unify_value, unify_constant)
  - Choice point code correctly references `choice_points` for state save/restore
  - builtin_call delegates to execute_builtin
  - Elixir idioms: pipe operators (`|>`), `Map.put`, `match?`
  - Immutable state updates (`%{state | ...}`)
  - Recursive run loop (not imperative)
- [x] External review confirmed all compound term instructions are full implementations, not stubs

🤖 Generated with [Claude Code](https://claude.com/claude-code)
