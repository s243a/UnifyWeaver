# ILAsm Target — Implementation Plan

## Context

The shared compile_expression framework (22 targets with hooks,
22 deepened with classify_goal_sequence) changes the implementation
order for new targets. Previously, you'd start with recursion
templates and bindings. Now, the shared framework provides
non-recursive pattern support immediately via hooks — so we start
with the scaffold and hooks, then add depth incrementally.

## Phase 1: Scaffold + Hooks (immediate non-recursive support)

**Goal:** ILAsm target compiles non-recursive predicates using the
shared compile_expression framework.

**Files to create:**

1. `src/unifyweaver/targets/ilasm_target.pl` — module with:
   - `compile_predicate_to_ilasm/3` — entry point
   - `native_ilasm_clause_body` — clause dispatcher
   - `native_ilasm_clause` — per-clause handler
   - `ilasm_guard_condition` — guard rendering (CIL comparisons)
   - `ilasm_output_goal` — output rendering (stloc)
   - `ilasm_branch_value` — branch value extraction
   - `ilasm_expr` — expression to CIL instructions
   - 4 compile_expression hooks (render_output_goal, etc.)
   - classify_goal_sequence integration from day one

2. `src/unifyweaver/core/cil_bytecode.pl` — shared CIL layer:
   - `cil_expr_to_instructions/3`
   - `cil_guard_to_instructions/4`
   - `cil_if_chain/3`
   - `cil_resolve_value/3`
   - CIL operator mapping

3. Wire into `recursive_compiler.pl`:
   - `use_module('../targets/ilasm_target', [])`
   - `compile_non_recursive(ilasm, ...)`

**Test predicates:**
- `classify_sign/2` — multi-clause with guards
- `label/2` — if-then-else output
- `safe_double/2` — guard + arithmetic output

**What works after Phase 1:**
- Multi-clause dispatch (if/br chains)
- Guards (comparison operators)
- Arithmetic output (add/sub/mul/div)
- If-then-else output (via compile_expression hooks)
- Disjunction output (via classify_goal_sequence)
- Guarded tail sequences (output then guard)

## Phase 2: Component Registration

**Goal:** Enable custom IL injection via the component system.

**Files:**

1. `src/unifyweaver/targets/ilasm_runtime/custom_ilasm.pl`:
   - `init_component/2`
   - `invoke_component/4`
   - `compile_component/4`
   - Register as `custom_ilasm` component type

Follow the pattern from `csharp_runtime/custom_csharp.pl` and
`jamaica_target.pl` component registration.

## Phase 3: Binding Registry

**Goal:** Map Prolog operations to .NET BCL calls.

**File:** `src/unifyweaver/bindings/cil_asm_bindings.pl`

Register as `ilasm` target in the binding registry. Follow the
pattern from `jvm_asm_bindings.pl` which registers dual bindings
for both Jamaica and Krakatau.

Key bindings:
- Arithmetic: `add`, `sub`, `mul`, `div`, `rem`
- Math: `System.Math::Sqrt`, `System.Math::Abs`, etc.
- String: `String::get_Length`, `String::Concat`
- I/O: `Console::WriteLine`, `Console::ReadLine`
- Conversion: `conv.i8`, `conv.r8`

Wire into `ilasm_output_goal` and `ilasm_guard_condition` so
classified goals can use bindings.

## Phase 4: Recursion Patterns (multifile dispatch)

**Goal:** Compile recursive predicates to CIL.

Register for each pattern:

### Tail Recursion
```prolog
tail_recursion:compile_tail_pattern(ilasm, PredStr, Arity, ..., Code) :-
    %% CIL has .tail prefix — can express tail calls directly
    %% OR transform to loop like JVM targets
```

CIL's `.tail` prefix is unique — it allows native tail call
optimization without loop transformation. Consider supporting both:
- `.tail call` for simple tail recursion
- Loop transformation for complex patterns

### Linear Recursion
```prolog
linear_recursion:compile_linear_pattern(ilasm, PredStr, Arity, ..., Code) :-
    %% Transform to iterative loop (like JVM targets)
```

### Tree Recursion
```prolog
tree_recursion:compile_tree_pattern(ilasm, Pattern, Pred, Arity, UseMemo, Code) :-
    %% Memoized with Dictionary<int64, int64>
```

### Mutual Recursion
```prolog
mutual_recursion:compile_mutual_pattern(ilasm, Predicates, ..., Code) :-
    %% Multiple static methods calling each other
```

## Phase 5: Transitive Closure Template

**Goal:** Composable TC templates for all 5 input modes.

**Files:**
```
templates/targets/ilasm/
    transitive_closure.mustache
    tc_definitions.mustache
    tc_input_stdin.mustache
    tc_input_embedded.mustache
    tc_input_file.mustache
    tc_input_vfs.mustache      (placeholder — ILAsm not in browser)
    tc_input_function.mustache
    tc_interface_cli.mustache
```

Wire into `compile_transitive_closure(ilasm, ...)` using
`compile_tc_from_template/6`.

## Phase 6: Type Integration

Add `resolve_type(Type, ilasm, CILType)` facts to
`type_declarations.pl` for optional type annotations in
generated IL method signatures.

## Dependencies

```
Phase 1 (scaffold + hooks) ← no dependencies, start here
  ↓
Phase 2 (components) ← independent of Phase 1
  ↓
Phase 3 (bindings) ← independent, can parallel with Phase 2
  ↓
Phase 4 (recursion) ← depends on Phase 1 for basic IL generation
  ↓
Phase 5 (TC templates) ← independent of Phase 4
  ↓
Phase 6 (types) ← independent, can do anytime
```

## Relationship to Existing C# Target

ILAsm does NOT replace the C# target. They serve different needs:

| | C# Target | ILAsm Target |
|---|---|---|
| Output | C# source code | CIL assembly text |
| Requires | C# compiler (csc/dotnet) | ILAsm assembler (ilasm) |
| Use case | High-level, readable | Low-level, precise control |
| Tail calls | Not native in C# | `.tail` prefix in CIL |
| Fallback from | — | Could serve as C# fallback |
| Component system | `custom_csharp` | `custom_ilasm` |

## CIL-Specific Considerations

### Stack Management
CIL is stack-based. Every expression pushes results onto the
evaluation stack. `.maxstack` must be declared. For complex
expressions, track max stack depth during compilation.

### Local Variables
`.locals init (int64 v0, int64 v1, ...)` declares all locals
upfront. VarMap indices map to local slot numbers.

### Method Signatures
`.method public static int64 predname(int64 arg0) cil managed`
— explicit types in signatures. Use `resolve_type` for type
selection.

### Tail Calls
`.tail` prefix before `call` enables tail call optimization.
This is a CIL advantage over JVM — tail recursion can be
expressed directly without loop transformation:
```il
.tail
call int64 PrologGenerated.Program::is_odd(int64)
ret
```
