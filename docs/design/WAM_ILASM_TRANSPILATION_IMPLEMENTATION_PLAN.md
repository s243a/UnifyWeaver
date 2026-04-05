# WAM-to-ILAsm Transpilation: Implementation Plan

## Phase 0: CIL WAM Binding Registry (PREREQUISITE)

**Goal:** Define CIL equivalents for the Prolog builtins used by
`wam_runtime.pl`, so the native lowering pipeline can translate them.

**Scope:**
- Register bindings for register access → `ldelem.ref`/`stelem.ref`
  on the `Value[]` array
- Register bindings for list operations → `List<T>` method calls
- Register bindings for arithmetic → CIL `add`/`sub`/`mul`/`div`
- Register bindings for type checks → `isinst` instructions
- Register bindings for `=../2` → `CompoundValue` field access

**Changes:**
1. Create `src/unifyweaver/bindings/cil_wam_bindings.pl`:
   ```prolog
   :- module(cil_wam_bindings, [cil_wam_binding/5, cil_wam_type_map/2]).

   cil_wam_type_map(assoc, 'class Value[]').
   cil_wam_type_map(list, 'class Value[]').
   cil_wam_type_map(value, 'class Value').
   cil_wam_type_map(atom, 'string').
   cil_wam_type_map(integer, 'int64').
   cil_wam_type_map(float, 'float64').

   cil_wam_binding(get_assoc/3,
       'ldelem.ref',
       [reg_idx-int32], [value],
       [pure, pattern(array_load)]).
   cil_wam_binding(put_assoc/4,
       'stelem.ref',
       [reg_idx-int32, val-value], [],
       [mutating, pattern(array_store)]).
   cil_wam_binding(atom/1,
       'isinst AtomValue',
       [val-value], [bool],
       [pure, pattern(type_check)]).
   ```

2. Reuse compile-time `reg_name_to_index/2` from `llvm_wam_bindings.pl`
   (same ABI: A1→0, X1→16).

**Effort:** Medium — mechanical mapping, simpler than LLVM (no GEP).

## Phase 1: Mustache Templates for CIL WAM Assembly

**Goal:** Define the CIL assembly skeleton.

**Scope:**
- `templates/targets/ilasm_wam/types.il.mustache` — Value class
  hierarchy, WamState, Instruction, ChoicePoint, TrailEntry
- `templates/targets/ilasm_wam/state.il.mustache` — WamState
  constructor, GetReg, SetReg, IncPC, TrailBinding, HeapPush
- `templates/targets/ilasm_wam/runtime.il.mustache` — step method
  (switch dispatch), run_loop (.tail call), helper methods
- `templates/targets/ilasm_wam/module.il.mustache` — full assembly
  wrapper (.assembly, .class, method aggregation)

**CIL-specific template considerations:**
- All types must be in a single `.class` (or multiple nested classes)
- Method signatures need explicit `.maxstack` declarations
- `.locals init (...)` declares all local variables upfront
- `.entrypoint` on the Main method
- `.assembly extern mscorlib {}` for BCL references

**Effort:** Medium — more boilerplate than LLVM (explicit class
structure), but simpler than Rust (no generics/lifetimes).

## Phase 2: Transpile `step_wam/3` to CIL `switch`

**Goal:** Compile the multi-clause `step_wam/3` predicate to a CIL
`switch` instruction, producing the core instruction dispatcher.

**Scope:**
- Each `step_wam` clause body → one switch label's CIL code
- State threading via `ldarg.0` (WamState) field access
- Register access via `ldfld Value[] WamState::Regs` + `ldelem.ref`

**Changes:**
1. Add `compile_step_wam_to_cil/2` in `wam_ilasm_target.pl`:
   ```prolog
   compile_step_wam_to_cil(Options, CILCode) :-
       findall(Case, compile_cil_step_case(Case), Cases),
       atomic_list_concat(Cases, '\n', CasesCode),
       format(atom(CILCode),
   '.method public static bool step(class WamState vm, class Instruction instr) cil managed {
       .maxstack 8
       ldarg.1
       ldfld int32 Instruction::Tag
       switch (~w)
       ldc.i4.0
       ret
   ~w
   }', [SwitchLabels, CasesCode]).
   ```

2. CIL instruction blocks use stack-based patterns:
   ```il
   L_get_constant:
       ldarg.0                      // vm
       ldfld class Value[] WamState::Regs
       ldarg.1                      // instr
       ldfld int64 Instruction::Op2
       conv.i4                      // reg index
       ldelem.ref                   // current value
       callvirt instance bool Value::IsUnbound()
       brtrue L_gc_bind
       // ... check equality ...
   ```

**Effort:** High — the core compilation challenge. Stack-based codegen
requires careful push/pop balancing per path.

**Depends on:** Phase 0, Phase 1.

## Phase 3: Transpile Helper Predicates

**Goal:** Compile the remaining `wam_runtime.pl` predicates to CIL
methods via native lowering.

**Scope:**
- `run_loop/2` → `.method static bool run_loop(WamState)` with
  `.tail call` for constant-stack execution
- `backtrack/2` → `.method static bool backtrack(WamState)` with
  `List<ChoicePoint>` access and register array clone
- `unwind_trail/4` → `.method static void unwind_trail(WamState, int32)`
  iterating `List<TrailEntry>` backward
- `eval_arith/5` → `.method static int64 eval_arith(WamState, Value)`
  with type dispatch via `isinst`
- `execute_builtin/3` → `.method static bool execute_builtin(WamState, int32, int32)`
  with `switch` dispatch on builtin op ID

**CIL-specific advantages:**
- `backtrack` uses `Value[].Clone()` for register save/restore —
  one method call instead of LLVM's `memcpy`
- `unwind_trail` iterates `List<TrailEntry>` with `get_Count()` and
  indexer — no manual pointer arithmetic
- `eval_arith` uses `isinst IntegerValue` / `isinst CompoundValue`
  for type dispatch — cleaner than tag integer comparison

**Effort:** Medium — most helpers are straightforward. GC simplifies
state management significantly.

**Depends on:** Phase 0.

## Phase 4: WAM Instruction Lowering to CIL

**Goal:** Compile WAM instructions from `compile_predicate_to_wam/3`
output into CIL `Instruction` constructor calls (static field arrays).

**Changes:**
1. Add `wam_to_cil_instruction/2`:
   ```prolog
   wam_to_cil_instruction(get_constant(C, Ai), CILCode) :-
       pack_value(C, PackedVal),
       reg_name_to_index(Ai, RegIdx),
       format(atom(CILCode),
           'new Instruction(0, ~wL, ~wL)', [PackedVal, RegIdx]).
   ```

2. Generate static field arrays:
   ```il
   .field public static class Instruction[] ancestor_code
   .method private static void .cctor() cil managed {
       ldc.i4 12
       newarr Instruction
       // ... store each instruction ...
       stsfld class Instruction[] ancestor_code
   }
   ```

3. Two-pass label resolution (same approach as LLVM target).

**Effort:** Low — mechanical translation, simpler than LLVM (no
struct literal syntax).

**Depends on:** Phase 0.

## Phase 5: WAM Fallback Integration in ILAsm Target

**Goal:** When `compile_predicate_to_ilasm` fails native lowering,
fall back to WAM compilation + CIL codegen.

**Scope:**
- Add final clause to `compile_predicate_to_ilasm`:
  ```prolog
  compile_predicate_to_ilasm(Pred/Arity, Options, ILCode) :-
      % All native tiers failed — fall back to WAM
      wam_target:compile_predicate_to_wam(Pred/Arity, Options, WamCode),
      compile_wam_predicate_to_cil(Pred/Arity, WamCode, Options, ILCode).
  ```

- `wam_fallback(false)` option to disable WAM fallback
- Register natively-lowered predicates as builtins for WAM interop
- `write_wam_ilasm_project/3` for full assembly generation

**Effort:** Medium.

**Depends on:** Phase 2, Phase 3, Phase 4.

## Phase 6: End-to-End Testing & Validation

**Goal:** Compile a mixed-complexity Prolog module to CIL, assemble
with `ilasm`, run on `mono`/`dotnet`, verify correctness.

**Scope:**
- Test module with facts, rules, recursive predicates, and predicates
  requiring WAM fallback
- `ilasm` assembly validation
- `mono`/`dotnet run` execution for correctness
- Comparison of Prolog-runtime results vs CIL-compiled results

**Changes:**
1. Create `tests/test_wam_ilasm_target.pl` — Prolog-side tests:
   - Step function generation (switch dispatch, all instruction cases)
   - Helper function generation (run_loop, backtrack, unwind_trail,
     execute_builtin, eval_arith)
   - WAM predicate wrapper generation
   - Label resolution
   - Full module assembly

2. Create `tests/integration/test_ilasm_wam_pipeline.sh`:
   - Generate CIL assembly from test Prolog
   - Assemble with `ilasm` (if available)
   - Run with `mono`/`dotnet` (if available)

**Effort:** Medium.

**Depends on:** Phase 5.

## Priority and Dependencies

```
Phase 0 (CIL WAM bindings) ← independent
  ↓
Phase 1 (Mustache templates for .il assembly) ← independent
  ↓
Phase 2 (step_wam/3 → CIL switch) ← depends on Phase 0, 1
  ↓
Phase 3 (helper predicates → CIL methods) ← depends on Phase 0
  ↓
Phase 4 (WAM instructions → CIL arrays) ← depends on Phase 0
  ↓
Phase 5 (WAM fallback in ILAsm target) ← depends on Phase 2, 3, 4
  ↓
Phase 6 (E2E testing) ← depends on Phase 5
```

Phases 0 and 1 can proceed in parallel. Phases 3 and 4 can proceed
in parallel with Phase 2 after Phase 0 is complete.

## Metrics

| Phase | Templates | Native Lowering | New Tests | Risk |
|-------|-----------|-----------------|-----------|------|
| 0 | 0 | 15+ bindings | 5 | Low |
| 1 | 4 files | 0 | 2 | Low |
| 2 | 0 | 1 major predicate | 5 | Medium |
| 3 | 0 | 5+ predicates | 10 | Low |
| 4 | 0 | instruction lowering | 5 | Low |
| 5 | 1 wrapper | 1 fallback path + interop | 5 | Medium |
| 6 | 0 | 0 | 15+ | Low |

## Phase Summary

| Phase | Description | Effort | Depends On |
|-------|-------------|--------|------------|
| 0 | CIL WAM bindings registry | Medium | — |
| 1 | Mustache templates for .il assembly | Medium | — |
| 2 | step_wam/3 → CIL switch dispatch | High | Phase 0, 1 |
| 3 | Helper predicates → CIL methods | Medium | Phase 0 |
| 4 | WAM instructions → CIL Instruction arrays | Low | Phase 0 |
| 5 | WAM fallback + interop bridge | Medium | Phase 2, 3, 4 |
| 6 | E2E testing (ilasm, mono/dotnet) | Medium | Phase 5 |

## Differences from LLVM Implementation

| Aspect | LLVM | CIL (ILAsm) |
|--------|------|-------------|
| Value type | `%Value = { i32, i64 }` | Class hierarchy (AtomValue, IntegerValue, ...) |
| Type checks | `icmp eq i32 %tag, N` | `isinst ClassName` |
| Register access | `getelementptr` + `load`/`store` | `ldelem.ref` / `stelem.ref` |
| Memory | Arena allocation + manual rewind | CLR GC (automatic) |
| Trail unwind | Manual pointer iteration | `List<TrailEntry>` iteration |
| Choice point save | `memcpy` 512 bytes | `Value[].Clone()` |
| Run loop | `musttail call` | `.tail call` |
| Instruction dispatch | `switch i32 %tag` | CIL `switch` (jump table) |
| Function defs | `define i1 @step(...)` | `.method static bool step(...)` |
| Build validation | `opt -passes=verify` | `ilasm /verify` |
| Execution | `lli` (JIT) | `mono` / `dotnet run` |
| Atom storage | Integer IDs (intern table) | `string` (CLR handles interning) |

### CIL Simplifications over LLVM

1. **No atom table needed**: CIL strings are reference types with
   built-in equality. `AtomValue.Name` is a `string` field — no
   integer packing or hash table.

2. **No arena allocator**: GC handles all allocation. Backtracking
   just restores references; orphaned objects are collected.

3. **No SSA constraints**: Stack-based IL doesn't need phi nodes,
   dominance, or unique variable names. Push/pop is enough.

4. **No `memcpy` for state save**: `Array.Clone()` on `Value[]`
   creates a copy in one CLR call.

5. **No pointer provenance issues**: All references are managed by
   the CLR — no `inttoptr`/`getelementptr` reasoning needed.

## Future Optimizations

### `List<T>` vs Pre-sized Arrays

The current implementation uses `List<StackEntry>`, `List<TrailEntry>`,
and `List<ChoicePoint>` for the WAM stack, trail, and choice point
collections. This is idiomatic CLR code and correct, but each `Add()`
call may trigger GC-visible allocation when the list resizes.

For high-performance WAM execution, these could be replaced with
pre-sized arrays (e.g., `StackEntry[256]`) and integer stack pointers,
matching the LLVM target's approach. This would reduce GC pressure at
the cost of a fixed capacity limit (or manual resize logic).

The trade-off: `List<T>` is simpler, correct, and sufficient for the
bootstrapping runtime. Pre-sized arrays are a performance optimization
to consider once profiling shows GC pause times are significant.

## Future Extensions

Once the CIL WAM pipeline works:

- **C# interop**: WAM-compiled predicates callable from C# via
  the assembly reference — enables hybrid C#/Prolog applications.
- **F# integration**: F#'s pattern matching + WAM backtracking.
- **Blazor/WASM**: .NET's Blazor compiles CIL to WASM — the WAM
  runtime could run in the browser via this path.
- **Mono AOT**: Ahead-of-time compilation for mobile (Xamarin/MAUI).
