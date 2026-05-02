# WAM-to-LLVM IR Transpilation: Implementation Plan

## Phase 0: LLVM WAM Binding Registry for Prolog Builtins (PREREQUISITE)

**Goal:** Define LLVM IR equivalents for the Prolog builtins used by
`wam_runtime.pl`, so the native lowering pipeline can translate them.

**Scope:**
- Register bindings for register access → `getelementptr` + `load`/`store`
  on the `[64 x %Value]` array
- Register bindings for list operations → runtime helper calls
- Register bindings for arithmetic → native LLVM instructions
  (`add`, `sub`, `mul`, `sdiv`) on extracted payloads
- Register bindings for type checks → `icmp eq` on tag field
- Register bindings for `=../2` → `%Compound` construction/destructure

**Changes:**
1. Create `src/unifyweaver/bindings/llvm_wam_bindings.pl`:
   ```prolog
   :- module(llvm_wam_bindings, [llvm_wam_binding/5]).

   %% llvm_wam_binding(+PrologPred/Arity, +LLVMPattern, +ArgMap, +ReturnMap, +Props)

   % Register access (replaces assoc operations)
   llvm_wam_binding(get_assoc/3,
       "getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 ~reg_idx",
       [reg_idx-i32], [value-value], [pure, pattern(gep_load)]).
   llvm_wam_binding(put_assoc/4,
       "store %Value ~val, %Value* ~ptr",
       [reg_idx-i32, val-value], [result-void], [mutating, pattern(gep_store)]).

   % Arithmetic on extracted payloads
   llvm_wam_binding('+'/3, "add i64 ~a, ~b",
       [a-i64, b-i64], [result-i64], [pure, pattern(binary_op)]).
   llvm_wam_binding('-'/3, "sub i64 ~a, ~b",
       [a-i64, b-i64], [result-i64], [pure, pattern(binary_op)]).
   llvm_wam_binding('*'/3, "mul i64 ~a, ~b",
       [a-i64, b-i64], [result-i64], [pure, pattern(binary_op)]).

   % Type checks on tag field
   llvm_wam_binding(atom/1, "icmp eq i32 ~tag, 0",
       [tag-i32], [result-i1], [pure, pattern(tag_check)]).
   llvm_wam_binding(number/1,
       "or i1 (icmp eq i32 ~tag, 1), (icmp eq i32 ~tag, 2)",
       [tag-i32], [result-i1], [pure, pattern(tag_check)]).
   llvm_wam_binding(compound/1, "icmp eq i32 ~tag, 3",
       [tag-i32], [result-i1], [pure, pattern(tag_check)]).
   llvm_wam_binding(is_list/1, "icmp eq i32 ~tag, 4",
       [tag-i32], [result-i1], [pure, pattern(tag_check)]).

   % Value construction
   llvm_wam_binding(empty_assoc/1, "zeroinitializer",
       [], [result-value], [pure]).

   % String/atom operations (delegate to C runtime)
   llvm_wam_binding(sub_atom/5, "call i8* @strstr(i8* ~haystack, i8* ~needle)",
       [haystack-ptr, needle-ptr], [result-ptr], [pure, requires(libc)]).
   llvm_wam_binding(format/2, "call i32 @snprintf(i8* ~buf, i64 ~len, i8* ~fmt, ...)",
       [buf-ptr, len-i64, fmt-ptr], [result-i32], [impure, requires(libc)]).
   ```

2. Create compile-time register name → index mapping:
   ```prolog
   %% reg_name_to_index(+Name, -Index)
   reg_name_to_index(Name, Index) :-
       atom_concat('A', NumAtom, Name), !,
       atom_number(NumAtom, Num),
       Index is Num - 1.              % A1 → 0, A2 → 1, ...
   reg_name_to_index(Name, Index) :-
       atom_concat('X', NumAtom, Name), !,
       atom_number(NumAtom, Num),
       Index is Num + 15.             % X1 → 16, X2 → 17, ...
   ```

3. Register these bindings with the LLVM target's `is_builtin_goal`
   detection.

**Effort:** Medium — the array-based register model is simpler than
the hash map model in Rust/Go, but type checks and boxing/unboxing
require more explicit code.

**Risk:** Low — mechanical mapping work.

## Phase 1: Mustache Templates for LLVM WAM Module

**Goal:** Define the LLVM IR module skeleton that the transpiled WAM
runtime will live in.

**Scope:**
- `templates/targets/llvm_wam/types.ll.mustache` — type definitions
- `templates/targets/llvm_wam/value.ll.mustache` — `%Value` constructors
  and inspectors
- `templates/targets/llvm_wam/state.ll.mustache` — `%WamState`
  management functions
- `templates/targets/llvm_wam/arena.ll.mustache` — arena allocator
- `templates/targets/llvm_wam/runtime.ll.mustache` — `@step`, `@run_loop`
- `templates/targets/llvm_wam/module.ll.mustache` — top-level assembly

**Changes:**
1. Create template files in `templates/targets/llvm_wam/`
2. Register templates in `template_system.pl` via `template/2` facts
3. Add `compile_wam_llvm_module/3` to a new `wam_llvm_target.pl`
   module that orchestrates template rendering + code generation

**Template composition strategy:**
- `types.ll.mustache` emits all struct definitions
- `value.ll.mustache` emits constructor/inspector functions
- `state.ll.mustache` emits `@wam_state_new`, register access, heap
  allocation
- `arena.ll.mustache` emits `@arena_alloc`, `@arena_rewind`
- `runtime.ll.mustache` has `{{step_switch_cases}}` placeholder filled
  by compiling `step_wam/3` clauses to LLVM `switch` cases
- `module.ll.mustache` ties everything together with `{{native_predicates}}`
  and `{{wam_predicates}}` sections

**LLVM-specific template considerations:**
- LLVM IR is a single flat file (no modules/packages/crates) — simpler
  than Rust/Go templates
- All type definitions must appear before use (forward references via
  opaque types if needed)
- External declarations (`declare`) for C runtime functions
- Target triple and datalayout are parameterized for cross-compilation

**Effort:** Medium — more boilerplate than Rust (explicit GEP, no
sugar) but only one output file.

**Risk:** Low — follows established template patterns.

## Phase 2: Transpile `step_wam/3` to LLVM `switch`

**Goal:** Compile the multi-clause `step_wam/3` predicate to an LLVM
`switch` instruction, producing the core instruction dispatcher.

**Scope:**
- Extend `wam_llvm_target.pl` to handle compound-head multi-clause
  dispatch as LLVM `switch i32 %tag, label %default [...]`
- Each `step_wam` clause body → one switch label's basic block, lowered
  via `clause_body_analysis` with LLVM WAM bindings
- Handle state threading: `wam_state(PC, R, S, H, T, CP, CPS, Code, L)`
  fields → `getelementptr %WamState, ... i32 0, i32 N` for field N

**Changes:**
1. Add `compile_step_wam_to_llvm/2` in `wam_llvm_target.pl`:
   ```prolog
   compile_step_wam_to_llvm(Clauses, LLVMCode) :-
       %% Group step_wam/3 clauses by first-arg functor
       group_by_instruction_type(Clauses, Groups),
       %% Each group → one basic block label
       maplist(compile_instruction_block, Groups, Blocks),
       %% Assemble into switch instruction
       build_switch_dispatch(Blocks, LLVMCode).
   ```

2. Instruction functor → tag mapping:
   ```prolog
   wam_instr_tag(get_constant, 0).
   wam_instr_tag(get_variable, 1).
   wam_instr_tag(get_value, 2).
   wam_instr_tag(get_structure, 3).
   wam_instr_tag(get_list, 4).
   wam_instr_tag(unify_variable, 5).
   wam_instr_tag(unify_value, 6).
   wam_instr_tag(unify_constant, 7).
   wam_instr_tag(put_constant, 8).
   wam_instr_tag(put_variable, 9).
   wam_instr_tag(put_value, 10).
   wam_instr_tag(put_structure, 11).
   wam_instr_tag(put_list, 12).
   wam_instr_tag(set_variable, 13).
   wam_instr_tag(set_value, 14).
   wam_instr_tag(set_constant, 15).
   wam_instr_tag(allocate, 16).
   wam_instr_tag(deallocate, 17).
   wam_instr_tag(call, 18).
   wam_instr_tag(execute, 19).
   wam_instr_tag(proceed, 20).
   wam_instr_tag(builtin_call, 21).
   wam_instr_tag(try_me_else, 22).
   wam_instr_tag(retry_me_else, 23).
   wam_instr_tag(trust_me, 24).
   wam_instr_tag(switch_on_constant, 25).
   wam_instr_tag(switch_on_structure, 26).
   wam_instr_tag(switch_on_constant_a2, 27).
   ```

3. State field GEP mapping:
   ```prolog
   wam_state_gep(pc,            "i32 0, i32 0").
   wam_state_gep(regs,          "i32 0, i32 1").
   wam_state_gep(stack,         "i32 0, i32 2").
   wam_state_gep(stack_size,    "i32 0, i32 3").
   wam_state_gep(heap,          "i32 0, i32 5").
   wam_state_gep(heap_size,     "i32 0, i32 6").
   wam_state_gep(trail,         "i32 0, i32 8").
   wam_state_gep(trail_size,    "i32 0, i32 9").
   wam_state_gep(cp,            "i32 0, i32 11").
   wam_state_gep(choice_points, "i32 0, i32 12").
   wam_state_gep(cp_count,      "i32 0, i32 13").
   wam_state_gep(code,          "i32 0, i32 15").
   wam_state_gep(code_length,   "i32 0, i32 16").
   wam_state_gep(labels,        "i32 0, i32 17").
   wam_state_gep(halted,        "i32 0, i32 19").
   ```

4. SSA variable naming: each basic block uses a fresh `%` prefix
   namespace (e.g., `%gc.0`, `%gc.1` for get_constant block). This
   avoids SSA name collisions across blocks.

**Effort:** High — this is the core compilation challenge. LLVM IR
requires explicit `getelementptr`, `load`, `store`, and SSA `phi`
nodes where Rust/Go use `self.field` and Go uses `vm.Field`.

**Risk:** Medium — clause body analysis handles most patterns, but
LLVM's SSA form requires careful handling of variable definitions
(every `%var` must dominate all uses).

**Depends on:** Phase 0, Phase 1.

## Phase 3: Transpile Helper Predicates

**Goal:** Compile the remaining `wam_runtime.pl` predicates to LLVM IR
functions via native lowering.

**Scope:**
- `run_loop/2` → `define i1 @run_loop(%WamState* %vm)` with `musttail`
- `backtrack/2` → `define i1 @backtrack(%WamState* %vm)`
- `unwind_trail/4` → `define void @unwind_trail(%WamState* %vm, i32 %saved)`
- `eval_arith/5` → `define %Value @eval_arith(%WamState* %vm, %Value %expr)`
- `deref_heap/3` → `define %Value @deref_heap(%WamState* %vm, %Value %val)`
- `is_unbound_var/1` → `define i1 @value_is_unbound(%Value %val)` (inline)
- `trail_binding/4` → `define void @trail_binding(%WamState* %vm, i32 %reg)`
- `unify/3` → `define i1 @unify(%WamState* %vm, %Value %a, %Value %b)`
  (recursive structural unification for compound terms)

**Changes:**
1. Each predicate → attempt native lowering first
2. For predicates that resist → WAM-compile as fallback
3. Wire into the template's `{{helper_functions}}` placeholder

**LLVM-specific considerations:**
- `run_loop` uses `musttail call i1 @run_loop(...)` for constant-stack
  execution — this is a key advantage over Rust/Go implementations
- `backtrack` restores choice point registers via bulk `memcpy` of
  the saved `[64 x %Value]` array
- `unwind_trail` iterates backward through trail entries, restoring
  register values via `getelementptr` + `store`
- `eval_arith` recurses on `%Value` tag: if integer → return payload;
  if compound (op node) → extract children, recurse, apply operation
- `unify` handles compound terms recursively: check functors match,
  then unify corresponding arguments

**Effort:** Medium — most helpers are straightforward. `unify` is the
most complex due to recursive compound term handling.

**Risk:** Low — these are simpler than `step_wam/3`.

**Depends on:** Phase 0.

## Phase 4: WAM Instruction Lowering to LLVM

**Goal:** Compile WAM instructions from `compile_predicate_to_wam/3`
output into LLVM `%Instruction` struct literals (global constant arrays).

**Changes:**
1. Add `wam_to_llvm_instruction/2` to translate each WAM instruction:
   ```prolog
   wam_to_llvm_instruction(get_constant(C, Ai), LLVMCode) :-
       value_to_packed_i64(C, PackedVal),
       reg_name_to_index(Ai, RegIdx),
       format(string(LLVMCode),
           '{ i32 0, i64 ~w, i64 ~w }',
           [PackedVal, RegIdx]).
   ```

2. Add `wam_to_llvm_code_array/3` to produce a global constant:
   ```llvm
   @ancestor_code = private constant [12 x %Instruction] [
     %Instruction { i32 0, i64 4294967296, i64 0 },  ; get_constant(parent, A1)
     %Instruction { i32 18, i64 ptrtoint(...), i64 2 }, ; call(parent/2, 2)
     ; ...
   ]
   @ancestor_code_len = private constant i32 12
   ```

3. Add `wam_to_llvm_labels/2` to produce label index array:
   ```llvm
   @ancestor_labels = private constant [2 x i32] [
     i32 0,   ; clause_1 → PC 0
     i32 5    ; clause_2 → PC 5
   ]
   @ancestor_label_count = private constant i32 2
   ```

**Effort:** Low — mechanical translation of instruction terms to struct
literals. Simpler than Rust/Go because instructions are flat structs
with integer fields rather than nested constructors.

**Depends on:** Phase 0.

## Phase 5: WAM Fallback Integration in LLVM Target

**Goal:** When `compile_predicate_to_llvm` fails native lowering, fall
back to WAM compilation + LLVM codegen.

**Scope:**
- Add final clause to `compile_predicate_to_llvm`:
  ```prolog
  compile_predicate_to_llvm(Pred/Arity, Options, LLVMCode) :-
      % All native tiers failed — fall back to WAM
      wam_target:compile_predicate_to_wam(Pred/Arity, Options, WamCode),
      wam_to_llvm_code_array(Pred/Arity, WamCode, InstrArray),
      wam_to_llvm_labels(Pred/Arity, WamCode, LabelArray),
      compile_wam_wrapper_llvm(Pred/Arity, InstrArray, LabelArray, LLVMCode).
  ```

- Generate wrapper function:
  ```llvm
  define i1 @ancestor(%WamState* %vm_or_null, %Value %a1, %Value %a2) {
  entry:
    ; Create fresh VM if caller didn't provide one
    %has_vm = icmp ne %WamState* %vm_or_null, null
    br i1 %has_vm, label %use_existing, label %create_new

  create_new:
    %new_vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([12 x %Instruction], [12 x %Instruction]* @ancestor_code, i32 0, i32 0),
      i32 12,
      i32* getelementptr ([2 x i32], [2 x i32]* @ancestor_labels, i32 0, i32 0),
      i32 2)
    br label %run

  use_existing:
    br label %run

  run:
    %vm = phi %WamState* [%new_vm, %create_new], [%vm_or_null, %use_existing]
    call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
    call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
    %result = call i1 @run_loop(%WamState* %vm)
    ret i1 %result
  }
  ```

- Register natively-lowered predicates as builtins so WAM-compiled
  code can call them via `BuiltinCall`

- Generate the builtin dispatch table:
  ```prolog
  compile_builtin_dispatch(NativePredicates, LLVMCode) :-
      %% For each native predicate, generate a strcmp check
      %% and unbox/call/box bridge in the builtin_call handler
      ...
  ```

**Effort:** Medium — assembly and routing logic, plus the
boxing/unboxing interop bridge.

**Risk:** Medium — interop between the two value tiers needs careful
attention to type safety.

**Depends on:** Phase 2, Phase 3, Phase 4.

## Phase 6: End-to-End Testing & Validation

**Goal:** Compile a mixed-complexity Prolog module to LLVM IR, run it
through `opt` + `llc` or `lli` (JIT), verify correctness.

**Scope:**
- Test module with facts, rules, recursive predicates, and predicates
  requiring WAM fallback
- `opt -verify` validation of generated IR (catches SSA violations)
- `lli` JIT execution for quick testing (no compilation needed)
- `opt -O2 | llc` native compilation for performance validation
- Comparison of Prolog-runtime results vs LLVM-compiled results

**Changes:**
1. Create `tests/test_wam_llvm_target.pl`:
   - `test_step_wam_generation` — verify `switch` cases for all
     instruction types
   - `test_helper_generation` — helper functions (run_loop, backtrack,
     unwind_trail, eval_arith)
   - `test_value_constructors` — boxing/unboxing roundtrips
   - `test_full_module_generation` — complete module with types + runtime

2. Create `tests/integration/test_llvm_wam_pipeline.sh`:
   - Compile test Prolog to `.ll`
   - Run `opt -verify` to validate IR
   - Run `lli` to execute and check output
   - Optionally: `opt -O2 | llc -filetype=obj` + link + execute

3. CI integration

**Effort:** Medium.

**Risk:** Low — validation is mechanical. `lli` provides a fast
feedback loop without requiring full compilation.

**Depends on:** Phase 5.

## Phase 7: Cross-Platform & WebAssembly (Future)

**Goal:** Extend the WAM-hybrid LLVM module to compile for WebAssembly
and other architectures.

**Scope:**
- The existing LLVM target already has WASM support (`compile_wasm_module/3`)
- Extend it to include WAM-compiled predicates in WASM output
- Arena allocator is WASM-friendly (linear memory, no GC needed)
- Replace C runtime calls (`malloc`, `strcmp`, `snprintf`) with WASM
  equivalents or embed minimal implementations

**Changes:**
1. Arena allocator uses WASM linear memory (`memory.grow`)
2. String comparison inlined (no libc dependency)
3. JavaScript/TypeScript bindings for WAM-compiled predicates
4. Target triple: `wasm32-unknown-unknown` or `wasm32-wasi`

**Effort:** Medium — mostly parameterizing existing templates.

**Risk:** Low — WASM is a well-supported LLVM backend.

**Depends on:** Phase 6.

## Priority and Dependencies

```
Phase 0 (LLVM WAM bindings) ← independent
  ↓
Phase 1 (Mustache templates for .ll module) ← independent
  ↓
Phase 2 (step_wam/3 → LLVM switch) ← depends on Phase 0, 1
  ↓
Phase 3 (helper predicates → LLVM functions) ← depends on Phase 0
  ↓
Phase 4 (WAM instructions → LLVM struct literals) ← depends on Phase 0
  ↓
Phase 5 (WAM fallback in LLVM target) ← depends on Phase 2, 3, 4
  ↓
Phase 6 (E2E testing) ← depends on Phase 5
  ↓
Phase 7 (WASM + cross-platform) ← depends on Phase 6
```

Phases 0 and 1 can proceed in parallel. Phases 3 and 4 can proceed
in parallel with Phase 2 after Phase 0 is complete.

## Metrics

| Phase | Templates | Native Lowering | New Tests | Risk |
|-------|-----------|-----------------|-----------|------|
| 0 | 0 | 15+ bindings | 5 | Low |
| 1 | 6 files | 0 | 2 | Low |
| 2 | 0 | 1 major predicate | 5 | Medium |
| 3 | 0 | 8+ predicates | 10 | Low |
| 4 | 0 | instruction lowering | 5 | Low |
| 5 | 1 wrapper | 1 fallback path + interop | 5 | Medium |
| 6 | 0 | 0 | 10+ | Low |
| 7 | 2 (WASM variants) | WASM adaptations | 5 | Low |

## Phase Summary

| Phase | Description | Effort | Depends On |
|-------|-------------|--------|------------|
| 0 | LLVM WAM bindings registry | Medium | — |
| 1 | Mustache templates for .ll module | Medium | — |
| 2 | step_wam/3 → LLVM switch dispatch | High | Phase 0, 1 |
| 3 | Helper predicates → LLVM functions | Medium | Phase 0 |
| 4 | WAM instructions → LLVM struct literals | Low | Phase 0 |
| 5 | WAM fallback + interop bridge | Medium | Phase 2, 3, 4 |
| 6 | E2E testing (opt -verify, lli, llc) | Medium | Phase 5 |
| 7 | WASM + cross-platform | Medium | Phase 6 |

## Future Extensions

Once the LLVM WAM pipeline works, it enables:

- **JIT compilation via LLVM ORC**: Load WAM-compiled predicates at
  runtime, JIT-compile them, swap out interpreted execution for native
  code. This is the holy grail: a Prolog system that starts interpreted
  and progressively compiles hot predicates to native code.

- **Link-time optimization (LTO)**: When both native and WAM-compiled
  predicates are in the same `.ll` module, `opt -O2` can inline the
  interop bridge, potentially eliminating boxing/unboxing overhead for
  frequently-called cross-tier predicates.

- **Profile-guided optimization (PGO)**: Use LLVM's PGO infrastructure
  to optimize the `switch` dispatch in `@step` based on actual
  instruction frequency data.

- **Shared `.so` with WAM**: The existing C ABI export (`dllexport`)
  extends naturally to WAM-compiled predicates, allowing other languages
  to call Prolog predicates that require backtracking via a C-compatible
  interface.
