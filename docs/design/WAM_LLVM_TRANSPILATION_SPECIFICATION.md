# WAM-to-LLVM IR Transpilation: Specification

## Overview

This document specifies the hybrid compilation strategy that produces
LLVM IR modules containing a mix of natively-lowered functions (unboxed
`i64`/`double`/`i1`) and WAM-compiled functions (boxed `%Value` tagged
unions), with a transpiled WAM runtime providing backtracking and
unification support.

## Architecture Layers

```
Layer 1: Predicate Classification
    → native_lowerable | wam_required | builtin

Layer 2: Compilation Strategy Selection
    → native_lowering(llvm) | wam_compile_then_lower(llvm) | builtin_binding(llvm)

Layer 3: Code Generation
    → Mustache templates (module structure) + native lowering (bodies)

Layer 4: WAM Runtime Transpilation
    → wam_runtime.pl → LLVM IR via same pipeline (self-application)

Layer 5: Assembly & Validation
    → opt -O2, llc, or lli (JIT)
```

## LLVM IR Value System

### Two-Tier Design

**Tier 1 — Unboxed (existing LLVM target):**
Natively-lowered predicates use typed LLVM values directly:
```llvm
i64      ; integers
double   ; floats
i1       ; booleans
i8*      ; strings/atoms (C-style)
```

**Tier 2 — Boxed (new WAM support):**
WAM-compiled predicates use a tagged-union `%Value`:

```llvm
; Value = { tag: i32, payload: i64 }
%Value = type { i32, i64 }

; Tag constants
@TAG_ATOM     = private constant i32 0
@TAG_INTEGER  = private constant i32 1
@TAG_FLOAT    = private constant i32 2
@TAG_COMPOUND = private constant i32 3
@TAG_LIST     = private constant i32 4
@TAG_REF      = private constant i32 5
@TAG_UNBOUND  = private constant i32 6
@TAG_BOOL     = private constant i32 7
```

### Value Construction

```llvm
; Integer value: { tag=1, payload=N }
define %Value @value_integer(i64 %n) {
  %v1 = insertvalue %Value { i32 1, i64 undef }, i64 %n, 1
  ret %Value %v1
}

; Float value: { tag=2, payload=bitcast(F) }
define %Value @value_float(double %f) {
  %bits = bitcast double %f to i64
  %v1 = insertvalue %Value { i32 2, i64 undef }, i64 %bits, 1
  ret %Value %v1
}

; Atom value: { tag=0, payload=ptrtoint(name) }
define %Value @value_atom(i8* %name) {
  %ptr = ptrtoint i8* %name to i64
  %v1 = insertvalue %Value { i32 0, i64 undef }, i64 %ptr, 1
  ret %Value %v1
}

; Ref value: { tag=5, payload=addr }
define %Value @value_ref(i64 %addr) {
  %v1 = insertvalue %Value { i32 5, i64 undef }, i64 %addr, 1
  ret %Value %v1
}

; Unbound value: { tag=6, payload=ptrtoint(name) }
define %Value @value_unbound(i8* %name) {
  %ptr = ptrtoint i8* %name to i64
  %v1 = insertvalue %Value { i32 6, i64 undef }, i64 %ptr, 1
  ret %Value %v1
}

; Bool value: { tag=7, payload=zext(b) }
define %Value @value_bool(i1 %b) {
  %ext = zext i1 %b to i64
  %v1 = insertvalue %Value { i32 7, i64 undef }, i64 %ext, 1
  ret %Value %v1
}
```

### Value Inspection

```llvm
define i32 @value_tag(%Value %v) {
  %tag = extractvalue %Value %v, 0
  ret i32 %tag
}

define i64 @value_payload(%Value %v) {
  %p = extractvalue %Value %v, 1
  ret i64 %p
}

define i1 @value_is_unbound(%Value %v) {
  %tag = extractvalue %Value %v, 0
  %is = icmp eq i32 %tag, 6
  ret i1 %is
}

define i1 @value_equals(%Value %a, %Value %b) {
  %tag_a = extractvalue %Value %a, 0
  %tag_b = extractvalue %Value %b, 0
  %tags_eq = icmp eq i32 %tag_a, %tag_b
  br i1 %tags_eq, label %check_payload, label %not_equal

check_payload:
  %pay_a = extractvalue %Value %a, 1
  %pay_b = extractvalue %Value %b, 1
  %pay_eq = icmp eq i64 %pay_a, %pay_b
  ret i1 %pay_eq

not_equal:
  ret i1 false
}
```

### Compound Types (Heap-Allocated)

```llvm
; Compound: functor name + arity + pointer to args array
%Compound = type { i8*, i32, %Value* }

; List: length + pointer to elements array
%List = type { i32, %Value* }
```

Construction of compound values allocates on the WAM heap:

```llvm
define %Value @value_compound(i8* %functor, i32 %arity, %Value* %args) {
  ; Allocate Compound struct on WAM heap
  %cp = call i8* @wam_heap_alloc(i64 24)  ; sizeof(Compound)
  %typed = bitcast i8* %cp to %Compound*
  %f_ptr = getelementptr %Compound, %Compound* %typed, i32 0, i32 0
  store i8* %functor, i8** %f_ptr
  %a_ptr = getelementptr %Compound, %Compound* %typed, i32 0, i32 1
  store i32 %arity, i32* %a_ptr
  %args_ptr = getelementptr %Compound, %Compound* %typed, i32 0, i32 2
  store %Value* %args, %Value** %args_ptr
  ; Pack pointer into Value
  %addr = ptrtoint %Compound* %typed to i64
  %v = insertvalue %Value { i32 3, i64 undef }, i64 %addr, 1
  ret %Value %v
}
```

## WAM State Structure

```llvm
%WamState = type {
  i32,                  ; 0: pc — program counter
  [32 x %Value],       ; 1: regs — argument/temp registers (A1..A16, X1..X16)
  %StackEntry*,        ; 2: stack — environment frames + unify contexts
  i32,                 ; 3: stack_size
  i32,                 ; 4: stack_cap
  %Value*,             ; 5: heap — term construction heap
  i32,                 ; 6: heap_size
  i32,                 ; 7: heap_cap
  %TrailEntry*,        ; 8: trail — binding trail for backtracking
  i32,                 ; 9: trail_size
  i32,                 ; 10: trail_cap
  i32,                 ; 11: cp — continuation pointer
  %ChoicePoint*,       ; 12: choice_points — backtracking stack
  i32,                 ; 13: cp_count
  i32,                 ; 14: cp_cap
  %Instruction*,       ; 15: code — compiled instructions
  i32,                 ; 16: code_length
  i32*,                ; 17: labels — label index → PC array
  i32,                 ; 18: label_count
  i1                   ; 19: halted
}

%StackEntry = type { i32, [16 x %Value] }  ; type tag + saved values

%ChoicePoint = type {
  i32,                 ; saved PC
  [32 x %Value],       ; saved registers
  i32,                 ; trail mark (trail size at creation)
  i32                  ; saved CP
}

%TrailEntry = type {
  i32,                 ; register index
  %Value               ; old value
}
```

### Register Mapping (Compile-Time)

Register names from WAM instructions map to fixed array indices:

```
A1 → 0,  A2 → 1,  A3 → 2,  ..., A16 → 15
X1 → 16, X2 → 17, X3 → 18, ..., X16 → 31
```

This eliminates the need for hash maps. Register access is:

```llvm
; Load register A1 (index 0):
%reg_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
%val = load %Value, %Value* %reg_ptr

; Store to register X3 (index 18):
%reg_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 18
store %Value %new_val, %Value* %reg_ptr
```

## Instruction Representation

```llvm
; Instruction = { tag: i32, operand1: i64, operand2: i64 }
; Two i64 operands suffice for all instruction types.
; Semantics depend on tag.
%Instruction = type { i32, i64, i64 }

; Instruction tag constants:
; Head unification
;   0 = GetConstant    (op1: Value packed, op2: reg index)
;   1 = GetVariable    (op1: Xn index, op2: Ai index)
;   2 = GetValue        (op1: Xn index, op2: Ai index)
;   3 = GetStructure    (op1: functor ptr, op2: Ai index)
;   4 = GetList          (op1: Ai index, op2: unused)
;   5 = UnifyVariable    (op1: Xn index, op2: unused)
;   6 = UnifyValue       (op1: Xn index, op2: unused)
;   7 = UnifyConstant    (op1: Value packed, op2: unused)
;
; Body construction
;   8 = PutConstant     (op1: Value packed, op2: reg index)
;   9 = PutVariable     (op1: Xn index, op2: Ai index)
;  10 = PutValue         (op1: Xn index, op2: Ai index)
;  11 = PutStructure     (op1: functor ptr, op2: Ai index)
;  12 = PutList           (op1: Ai index, op2: unused)
;  13 = SetVariable       (op1: Xn index, op2: unused)
;  14 = SetValue          (op1: Xn index, op2: unused)
;  15 = SetConstant       (op1: Value packed, op2: unused)
;
; Control
;  16 = Allocate          (op1: unused, op2: unused)
;  17 = Deallocate        (op1: unused, op2: unused)
;  18 = Call              (op1: pred name ptr, op2: arity)
;  19 = Execute           (op1: pred name ptr, op2: unused)
;  20 = Proceed           (op1: unused, op2: unused)
;  21 = BuiltinCall       (op1: op name ptr, op2: arity)
;
; Choice points
;  22 = TryMeElse         (op1: label index, op2: unused)
;  23 = RetryMeElse       (op1: label index, op2: unused)
;  24 = TrustMe           (op1: unused, op2: unused)
;
; Indexing
;  25 = SwitchOnConstant  (op1: case table ptr, op2: case count)
;  26 = SwitchOnStructure (op1: case table ptr, op2: case count)
;  27 = SwitchOnConstantA2 (op1: case table ptr, op2: case count)
```

## Predicate Classification

```prolog
%% classify_for_llvm(+Pred/Arity, -Strategy)
classify_for_llvm(Pred/Arity, native) :-
    compile_predicate_to_llvm(Pred/Arity, [], _), !.
classify_for_llvm(Pred/Arity, wam) :-
    compile_predicate_to_wam(Pred/Arity, [], _), !.
classify_for_llvm(Pred/Arity, builtin) :-
    is_builtin_pred(Pred, Arity).
```

## Compilation Pipelines

### For natively-lowered predicates (no change):

```
Prolog clause → clause_body_analysis → LLVM IR function (typed i64/double)
```

### For WAM-compiled predicates:

```
Prolog clause → wam_target:compile_predicate_to_wam → WAM instructions
    → wam_to_llvm_instructions → LLVM %Instruction struct literals
    → wrapped in @predicate(%WamState*) → LLVM function
```

### For the WAM runtime itself:

```
wam_runtime.pl predicates
    → clause_body_analysis + LLVM native lowering
    → i1 @step(%WamState*, %Instruction*) { switch i32 %tag { ... } }
```

## `step_wam/3` Lowering Strategy

The `step_wam/3` predicate is a multi-clause dispatch on the first
argument (instruction type). This maps to LLVM's `switch` instruction:

```prolog
% Prolog (wam_runtime.pl):
step_wam(get_constant(C, Ai), State0, State1) :- ...
step_wam(get_variable(Xn, Ai), State0, State1) :- ...
```

```llvm
; LLVM IR (transpiled):
define i1 @step(%WamState* %vm, %Instruction* %instr) {
entry:
  %tag_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 0
  %tag = load i32, i32* %tag_ptr
  switch i32 %tag, label %default [
    i32 0, label %get_constant
    i32 1, label %get_variable
    i32 2, label %get_value
    ; ... all 28 instruction types
  ]

get_constant:
  ; op1 = value (packed), op2 = register index
  %op1_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 1
  %val_bits = load i64, i64* %op1_ptr
  %c = bitcast i64 %val_bits to %Value        ; unpack constant
  %op2_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 2
  %ai_idx = load i64, i64* %op2_ptr
  %ai = trunc i64 %ai_idx to i32
  ; Load current register value
  %reg_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 %ai
  %current = load %Value, %Value* %reg_ptr
  ; Check if unbound
  %is_unb = call i1 @value_is_unbound(%Value %current)
  br i1 %is_unb, label %gc_bind, label %gc_check

gc_bind:
  store %Value %c, %Value* %reg_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc_check:
  %eq = call i1 @value_equals(%Value %current, %Value %c)
  br i1 %eq, label %gc_match, label %gc_fail

gc_match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc_fail:
  ret i1 false

get_variable:
  ; Copy Ai to Xn
  ; ...
  ret i1 true

default:
  ret i1 false
}
```

Each clause body is lowered via `clause_body_analysis`:
- `get_assoc(Ai, R, Val)` → `getelementptr` + `load` on register array
- `put_assoc(Ai, R, C, NR)` → `getelementptr` + `store` on register array
- `Val == C` → `call i1 @value_equals(%Value, %Value)`
- `is_unbound_var(Val)` → `call i1 @value_is_unbound(%Value)`
- `NPC is PC + 1` → `add i32 %pc, 1` + `store`

## Builtin Mapping Table

| Prolog builtin | LLVM IR equivalent |
|----------------|--------------------|
| `get_assoc/3` | `getelementptr` + `load` (reg array) |
| `put_assoc/4` | `getelementptr` + `store` (reg array) |
| `nth0/3` | `getelementptr` + `load` (array index) |
| `nth1/3` | `sub i32 %i, 1` + `getelementptr` + `load` |
| `append/3` | `call @wam_list_append(...)` (runtime helper) |
| `length/2` | load length field from `%List` |
| `member/2` | loop over list elements with `icmp eq` |
| `format/2` | `call i32 @snprintf(...)` (C runtime) |
| `=../2` (univ) | `%Compound` construction/destructure |
| `atom/1` | `icmp eq i32 %tag, 0` |
| `number/1` | `icmp eq i32 %tag, 1` or `icmp eq i32 %tag, 2` |
| `compound/1` | `icmp eq i32 %tag, 3` |
| `is_list/1` | `icmp eq i32 %tag, 4` |
| `empty_assoc/1` | zeroinitializer for register subset |
| `sub_atom/5` | `call i8* @strstr(...)` (C runtime) |

## Mustache Templates

### Module structure: `templates/targets/llvm_wam/`

**`module.ll.mustache`:**
```llvm
; ModuleID = '{{module_name}}'
source_filename = "{{module_name}}.ll"
target datalayout = "{{target_datalayout}}"
target triple = "{{target_triple}}"

; === Type Definitions ===
{{type_definitions}}

; === External Declarations ===
declare i8* @malloc(i64)
declare void @free(i8*)
declare i32 @snprintf(i8*, i64, i8*, ...)
declare i32 @strcmp(i8*, i8*)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

; === Value Constructors & Inspectors ===
{{value_functions}}

; === WAM State Management ===
{{state_functions}}

; === WAM Runtime (step + run_loop) ===
{{runtime_functions}}

; === WAM Helper Functions ===
{{helper_functions}}

; === Native Predicates (unboxed) ===
{{native_predicates}}

; === WAM-Compiled Predicates (boxed) ===
{{wam_predicates}}

; === Interop Bridge ===
{{interop_bridge}}
```

**`types.ll.mustache`:** `%Value`, `%WamState`, `%Instruction`,
`%ChoicePoint`, `%TrailEntry`, `%StackEntry`, `%Compound`, `%List`.

**`value.ll.mustache`:** Constructor functions (`@value_integer`,
`@value_atom`, etc.), inspection functions (`@value_tag`,
`@value_is_unbound`, `@value_equals`), boxing/unboxing bridge
(`@box_integer`, `@unbox_integer`, etc.).

**`state.ll.mustache`:** `@wam_state_new`, `@wam_set_reg`,
`@wam_get_reg`, `@wam_inc_pc`, `@wam_push_trail`, `@wam_push_cp`,
`@wam_heap_alloc`.

**`runtime.ll.mustache`:** `@step` (transpiled from `step_wam/3`),
`@run_loop` (transpiled from `run_loop/2` with `musttail`),
`@backtrack` (transpiled from `backtrack/2`).

## Interop Calling Convention

### Native calls WAM-compiled predicate:

```llvm
define i1 @query_ancestor(i8* %a, i8* %b) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* @ancestor_code, i32 @ancestor_code_len,
    i32* @ancestor_labels, i32 @ancestor_label_count)
  %val_a = call %Value @value_atom(i8* %a)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %val_a)
  %val_b = call %Value @value_atom(i8* %b)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %val_b)
  %result = call i1 @run_loop(%WamState* %vm)
  call void @free(i8* bitcast (%WamState* %vm to i8*))
  ret i1 %result
}
```

### WAM-compiled calls native predicate:

```llvm
; Inside the BuiltinCall case of @step:
builtin_call:
  %op_ptr = ; load op name string pointer from instruction
  ; Compare op name to known builtins
  %is_factorial = call i32 @strcmp(i8* %op_ptr, i8* @str_factorial_2)
  %cmp = icmp eq i32 %is_factorial, 0
  br i1 %cmp, label %call_factorial, label %next_builtin

call_factorial:
  ; Unbox A1 from register
  %a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %n = extractvalue %Value %a1, 1              ; payload as i64
  ; Call native (unboxed) function
  %result = call i64 @factorial(i64 %n)
  ; Box result and store in A2
  %boxed = call %Value @value_integer(i64 %result)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %boxed)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true
```

## Arena Memory Management

```llvm
%Arena = type {
  i8*,    ; base pointer (from malloc)
  i64,    ; capacity
  i64     ; current offset (bump pointer)
}

define i8* @arena_alloc(%Arena* %arena, i64 %size) {
  %off_ptr = getelementptr %Arena, %Arena* %arena, i32 0, i32 2
  %off = load i64, i64* %off_ptr
  %base_ptr = getelementptr %Arena, %Arena* %arena, i32 0, i32 0
  %base = load i8*, i8** %base_ptr
  %ptr = getelementptr i8, i8* %base, i64 %off
  %new_off = add i64 %off, %size
  store i64 %new_off, i64* %off_ptr
  ret i8* %ptr
}

define void @arena_rewind(%Arena* %arena, i64 %mark) {
  %off_ptr = getelementptr %Arena, %Arena* %arena, i32 0, i32 2
  store i64 %mark, i64* %off_ptr
  ret void
}
```

The WAM heap, trail, and stack can all use arena allocation:
- **Forward execution**: bump-allocate from the arena
- **Backtracking**: rewind the arena to the saved mark
- **Cleanup**: free the entire arena at end of query

## Target Capability Matrix

| Capability | Native LLVM | WAM-Compiled | Both |
|------------|------------|-------------|------|
| Arithmetic | yes (unboxed i64) | yes (boxed) | native preferred |
| Guards/comparisons | yes | yes | native preferred |
| Facts (lookup) | yes | yes | native preferred |
| Tail recursion | yes (musttail) | yes (run_loop musttail) | native preferred |
| Transitive closure | yes (BFS worklist) | yes | native preferred |
| If-then-else | yes (br/phi) | yes | native preferred |
| C ABI export | yes (dllexport) | via wrapper | native preferred |
| WebAssembly | yes (wasm triple) | via wrapper | native preferred |
| Choice points | no | yes | WAM only |
| Deep unification | no | yes | WAM only |
| Mutual recursion + BT | partial | yes | WAM fallback |
| Meta-predicates | no | yes | WAM only |

## Differences from Rust and Go

| Aspect | Rust | Go | LLVM IR |
|--------|------|-----|---------|
| Value type | `enum Value` | `interface Value` | `%Value = { i32, i64 }` (tagged union) |
| Dispatch | `match` expression | `switch type` | `switch i32 %tag` |
| Memory | Ownership/borrowing | GC | Arena allocation |
| Registers | `HashMap<String, Value>` | `map[string]Value` | `[32 x %Value]` fixed array |
| Concurrency | Not planned | Goroutines | Not planned (OS threads via pthreads possible) |
| Package format | Cargo crate | Go module | `.ll` file (single module) |
| Template files | `.rs.mustache` | `.go.mustache` | `.ll.mustache` |
| Build validation | `cargo check` | `go build` | `opt -verify` + `llc` |
| Optimization | Rust compiler (LLVM backend) | Go compiler (gc) | Direct LLVM passes (`opt -O2`) |
| Boxing overhead | All WAM values boxed | All WAM values boxed | Only WAM values boxed; native stays unboxed |
| Run loop | while loop | for loop | `musttail` trampoline |

### LLVM-Specific Advantages

1. **Direct access to LLVM optimizer**: No language frontend overhead.
   `opt -O2` directly optimizes the WAM dispatch loop, potentially
   inlining instruction handlers and eliminating tag checks.

2. **Unboxed native tier preserved**: Unlike Rust/Go where WAM values
   are always boxed, LLVM's two-tier approach keeps natively-lowered
   predicates fully unboxed — zero overhead for the common case.

3. **Register array vs hash map**: Compile-time register index mapping
   turns hash lookups into array indexing (`getelementptr`), which
   LLVM can further optimize to direct register allocation.

4. **`musttail` run loop**: Guaranteed constant-stack execution of the
   WAM interpreter loop, matching hand-optimized C performance.

5. **Cross-platform from single source**: The same `.ll` file compiles
   to native code for any LLVM-supported architecture.
