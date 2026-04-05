# WAM-to-ILAsm Transpilation: Specification

## Overview

This document specifies the hybrid compilation strategy that produces
CIL assembly modules containing a mix of natively-lowered methods and
WAM-compiled methods, with a transpiled WAM runtime providing
backtracking and unification support on the .NET CLR.

## Architecture Layers

```
Layer 1: Predicate Classification
    → native_lowerable | wam_required | builtin

Layer 2: Compilation Strategy Selection
    → native_lowering(ilasm) | wam_compile_then_lower(ilasm) | builtin_binding(ilasm)

Layer 3: Code Generation
    → Mustache templates (assembly structure) + native lowering (bodies)

Layer 4: WAM Runtime Transpilation
    → wam_runtime.pl → CIL via same pipeline (self-application)

Layer 5: Assembly & Validation
    → ilasm assembly, mono/dotnet execution
```

## CIL Value System: Class Hierarchy

The WAM operates on a universal `Value` type. In CIL, using a class
hierarchy with virtual dispatch:

```il
.class public abstract auto ansi Value extends [mscorlib]System.Object {
    .method public hidebysig specialname rtspecialname
        instance void .ctor() cil managed {
        ldarg.0
        call instance void [mscorlib]System.Object::.ctor()
        ret
    }
    .method public virtual instance bool Equals(class Value other) cil managed {
        ldc.i4.0
        ret
    }
    .method public virtual instance bool IsUnbound() cil managed {
        ldc.i4.0
        ret
    }
}

.class public auto ansi AtomValue extends Value {
    .field public string Name
    .method public instance void .ctor(string name) cil managed {
        ldarg.0
        call instance void Value::.ctor()
        ldarg.0
        ldarg.1
        stfld string AtomValue::Name
        ret
    }
}

.class public auto ansi IntegerValue extends Value {
    .field public int64 Val
}

.class public auto ansi FloatValue extends Value {
    .field public float64 Val
}

.class public auto ansi CompoundValue extends Value {
    .field public string Functor
    .field public int32 Arity
    .field public class Value[] Args
}

.class public auto ansi ListValue extends Value {
    .field public class Value[] Elements
}

.class public auto ansi RefValue extends Value {
    .field public int32 Addr
}

.class public auto ansi UnboundValue extends Value {
    .field public string Name
    .method public virtual instance bool IsUnbound() cil managed {
        ldc.i4.1
        ret
    }
}
```

## WAM State Structure

```il
.class public auto ansi WamState extends [mscorlib]System.Object {
    .field public int32 PC
    .field public class Value[] Regs       // 32-element array
    .field public class [mscorlib]System.Collections.Generic.List`1<class StackEntry> Stack
    .field public class Value[] Heap       // dynamic array
    .field public int32 HeapSize
    .field public class [mscorlib]System.Collections.Generic.List`1<class TrailEntry> Trail
    .field public int32 CP                 // continuation pointer
    .field public class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> ChoicePoints
    .field public class Instruction[] Code
    .field public int32 CodeLength
    .field public int32[] Labels           // label index → PC mapping
    .field public int32 LabelCount
    .field public bool Halted
}
```

### Register Access

Registers map to fixed array indices (same ABI as LLVM target):

```
A1 → 0, A2 → 1, ..., A16 → 15
X1 → 16, X2 → 17, ..., X16 → 31
```

```il
// Get register A1 (index 0):
ldarg.0                          // push WamState
ldfld class Value[] WamState::Regs
ldc.i4.0                        // index 0
ldelem.ref                      // load element

// Set register A2 (index 1):
ldarg.0
ldfld class Value[] WamState::Regs
ldc.i4.1
ldarg.2                          // new value
stelem.ref                       // store element
```

## Instruction Representation

```il
.class public auto ansi Instruction extends [mscorlib]System.Object {
    .field public int32 Tag
    .field public int64 Op1
    .field public int64 Op2
}
```

Instruction tag constants (same as LLVM target):

```
Head unification:  0=GetConstant  1=GetVariable  2=GetValue
                   3=GetStructure 4=GetList      5=UnifyVariable
                   6=UnifyValue   7=UnifyConstant
Body construction: 8=PutConstant  9=PutVariable  10=PutValue
                   11=PutStructure 12=PutList     13=SetVariable
                   14=SetValue    15=SetConstant
Control:           16=Allocate    17=Deallocate   18=Call
                   19=Execute     20=Proceed      21=BuiltinCall
Choice points:     22=TryMeElse  23=RetryMeElse  24=TrustMe
Indexing:          25=SwitchOnConstant  26=SwitchOnStructure
                   27=SwitchOnConstantA2
```

## Supporting Types

```il
.class public auto ansi StackEntry extends [mscorlib]System.Object {
    .field public int32 Type      // 0=EnvFrame, 1=UnifyCtx, 2=WriteCtx
    .field public int32 Aux       // saved CP or write count
    .field public class Value[] Data
}

.class public auto ansi TrailEntry extends [mscorlib]System.Object {
    .field public int32 RegIndex
    .field public class Value OldValue
}

.class public auto ansi ChoicePoint extends [mscorlib]System.Object {
    .field public int32 NextPC
    .field public class Value[] SavedRegs    // 32-element clone
    .field public int32 TrailMark
    .field public int32 SavedCP
}
```

## Predicate Classification

```prolog
%% classify_for_ilasm(+Pred/Arity, -Strategy)
classify_for_ilasm(Pred/Arity, native) :-
    compile_predicate_to_ilasm(Pred/Arity, [], _), !.
classify_for_ilasm(Pred/Arity, wam) :-
    compile_predicate_to_wam(Pred/Arity, [], _), !.
classify_for_ilasm(Pred/Arity, builtin) :-
    is_builtin_pred(Pred, Arity).
```

## Compilation Pipelines

### For natively-lowered predicates (no change):

```
Prolog clause → clause_body_analysis → CIL method
```

### For WAM-compiled predicates:

```
Prolog clause → wam_target:compile_predicate_to_wam → WAM instructions
    → wam_to_cil_instructions → CIL Instruction array
    → wrapped in .method predicate(WamState) → CIL method
```

### For the WAM runtime itself:

```
wam_runtime.pl predicates
    → clause_body_analysis + CIL native lowering
    → .method step(WamState, Instruction) { switch ... }
```

## `step_wam/3` Lowering Strategy

The `step_wam/3` predicate maps to a CIL `switch` instruction:

```prolog
% Prolog (wam_runtime.pl):
step_wam(get_constant(C, Ai), State0, State1) :- ...
step_wam(get_variable(Xn, Ai), State0, State1) :- ...
```

```il
// CIL (transpiled):
.method public static bool step(class WamState vm, class Instruction instr) cil managed {
    .maxstack 8
    ldarg.1
    ldfld int32 Instruction::Tag
    switch (L_get_constant, L_get_variable, L_get_value,
            L_get_structure, L_get_list, L_unify_variable,
            L_unify_value, L_unify_constant,
            L_put_constant, L_put_variable, L_put_value,
            L_put_structure, L_put_list, L_set_variable,
            L_set_value, L_set_constant,
            L_allocate, L_deallocate, L_call, L_execute,
            L_proceed, L_builtin_call,
            L_try_me_else, L_retry_me_else, L_trust_me)
    ldc.i4.0
    ret                              // default: return false

L_get_constant:
    // Load register value
    ldarg.0
    ldfld class Value[] WamState::Regs
    ldarg.1
    ldfld int64 Instruction::Op2     // register index
    conv.i4
    ldelem.ref                       // current value
    // Check if unbound
    callvirt instance bool Value::IsUnbound()
    brtrue L_gc_bind
    // Check equality
    // ...
}
```

## Builtin Mapping Table

| Prolog builtin | CIL equivalent |
|----------------|---------------|
| `get_assoc/3` | `ldelem.ref` on Regs array |
| `put_assoc/4` | `stelem.ref` on Regs array |
| `nth0/3` | `ldelem.ref` on Value[] |
| `append/3` | `List.AddRange()` |
| `length/2` | `.Length` or `get_Count()` |
| `member/2` | `List.Contains()` |
| `format/2` | `String.Format()` |
| `=../2` (univ) | CompoundValue field access |
| `atom/1` | `isinst AtomValue` |
| `number/1` | `isinst IntegerValue` or `isinst FloatValue` |
| `compound/1` | `isinst CompoundValue` |
| `is_list/1` | `isinst ListValue` |
| `empty_assoc/1` | `new Value[32]` |
| `sub_atom/5` | `String.Contains()` |
| `==/2` | `callvirt Value::Equals(Value)` |

## Interop Calling Convention

### Native calls WAM-compiled predicate:

```il
.method public static bool query_ancestor(string a, string b) cil managed {
    // Create WamState
    ldsfld class Instruction[] PrologGenerated::ancestor_code
    ldsfld int32[] PrologGenerated::ancestor_labels
    newobj instance void WamState::.ctor(class Instruction[], int32[])
    stloc.0
    // Set A1
    ldloc.0
    ldc.i4.0                         // reg index 0 = A1
    ldarg.0
    newobj instance void AtomValue::.ctor(string)
    call void WamState::SetReg(int32, class Value)
    // Set A2
    ldloc.0
    ldc.i4.1                         // reg index 1 = A2
    ldarg.1
    newobj instance void AtomValue::.ctor(string)
    call void WamState::SetReg(int32, class Value)
    // Run
    ldloc.0
    call bool PrologGenerated::run_loop(class WamState)
    ret
}
```

### WAM-compiled calls native predicate:

```il
// Inside the BuiltinCall case of step():
L_builtin_call:
    ldarg.1
    ldfld int64 Instruction::Op1    // builtin op id
    conv.i4
    switch (L_bi_is, L_bi_gt, L_bi_lt, ...)
    ldc.i4.0
    ret

L_bi_is:
    // Get A2, call eval_arith, bind A1
    ldarg.0
    ldc.i4.1
    call class Value WamState::GetReg(int32)
    call int64 PrologGenerated::eval_arith(class WamState, class Value)
    // ... box result, bind to A1
```

## Target Capability Matrix

| Capability | Native ILAsm | WAM-Compiled | Both |
|------------|-------------|-------------|------|
| Arithmetic | yes (CIL ops) | yes (eval_arith) | native preferred |
| Guards/comparisons | yes (bgt/blt/beq) | yes | native preferred |
| Facts (lookup) | yes | yes | native preferred |
| Tail recursion | yes (.tail prefix) | yes (run_loop .tail) | native preferred |
| Linear recursion | yes (iterative loop) | yes | native preferred |
| Tree recursion | yes (direct calls) | yes | native preferred |
| Mutual recursion | yes (.tail call) | yes | native preferred |
| Transitive closure | yes (BFS templates) | yes | native preferred |
| If-then-else | yes (br/labels) | yes | native preferred |
| Choice points | no | yes | WAM only |
| Deep unification | no | yes | WAM only |
| Meta-predicates | no | yes | WAM only |
| .NET interop | yes (BCL calls) | via builtin | native preferred |

## Differences from Other Hybrid Targets

| Aspect | Rust | Go | LLVM | CIL (ILAsm) |
|--------|------|-----|------|-------------|
| Value type | `enum Value` | `interface Value` | `{i32, i64}` | Class hierarchy |
| Dispatch | `match` | type switch | `switch i32` | CIL `switch` |
| Memory | Ownership | GC | Arena | **CLR GC** |
| Registers | `HashMap` | `map[string]Value` | `[32 x %Value]` | `Value[]` (array) |
| Tail calls | while loop | for loop | `musttail` | **`.tail call`** |
| Type checks | `matches!()` | type assertion | `icmp eq tag` | **`isinst`** |
| Backtracking | Manual trail | Manual trail | Manual trail | **GC + clone** |
| Package | Cargo crate | Go module | `.ll` file | `.il` assembly |
| Build tool | `cargo check` | `go build` | `opt`+`llc` | `ilasm` |
| Concurrency | — | Goroutines | — | `Task`/`async` (future) |

### CIL-Specific Advantages

1. **GC eliminates memory management**: No arena allocator, no manual
   trail entry deallocation, no ownership reasoning. Choice point
   saves are just array clones; GC reclaims old state.

2. **`.tail call` for run loop**: Native tail call support means the
   WAM interpreter loop runs in constant stack space without loop
   transformation.

3. **`isinst` for type dispatch**: Type checks on Value subclasses
   are a single CIL instruction — more expressive than tag integer
   comparison.

4. **Rich BCL interop**: WAM-compiled predicates can call any .NET
   BCL method via `BuiltinCall` dispatch — string operations,
   collections, I/O, networking are all available.

5. **Stack-based codegen is simpler**: No SSA variable naming, no
   phi nodes, no dominance requirements. Push/pop is straightforward
   from Prolog clause body analysis.
