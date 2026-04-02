# WAM-to-Go Transpilation: Specification

## Overview

This document specifies the hybrid compilation strategy that produces
Go packages containing a mix of natively-lowered functions and WAM-
compiled functions, with a transpiled WAM runtime providing backtracking
and unification support.

## Architecture Layers

```
Layer 1: Predicate Classification
    → native_lowerable | wam_required | builtin

Layer 2: Compilation Strategy Selection
    → native_lowering(go) | wam_compile_then_lower(go) | builtin_binding(go)

Layer 3: Code Generation
    → Mustache templates (package structure) + native lowering (bodies)

Layer 4: WAM Runtime Transpilation
    → wam_runtime.pl → Go via same pipeline (self-application)

Layer 5: Assembly & Validation
    → go build, go test
```

## Go Value System

The WAM operates on a universal `Value` type. In Go, using a sealed
interface:

```go
// Value is the universal Prolog term type.
type Value interface {
    valueTag()  // sealed marker
    String() string
}

type Atom struct{ Name string }
type Integer struct{ Val int64 }
type Float struct{ Val float64 }
type Compound struct{ Functor string; Args []Value }
type List struct{ Elements []Value }
type Ref struct{ Addr int }
type Unbound struct{ Name string }
type Bool struct{ Val bool }

func (Atom) valueTag()     {}
func (Integer) valueTag()  {}
func (Float) valueTag()    {}
func (Compound) valueTag() {}
func (List) valueTag()     {}
func (Ref) valueTag()      {}
func (Unbound) valueTag()  {}
func (Bool) valueTag()     {}
```

## WAM State Structure

Mirrors the 9-field `wam_state` tuple from `wam_runtime.pl`:

```go
type WamState struct {
    PC           int                    // program counter
    Regs         map[string]Value       // registers (Ai, Xi)
    Stack        []StackEntry           // env frames + unify contexts
    Heap         []Value                // term construction heap
    Trail        []TrailEntry           // binding trail for backtrack
    CP           int                    // continuation pointer
    ChoicePoints []ChoicePoint          // backtracking stack
    Code         []Instruction          // compiled instructions
    Labels       map[string]int         // label → PC mapping
}

type StackEntry struct {
    Type    string // "env" or "unify_context"
    Env     *Environment
    Context *UnifyContext
}

type ChoicePoint struct {
    PC    int
    Regs  map[string]Value  // saved registers
    Trail int               // trail mark
    CP    int               // saved continuation
}

type TrailEntry struct {
    Var   string
    Value Value
}
```

## Instruction Interface

Each WAM instruction is a Go struct implementing the `Instruction`
interface. Dispatch uses type switch:

```go
type Instruction interface {
    instrTag()
}

// Head unification
type GetConstant struct{ C Value; Ai string }
type GetVariable struct{ Xn, Ai string }
type GetValue struct{ Xn, Ai string }
type GetStructure struct{ Functor string; Ai string }
type GetList struct{ Ai string }
type UnifyVariable struct{ Xn string }
type UnifyValue struct{ Xn string }
type UnifyConstant struct{ C Value }

// Body construction
type PutConstant struct{ C Value; Ai string }
type PutVariable struct{ Xn, Ai string }
type PutValue struct{ Xn, Ai string }
type PutStructure struct{ Functor string; Ai string }
type PutList struct{ Ai string }
type SetVariable struct{ Xn string }
type SetValue struct{ Xn string }
type SetConstant struct{ C Value }

// Control
type Allocate struct{}
type Deallocate struct{}
type Call struct{ Pred string; Arity int }
type Execute struct{ Pred string }
type Proceed struct{}
type BuiltinCall struct{ Op string; Arity int }

// Choice points
type TryMeElse struct{ Label string }
type RetryMeElse struct{ Label string }
type TrustMe struct{}

// Indexing
type SwitchOnConstant struct{ Cases []ConstCase }
type SwitchOnStructure struct{ Cases []StructCase }
type SwitchOnConstantA2 struct{ Cases []ConstCase }

type ConstCase struct{ Val Value; Label string }
type StructCase struct{ Functor, Label string }

func (GetConstant) instrTag()       {}
func (GetVariable) instrTag()       {}
// ... (all instruction types)
```

## Predicate Classification

```prolog
%% classify_for_go(+Pred/Arity, -Strategy)
classify_for_go(Pred/Arity, native) :-
    compile_predicate_to_go_normal(Pred, Arity, _, _), !.
classify_for_go(Pred/Arity, wam) :-
    compile_predicate_to_wam(Pred/Arity, [], _), !.
classify_for_go(Pred/Arity, builtin) :-
    is_builtin_pred(Pred, Arity).
```

## Compilation Pipeline

### For natively-lowered predicates (no change):

```
Prolog clause → clause_body_analysis → Go function
```

### For WAM-compiled predicates:

```
Prolog clause → wam_target:compile_predicate_to_wam → WAM instructions
    → wam_to_go_instructions → Go Instruction struct literals
    → wrapped in func predicate(vm *WamState) → Go function
```

### For the WAM runtime itself:

```
wam_runtime.pl predicates
    → clause_body_analysis + Go native lowering
    → func (vm *WamState) Step(instr Instruction) bool { switch ... }
```

## `step_wam/3` Lowering Strategy

The `step_wam/3` predicate maps to a Go type switch:

```prolog
% Prolog (wam_runtime.pl):
step_wam(get_constant(C, Ai), State0, State1) :- ...
step_wam(get_variable(Xn, Ai), State0, State1) :- ...
```

```go
// Go (transpiled):
func (vm *WamState) Step(instr Instruction) bool {
    switch i := instr.(type) {
    case *GetConstant:
        val := vm.Regs[i.Ai]
        if isUnbound(val) {
            vm.Regs[i.Ai] = i.C
            vm.PC++
            return true
        }
        if equals(val, i.C) {
            vm.PC++
            return true
        }
        return false
    case *GetVariable:
        vm.Regs[i.Xn] = vm.Regs[i.Ai]
        vm.PC++
        return true
    // ...
    default:
        return false
    }
}
```

## Builtin Mapping Table

| Prolog builtin | Go equivalent |
|----------------|---------------|
| `get_assoc/3` | `regs[key]` (map lookup) |
| `put_assoc/4` | `regs[key] = val` (map insert) |
| `nth0/3` | `slice[i]` |
| `nth1/3` | `slice[i-1]` |
| `append/3` | `append(a, b...)` |
| `length/2` | `len(slice)` |
| `member/2` | `slices.Contains()` or loop |
| `format/2` | `fmt.Sprintf()` |
| `=../2` (univ) | `Compound` construction/destructure |
| `atom/1` | `_, ok := val.(*Atom)` |
| `number/1` | type switch `*Integer \| *Float` |
| `compound/1` | `_, ok := val.(*Compound)` |
| `is_list/1` | `_, ok := val.(*List)` |
| `empty_assoc/1` | `make(map[string]Value)` |
| `sub_atom/5` | `strings.Contains` / `strings.Index` |

## Mustache Templates

### Package structure: `templates/targets/go_wam/`

**`go.mod.mustache`:**
```
module {{module_name}}

go 1.21
```

**`value.go.mustache`:** The `Value` interface and concrete types.

**`state.go.mustache`:** The `WamState` struct with register/stack/heap
management methods.

**`instructions.go.mustache`:** The `Instruction` interface and all
instruction struct types.

**`runtime.go.mustache`:** The `Step` method (transpiled from
`step_wam/3`) and `Run` loop (transpiled from `run_loop/2`).

## Interop Calling Convention

### Native calls WAM-compiled predicate:

```go
func queryAncestor(a, b string) bool {
    vm := NewWamState(ancestorCode, ancestorLabels)
    vm.SetReg("A1", &Atom{a})
    vm.SetReg("A2", &Atom{b})
    return vm.Run()
}
```

### WAM-compiled calls native predicate:

```go
case *BuiltinCall:
    switch i.Op {
    case "is/2":
        vm.builtinIs()
    case ">/2":
        vm.builtinGt()
    case "factorial/2":
        n := vm.GetRegInt("A1")
        result := factorial(n)
        vm.SetReg("A2", &Integer{result})
        vm.PC++
    default:
        return false
    }
```

## Order-Independence Declarations

Predicates can be marked order-independent to enable safe parallelism:

```prolog
%% User declaration — a promise that clause order doesn't matter
:- order_independent(node_score/2).
:- parallel_safe(feature_a/2).

%% Or: proven by static analysis (no side effects, disjoint bindings)
```

### Analysis Predicate

```prolog
%% is_order_independent(+Pred/Arity, -Evidence)
%%   Evidence: declared | proven(Reasons)
is_order_independent(Pred/Arity, declared) :-
    order_independent(Pred/Arity), !.
is_order_independent(Pred/Arity, proven(Reasons)) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(user:Head, Body), Clauses),
    all_clauses_pure(Clauses, R1),      % no I/O, no assert/retract
    all_goals_independent(Clauses, R2),  % disjoint variable sets
    append(R1, R2, Reasons),
    Reasons \= [].
```

### Parallelism Classification

```prolog
%% classify_parallelism(+Pred/Arity, -Strategy)
classify_parallelism(Pred/Arity, goal_parallel(Goals)) :-
    %% Body goals bind disjoint variables — run concurrently
    single_clause_with_independent_goals(Pred/Arity, Goals), !.
classify_parallelism(Pred/Arity, clause_parallel) :-
    %% All clauses are order-independent — collect solutions concurrently
    is_order_independent(Pred/Arity, _), !.
classify_parallelism(_, sequential).
```

### Go Code Generation for Order-Independent Goals

Goal-level parallelism (independent body goals in one clause):

```go
// From: score(X, S) :- feature_a(X, A), feature_b(X, B), S is A + B.
// With: :- order_independent(score/2).
func score(x Value) Value {
    var a, b Value
    var wg sync.WaitGroup
    wg.Add(2)
    go func() { defer wg.Done(); a = featureA(x) }()
    go func() { defer wg.Done(); b = featureB(x) }()
    wg.Wait()
    return &Integer{a.(*Integer).Val + b.(*Integer).Val}
}
```

Clause-level parallelism (multiple clauses, all solutions):

```go
// From: reachable(X, Y) :- edge(X, Y).
//       reachable(X, Y) :- edge(X, Z), reachable(Z, Y).
// With: :- order_independent(reachable/2).
func reachableAll(x Value) <-chan Value {
    results := make(chan Value, 100)
    go func() {
        defer close(results)
        var wg sync.WaitGroup
        // Each clause explores independently
        wg.Add(2)
        go func() { defer wg.Done(); /* clause 1 */ }()
        go func() { defer wg.Done(); /* clause 2 */ }()
        wg.Wait()
    }()
    return results
}
```

## Target Capability Matrix

| Capability | Native Go | WAM-Compiled | Both |
|------------|-----------|-------------|------|
| Arithmetic | yes | yes | native preferred |
| Guards/comparisons | yes | yes | native preferred |
| Facts (lookup) | yes | yes | native preferred |
| Tail recursion | yes | yes | native preferred |
| Transitive closure | yes | yes | native preferred |
| If-then-else | yes | yes | native preferred |
| Pipeline/streaming | yes | no | native only |
| Choice points | no | yes | WAM only |
| Deep unification | no | yes | WAM only |
| Mutual recursion | partial | yes | WAM fallback |
| Meta-predicates | no | yes | WAM only |
| Goroutine search | planned | no | native extension |
| Order-independent parallel | yes | no | native (goroutines) |
