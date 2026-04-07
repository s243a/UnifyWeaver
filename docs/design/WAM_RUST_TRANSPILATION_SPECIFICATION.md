# WAM-to-Rust Transpilation: Specification

## Overview

This document specifies the hybrid compilation strategy that produces
Rust modules containing a mix of natively-lowered functions and WAM-
compiled functions, with a transpiled WAM runtime providing backtracking
and unification support.

## Architecture Layers

```
Layer 1: Predicate Classification
    → native_lowerable | wam_required | builtin

Layer 2: Compilation Strategy Selection
    → native_lowering(rust) | wam_compile_then_lower(rust) | builtin_binding(rust)

Layer 3: Code Generation
    → Mustache templates (crate structure) + native lowering (bodies)

Layer 4: WAM Runtime Transpilation
    → wam_runtime.pl → Rust via same pipeline (self-application)

Layer 5: Assembly & Validation
    → cargo check, test execution
```

## Rust Value System

The WAM operates on a universal `Value` type. In Rust:

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Atom(String),
    Integer(i64),
    Float(f64),
    Str(String, Vec<Value>),  // compound: functor + args
    List(Vec<Value>),
    Ref(usize),               // heap reference
    Unbound(String),           // unbound variable (_V5, _H3)
    Bool(bool),
}
```

## WAM State Structure

Mirrors the 9-field `wam_state` tuple from `wam_runtime.pl`:

```rust
pub struct WamState {
    pub pc: usize,                          // program counter
    pub regs: HashMap<String, Value>,       // registers (Ai, Xi)
    pub stack: Vec<StackEntry>,             // env frames + unify contexts
    pub heap: Vec<Value>,                   // term construction heap
    pub trail: Vec<TrailEntry>,             // binding trail for backtrack
    pub cp: usize,                          // continuation pointer
    pub choice_points: Vec<ChoicePoint>,    // backtracking stack
    pub code: Vec<Instruction>,             // compiled instructions
    pub labels: HashMap<String, usize>,     // label → PC mapping
}
```

## Instruction Enum

Each WAM instruction maps to a Rust enum variant:

```rust
pub enum Instruction {
    // Head unification
    GetConstant(Value, String),          // get_constant C, Ai
    GetVariable(String, String),         // get_variable Xn, Ai
    GetValue(String, String),            // get_value Xn, Ai
    GetStructure(String, String),        // get_structure F/N, Ai
    GetList(String),                     // get_list Ai
    UnifyVariable(String),              // unify_variable Xn
    UnifyValue(String),                 // unify_value Xn
    UnifyConstant(Value),               // unify_constant C

    // Body construction
    PutConstant(Value, String),         // put_constant C, Ai
    PutVariable(String, String),        // put_variable Xn, Ai
    PutValue(String, String),           // put_value Xn, Ai
    PutStructure(String, String),       // put_structure F/N, Ai
    PutList(String),                    // put_list Ai
    SetVariable(String),               // set_variable Xn
    SetValue(String),                   // set_value Xn
    SetConstant(Value),                 // set_constant C

    // Control
    Allocate,
    Deallocate,
    Call(String, usize),                // call P/N, Arity
    Execute(String),                    // execute P/N
    Proceed,
    BuiltinCall(String, usize),         // builtin_call Op, Arity

    // Choice points
    TryMeElse(String),                  // try_me_else Label
    RetryMeElse(String),               // retry_me_else Label
    TrustMe,

    // Indexing
    SwitchOnConstant(Vec<(Value, String)>),
    SwitchOnStructure(Vec<(String, String)>),
    SwitchOnConstantA2(Vec<(Value, String)>),
}
```

## Predicate Classification

```prolog
%% classify_for_rust(+Pred/Arity, -Strategy)
classify_for_rust(Pred/Arity, native) :-
    compile_predicate_to_rust_normal(Pred, Arity, _, _), !.
classify_for_rust(Pred/Arity, wam) :-
    compile_predicate_to_wam(Pred/Arity, [], _), !.
classify_for_rust(Pred/Arity, builtin) :-
    is_builtin_pred(Pred, Arity).
```

## Compilation Pipeline

### For natively-lowered predicates (no change):

```
Prolog clause → clause_body_analysis → Rust function
```

### For WAM-compiled predicates:

```
Prolog clause → wam_target:compile_predicate_to_wam → WAM instructions
    → wam_to_rust_instructions → Rust Instruction enum literals
    → wrapped in fn predicate(vm: &mut WamState) → Rust function
```

### For the WAM runtime itself:

```
wam_runtime.pl predicates
    → clause_body_analysis + Rust native lowering
    → Rust impl WamState { fn step(&mut self) → match ... }
```

## Mustache Templates

### Crate structure: `templates/targets/rust_wam/`

**`Cargo.toml.mustache`:**
```toml
[package]
name = "{{module_name}}"
version = "0.1.0"
edition = "2021"
```

**`lib.rs.mustache`:**
```rust
mod value;
mod state;
mod instructions;
mod runtime;
{{#native_predicates}}
mod {{module_name}};
{{/native_predicates}}

{{predicates_code}}
```

**`value.rs.mustache`:** The `Value` enum and helper methods.

**`state.rs.mustache`:** The `WamState` struct with register/stack/heap
management methods.

**`runtime.rs.mustache`:** The `step` method (transpiled from
`step_wam/3`) and `run_loop` (transpiled from `run_loop/2`).

## `step_wam/3` Lowering Strategy

The `step_wam/3` predicate is a multi-clause dispatch on the first
argument (instruction type). This maps directly to a Rust `match`:

```prolog
% Prolog (wam_runtime.pl):
step_wam(get_constant(C, Ai), State0, State1) :- ...
step_wam(get_variable(Xn, Ai), State0, State1) :- ...
```

```rust
// Rust (transpiled):
impl WamState {
    fn step(&mut self, instr: &Instruction) -> bool {
        match instr {
            Instruction::GetConstant(c, ai) => { ... }
            Instruction::GetVariable(xn, ai) => { ... }
            ...
        }
    }
}
```

Each clause body is lowered via `clause_body_analysis`:
- `get_assoc(Ai, R, Val)` → `self.regs.get(ai)`
- `put_assoc(Ai, R, C, NR)` → `self.regs.insert(ai, c)`
- `Val == C` → `val == c`
- `is_unbound_var(Val)` → `val.is_unbound()`
- `NPC is PC + 1` → `self.pc += 1`

## Builtin Mapping Table

| Prolog builtin | Rust equivalent |
|----------------|-----------------|
| `get_assoc/3` | `HashMap::get` |
| `put_assoc/4` | `HashMap::insert` |
| `nth0/3` | `Vec` indexing `[]` |
| `nth1/3` | `Vec` indexing `[i-1]` |
| `append/3` | `Vec::extend` / `[a, b].concat()` |
| `length/2` | `Vec::len()` |
| `member/2` | `.contains()` / `.iter().any()` |
| `format/2` | `format!()` |
| `=../2` (univ) | `Value::Str` construction/destructure |
| `atom/1` | `matches!(val, Value::Atom(_))` |
| `number/1` | `matches!(val, Value::Integer(_) \| Value::Float(_))` |
| `compound/1` | `matches!(val, Value::Str(_, _))` |
| `is_list/1` | `matches!(val, Value::List(_))` |
| `empty_assoc/1` | `HashMap::new()` |
| `assoc_to_list/2` | `.iter().collect()` |
| `sub_atom/5` | `str::contains` / `str::find` |

## Interop Calling Convention

### Native calls WAM-compiled predicate:

```rust
fn query_ancestor(a: &str, b: &str) -> bool {
    let mut vm = WamState::new(&ANCESTOR_CODE, &ANCESTOR_LABELS);
    vm.set_reg("A1", Value::Atom(a.into()));
    vm.set_reg("A2", Value::Atom(b.into()));
    vm.run()
}
```

### WAM-compiled calls native predicate:

The `BuiltinCall` instruction dispatches to native Rust:

```rust
Instruction::BuiltinCall(op, _arity) => {
    match op.as_str() {
        "is/2" => self.builtin_is(),
        ">/2"  => self.builtin_gt(),
        "factorial/2" => {  // natively-lowered predicate
            let n = self.get_reg_int("A1");
            let result = factorial(n);  // call native fn
            self.set_reg("A2", Value::Integer(result));
            self.pc += 1;
            true
        }
        _ => false
    }
}
```

## Target Capability Matrix

| Capability | Native Lowering | WAM-Compiled | Both |
|------------|----------------|-------------|------|
| Arithmetic | yes | yes | native preferred |
| Guards/comparisons | yes | yes | native preferred |
| Facts (lookup) | yes | yes | native preferred |
| Tail recursion | yes | yes | native preferred |
| Transitive closure | yes | yes | native preferred |
| If-then-else | yes | yes | native preferred |
| Choice points | no | yes | WAM only |
| Deep unification | no | yes | WAM only |
| Mutual recursion | partial | yes | WAM fallback |
| Meta-predicates | no | yes | WAM only |
