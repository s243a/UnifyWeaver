# Code Generation Specification — Hybrid Template + Lowering Architecture

## Overview

This document specifies how to combine template-based and native
lowering approaches in UnifyWeaver's code generation pipeline. The
goal is to produce idiomatic code (templates) that handles arbitrary
predicate structures (native lowering) with type awareness (TypR's
contribution).

## Architecture

### Compilation Layers

```
Layer 1: Pattern Detection
  classify_predicate → transitive_closure | tail_recursion | ...

Layer 2: Structural Template Selection
  compile_tc_from_template/6 → tc_definitions + tc_input_* + tc_interface_*

Layer 3: Template Variable Computation
  Type-aware expressions: empty_collection_expr, annotation_suffix, seed_code
  Context-aware: input mode, CLI vs notebook

Layer 4: Logic Lowering (within template bodies)
  native_lowering(Target, GoalSequence, Code) — goal-by-goal translation
  wrapped_fallback(Target, FallbackTarget, Code) — compile to fallback, embed

Layer 5: Validation (optional)
  cargo check, tsc --noEmit, typr check, go vet
```

### Current Implementation by Target

| Target | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
|--------|---------|---------|---------|---------|---------|
| Bash | detect | template_system | strategy | stream compiler | shellcheck (manual) |
| Lua | detect | composable | basic vars | none | none |
| Python | detect | composable | basic + pred_cap | none | none |
| R | detect | composable | basic vars | none | none |
| TypR | detect | monolithic | type-aware (10 vars) | native + R fallback | typr check |
| Rust | detect | composable | basic + pred_cap | none | cargo check (manual) |
| Go | detect | composable | basic + pred_cap | none | none |
| Others | detect | composable | basic vars | none | none |

### Target: All Targets at Layer 3+

Every target should compute type-aware template variables. This
requires extending `input_source.pl` and `recursive_compiler.pl`
with per-target type resolution.

## Type-Aware Template Variables

### Current Variables (all targets)

```prolog
Dict = [
    pred = "ancestor",
    base = "parent",
    pred_cap = "Ancestor"      % where applicable
]
```

### Extended Variables (TypR model, generalize to all)

```prolog
Dict = [
    pred = "ancestor",
    base = "parent",
    pred_cap = "Ancestor",
    %% Type-aware (new)
    empty_collection = "[]",            % target-specific empty list
    node_type = "String",               % resolved node type
    collection_type = "Vec<String>",    % resolved collection type
    seed_code = "add_fact(\"a\", \"b\")\n..."
]
```

### Per-Target Type Resolution

```prolog
%% empty_collection_expr(+NodeType, +Target, -Expr)
empty_collection_expr(atom, rust, "Vec::new()").
empty_collection_expr(atom, go, "[]string{}").
empty_collection_expr(atom, python, "[]").
empty_collection_expr(atom, lua, "{}").
empty_collection_expr(atom, r, "character(0)").
empty_collection_expr(atom, typescript, "[] as string[]").
empty_collection_expr(integer, rust, "Vec::<i64>::new()").
empty_collection_expr(integer, go, "[]int{}").
empty_collection_expr(integer, python, "[]").  % Python: same, untyped
empty_collection_expr(integer, lua, "{}").     % Lua: same, untyped
empty_collection_expr(integer, r, "integer(0)").
empty_collection_expr(integer, typescript, "[] as number[]").

%% node_type_name(+NodeType, +Target, -TypeName)
node_type_name(atom, rust, "String").
node_type_name(atom, go, "string").
node_type_name(atom, typescript, "string").
node_type_name(atom, python, "str").
node_type_name(integer, rust, "i64").
node_type_name(integer, go, "int").
node_type_name(integer, typescript, "number").
```

### Node Type Inference

The node type comes from the Prolog predicate:

```prolog
%% resolve_node_type(+PredSpec, +BasePredSpec, -NodeType)
%  Check declared types first, then infer from facts.
resolve_node_type(Pred/Arity, BasePred/2, NodeType) :-
    (   declared_argument_type(BasePred/2, 1, Type)
    ->  NodeType = Type
    ;   infer_node_type_from_facts(BasePred, NodeType)
    ).

infer_node_type_from_facts(BasePred, NodeType) :-
    functor(H, BasePred, 2),
    clause(user:H, true),
    H =.. [_, From, _],
    (   number(From) -> NodeType = integer
    ;   atom(From) -> NodeType = atom
    ;   NodeType = any
    ), !.
infer_node_type_from_facts(_, atom).  % default
```

## Native Lowering Extension Points

### Goal-to-Expression Translation

TypR's `native_typr_output_expr` and `native_typr_guard_goal` can
be generalized. The pattern is:

```prolog
%% native_goal(+Target, +Goal, +VarMap, -Code, -VarMap1)
native_goal(Target, (A =:= B), VM, Code, VM) :-
    native_expr(Target, A, VM, ACode),
    native_expr(Target, B, VM, BCode),
    comparison_op(Target, '=:=', Op),
    format(string(Code), '~w ~w ~w', [ACode, Op, BCode]).
```

Each target registers its own expression translations:

```prolog
comparison_op(rust, '=:=', "==").
comparison_op(python, '=:=', "==").
comparison_op(lua, '=:=', "==").
comparison_op(r, '=:=', "==").
comparison_op(go, '=:=', "==").
```

### Binding Registry

TypR uses R bindings to translate Prolog predicates to R calls.
This can be generalized:

```prolog
%% target_binding(+Target, +Pred/Arity, +ArgPattern, -CodePattern)
target_binding(python, length/2, [List, Len], "~Len = len(~List)").
target_binding(rust, length/2, [List, Len], "let ~Len = ~List.len();").
target_binding(lua, length/2, [List, Len], "local ~Len = #~List").
target_binding(go, length/2, [List, Len], "~Len := len(~List)").
```

### Fallback Chains

Not every target has an obvious fallback, but some do:

```prolog
%% fallback_target(+Target, -FallbackTarget)
fallback_target(typr, r).           % TypR → R (existing)
fallback_target(typescript, javascript).  % TS → JS
fallback_target(kotlin, java).      % Kotlin → Java
fallback_target(jython, python).    % Jython → Python
fallback_target(cpp, c).            % C++ → C (limited)
```

When native lowering fails and a fallback exists, compile to the
fallback target and embed via the target's FFI mechanism.

## Composable Templates + Type Awareness

### Current Composable Structure

```
tc_definitions.mustache     — functions (add_fact, find_all, check)
tc_input_stdin.mustache     — for line in io.lines() do ...
tc_input_embedded.mustache  — {{seed_code}}
tc_input_file.mustache      — for line in io.lines(path) do ...
tc_input_vfs.mustache       — nb.read(cell, prop)
tc_input_function.mustache  — ancestor_from_pairs(pairs)
tc_interface_cli.mustache   — arg parsing + query dispatch
```

### Proposed: Type-Aware Templates

Templates can use type-aware variables without changing structure:

```mustache
-- Current (Lua):
local rel = {}

-- Proposed (Lua, with type awareness):
local rel = {}  -- {{node_type}} → {{node_type}} adjacency list
```

```mustache
-- Current (Rust):
let mut rel: HashMap<String, Vec<String>> = HashMap::new();

-- Proposed (Rust, with type awareness):
let mut rel: HashMap<{{node_type}}, Vec<{{node_type}}>> = HashMap::new();
```

For dynamically typed targets (Lua, Python, Ruby, Perl), type
awareness adds documentation but doesn't change the code. For
statically typed targets (Rust, Go, C, C++, TypeScript), it
produces correct type signatures.

## Validation Integration

### Per-Target Validators

```prolog
%% validate_generated(+Target, +Code, -Result)
validate_generated(typr, Code, Result) :-
    run_typr_check(Code, Result).
validate_generated(rust, Code, Result) :-
    write_temp_file(Code, "rs", Path),
    run_command(["rustc", "--edition=2021", "--crate-type=lib", Path], Result).
validate_generated(go, Code, Result) :-
    write_temp_file(Code, "go", Path),
    run_command(["go", "vet", Path], Result).
validate_generated(typescript, Code, Result) :-
    write_temp_file(Code, "ts", Path),
    run_command(["tsc", "--noEmit", Path], Result).
```

This catches type mismatches, syntax errors, and import issues
before the user runs the generated code.
