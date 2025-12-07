<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 3: Target Registry and Mapping

## Overview

The target registry is the foundation of cross-target glue. It manages:

- **Target metadata** - What targets exist and their capabilities
- **Runtime families** - How targets are grouped
- **Predicate mappings** - Which target compiles each predicate
- **Location resolution** - Where predicates execute

## The Target Registry Module

Located at `src/unifyweaver/core/target_registry.pl`, this module provides:

```prolog
:- module(target_registry, [
    register_target/3,        % Register a new target
    unregister_target/1,      % Remove a target
    target_exists/1,          % Check if target exists
    target_family/2,          % Query target's family
    target_capabilities/2,    % Query capabilities
    targets_same_family/2,    % Check family membership
    family_targets/2,         % List targets in family
    list_targets/1,           % List all targets
    list_families/1,          % List all families
    default_location/2,       % Default location for target
    default_transport/3       % Default transport between locations
]).
```

## Built-in Targets

The registry comes pre-configured with 15+ targets:

### Shell Family

```prolog
target_family(bash, shell).
target_family(awk, shell).
target_family(sed, shell).
target_family(perl, shell).
```

**Characteristics:**
- Always run as separate processes
- Communicate via pipes
- Best for text processing

### Python Family

```prolog
target_family(python, python).
target_family(ironpython, dotnet).  % Note: IronPython is .NET family
```

**Characteristics:**
- CPython runs as separate process
- IronPython can be in-process with .NET

### .NET Family

```prolog
target_family(csharp, dotnet).
target_family(fsharp, dotnet).
target_family(powershell, dotnet).
target_family(ironpython, dotnet).
```

**Characteristics:**
- Can communicate in-process
- Share the CLR runtime
- Zero serialization between them

### JVM Family

```prolog
target_family(java, jvm).
target_family(scala, jvm).
target_family(clojure, jvm).
target_family(jython, jvm).
```

**Characteristics:**
- Can communicate in-process
- Share the JVM runtime

### Native Family

```prolog
target_family(go, native).
target_family(rust, native).
target_family(c, native).
target_family(cpp, native).
```

**Characteristics:**
- Compiled to machine code
- Communicate via pipes or shared memory
- Best for performance-critical code

### Database Family

```prolog
target_family(sql, database).
target_family(sqlite, database).
target_family(postgresql, database).
```

**Characteristics:**
- Execute queries against databases
- Typically accessed via network or file

## Querying the Registry

### Check if a Target Exists

```prolog
?- target_exists(python).
true.

?- target_exists(fortran).
false.
```

### Query Target Family

```prolog
?- target_family(powershell, Family).
Family = dotnet.

?- target_family(Target, shell).
Target = bash ;
Target = awk ;
Target = sed ;
Target = perl.
```

### Check Same Family

```prolog
?- targets_same_family(csharp, powershell).
true.

?- targets_same_family(python, rust).
false.
```

### List Family Members

```prolog
?- family_targets(native, Targets).
Targets = [go, rust, c, cpp].
```

## Registering Custom Targets

You can register your own targets:

```prolog
% Register a custom Rust target with specific capabilities
register_target(myrust, native, [compiled, typed, async, simd]).

% Register a custom Python target
register_target(micropython, python, [embedded, limited_stdlib]).
```

### Capabilities

Capabilities describe what a target can do:

| Capability | Meaning |
|------------|---------|
| `compiled` | Compiles to machine code |
| `typed` | Statically typed |
| `async` | Supports async/await |
| `streaming` | Can process streams |
| `pipes` | Can use pipe I/O |
| `http` | Can make HTTP requests |
| `simd` | Supports SIMD operations |

Query capabilities:

```prolog
?- target_capabilities(go, Caps).
Caps = [compiled, typed, async, streaming, pipes, http].
```

## The Target Mapping Module

Located at `src/unifyweaver/core/target_mapping.pl`:

```prolog
:- module(target_mapping, [
    declare_target/2,          % Basic declaration
    declare_target/3,          % Declaration with options
    undeclare_target/1,        % Remove declaration
    declare_location/2,        % Specify location
    declare_connection/3,      % Specify connection
    predicate_target/2,        % Query mapping
    predicate_location/2,      % Query location
    resolve_location/2,        % Resolve with defaults
    resolve_transport/3,       % Resolve transport
    list_mappings/1,           % List all mappings
    validate_mapping/2         % Validate configuration
]).
```

## Declaring Target Mappings

### Basic Declaration

```prolog
% Declare that filter/2 compiles to AWK
:- declare_target(filter/2, awk).

% Declare that transform/2 compiles to Python
:- declare_target(transform/2, python).
```

### Declaration with Options

```prolog
:- declare_target(analyze/2, python, [
    location(local_process),
    format(json),
    timeout(30)
]).
```

Available options:

| Option | Values | Description |
|--------|--------|-------------|
| `location(L)` | `in_process`, `local_process`, `remote(Host)` | Execution location |
| `transport(T)` | `direct`, `pipe`, `socket`, `http` | Communication method |
| `format(F)` | `tsv`, `json`, `binary` | Data format |
| `timeout(S)` | Integer | Timeout in seconds |

## Declaring Locations

Override where a predicate executes:

```prolog
% Run on a specific remote host
:- declare_location(heavy_compute/2, [
    host('gpu-worker.example.com'),
    port(8080),
    transport(http)
]).

% Force separate process for isolation
:- declare_location(untrusted/2, [
    process(separate),
    sandbox(true)
]).
```

## Declaring Connections

Specify how two predicates communicate:

```prolog
% Use JSON format between these predicates
:- declare_connection(producer/2, consumer/2, [
    format(json),
    buffer(line)
]).

% Use socket for bidirectional communication
:- declare_connection(client/2, server/2, [
    transport(socket),
    format(json)
]).
```

## Resolution Rules

When explicit configuration isn't provided, defaults are computed.

### Location Resolution

```prolog
resolve_location(Pred/Arity, Location) :-
    % 1. Check explicit declaration
    predicate_location(Pred/Arity, Location), !.

resolve_location(Pred/Arity, Location) :-
    % 2. Use target's default
    predicate_target(Pred/Arity, Target),
    default_location(Target, Location).
```

### Default Location Rules

```prolog
% Same family = in-process
default_location(Target, in_process) :-
    target_family(Target, Family),
    member(Family, [dotnet, jvm]).

% Different family = separate process
default_location(Target, local_process) :-
    target_family(Target, Family),
    member(Family, [shell, python, native]).
```

### Transport Resolution

```prolog
resolve_transport(Pred1, Pred2, Transport) :-
    resolve_location(Pred1, Loc1),
    resolve_location(Pred2, Loc2),
    default_transport(Loc1, Loc2, Transport).
```

### Default Transport Rules

```prolog
% In-process = direct calls
default_transport(in_process, in_process, direct).

% Local processes = pipes
default_transport(local_process, local_process, pipe).

% Remote = HTTP
default_transport(local_process, remote(_), http).
default_transport(remote(_), remote(_), http).
```

## Querying Mappings

### Find Target for Predicate

```prolog
?- predicate_target(filter/2, Target).
Target = awk.
```

### Find Location for Predicate

```prolog
?- resolve_location(filter/2, Location).
Location = local_process.
```

### Find Transport Between Predicates

```prolog
?- resolve_transport(filter/2, transform/2, Transport).
Transport = pipe.
```

### List All Mappings

```prolog
?- list_mappings(Mappings).
Mappings = [
    mapping(filter/2, awk, local_process),
    mapping(transform/2, python, local_process),
    mapping(store/2, sql, remote('db.local'))
].
```

## Validation

Validate your configuration:

```prolog
?- validate_mapping(unknown_pred/2, Errors).
Errors = [no_target_declared].

?- validate_mapping(filter/2, Errors).
Errors = [].  % No errors
```

Common validation errors:

| Error | Meaning |
|-------|---------|
| `no_target_declared` | Predicate has no target mapping |
| `unknown_target` | Target doesn't exist in registry |
| `invalid_location` | Location specification is invalid |
| `incompatible_transport` | Transport not valid for locations |

## Practical Example

Let's configure a complete pipeline:

```prolog
% file: pipeline_config.pl

:- use_module('src/unifyweaver/core/target_registry').
:- use_module('src/unifyweaver/core/target_mapping').

% Declare targets for each predicate
:- declare_target(fetch_logs/2, bash).
:- declare_target(parse_logs/2, awk).
:- declare_target(analyze_logs/2, python, [format(json)]).
:- declare_target(generate_report/2, python).
:- declare_target(store_report/2, sql).

% Configure remote database
:- declare_location(store_report/2, [
    host('db.production.local'),
    port(5432)
]).

% Specify JSON between parse and analyze
:- declare_connection(parse_logs/2, analyze_logs/2, [
    format(json)
]).
```

Query the configuration:

```prolog
?- resolve_location(fetch_logs/2, L).
L = local_process.

?- resolve_location(store_report/2, L).
L = remote('db.production.local').

?- resolve_transport(parse_logs/2, analyze_logs/2, T).
T = pipe.
```

## Chapter Summary

- **Target Registry** manages target metadata and families
- **Target Mapping** connects predicates to targets
- **Location** specifies where predicates run
- **Transport** specifies how they communicate
- **Defaults** are computed from family membership
- **Validation** catches configuration errors early

## Next Steps

In Chapter 4, we'll explore the pipe protocols in detail:
- TSV format specification
- JSON Lines format
- Header negotiation
- Field mapping between targets

## Exercises

1. **Registry query**: Write a query to find all targets that support `async`.

2. **Custom target**: Register a custom `nodejs` target in the `native` family with capabilities `[interpreted, async, http]`.

3. **Configuration**: Configure a pipeline where:
   - `input/1` is Bash
   - `process/2` is Go (with parallel(4))
   - `output/1` is remote SQL at 'db.local:5432'

4. **Resolution**: Given `csharp` calling `powershell`, trace through the resolution rules to determine location and transport.

## Advanced Example: Type Inference Pipeline with Target Mapping

This example demonstrates the **target registry and mapping system** by implementing a simple type inference pipeline. UnifyWeaver compiles Prolog predicates to optimal targets based on their characteristics.

### The Problem

Given expressions like `add(x, 1)` where `x` is unknown, infer the types. This is a simplified version of Hindley-Milner type inference.

### Step 1: Define the Type Inference Logic in Prolog

```prolog
% type_inference.pl - Type inference predicates
:- module(type_inference, [
    type_of/2,        % type_of(Expr, Type)
    unify_types/3,    % unify_types(T1, T2, Unified)
    infer_expr/3      % infer_expr(Expr, Env, NewEnv)
]).

% Built-in function signatures
signature(add, [int, int], int).
signature(concat, [str, str], str).
signature(length, [str], int).
signature(tostring, [int], str).

% Type unification (core of Hindley-Milner)
unify_types(T, T, T) :- !.
unify_types(var(X), T, T) :- \+ occurs_in(X, T), !.
unify_types(T, var(X), T) :- \+ occurs_in(X, T), !.
unify_types(_, _, fail).

% Occurs check for sound unification
occurs_in(X, var(X)) :- !.
occurs_in(X, compound(_, Args)) :- member(A, Args), occurs_in(X, A).

% Type inference for expressions
type_of(int(_), int).
type_of(str(_), str).
type_of(var(X), var(X)).
type_of(app(F, Args), RetType) :-
    signature(F, ArgTypes, RetType),
    maplist(type_of, Args, InferredTypes),
    maplist(unify_types, ArgTypes, InferredTypes, _).

% Infer with environment threading
infer_expr(Expr, EnvIn, EnvOut) :-
    type_of(Expr, Type),
    update_env(Expr, Type, EnvIn, EnvOut).
```

### Step 2: Declare Target Mappings

The target registry lets you specify which target compiles each predicate:

```prolog
% type_inference_config.pl
:- use_module('src/unifyweaver/core/target_registry').
:- use_module('src/unifyweaver/core/target_mapping').
:- use_module('src/unifyweaver/targets/python_target').
:- use_module('src/unifyweaver/targets/awk_target').

% Register targets with capabilities
:- register_target(python3, python, [recursion, memoization, complex_data]).
:- register_target(gawk, shell, [text_processing, streaming, fast_startup]).

% Declare which target handles each predicate
:- declare_target(parse_expr/2, awk).         % Fast text parsing
:- declare_target(type_of/2, python).         % Needs recursion
:- declare_target(unify_types/3, python).     % Needs backtracking
:- declare_target(infer_expr/3, python).      % Complex state
:- declare_target(format_output/2, awk).      % Simple formatting

% Configure connections
:- declare_connection(parse_expr/2, infer_expr/3, [format(json)]).
:- declare_connection(infer_expr/3, format_output/2, [format(tsv)]).
```

### Step 3: Compile Predicates to Targets

UnifyWeaver compiles your Prolog to the declared targets:

```prolog
% compile_type_inference.pl
compile_all :-
    % Compile type inference predicates to Python
    compile_predicate_to_python(type_of/2, [
        record_format(json),
        mode(function)
    ], TypeOfCode),

    compile_predicate_to_python(unify_types/3, [
        record_format(json),
        mode(function)
    ], UnifyCode),

    compile_predicate_to_python(infer_expr/3, [
        record_format(json),
        mode(generator),
        include_predicates([type_of/2, unify_types/3])
    ], InferCode),

    % Write to file
    open('infer_types.py', write, S),
    write(S, InferCode),
    close(S).
```

### Step 4: Generated Python (by UnifyWeaver)

UnifyWeaver generates this Python from your Prolog - you don't write it:

```python
#!/usr/bin/env python3
"""
Generated by UnifyWeaver from type_inference module
Predicates: type_of/2, unify_types/3, infer_expr/3
"""
import sys
import json

# signature/3 compiled as dictionary
_signature = {
    "add": (["int", "int"], "int"),
    "concat": (["str", "str"], "str"),
    "length": (["str"], "int"),
    "tostring": (["int"], "str"),
}

def unify_types(t1, t2):
    """unify_types(T1, T2, Unified) - Prolog unification"""
    if t1 == t2:
        return t1
    if isinstance(t1, str) and t1.startswith("?"):
        return t2
    if isinstance(t2, str) and t2.startswith("?"):
        return t1
    return None  # fail

def type_of(expr):
    """type_of(Expr, Type) - infer type of expression"""
    if isinstance(expr, int) or (isinstance(expr, str) and expr.isdigit()):
        return "int"
    if isinstance(expr, str) and expr.startswith('"'):
        return "str"
    if isinstance(expr, str) and expr[0].isupper():
        return f"?{expr}"  # Type variable
    if isinstance(expr, dict) and "func" in expr:
        func = expr["func"]
        args = expr.get("args", [])
        if func in _signature:
            arg_types, ret_type = _signature[func]
            inferred = [type_of(a) for a in args]
            for expected, got in zip(arg_types, inferred):
                if unify_types(expected, got) is None:
                    return "type_error"
            return ret_type
    return "?unknown"

def infer_expr(expr, env):
    """infer_expr(Expr, EnvIn, EnvOut) - infer with environment"""
    t = type_of(expr)
    new_env = env.copy()
    # Update environment with inferred bindings
    if isinstance(expr, str) and expr[0].isupper():
        if t and not t.startswith("?"):
            new_env[expr] = t
    return t, new_env

def process_stream():
    env = {}
    for line in sys.stdin:
        data = json.loads(line)
        expr = data.get("expr")
        result_type, env = infer_expr(expr, env)
        bindings = ", ".join(f"{k}:{v}" for k, v in env.items())
        print(f"{data.get('name', 'expr')}\t{result_type}\t{bindings}")

if __name__ == "__main__":
    process_stream()
```

### Step 5: Generate the Complete Pipeline

```prolog
% type_pipeline.pl
type_inference_pipeline(Script) :-
    generate_pipeline(
        [
            % Stage 1: Parse expressions (AWK)
            step(parse_expr, awk, '
                {
                    gsub(/[(),]/, "\t")
                    gsub(/[[:space:]]+/, "\t")
                    print
                }
            ', []),

            % Stage 2: Compiled type inference (Python)
            step(infer_types, python, 'infer_types.py', []),

            % Stage 3: Format output (AWK)
            step(format_output, awk, '
                {
                    printf "%s has type %s", $1, $2
                    if ($3 != "") printf " (where %s)", $3
                    print ""
                }
            ', [])
        ],
        [input('expressions.txt')],
        Script
    ).
```

### Sample Input (`expressions.txt`)

```
add(x, 1)
add(y, z)
concat(s, t)
length(s)
add(length(s), 1)
```

### Expected Output

```
add has type int (where x:int)
add has type int (where x:int, y:int, z:int)
concat has type str (where x:int, y:int, z:int, s:str, t:str)
length has type int (where x:int, y:int, z:int, s:str, t:str)
add has type int (where x:int, y:int, z:int, s:str, t:str)
```

### How Target Mapping Works Here

Query the mappings:

```prolog
?- predicate_target(parse_expr/2, T).
T = awk.

?- predicate_target(infer_types/2, T).
T = python.

?- resolve_transport(parse_expr/2, infer_types/2, Transport).
Transport = pipe.

?- targets_same_family(awk, python).
false.  % Different families, so pipes are used
```

### Why These Targets?

| Stage | Target | Reason |
|-------|--------|--------|
| `parse_expr` | AWK | Fast regex-based text parsing |
| `infer_types` | Python | Complex recursion and data structures |
| `format_output` | AWK | Simple string formatting |

### Modification Exercise

**Task**: Add a new target mapping for a **validation stage** that checks types are consistent.

**What to do**:
1. Add a new declaration: `:- declare_target(validate_types/2, python).`
2. Add a new pipeline step between `infer_types` and `format_output`
3. The validation step should check that all uses of a variable have the same type

**Hints**:
- The validation stage reads the type bindings from column 3
- It should output an error if the same variable has conflicting types
- Add `[error_on_failure(true)]` to the step options

**Sample validation output**:
```
add has type int (where x:int) [OK]
add has type int (where y:int, z:int) [OK]
ERROR: Variable 's' has conflicting types: str vs int
```

## API Reference

See `docs/design/cross-target-glue/04-api-reference.md` for complete predicate documentation.

---

## Navigation

**←** [Previous: Chapter 2: Philosophy and Design Principles](02_philosophy) | [📖 Book 7: Cross-Target Glue](./) | [Next: Chapter 4: Pipe Protocols and Data Formats →](04_pipe_protocols)
