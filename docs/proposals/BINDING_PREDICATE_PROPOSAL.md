# Proposal: Foreign Function Binding Predicate

**Status:** Draft
**Author:** Claude (via Claude Code)
**Date:** 2025-12-09
**Related:** `constraint_analyzer.pl`, `target_registry.pl`

## Summary

This proposal introduces a `binding/6` predicate to map Prolog predicates to foreign functions in target languages. This enables the transpiler to correctly translate calls to built-in or library functions while preserving semantic information about effects, types, and execution characteristics.

## Motivation

When transpiling Prolog to target languages (Bash, Go, Python, C#, etc.), we need to:

1. Map Prolog predicates to equivalent target language functions
2. Track side effects for optimization decisions (e.g., `order_independent` evaluation)
3. Handle target-specific semantics (exit codes in Bash, error returns in Go, exceptions in Python)
4. Preserve type information for statically-typed targets
5. Enable correct code generation for different calling conventions

## Proposed Design

### Core Predicate

```prolog
%% binding(+Target, +PrologPred, +TargetName, +Inputs, +Outputs, +Options)
%
% Target:     atom - target language (go, bash, python, csharp, sql, etc.)
% PrologPred: Name/Arity - the Prolog predicate being mapped
% TargetName: string/atom - the target language function/command
% Inputs:     list of input argument specifications
% Outputs:    list of output argument specifications
% Options:    list of options/effects/annotations
```

### Argument Specifications

Input and output arguments can be specified as:

| Form | Description | Example |
|------|-------------|---------|
| `atom` | Simple type name | `string`, `int`, `list`, `path` |
| `type(T)` | Explicit type for typed targets | `type('[]byte')` |
| `mode(M)` | Argument mode (+/-/?) | `mode(+)` for input |

### Options/Effects

| Option | Description |
|--------|-------------|
| `pure` | No side effects - can be reordered, memoized, parallelized |
| `effect(io)` | Performs I/O operations |
| `effect(state)` | Mutates state |
| `effect(throws)` | Can raise exceptions/errors |
| `nondeterministic` | May produce multiple results |
| `deterministic` | Produces exactly one result |
| `order_independent` | Results can be accumulated in any order |
| `exit_status(Mapping)` | How to interpret shell exit codes |
| `returns_error` | Returns error as additional value (Go-style) |
| `variadic` | Accepts variable number of arguments |

### Exit Status Handling

For shell targets, exit status is a critical concept:

```prolog
% Option A: Exit status as output argument
binding(bash, file_exists/2, "test -f", [path], [exit_status], []).

% Option B: Exit status as option with mapping
binding(bash, file_exists/1, "test -f", [path], [],
        [effect(io), exit_status(0=true, _=false)]).

% Option C: Boolean interpretation
binding(bash, file_exists/1, "test -f", [path], [],
        [effect(io), exit_status(success_is_true)]).
```

For Go, error handling is idiomatic:

```prolog
binding(go, file_exists/2, "os.Stat", [path], [bool],
        [effect(io), returns_error]).
```

## Examples

### Pure Functions

```prolog
% Length/size operations - pure, deterministic
binding(go, length/2, "len", [list], [int], [pure, deterministic]).
binding(python, length/2, "len", [list], [int], [pure, deterministic]).
binding(bash, length/2, "wc -l", [list], [int], [pure, deterministic]).
binding(csharp, length/2, ".Count", [list], [int], [pure, deterministic]).
```

### I/O Operations

```prolog
% File existence check
binding(bash, file_exists/1, "test -f", [path], [],
        [effect(io), exit_status(0=true, _=false)]).
binding(go, file_exists/2, "os.Stat", [path], [bool],
        [effect(io), returns_error]).
binding(python, file_exists/1, "os.path.exists", [path], [bool],
        [effect(io)]).

% File reading
binding(bash, read_file/2, "cat", [path], [string], [effect(io)]).
binding(go, read_file/2, "os.ReadFile", [path], [bytes],
        [effect(io), returns_error]).
binding(python, read_file/2, "open(...).read()", [path], [string],
        [effect(io), effect(throws)]).
```

### Nondeterministic Operations

```prolog
% Glob/file listing - produces multiple results
binding(bash, glob_files/2, "ls", [pattern], [path],
        [effect(io), nondeterministic, order_independent]).
binding(python, glob_files/2, "glob.glob", [pattern], [list(path)],
        [effect(io)]).
binding(go, glob_files/2, "filepath.Glob", [pattern], [list(path)],
        [effect(io), returns_error]).
```

### State Mutation

```prolog
% Dictionary/map operations
binding(python, dict_set/3, "dict.__setitem__", [dict, key, value], [],
        [effect(state)]).
binding(go, map_set/3, "map[k]=v", [map, key, value], [],
        [effect(state)]).
```

### Typed Targets

For statically-typed targets like Go, C#, and Rust:

```prolog
% With explicit type annotations
binding(go, parse_int/2, "strconv.Atoi",
        [type(string)], [type(int)],
        [effect(throws), returns_error]).

binding(csharp, parse_int/2, "int.Parse",
        [type(string)], [type(int)],
        [effect(throws)]).

binding(rust, parse_int/2, "str::parse::<i32>",
        [type('&str')], [type('Result<i32, ParseIntError>')],
        []).
```

## Integration with Existing Systems

### Constraint Analyzer

The `order_independent` option integrates with `constraint_analyzer.pl`:

```prolog
% If a binding is order_independent and the predicate has unordered(true),
% the compiler can use parallel evaluation or set-based accumulation
binding(bash, find_files/2, "find", [path], [path],
        [effect(io), nondeterministic, order_independent]).
```

### Target Registry

Bindings are organized by target, which aligns with `target_registry.pl`:

```prolog
% Query all bindings for a target
bindings_for_target(Target, Bindings) :-
    findall(binding(Target, P, N, I, O, Opts),
            binding(Target, P, N, I, O, Opts),
            Bindings).
```

## Open Questions

1. **Arity handling**: Should bindings support multiple arities for the same predicate?
   ```prolog
   binding(bash, echo/1, "echo", [string], [], [effect(io)]).
   binding(bash, echo/2, "echo -n", [string], [], [effect(io)]).  % No newline
   ```

2. **Bidirectional predicates**: How to handle Prolog predicates that work in multiple modes?
   ```prolog
   % append/3 can be used to split or join
   binding(python, append/3, "???", [list, list], [list], [pure]).
   ```

3. **Target-specific imports**: Should bindings declare required imports?
   ```prolog
   binding(go, read_file/2, "os.ReadFile", [path], [bytes],
           [effect(io), returns_error, import("os")]).
   ```

4. **Fallback chains**: What if a target doesn't have a direct equivalent?
   ```prolog
   % Primary binding
   binding(go, str_reverse/2, "???", [string], [string], [pure]).
   % Could fall back to generated code
   ```

## Implementation Plan

1. **Phase 1**: Define `binding/6` predicate and storage mechanism
2. **Phase 2**: Populate bindings for core operations across all targets
3. **Phase 3**: Integrate with code generators to use bindings
4. **Phase 4**: Add effect analysis for optimization decisions

## Alternatives Considered

### Alternative A: Separate predicates per aspect

```prolog
foreign_name(go, length/2, "len").
foreign_type(go, length/2, [list], [int]).
foreign_effect(go, length/2, pure).
```

**Rejected**: Too fragmented, harder to maintain consistency.

### Alternative B: Target-specific modules

```prolog
% In go_bindings.pl
:- module(go_bindings, [...]).
binding(length/2, "len", [list], [int], [pure]).
```

**Consideration**: Could be used in addition to centralized storage for organization.

## References

- `src/unifyweaver/core/constraint_analyzer.pl` - Existing constraint system
- `src/unifyweaver/core/target_registry.pl` - Target registration system
- `docs/EXTENDED_README.md` - Project documentation
