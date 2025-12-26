# Proposal: Foreign Function Binding Predicate

**Status:** Draft
**Author:** John William Creighton (@s243a)
**Co-Author:** Claude Code (Opus 4.5)
**Date:** 2025-12-09
**Updated:** 2025-12-09 (v3)
**Related:** `constraint_analyzer.pl`, `target_registry.pl`, `firewall_v2.pl`

## Summary

This proposal introduces a `binding/6` predicate to map Prolog predicates to foreign functions in target languages. This enables the transpiler to correctly translate calls to built-in or library functions while preserving semantic information about effects, types, and execution characteristics.

## Terminology Note

The term "binding" is established terminology in this domain:

- **Ada** uses "language bindings" for inter-language calls
- **SWI-Prolog** documentation refers to "foreign language interface" and "bindings"
- **Scryer Prolog** FFI uses "bind" terminology for function definitions
- A "binding" in computing generally refers to an API providing glue code to use a library or service in a given language

Alternative terms considered:
- `foreign/6` - more Prolog-traditional but less descriptive
- `mapping/6` - generic, doesn't convey the FFI nature
- `extern/6` - C-influenced, less idiomatic for Prolog

**Decision:** `binding/6` is appropriate and well-established terminology.

## Motivation

When transpiling Prolog to target languages (Bash, Go, Python, C#, etc.), we need to:

1. Map Prolog predicates to equivalent target language functions
2. Track side effects for optimization decisions (e.g., `order_independent` evaluation)
3. Handle target-specific semantics (exit codes in Bash, error returns in Go, exceptions in Python)
4. Preserve type information for statically-typed targets
5. Enable correct code generation for different calling conventions
6. Declare imports for firewall validation

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
| `import(Module)` | Required import/module for firewall validation |
| `pattern(Name)` | Design pattern for non-standard usage (see below) |
| `total` | Function always succeeds for valid inputs |
| `partial` | Function may fail/return no result |

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

## Supporting Predicates

### Effect Annotations (Separate Predicate)

For declaring effects independently of bindings, use a separate `effect/2` predicate:

```prolog
%% effect(+PrologPred, +Effects)
%
% Declares effects for a Prolog predicate without specifying a target binding.
% Useful for user-defined predicates or when effects are target-independent.

effect(my_io_predicate/2, [effect(io), nondeterministic]).
effect(pure_computation/3, [pure, deterministic]).
```

This separates concerns: `binding/6` maps to foreign functions, while `effect/2` annotates any predicate with effect information.

### Bidirectional Predicates

For Prolog predicates that work in multiple modes (e.g., `append/3`):

```prolog
%% bidirectional(+PrologPred, +Modes)
%
% Declares that a predicate supports multiple argument modes.
% Each mode specifies which arguments are inputs (+) vs outputs (-).

bidirectional(append/3, [
    mode(+, +, -),  % Join: append([1,2], [3,4], X) -> X = [1,2,3,4]
    mode(-, -, +)   % Split: append(X, Y, [1,2,3]) -> multiple solutions
]).

%% binding_mode(+Target, +PrologPred, +Mode, +TargetName, +Options)
%
% Bind a specific mode of a bidirectional predicate to a target function.

binding_mode(python, append/3, mode(+, +, -), "list.__add__", [pure]).
binding_mode(python, append/3, mode(-, -, +), "???", [nondeterministic]).
```

A helper rule can flip arguments for reversed modes:

```prolog
% Generate reversed binding from forward binding
derive_reverse_binding(Target, Pred, ForwardName, ReverseName) :-
    bidirectional(Pred, Modes),
    member(mode(-, -, +), Modes),
    binding_mode(Target, Pred, mode(+, +, -), ForwardName, _),
    % Generate or lookup reverse implementation
    ...
```

### Import Declarations for Firewall

Bindings can declare required imports, which integrates with the firewall module:

```prolog
binding(go, read_file/2, "os.ReadFile", [path], [bytes],
        [effect(io), returns_error, import("os")]).

binding(python, json_parse/2, "json.loads", [string], [dict],
        [pure, import("json")]).

binding(python, http_get/2, "requests.get", [url], [response],
        [effect(io), effect(throws), import("requests")]).
```

The firewall can then validate:

```prolog
% Check if import is allowed before code generation
validate_binding_imports(Target, Pred) :-
    binding(Target, Pred, _, _, _, Options),
    member(import(Module), Options),
    (   allowed_import(Target, Module)
    ->  true
    ;   throw(firewall_error(import_denied(Target, Module)))
    ).
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
        [effect(io), returns_error, import("os")]).
binding(python, file_exists/1, "os.path.exists", [path], [bool],
        [effect(io), import("os.path")]).

% File reading
binding(bash, read_file/2, "cat", [path], [string], [effect(io)]).
binding(go, read_file/2, "os.ReadFile", [path], [bytes],
        [effect(io), returns_error, import("os")]).
binding(python, read_file/2, "open(...).read()", [path], [string],
        [effect(io), effect(throws)]).
```

### Nondeterministic Operations

```prolog
% Glob/file listing - produces multiple results
binding(bash, glob_files/2, "ls", [pattern], [path],
        [effect(io), nondeterministic, order_independent]).
binding(python, glob_files/2, "glob.glob", [pattern], [list(path)],
        [effect(io), import("glob")]).
binding(go, glob_files/2, "filepath.Glob", [pattern], [list(path)],
        [effect(io), returns_error, import("path/filepath")]).
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
        [effect(throws), returns_error, import("strconv")]).

binding(csharp, parse_int/2, "int.Parse",
        [type(string)], [type(int)],
        [effect(throws)]).

binding(rust, parse_int/2, "str::parse::<i32>",
        [type('&str')], [type('Result<i32, ParseIntError>')],
        []).
```

### Multiple Arities

Bindings naturally support multiple arities for the same predicate name:

```prolog
binding(bash, echo/1, "echo", [string], [], [effect(io)]).
binding(bash, echo/2, "echo -n", [string, string], [], [effect(io)]).

binding(python, print/1, "print", [any], [], [effect(io), import("builtins")]).
binding(python, print/2, "print", [any, any], [], [effect(io), import("builtins")]).
binding(python, print/3, "print", [any, any, any], [], [effect(io), import("builtins")]).
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

### Firewall Integration

The `import/1` option enables firewall validation:

```prolog
% Extract all imports required by bindings used in a program
required_imports(Target, Program, Imports) :-
    findall(Module,
            (member(Pred, Program),
             binding(Target, Pred, _, _, _, Opts),
             member(import(Module), Opts)),
            Imports).

% Validate against firewall policy
validate_imports(Target, Imports) :-
    forall(member(M, Imports),
           firewall:check_import(Target, M)).
```

## Target-Specific Modules (Alternative Organization)

While the core `binding/6` predicate is centralized, target-specific modules can provide convenience predicates with reduced arity:

```prolog
% In go_bindings.pl
:- module(go_bindings, [go_binding/5]).

% Internal reduced-arity form (not exported)
go_binding_(Pred, Name, In, Out, Opts) :-
    binding(go, Pred, Name, In, Out, Opts).

% Exported convenience predicate
go_binding(Pred, Name, In, Out, Opts) :-
    go_binding_(Pred, Name, In, Out, Opts).

% Or generate from central registry
go_binding(Pred, Name, In, Out, Opts) :-
    binding(go, Pred, Name, In, Out, Opts).
```

This pattern allows target modules to:
1. Add target-specific helper rules
2. Provide a cleaner API for target-specific code
3. Map reduced-arity forms to the general `binding/6`

## User-Defined Binding Directives

Users can declare their own bindings directly in Prolog code using target-specific directives. These directives use Prolog's `term_expansion/2` mechanism to register bindings at load time.

### Supported Targets

| Target | Directive | Status |
|--------|-----------|--------|
| Go | `:- go_binding(...)` | ✅ Implemented |
| Python | `:- py_binding(...)` | ✅ Implemented |
| Rust | `:- rs_binding(...)` | ✅ Implemented |
| C# | `:- cs_binding(...)` | ✅ Implemented |

### Syntax

```prolog
%% :- <target>_binding(Pred, TargetName, Inputs, Outputs, Options)
%
% Pred:       Name/Arity - the Prolog predicate being bound
% TargetName: string/atom - the target language function/method
% Inputs:     list of input types
% Outputs:    list of output types
% Options:    list of options (pure, effect(io), import(...), etc.)
```

### Examples

```prolog
% Go binding for a custom hash function
:- go_binding(my_hash/2, 'mylib.Hash', [string], [string], [
    pure,
    import("mylib")
]).

% Python binding for a machine learning prediction
:- py_binding(predict/2, 'model.predict', [list], [list], [
    effect(io),
    import("model")
]).

% Rust binding for a crypto operation
:- rs_binding(encrypt/3, 'crypto::encrypt', [string, string], [string], [
    pure,
    import("crypto")
]).

% C# binding for a .NET library call
:- cs_binding(format_date/2, 'DateTime.Parse', [string], [datetime], [
    pure,
    effect(throws),
    using("System")
]).
```

### Implementation

Each target's bindings module provides a `term_expansion/2` clause that transforms the directive into a binding registration:

```prolog
% In python_bindings.pl
user:term_expansion(
    (:- py_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(
        python, Pred, TargetName, Inputs, Outputs, Options)))
).
```

This approach allows bindings to be declared alongside the predicates that use them, making code more self-contained and portable.

## Fallback Chains

When multiple target preferences are specified, bindings can fall back to alternative targets:

```prolog
%% resolve_binding(+Preferences, +Pred, -Target, -Binding)
%
% Resolve a binding given a list of preferred targets.
% Falls back through the list until a binding is found.

resolve_binding([Target|_], Pred, Target, Binding) :-
    binding(Target, Pred, Name, In, Out, Opts),
    !,
    Binding = binding(Target, Pred, Name, In, Out, Opts).
resolve_binding([_|Rest], Pred, Target, Binding) :-
    resolve_binding(Rest, Pred, Target, Binding).

% Usage:
% ?- resolve_binding([rust, go, python], str_reverse/2, Target, Binding).
% Target = python,
% Binding = binding(python, str_reverse/2, "[::-1]", [string], [string], [pure]).
```

This enables graceful degradation when a preferred target lacks a binding.

## Design Patterns for Non-Standard Usage

Languages often use functions in non-standard ways. For instance, in Bash we might use `echo` or a variable assignment to return a result rather than a true function return. The binding should declare the **semantic intent** (design pattern) rather than the literal language construct.

### Pattern-Based Type Semantics

```prolog
%% pattern(+PatternName, +Description)
%
% Declares a design pattern for non-standard function usage.

pattern(stdout_return, "Return value via stdout (shell idiom)").
pattern(variable_return, "Return value via variable assignment").
pattern(exit_code_bool, "Boolean result via exit code").
pattern(pipe_transform, "Transform data through pipe").
pattern(accumulator, "Build result via accumulator parameter").

%% binding with pattern annotation
binding(bash, get_value/2, "echo $1", [varname], [string],
        [pattern(stdout_return), pure]).

binding(bash, set_value/2, "$1=$2", [varname, value], [],
        [pattern(variable_return), effect(state)]).

binding(bash, is_empty/1, "test -z", [string], [],
        [pattern(exit_code_bool), pure]).
```

### Types Match Intent, Not Literals

The input/output types should reflect the **logical meaning** of the operation:

```prolog
% Even though bash `wc -l` outputs to stdout, the binding declares
% the semantic output type as 'int' because that's the intent
binding(bash, line_count/2, "wc -l", [file], [int],
        [pattern(stdout_return), effect(io)]).

% The AWK print returns via stdout, but semantically it's outputting records
binding(bash, emit_record/1, "print", [record], [],
        [pattern(stdout_return), effect(io)]).
```

### Composability Through Patterns

Patterns enable reusable composition rules:

```prolog
% Rule: stdout_return patterns can be piped
composable_pipe(B1, B2) :-
    binding(bash, _, _, _, _, Opts1),
    member(pattern(stdout_return), Opts1),
    binding(bash, _, _, _, _, Opts2),
    member(pattern(pipe_transform), Opts2).
```

This approach was inspired by the PowerShell target educational materials, specifically:

- **Chapter 3: Cmdlet Generation** (`book-12-powershell-target/03_cmdlet_generation.md`)
  - Shows how compilation options map to PowerShell cmdlet attributes
  - Demonstrates `cmdlet_name`, `cmdlet_binding`, `verbose_output` as binding-like declarations
  - Parameter attributes (`Mandatory`, `ValueFromPipeline`) parallel our input/output specifications

- **Chapter 5: Windows Automation** (`book-12-powershell-target/05_windows_automation.md`)
  - Models system state (services, registry, events) as Prolog facts
  - Rules like `low_disk/1`, `memory_pressure/0` show semantic predicates over system state
  - Generated PowerShell uses patterns (CIM queries, registry providers) that need binding declarations

- **PowerShell Semantic Target** (`book-03-csharp-target/06_powershell_semantic.md`)
  - XML streaming via .NET, vector search strategies
  - Emphasizes composability and reusability across runtime environments

These materials demonstrate that bindings need to capture not just function names, but also parameter semantics, pipeline behavior, and platform-specific patterns.

## Design Rationale: Haskell Type System Connection

The design mirrors concepts from Haskell's type system, which is rooted in logic:

| Haskell Concept | This Proposal | Purpose |
|-----------------|---------------|---------|
| Type classes | Target-specific bindings | Polymorphism over targets |
| Effect types (IO, State) | `effect(io)`, `effect(state)` | Track computational effects |
| Pure functions | `pure` option | Enable optimizations |
| Type signatures | Input/Output lists | Document interfaces |

This connection is natural given Prolog's logical foundations and enables similar reasoning about program properties.

## Database Semantics: Bindings as Constraints

Given UnifyWeaver's deep connection between Prolog and database operations, bindings can be viewed through a relational/constraint lens.

### Bindings as Relations That Always Succeed

In database terms, a binding defines a **functional dependency** or **view** that always returns results (never fails in the Prolog sense of "no solutions"). The binding declares:

> "For these inputs, this target function will produce these outputs"

```prolog
% This binding asserts: for any valid string input, len() returns an int
% It's a total function - always succeeds for valid inputs
binding(python, length/2, "len", [string], [int], [pure, deterministic]).
```

### Constraint Interpretation

| Database Concept | Binding Interpretation |
|------------------|------------------------|
| Functional dependency | Deterministic binding (inputs → outputs) |
| View definition | Binding = view over target language primitives |
| NOT NULL constraint | No `effect(throws)` = guaranteed result |
| CHECK constraint | Type specifications = domain constraints |
| Foreign key | `import(Module)` = external dependency |

### Bindings as Datalog-Style Rules

A binding can be seen as a Datalog rule that always holds:

```prolog
% Conceptually, this binding is like asserting:
% ∀ X, Y: python_len(X, Y) ← is_string(X) ∧ Y = len(X)
%
% The binding never "fails" - it defines a mapping that exists
binding(python, length/2, "len", [string], [int], [pure]).
```

### Implications for Query Planning

If bindings always succeed (for valid inputs), the query planner can:

1. **Reorder freely** - Pure bindings with no effects can be evaluated in any order
2. **Push down filters** - Bindings can be treated as base relations for join optimization
3. **Materialize or stream** - Choose execution strategy based on binding properties

```prolog
% Query: find files larger than 1MB
% The binding tells us file_size/2 always produces a result for valid paths
% So we can safely use it in a filter without worrying about failure

large_file(Path) :-
    glob_files(Pattern, Path),      % nondeterministic
    file_size(Path, Size),          % deterministic - always succeeds
    Size > 1000000.                 % filter
```

### Partial Functions and Failure

For bindings that might not produce a result, use explicit annotations:

```prolog
% This can fail (file might not exist)
binding(bash, read_file/2, "cat", [path], [string],
        [effect(io), effect(throws), partial]).

% vs. this always succeeds (returns empty string for missing file)
binding(bash, read_file_safe/2, "cat 2>/dev/null || echo ''", [path], [string],
        [effect(io), total]).
```

The `partial` vs `total` distinction maps to database NULL semantics and helps query planners handle potential failures.

## Implementation Plan

1. **Phase 1**: Define `binding/6`, `effect/2`, `bidirectional/2` predicates and storage
2. **Phase 2**: Populate bindings for core operations across all targets
3. **Phase 3**: Integrate with code generators to use bindings
4. **Phase 4**: Add effect analysis for optimization decisions
5. **Phase 5**: Integrate import declarations with firewall module

## Open Questions (Resolved)

1. ~~**Arity handling**~~ → Multiple arities are naturally supported; each is a separate binding.

2. ~~**Bidirectional predicates**~~ → Use `bidirectional/2` to declare modes and `binding_mode/5` for mode-specific bindings.

3. ~~**Target-specific imports**~~ → Use `import(Module)` option; integrates with firewall for validation.

4. ~~**Fallback chains**~~ → Use `resolve_binding/4` with preference list to fall back through targets.

## References

### Internal Documentation
- `src/unifyweaver/core/constraint_analyzer.pl` - Existing constraint system
- `src/unifyweaver/core/target_registry.pl` - Target registration system
- `src/unifyweaver/core/firewall_v2.pl` - Firewall and security system
- `docs/EXTENDED_README.md` - Project documentation

### Educational Materials
- `education/UnifyWeaver_Education/book-12-powershell-target/03_cmdlet_generation.md` - Cmdlet generation patterns
- `education/UnifyWeaver_Education/book-12-powershell-target/05_windows_automation.md` - Windows automation patterns
- `education/UnifyWeaver_Education/book-03-csharp-target/06_powershell_semantic.md` - PowerShell semantic target

### External References
- [SWI-Prolog Foreign Language Interface](https://www.swi-prolog.org/pldoc/man?section=foreign)
- [Scryer Prolog FFI](https://www.scryer.pl/ffi)
- [Foreign Function Interface - Wikipedia](https://en.wikipedia.org/wiki/Foreign_function_interface)

## Acknowledgements

This proposal was developed collaboratively. The technical content, including the predicate designs, effect system, pattern-based semantics, database constraint interpretation, and code examples, was extensively contributed by Claude Code (Opus 4.5). The conceptual direction, design requirements, and feedback on non-standard usage patterns, composability goals, and database semantics were provided by John William Creighton (@s243a).

The binding concept was inspired by earlier work on the PowerShell target (book-12-powershell-target, Chapters 3 and 5), which explored making code more reusable and composable across different runtime environments.
