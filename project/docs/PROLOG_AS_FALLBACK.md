# Configuring Prolog as Fallback Target

**Status:** ✅ Dialect System Ready
**Integration:** Preferences + Firewall Systems

## Overview

UnifyWeaver's Prolog target can serve as a **universal fallback** when transpilation to other targets (Bash, C#) fails. This is configured through the existing **preferences** and **firewall** systems.

## Why Prolog as Fallback?

1. **Universal compatibility** - Any valid Prolog predicate can be copied verbatim
2. **No transpilation complexity** - Direct predicate copying when other methods fail
3. **Preserves semantics** - Guaranteed correct behavior (same Prolog code)
4. **Compilation option** - GNU Prolog can compile to native binaries
5. **Graceful degradation** - System never fails, always produces executable code

## Configuration Layers

### Layer 1: Global Defaults (Broadest)

Set system-wide preferences for all predicates:

```prolog
:- use_module('src/unifyweaver/core/preferences').
:- use_module('src/unifyweaver/core/firewall').

% Preferences: Try Bash first, fall back to Prolog
:- assertz(preferences:preferences_default([
    prefer([bash]),
    fallback_order([prolog_swi, prolog_gnu])
])).

% Firewall: Allow both Bash and Prolog execution
:- assertz(firewall:firewall_default([
    execution([bash, prolog_swi, prolog_gnu])
])).
```

### Layer 2: Predicate-Specific Rules

Override defaults for specific predicates:

```prolog
% Complex predicate that might fail in Bash
:- assertz(preferences:rule_preferences(complex_logic/3, [
    prefer([bash]),
    fallback_order([prolog_swi]),  % Use interpreted Prolog for debugging
    optimization(balance)
])).

% Performance-critical predicate
:- assertz(preferences:rule_preferences(hot_path/2, [
    prefer([bash]),
    fallback_order([prolog_gnu]),  % Use compiled Prolog if Bash fails
    optimization(speed)
])).

% Security-sensitive predicate
:- assertz(firewall:rule_firewall(sensitive_data/2, [
    execution([prolog_swi]),  % Only allow interpreted Prolog (no Bash)
    denied([bash])             % Explicitly deny Bash
])).
```

### Layer 3: Runtime Options (Highest Precedence)

Override at compilation time:

```prolog
% Force Prolog target at runtime
compile_recursive(my_pred/2, [target(prolog_gnu), compile(true)], Code).

% Force fallback behavior
compile_recursive(my_pred/2, [force_fallback(true)], Code).
```

## Target Naming Convention

### Prolog Targets

| Target Name | Dialect | Compilation | Use Case |
|------------|---------|-------------|----------|
| `prolog_swi` | SWI-Prolog | Interpreted | Development, debugging, full features |
| `prolog_gnu` | GNU Prolog | Compiled (gplc) | Production, performance, stack algorithms |
| `prolog` | Configurable | Varies | Generic alias (see below) |

### Configuring the `prolog` Alias

The `prolog` target is a **configurable alias** that expands to one or more concrete dialects based on your preference:

```prolog
% Set what 'prolog' means
:- use_module('src/unifyweaver/targets/prolog_dialects').

% Option 1: SWI-Prolog only (default)
:- set_prolog_default(swi).

% Option 2: GNU Prolog only
:- set_prolog_default(gnu).

% Option 3: Try GNU, fall back to SWI
:- set_prolog_default(gnu_fallback_swi).

% Option 4: Try SWI, fall back to GNU
:- set_prolog_default(swi_fallback_gnu).

% Option 5: Custom dialect list
:- set_prolog_default([gnu, swi]).
```

**Usage Example:**
```prolog
% After setting gnu_fallback_swi:
fallback_order([prolog])
% Expands to: [gnu, swi] - try compilation first, fall back to interpreted

% Without configuration:
fallback_order([prolog])
% Expands to: [swi] - interpreted only (default)
```

### Other Targets

| Target Name | Description |
|------------|-------------|
| `bash` | Bash script generation |
| `csharp` | C# LINQ-style compilation |
| `powershell` | PowerShell script generation |

## Fallback Logic Flow

```
1. Check Runtime Options
   ├─ target(X) specified? → Use X
   └─ force_fallback(true)? → Skip to fallback

2. Check Firewall Policy
   ├─ Is preferred target allowed?
   │  ├─ Yes → Try compilation
   │  └─ No → Try next in preference order
   └─ All denied? → Error

3. Try Preferred Target
   ├─ Compilation succeeds? → Done ✓
   └─ Compilation fails? → Try fallback

4. Try Fallback Targets (in order)
   ├─ For each target in fallback_order:
   │  ├─ Firewall allows?
   │  │  ├─ Yes → Try compilation
   │  │  │  ├─ Success? → Done ✓
   │  │  │  └─ Failure? → Next fallback
   │  │  └─ No → Next fallback
   └─ All fallbacks exhausted? → Error
```

## Example Configurations

### Example 1: Bash with Prolog Fallback (Development)

```prolog
% Global: Prefer Bash, fall back to SWI-Prolog for development
:- assertz(preferences:preferences_default([
    prefer([bash]),
    fallback_order([prolog]),  % Using generic 'prolog' alias
    optimization(balance)
])).

:- assertz(firewall:firewall_default([
    execution([bash, prolog_swi, prolog_gnu])
])).

% With default prolog configuration (swi):
% Result:
% - Simple predicates → Bash
% - Complex predicates (transpilation fails) → SWI-Prolog (interpreted)
% - Allows interactive debugging with SWI-Prolog
```

**Alternative with Explicit Dialect:**
```prolog
fallback_order([prolog_swi])  % Explicit SWI-Prolog
```

### Example 2: Bash with GNU Prolog Fallback (Production)

```prolog
% Configure prolog alias for production: try compilation, fall back to interpreted
:- use_module('src/unifyweaver/targets/prolog_dialects').
:- set_prolog_default(gnu_fallback_swi).

% Global: Prefer Bash, fall back to Prolog
:- assertz(preferences:preferences_default([
    prefer([bash]),
    fallback_order([prolog]),  % Will expand to [gnu, swi]
    optimization(speed)
])).

:- assertz(firewall:firewall_default([
    execution([bash, prolog_gnu, prolog_swi])
])).

% Result with gnu_fallback_swi:
% - Simple predicates → Bash
% - Complex predicates → Try GNU Prolog compilation
% - If GNU compilation fails → Try SWI-Prolog interpreted
% - Production-ready with graceful degradation
```

**Alternative with Explicit Dialect:**
```prolog
fallback_order([prolog_gnu])  % GNU only (no SWI fallback)
```

### Example 3: Multi-Target with Cascading Fallback

```prolog
% Try C# first, then Bash, finally Prolog
:- assertz(preferences:preferences_default([
    prefer([csharp, bash]),
    fallback_order([prolog_gnu, prolog_swi]),
    optimization(speed)
])).

:- assertz(firewall:firewall_default([
    execution([csharp, bash, prolog_gnu, prolog_swi])
])).

% Fallback cascade:
% 1. Try C# (LINQ compilation)
% 2. If fails → Try Bash
% 3. If fails → Try GNU Prolog (compiled)
% 4. If fails → Try SWI-Prolog (guaranteed to work)
```

### Example 4: Predicate-Specific Fallback Strategy

```prolog
% Default: Bash only
:- assertz(preferences:preferences_default([
    prefer([bash])
])).

% Recursive predicates: Allow Prolog fallback
:- assertz(preferences:rule_preferences(ancestor/2, [
    prefer([bash]),
    fallback_order([prolog_swi])
])).

% Constraint solving: Prefer Prolog from start
:- assertz(preferences:rule_preferences(solve_puzzle/2, [
    prefer([prolog_swi]),
    fallback_order([])  % No fallback, must use Prolog
])).

% Data processing: Multi-stage fallback
:- assertz(preferences:rule_preferences(process_data/3, [
    prefer([bash]),
    fallback_order([csharp, prolog_gnu])
])).
```

### Example 5: Security-Constrained Fallback

```prolog
% Sensitive predicates: Only allow Prolog (sandboxed)
:- assertz(firewall:rule_firewall(handle_credentials/2, [
    execution([prolog_swi]),
    denied([bash, csharp]),  % Never allow shell or compiled code
    network_access(denied)
])).

% Public predicates: Allow all targets
:- assertz(firewall:rule_firewall(public_api/2, [
    execution([bash, csharp, prolog_swi, prolog_gnu])
])).

% Result:
% - Sensitive predicates forced to SWI-Prolog only
% - Public predicates can use any target with fallback
```

## Implementation in Main Compiler

The main compilation logic should respect preferences and firewall:

```prolog
compile_with_fallback(Predicate, RuntimeOptions, FinalCode) :-
    % Get merged preferences
    preferences:get_final_options(Predicate, RuntimeOptions, Options),
    firewall:get_firewall_policy(Predicate, FirewallPolicy),

    % Extract target preferences
    option(prefer(PreferredTargets), Options, [bash]),
    option(fallback_order(FallbackTargets), Options, []),

    % Try targets in order
    append(PreferredTargets, FallbackTargets, AllTargets),
    try_targets_in_order(AllTargets, Predicate, Options, FirewallPolicy, FinalCode).

try_targets_in_order([], Predicate, _Options, _Firewall, _Code) :-
    format(atom(Error), 'All targets exhausted for ~w', [Predicate]),
    throw(error(compilation_failed, Error)).

try_targets_in_order([Target|Rest], Predicate, Options, Firewall, Code) :-
    % Check firewall allows this target
    (   validate_target(Target, Firewall)
    ->  % Try compilation
        (   catch(
                compile_to_target(Target, Predicate, Options, Code),
                Error,
                (format('[Fallback] ~w failed: ~w, trying next target~n', [Target, Error]), fail)
            )
        ->  format('[Success] Compiled ~w to ~w~n', [Predicate, Target])
        ;   % Try next target
            try_targets_in_order(Rest, Predicate, Options, Firewall, Code)
        )
    ;   % Target not allowed by firewall, skip
        format('[Firewall] Target ~w denied for ~w, trying next~n', [Target, Predicate]),
        try_targets_in_order(Rest, Predicate, Options, Firewall, Code)
    ).

validate_target(Target, FirewallPolicy) :-
    % Check if target is in execution whitelist
    (   member(execution(Allowed), FirewallPolicy)
    ->  member(Target, Allowed)
    ;   true  % No restriction, allow
    ),
    % Check if target is not in denied list
    (   member(denied(Denied), FirewallPolicy)
    ->  \+ member(Target, Denied)
    ;   true  % No denied list
    ).

compile_to_target(prolog_swi, Predicate, Options, Code) :-
    % Use Prolog target with SWI dialect
    prolog_target:generate_prolog_script([Predicate], [dialect(swi)|Options], Code).

compile_to_target(prolog_gnu, Predicate, Options, Code) :-
    % Use Prolog target with GNU dialect
    prolog_target:generate_prolog_script([Predicate], [dialect(gnu)|Options], Code).

compile_to_target(bash, Predicate, Options, Code) :-
    % Use existing Bash compilation
    bash_compiler:compile(Predicate, Options, Code).

compile_to_target(csharp, Predicate, Options, Code) :-
    % Use C# compilation
    csharp_compiler:compile(Predicate, Options, Code).
```

## Testing Fallback Behavior

```prolog
% Test 1: Successful Bash compilation (no fallback needed)
?- compile_with_fallback(simple_pred/2, [], Code).
% [Success] Compiled simple_pred/2 to bash

% Test 2: Bash fails, falls back to Prolog
?- compile_with_fallback(complex_pred/3, [], Code).
% [Fallback] bash failed: unsupported_feature, trying next target
% [Success] Compiled complex_pred/3 to prolog_swi

% Test 3: Force Prolog target
?- compile_with_fallback(my_pred/2, [target(prolog_gnu)], Code).
% [Success] Compiled my_pred/2 to prolog_gnu

% Test 4: Firewall denies Bash, forces Prolog
?- assertz(firewall:rule_firewall(secure/2, [execution([prolog_swi])])),
   compile_with_fallback(secure/2, [], Code).
% [Firewall] Target bash denied for secure/2, trying next
% [Success] Compiled secure/2 to prolog_swi
```

## Benefits

1. **Never fails** - Prolog fallback guarantees compilable output
2. **Flexible** - Configure globally or per-predicate
3. **Secure** - Firewall enforces security boundaries
4. **Performant** - Can use compiled Prolog (GNU) for fallback
5. **Debuggable** - SWI-Prolog fallback enables interactive debugging

## See Also

- [Prolog Dialect System](PROLOG_DIALECTS.md)
- [Preferences Module](../src/unifyweaver/core/preferences.pl)
- [Firewall Module](../src/unifyweaver/core/firewall.pl)
- [Prolog Target Implementation](../src/unifyweaver/targets/prolog_target.pl)
