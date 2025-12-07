# Chapter 7: The Firewall System

## Introduction to UnifyWeaver's Firewall

UnifyWeaver includes a comprehensive security system called the "firewall" that controls what operations generated code can perform. This chapter explores how the firewall integrates with the Prolog target to enforce security policies.

## Why a Firewall for Code Generation?

Code generation systems can be security risks:

1. **Arbitrary Code Execution**: Generated code could perform unintended operations
2. **Resource Access**: Scripts might access files, network, or system resources
3. **Dialect Capabilities**: Some dialects have more powerful features than others
4. **Compilation Security**: Compiled binaries are harder to audit than interpreted scripts

The firewall provides:
- **Declarative Security**: Express policies as Prolog rules
- **Fine-Grained Control**: Control specific features, not just all-or-nothing
- **Audit Trail**: Log all policy decisions
- **Fail-Safe**: Deny by default, allow explicitly

## Firewall Architecture

### Policy Files

Firewall policies are defined in `.firewall` files:

```prolog
% .firewall
% Security policy for this project

% Deny network access
:- disallow(network_access).

% Deny native compilation
:- disallow(compile(native)).

% Allow only SWI-Prolog
:- allow(prolog_target(swi)).
:- disallow(prolog_target(gnu)).

% Require all data sources to be local files
:- require(data_source(file(_))).
```

### Policy Predicates

**allow/1**: Explicitly permit an operation
```prolog
:- allow(prolog_target(swi)).
```

**disallow/1**: Explicitly deny an operation
```prolog
:- disallow(prolog_target(gnu)).
```

**require/1**: Make an operation mandatory
```prolog
:- require(dialect(swi)).
```

**implies/2**: Create conditional policies
```prolog
:- implies(prolog_target(gnu), compile(false)).
% "If using GNU Prolog, compilation must be disabled"
```

## Prolog Target Firewall Points

The Prolog target integrates with the firewall at several points:

### 1. Dialect Selection

**Check Point**: Before generating code for a dialect

```prolog
generate_prolog_script(Predicates, Options, Code) :-
    option(dialect(Dialect), Options, swi),

    % Firewall check
    (   firewall_allows(prolog_target(Dialect))
    ->  true
    ;   throw(error(firewall_denied(prolog_target(Dialect)),
                   'Dialect blocked by security policy'))
    ),

    % ... continue generation
```

**Policy Examples**:

```prolog
% Only allow SWI-Prolog
:- allow(prolog_target(swi)).
:- disallow(prolog_target(gnu)).

% Allow both but log usage
:- allow(prolog_target(swi)).
:- allow_with_audit(prolog_target(gnu)).
```

### 2. Compilation Control

**Check Point**: Before invoking compiler

```prolog
compile_script_safe(Dialect, ScriptPath, Options) :-
    % Check if compilation is allowed
    (   firewall_allows(compile(native))
    ->  compile_script(Dialect, ScriptPath)
    ;   format('[Firewall] DENIED: Native compilation blocked~n'),
        throw(error(firewall_denied(compile(native)),
                   'Compilation blocked by security policy'))
    ).
```

**Policy Examples**:

```prolog
% Disable all compilation
:- disallow(compile(native)).

% Allow compilation only for specific dialects
:- implies(compile(native), prolog_target(gnu)).

% Require compilation for production
:- implies(environment(production), compile(native)).
```

### 3. Module Imports

**Check Point**: Before adding dependencies

```prolog
add_dependency(Module, AllowedDeps, FinalDeps) :-
    (   firewall_allows(import_module(Module))
    ->  FinalDeps = [Module|AllowedDeps]
    ;   format('[Firewall] DENIED: Module ~w blocked~n', [Module]),
        FinalDeps = AllowedDeps
    ).
```

**Policy Examples**:

```prolog
% Block network modules
:- disallow(import_module(library(http/_))).

% Block database access
:- disallow(import_module(library(odbc))).

% Allow only approved modules
:- allow(import_module(library(lists))).
:- allow(import_module(library(filesex))).
:- disallow(import_module(_)).  % Deny all others
```

### 4. Script Capabilities

**Check Point**: Before generating main/0

```prolog
generate_main_with_restrictions(Options, MainCode) :-
    % Check what capabilities are allowed
    (   firewall_allows(file_access(write))
    ->  IncludeFileWrite = true
    ;   IncludeFileWrite = false
    ),

    (   firewall_allows(network_access)
    ->  IncludeNetwork = true
    ;   IncludeNetwork = false
    ),

    % Generate restricted main/0
    generate_main_predicate([
        file_write(IncludeFileWrite),
        network(IncludeNetwork)
        | Options
    ], MainCode).
```

## Firewall Modes

### 1. Guidance Mode (Default)

**Behavior**: Warn but allow

```prolog
:- firewall_mode(guidance).

:- guidance(prolog_target(gnu),
           'GNU Prolog has limited library support').
```

**Output**:
```
[Firewall] GUIDANCE: prolog_target(gnu)
  Reason: GNU Prolog has limited library support
  Action: Proceeding with caution
```

### 2. Enforce Mode

**Behavior**: Block violations

```prolog
:- firewall_mode(enforce).

:- disallow(prolog_target(gnu)).
```

**Output**:
```
[Firewall] DENIED: prolog_target(gnu)
  Reason: Blocked by security policy
  ERROR: Operation cannot proceed
```

### 3. Audit Mode

**Behavior**: Allow but log everything

```prolog
:- firewall_mode(audit).

:- audit_all.
```

**Output**:
```
[Firewall] AUDIT: prolog_target(gnu) - ALLOWED
[Firewall] AUDIT: compile(native) - ALLOWED
[Firewall] AUDIT: import_module(library(lists)) - ALLOWED
```

## Policy Composition

Policies can be combined using logical operators:

### AND (Both conditions required)

```prolog
:- allow(prolog_target(gnu)) :-
    environment(production),
    code_reviewed(true).
% Only allow GNU Prolog in production if code was reviewed
```

### OR (Either condition sufficient)

```prolog
:- allow(compile(native)) :-
    environment(production).

:- allow(compile(native)) :-
    explicit_permission(user).
% Allow compilation in production OR with explicit permission
```

### NOT (Invert condition)

```prolog
:- disallow(network_access) :-
    \+ environment(development).
% Disallow network except in development
```

### IMPLIES (Conditional requirements)

```prolog
:- implies(prolog_target(gnu), validation(strict)).
% If using GNU Prolog, strict validation is required

:- implies(compile(native), code_review(required)).
% If compiling, code review is mandatory
```

## Common Policy Patterns

### Pattern 1: Dialect Restriction

**Requirement**: Only allow SWI-Prolog

```prolog
% .firewall
:- allow(prolog_target(swi)).
:- disallow(prolog_target(gnu)).
:- disallow(prolog_target(_)).  % Deny any other dialects
```

### Pattern 2: Development vs Production

**Requirement**: Different rules for different environments

```prolog
% .firewall
% Development: permissive
:- allow(prolog_target(_)) :- environment(development).
:- allow(compile(_)) :- environment(development).

% Production: strict
:- allow(prolog_target(gnu)) :- environment(production).
:- require(compile(native)) :- environment(production).
:- require(code_review(passed)) :- environment(production).
```

### Pattern 3: Data Source Restrictions

**Requirement**: Only local files, no network

```prolog
% .firewall
:- allow(data_source(csv(file(_)))).
:- allow(data_source(json(file(_)))).
:- allow(data_source(sqlite(file(_)))).
:- disallow(data_source(http(_))).
:- disallow(network_access).
```

### Pattern 4: Compilation with Validation

**Requirement**: Require validation before compilation

```prolog
% .firewall
:- implies(compile(native), validation(passed)).
:- implies(compile(native), tests(passed)).
:- disallow(compile(native)) :- validation(failed).
```

### Pattern 5: Module Whitelist

**Requirement**: Only allow specific modules

```prolog
% .firewall
% Whitelist
:- allow(import_module(library(lists))).
:- allow(import_module(library(filesex))).
:- allow(import_module(unifyweaver(core/partitioner))).

% Deny all others
:- disallow(import_module(_)).
```

## Firewall Integration Example

Complete example showing firewall integration:

```prolog
% my_project/.firewall
:- firewall_mode(enforce).

% Only allow SWI-Prolog by default
:- allow(prolog_target(swi)).

% Allow GNU Prolog only if explicitly requested and validated
:- allow(prolog_target(gnu)) :-
    option(explicit_gnu_request(true)),
    validate_for_dialect(gnu, Predicates, []).

% Disable network access
:- disallow(network_access).
:- disallow(import_module(library(http/_))).

% Require validation
:- require(validation(strict)).

% Generation code
generate_secure(Predicates, Code) :-
    % Load firewall
    load_firewall_policy('.firewall'),

    % Attempt generation (firewall checks automatically)
    catch(
        generate_prolog_script(Predicates,
                              [dialect(gnu), compile(true)],
                              Code),
        error(firewall_denied(Operation), Reason),
        (   format('[Security] Operation ~w denied: ~w~n', [Operation, Reason]),
            fail
        )
    ).
```

## Implementing Custom Policies

You can create custom policy predicates:

```prolog
% custom_policy.pl

%% safe_for_production(+Dialect, +Options)
%  Check if configuration is safe for production
safe_for_production(Dialect, Options) :-
    % Must be GNU Prolog
    Dialect = gnu,

    % Must be compiled
    option(compile(true), Options),

    % Must have been validated
    option(validation(passed), Options),

    % Must have test coverage > 80%
    option(test_coverage(Coverage), Options),
    Coverage >= 80,

    % Must not use network
    \+ option(network_access(true), Options).

% Use in firewall
:- allow(deployment(production)) :-
    safe_for_production(Dialect, Options).
```

## Debugging Firewall Policies

### Trace Mode

```prolog
:- set_firewall_flag(trace, true).

% Shows decision process
[Firewall] Checking: prolog_target(gnu)
[Firewall] Rule 1: allow(prolog_target(swi)) - No match
[Firewall] Rule 2: disallow(prolog_target(gnu)) - MATCH
[Firewall] RESULT: DENIED
```

### Policy Query

```prolog
% Check what would be allowed
?- firewall_allows(prolog_target(gnu)).
false.

?- firewall_allows(prolog_target(swi)).
true.

% Check why something is denied
?- firewall_explain_denial(prolog_target(gnu), Reason).
Reason = 'Explicitly disallowed by policy rule 2'.
```

## What's Next?

Chapter 8 explores fallback mechanisms, showing how the system gracefully handles firewall denials and other failures by trying alternative approaches.

---

**Key Takeaways:**
- Firewall provides declarative security for code generation
- Policies control dialects, compilation, modules, and capabilities
- Three modes: guidance, enforce, audit
- Integrates at multiple points in generation pipeline
- Supports complex policy composition
- Essential for production deployments
- Can be customized for specific security requirements

---

## Navigation

**←** [Previous: Chapter 3: Understanding Prolog Dialects](03_dialects) | [📖 Book 11: Prolog Target](README) | [Next: Chapter 8: Dialect Fallback Mechanisms →](08_fallback_mechanisms)
