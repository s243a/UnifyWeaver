<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 2: Firewall Policies

This chapter covers the declarative firewall policy system that controls what operations generated code can perform.

## Policy Files

Firewall policies are defined in `.firewall` files using Prolog syntax:

```prolog
% .firewall - Project security policy

% Set enforcement mode
:- firewall_mode(enforce).

% Target restrictions
:- allow(target(bash)).
:- allow(target(go)).
:- allow(target(rust)).
:- deny(target(python)).

% Compilation control
:- deny(compile(native)).

% Module restrictions
:- deny(import_module(library(http/_))).
```

## Policy Predicates

### allow/1 - Explicitly Permit

```prolog
% Allow specific targets
:- allow(target(bash)).
:- allow(target(go)).

% Allow specific dialects
:- allow(prolog_target(swi)).

% Allow specific modules
:- allow(import_module(library(lists))).
```

### deny/1 - Explicitly Deny

```prolog
% Deny specific targets
:- deny(target(python)).

% Deny compilation
:- deny(compile(native)).

% Deny network modules
:- deny(import_module(library(http/_))).
:- deny(import_module(library(socket))).
```

### require/1 - Make Mandatory

```prolog
% Require validation before compilation
:- require(validation(passed)).

% Require specific dialect
:- require(prolog_target(gnu)).

% Require code review
:- require(code_review(approved)).
```

### implies/2 - Conditional Policies

```prolog
% If using GNU Prolog, compilation must be disabled
:- implies(prolog_target(gnu), compile(false)).

% If compiling native, code review is required
:- implies(compile(native), code_review(required)).

% Production requires validation
:- implies(environment(production), validation(strict)).
```

## Firewall Modes

### Guidance Mode (Default)

Warns but allows operations:

```prolog
:- firewall_mode(guidance).
:- guidance(target(python), 'Python not recommended for production').
```

**Output**:
```
[Firewall] GUIDANCE: target(python)
  Reason: Python not recommended for production
  Action: Proceeding with caution
```

### Enforce Mode

Blocks policy violations:

```prolog
:- firewall_mode(enforce).
:- deny(target(python)).
```

**Output**:
```
[Firewall] DENIED: target(python)
  Reason: Blocked by security policy
  ERROR: Operation cannot proceed
```

### Audit Mode

Allows but logs everything:

```prolog
:- firewall_mode(audit).
:- audit_all.
```

**Output**:
```
[Firewall] AUDIT: target(python) - ALLOWED
[Firewall] AUDIT: compile(native) - ALLOWED
[Firewall] AUDIT: import_module(library(lists)) - ALLOWED
```

## Policy Composition

### AND (Both Required)

```prolog
% Allow GNU Prolog only if in production AND code reviewed
:- allow(prolog_target(gnu)) :-
    environment(production),
    code_review(passed).
```

### OR (Either Sufficient)

```prolog
% Allow compilation in production OR with explicit permission
:- allow(compile(native)) :- environment(production).
:- allow(compile(native)) :- explicit_permission(admin).
```

### NOT (Invert)

```prolog
% Deny network except in development
:- deny(network_access) :- \+ environment(development).
```

## Common Policy Patterns

### Pattern 1: Target Whitelist

Only allow approved targets:

```prolog
% .firewall
:- allow(target(bash)).
:- allow(target(go)).
:- allow(target(rust)).
:- deny(target(_)).  % Deny all others
```

### Pattern 2: Development vs Production

Different rules per environment:

```prolog
% .firewall

% Development: permissive
:- allow(target(_)) :- environment(development).
:- allow(compile(_)) :- environment(development).

% Production: strict
:- allow(target(go)) :- environment(production).
:- allow(target(rust)) :- environment(production).
:- deny(target(python)) :- environment(production).
:- require(validation(passed)) :- environment(production).
```

### Pattern 3: Module Whitelist

Only allow approved imports:

```prolog
% .firewall

% Whitelist
:- allow(import_module(library(lists))).
:- allow(import_module(library(filesex))).
:- allow(import_module(library(pcre))).

% Blacklist dangerous modules
:- deny(import_module(library(http/_))).
:- deny(import_module(library(socket))).
:- deny(import_module(library(process))).

% Deny all others
:- deny(import_module(_)).
```

### Pattern 4: Conditional Compilation

Require validation before compilation:

```prolog
% .firewall
:- implies(compile(native), validation(passed)).
:- implies(compile(native), tests(passed)).
:- deny(compile(native)) :- validation(failed).
```

### Pattern 5: Data Source Restrictions

Only local files, no network:

```prolog
% .firewall
:- allow(data_source(csv(file(_)))).
:- allow(data_source(json(file(_)))).
:- allow(data_source(sqlite(file(_)))).
:- deny(data_source(http(_))).
:- deny(network_access).
```

## Firewall Integration Points

The firewall checks at multiple points in the code generation pipeline:

### 1. Target Selection

```prolog
generate_code(Predicates, Target, Code) :-
    % Firewall check
    (firewall_allows(target(Target))
    ->  true
    ;   throw(error(firewall_denied(target(Target)),
                   'Target blocked by policy'))
    ),
    % ... continue generation
```

### 2. Dialect Selection (Prolog Target)

```prolog
generate_prolog_script(Predicates, Options, Code) :-
    option(dialect(Dialect), Options, swi),
    (firewall_allows(prolog_target(Dialect))
    ->  true
    ;   throw(error(firewall_denied(prolog_target(Dialect)),
                   'Dialect blocked by policy'))
    ),
    % ... continue generation
```

### 3. Compilation Control

```prolog
compile_if_allowed(Target, Script, Options) :-
    (firewall_allows(compile(native))
    ->  compile_to_native(Target, Script)
    ;   format('[Firewall] Native compilation blocked~n'),
        % Continue with interpreted script
        true
    ).
```

### 4. Module Imports

```prolog
add_import(Module, Imports, FinalImports) :-
    (firewall_allows(import_module(Module))
    ->  FinalImports = [Module|Imports]
    ;   format('[Firewall] Module ~w blocked~n', [Module]),
        FinalImports = Imports
    ).
```

## Using Firewall Policies

### Loading Policies

```prolog
% Load from default location
:- load_firewall_policy('.firewall').

% Load from specific file
:- load_firewall_policy('/etc/unifyweaver/production.firewall').
```

### Querying Policies

```prolog
% Check if operation is allowed
?- firewall_allows(target(python)).
false.

?- firewall_allows(target(go)).
true.

% Explain denial
?- firewall_explain_denial(target(python), Reason).
Reason = 'Explicitly denied by policy rule 4'.
```

### Compile-Time Integration

```prolog
secure_compile(Predicates, Target, Code) :-
    % Load firewall policy
    load_firewall_policy('.firewall'),

    % Attempt generation with policy enforcement
    catch(
        generate_code(Predicates, Target, Code),
        error(firewall_denied(Operation), Reason),
        (format('[Security] ~w denied: ~w~n', [Operation, Reason]),
         fail)
    ).
```

## Debugging Firewall Policies

### Trace Mode

```prolog
:- set_firewall_flag(trace, true).

% Shows decision process
% [Firewall] Checking: target(python)
% [Firewall] Rule 1: allow(target(bash)) - No match
% [Firewall] Rule 2: allow(target(go)) - No match
% [Firewall] Rule 3: deny(target(python)) - MATCH
% [Firewall] RESULT: DENIED
```

### Policy Listing

```prolog
?- list_firewall_rules.
1. allow(target(bash))
2. allow(target(go))
3. deny(target(python))
4. deny(compile(native))
5. deny(import_module(library(http/_)))
```

### Dry Run

Test policies without executing:

```prolog
?- firewall_dry_run([
    check(target(python)),
    check(target(go)),
    check(compile(native))
], Results).
Results = [
    denied(target(python)),
    allowed(target(go)),
    denied(compile(native))
].
```

## Custom Policy Predicates

Create custom policy logic:

```prolog
% custom_policy.pl

%% safe_for_production(+Config)
%  Check if configuration is safe for production
safe_for_production(Config) :-
    % Must use approved target
    option(target(Target), Config),
    member(Target, [go, rust]),

    % Must have validation
    option(validation(passed), Config),

    % Must not use network
    \+ option(network_access(true), Config).

% Use in firewall policy
:- allow(deployment(production)) :-
    safe_for_production(Config).
```

## Summary

Firewall policies provide:

- **Declarative Security**: Express policies as Prolog rules
- **Fine-Grained Control**: Target, module, capability level
- **Three Modes**: Guidance, enforce, audit
- **Policy Composition**: AND, OR, NOT, implies
- **Integration Points**: Multiple checkpoints in pipeline

The next chapter covers lifecycle hooks for runtime security.

---

## Navigation

**←** [Previous: Chapter 1: Introduction to Code Generation Securit...](01_introduction) | [📖 Book 8: Security & Firewall](./) | [Next: Chapter 3: Lifecycle Hooks →](03_lifecycle_hooks)
