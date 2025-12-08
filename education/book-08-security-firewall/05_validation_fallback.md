<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 5: Validation & Fallback

This chapter covers the validation systems that ensure generated code meets security requirements, and fallback mechanisms that provide graceful degradation when issues occur.

## Validation Overview

UnifyWeaver validates at three points:

```
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                       │
│                                                              │
│  Source Code                                                 │
│      ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         PRE-GENERATION VALIDATION                     │   │
│  │  • Predicate compatibility                            │   │
│  │  • Target feature support                             │   │
│  │  • Firewall policy check                              │   │
│  └──────────────────────────────────────────────────────┘   │
│      ↓                                                       │
│  Code Generation                                             │
│      ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         POST-GENERATION VALIDATION                    │   │
│  │  • Syntax verification                                │   │
│  │  • Security pattern check                             │   │
│  │  • Complexity analysis                                │   │
│  └──────────────────────────────────────────────────────┘   │
│      ↓                                                       │
│  Deployment                                                  │
│      ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         DEPLOYMENT VALIDATION                         │   │
│  │  • Security requirements (HTTPS)                      │   │
│  │  • Configuration validation                           │   │
│  │  • Health check verification                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Pre-Generation Validation

### Target Compatibility

Check if predicates are compatible with the target:

```prolog
validate_for_target(Predicates, Target, Issues) :-
    findall(Issue,
        (member(Pred, Predicates),
         incompatible_feature(Pred, Target, Issue)),
        Issues).

% Example incompatibilities
incompatible_feature(Pred, bash, issue(tabling, Pred)) :-
    uses_tabling(Pred).
incompatible_feature(Pred, awk, issue(recursion_depth, Pred)) :-
    deep_recursion(Pred).
incompatible_feature(Pred, python, issue(mutual_recursion, Pred)) :-
    mutual_recursion(Pred),
    \+ python_supports_mutual_recursion.
```

### Feature Support Check

```prolog
% Check if target supports required features
validate_features(Predicates, Target, Issues) :-
    required_features(Predicates, Features),
    findall(issue(unsupported, F),
        (member(F, Features),
         \+ target_supports(Target, F)),
        Issues).

% Feature requirements
required_features(Predicates, Features) :-
    findall(F, (member(P, Predicates), predicate_feature(P, F)), Features).

% Target capabilities
target_supports(go, parallel).
target_supports(go, streaming).
target_supports(rust, parallel).
target_supports(rust, memory_safe).
target_supports(python, tabling).
```

### Firewall Pre-Check

```prolog
% Validate against firewall before generation
validate_firewall(Target, Options, Issues) :-
    findall(issue(firewall, denied(Op)),
        (required_operation(Target, Options, Op),
         \+ firewall_allows(Op)),
        Issues).

required_operation(Target, _, target(Target)).
required_operation(_, Options, compile(native)) :-
    option(compile(true), Options).
required_operation(_, Options, network_access) :-
    option(network(true), Options).
```

## Post-Generation Validation

### Syntax Verification

```prolog
% Verify generated code syntax
validate_syntax(Code, Target, Issues) :-
    target_validator(Target, Validator),
    call(Validator, Code, SyntaxIssues),
    (SyntaxIssues == []
    ->  Issues = []
    ;   Issues = [syntax_errors(SyntaxIssues)]
    ).

% Target-specific validators
target_validator(bash, validate_bash_syntax).
target_validator(python, validate_python_syntax).
target_validator(go, validate_go_syntax).

validate_bash_syntax(Code, Issues) :-
    % Use bash -n for syntax check
    process_create(path(bash), ['-n'],
                   [stdin(pipe(In)), stderr(pipe(Err))]),
    write(In, Code), close(In),
    read_string(Err, _, ErrMsg),
    (ErrMsg == "" -> Issues = [] ; Issues = [ErrMsg]).
```

### Security Pattern Check

```prolog
% Check for dangerous patterns in generated code
validate_security_patterns(Code, Target, Issues) :-
    findall(issue(pattern, P),
        dangerous_pattern(Target, P, Code),
        Issues).

% Bash dangerous patterns
dangerous_pattern(bash, eval, Code) :-
    sub_string(Code, _, _, _, "eval ").
dangerous_pattern(bash, backtick, Code) :-
    sub_string(Code, _, _, _, "`").
dangerous_pattern(bash, unquoted_var, Code) :-
    regex_match("\\$[a-zA-Z_][^\"']", Code).

% Python dangerous patterns
dangerous_pattern(python, exec, Code) :-
    sub_string(Code, _, _, _, "exec(").
dangerous_pattern(python, eval, Code) :-
    sub_string(Code, _, _, _, "eval(").
dangerous_pattern(python, import_star, Code) :-
    sub_string(Code, _, _, _, "from * import").
```

## Deployment Validation

### Security Requirements

Remote services must use encryption:

```prolog
validate_security(Service, Errors) :-
    service_config(Service, Config),
    findall(Error, security_error(Config, Error), Errors).

security_error(Config, remote_requires_encryption) :-
    option(host(Host), Config),
    Host \== localhost,
    Host \== '127.0.0.1',
    \+ option(transport(https), Config).

security_error(Config, missing_auth) :-
    option(host(Host), Config),
    Host \== localhost,
    \+ option(auth(_), Config).

security_error(Config, weak_timeout) :-
    option(timeout(T), Config),
    T > 300.  % > 5 minutes is suspicious
```

### Configuration Validation

```prolog
validate_config(Service, Errors) :-
    service_config(Service, Config),
    findall(Error, config_error(Config, Error), Errors).

config_error(Config, missing_required(Field)) :-
    required_field(Field),
    \+ option(Field, Config).

config_error(Config, invalid_port(Port)) :-
    option(port(Port), Config),
    (Port < 1 ; Port > 65535).

config_error(Config, invalid_host(Host)) :-
    option(host(Host), Config),
    \+ valid_hostname(Host).

required_field(host(_)).
required_field(port(_)).
required_field(target(_)).
```

## Fallback Mechanisms

### Compilation Fallback

When compilation fails, fall back to interpreted:

```prolog
compile_with_fallback(Target, Script, Options, Result) :-
    catch(
        (compile_to_native(Target, Script),
         Result = compiled),
        error(compilation_failed(_, _), _),
        (format('[Fallback] Compilation failed, using interpreted~n'),
         Result = interpreted)
    ).
```

### Dialect Fallback

When one dialect fails, try alternatives:

```prolog
generate_with_dialect_fallback(Predicates, Dialects, Options, Code) :-
    member(Dialect, Dialects),
    catch(
        (generate_for_dialect(Predicates, Dialect, Options, Code),
         format('[Success] Generated for ~w~n', [Dialect])),
        Error,
        (format('[Fallback] ~w failed: ~w~n', [Dialect, Error]),
         fail)
    ),
    !.  % Stop at first success

generate_with_dialect_fallback(_, Dialects, _, _) :-
    format('[Error] All dialects failed: ~w~n', [Dialects]),
    fail.
```

### Target Fallback

When preferred target fails, try alternatives:

```prolog
generate_with_target_fallback(Predicates, Targets, Options, Code, UsedTarget) :-
    member(Target, Targets),
    validate_for_target(Predicates, Target, Issues),
    (Issues == []
    ->  (firewall_allows(target(Target))
        ->  generate_code(Predicates, Target, Options, Code),
            UsedTarget = Target
        ;   format('[Firewall] ~w blocked, trying next~n', [Target]),
            fail)
    ;   format('[Validation] ~w incompatible: ~w~n', [Target, Issues]),
        fail
    ),
    !.
```

### Service Fallback

For failed services, use fallback values or services:

```prolog
% Declare fallback for a service
:- declare_fallback(analytics_api, [
    default_value(#{status: 'unavailable', data: []}),
    fallback_service(analytics_backup),
    cache_timeout(300)
]).

% Call with fallback
call_with_fallback(Service, Request, Response) :-
    catch(
        call_service(Service, Request, Response),
        _Error,
        get_fallback_response(Service, Request, Response)
    ).

get_fallback_response(Service, Request, Response) :-
    (fallback_service(Service, FallbackService)
    ->  call_service(FallbackService, Request, Response)
    ;   default_value(Service, Response)
    ).
```

## Error Resilience

### Retry Policies

Automatic retry on transient failures:

```prolog
:- declare_retry_policy(api_service, [
    max_retries(3),
    retry_delay(exponential(1000, 2, 10000)),
    retry_on([connection_error, timeout, server_error])
]).

% Execute with retry
with_retry(Service, Goal, Result) :-
    retry_policy(Service, Policy),
    option(max_retries(Max), Policy, 3),
    retry_loop(Service, Goal, 1, Max, Policy, Result).

retry_loop(Service, Goal, Attempt, Max, Policy, Result) :-
    catch(
        (call(Goal, ServiceResult),
         Result = success(ServiceResult)),
        Error,
        (should_retry(Error, Policy)
        ->  (Attempt < Max
            ->  delay_before_retry(Attempt, Policy),
                NextAttempt is Attempt + 1,
                retry_loop(Service, Goal, NextAttempt, Max, Policy, Result)
            ;   Result = error(max_retries_exceeded(Error)))
        ;   Result = error(non_retryable(Error)))
    ).
```

### Circuit Breaker

Prevent cascading failures:

```prolog
:- declare_circuit_breaker(api_service, [
    failure_threshold(5),
    reset_timeout(60),
    half_open_requests(3)
]).

% Call through circuit breaker
with_circuit_breaker(Service, Goal, Result) :-
    circuit_state(Service, State),
    (State == open
    ->  Result = error(circuit_open)
    ;   catch(
            (call(Goal, ServiceResult),
             record_success(Service),
             Result = success(ServiceResult)),
            Error,
            (record_failure(Service),
             maybe_open_circuit(Service),
             Result = error(Error))
        )
    ).
```

### Protected Calls

Combine all protections:

```prolog
% Full protection: circuit breaker → timeout → retry → fallback
protected_call(Service, Goal, Options, Result) :-
    with_circuit_breaker(Service,
        with_timeout(Service,
            with_retry(Service, Goal)),
        ProtectedResult),
    (ProtectedResult = success(R)
    ->  Result = R
    ;   get_fallback_response(Service, Goal, Result)
    ).
```

## Validation Pipeline Example

Complete validation workflow:

```prolog
secure_generate(Predicates, Target, Options, Code) :-
    format('Starting secure generation...~n'),

    % 1. Pre-generation validation
    format('  [1/4] Pre-generation validation~n'),
    validate_for_target(Predicates, Target, TargetIssues),
    validate_firewall(Target, Options, FirewallIssues),
    append(TargetIssues, FirewallIssues, PreIssues),
    (PreIssues \== []
    ->  format('    FAILED: ~w~n', [PreIssues]), fail
    ;   format('    PASSED~n')),

    % 2. Generate code
    format('  [2/4] Code generation~n'),
    generate_code(Predicates, Target, Options, Code),
    format('    Generated ~w bytes~n', [_]),

    % 3. Post-generation validation
    format('  [3/4] Post-generation validation~n'),
    validate_syntax(Code, Target, SyntaxIssues),
    validate_security_patterns(Code, Target, SecurityIssues),
    append(SyntaxIssues, SecurityIssues, PostIssues),
    (PostIssues \== []
    ->  format('    FAILED: ~w~n', [PostIssues]), fail
    ;   format('    PASSED~n')),

    % 4. Compilation (if requested)
    format('  [4/4] Compilation~n'),
    (option(compile(true), Options)
    ->  compile_with_fallback(Target, Code, Options, CompileResult),
        format('    Result: ~w~n', [CompileResult])
    ;   format('    Skipped (interpreted mode)~n')),

    format('Secure generation complete.~n').
```

## Summary

Validation and fallback provide defense in depth:

**Validation Points**:
1. **Pre-generation**: Target compatibility, feature support, firewall
2. **Post-generation**: Syntax, security patterns, complexity
3. **Deployment**: Security requirements, configuration

**Fallback Mechanisms**:
1. **Compilation fallback**: Native → interpreted
2. **Dialect fallback**: Try alternative dialects
3. **Target fallback**: Try alternative targets
4. **Service fallback**: Default values, backup services

**Error Resilience**:
1. **Retry policies**: Automatic retry with backoff
2. **Circuit breaker**: Prevent cascading failures
3. **Protected calls**: Combined protection

The next chapter covers production hardening.

---

## Navigation

**←** [Previous: Chapter 4: Target Security](04_target_security) | [📖 Book 8: Security & Firewall](./) | [Next: Chapter 6: Production Hardening →](06_production_hardening)
