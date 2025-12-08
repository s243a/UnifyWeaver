<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 1: Introduction to Code Generation Security

Code generation systems create executable code from high-level specifications. This power comes with security responsibilities. This chapter introduces UnifyWeaver's security architecture and the threats it addresses.

## Why Secure Code Generation?

Code generation systems can be security risks:

| Threat | Description | Example |
|--------|-------------|---------|
| **Arbitrary Execution** | Generated code performs unintended operations | Malicious predicate generates shell commands |
| **Resource Access** | Scripts access files, network, system | Generated script reads sensitive files |
| **Dialect Capabilities** | Some targets have dangerous features | GNU Prolog can compile to native binaries |
| **Compilation Security** | Compiled binaries are harder to audit | Native code may contain hidden behavior |
| **Cross-Target Leakage** | Secrets flow between pipeline stages | API key appears in AWK script |

## Security Architecture Overview

UnifyWeaver provides defense in depth across three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPILE-TIME SECURITY                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Firewall Policy Evaluation               │   │
│  │  • Target restrictions (allow/deny)                   │   │
│  │  • Module whitelisting                                │   │
│  │  • Capability control                                 │   │
│  │  • Validation gates                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│              Code Generation Pipeline                        │
│         (with firewall checks at each point)                 │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                   DEPLOYMENT SECURITY                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Service Configuration                    │   │
│  │  • Encryption enforcement (HTTPS for remote)          │   │
│  │  • Location validation                                │   │
│  │  • Authentication setup                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Lifecycle Hooks                          │   │
│  │  • Pre/post deployment hooks                          │   │
│  │  • Health checks                                      │   │
│  │  • Graceful shutdown                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    RUNTIME SECURITY                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Error Resilience                         │   │
│  │  • Circuit breaker                                    │   │
│  │  • Retry policies                                     │   │
│  │  • Fallback mechanisms                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Monitoring & Audit                       │   │
│  │  • Health checks                                      │   │
│  │  • Metrics collection                                 │   │
│  │  • Structured logging                                 │   │
│  │  • Alert triggering                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## The Firewall System

The firewall is UnifyWeaver's primary compile-time security mechanism:

### What It Controls

| Control Point | Examples |
|---------------|----------|
| **Target Selection** | Allow bash, deny python for production |
| **Dialect Selection** | Allow SWI-Prolog, deny GNU Prolog |
| **Module Imports** | Block network libraries, database access |
| **Compilation** | Prevent native binary compilation |
| **Capabilities** | Disable file write, network access |

### Policy Declaration

Firewall policies are declarative Prolog rules in `.firewall` files:

```prolog
% .firewall
:- firewall_mode(enforce).

% Target restrictions
:- allow(target(bash)).
:- allow(target(go)).
:- deny(target(python)).  % Not approved for production

% Compilation control
:- deny(compile(native)).

% Module restrictions
:- deny(import_module(library(http/_))).
:- deny(import_module(library(odbc))).
```

### Three Operating Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Guidance** | Warn but allow | Development |
| **Enforce** | Block violations | Production |
| **Audit** | Allow but log everything | Compliance |

## Lifecycle Hooks

Lifecycle hooks provide runtime security checkpoints:

```prolog
% Declare hooks for a service
:- declare_lifecycle_hook(api_service, pre_shutdown, drain_connections).
:- declare_lifecycle_hook(api_service, post_deploy, health_check).
:- declare_lifecycle_hook(api_service, on_health_failure, custom('alert.sh')).
```

### Available Hook Events

| Event | When | Use For |
|-------|------|---------|
| `pre_deploy` | Before deployment | Validation, backup |
| `post_deploy` | After deployment | Health check, cache warming |
| `pre_shutdown` | Before stopping | Connection draining |
| `post_shutdown` | After stopping | Cleanup, notification |
| `on_health_failure` | Health check fails | Alerting, recovery |

### Built-in Hook Actions

| Action | Description |
|--------|-------------|
| `drain_connections` | Wait for active connections to complete |
| `health_check` | Verify service is responding |
| `warm_cache` | Pre-populate caches |
| `save_state` | Persist service state |
| `custom(Command)` | Run arbitrary command |

## Target-Specific Security

Different compilation targets have different security characteristics:

| Target | Security Considerations |
|--------|------------------------|
| **Bash** | Command injection, shell escaping |
| **Python** | Import restrictions, sandbox limitations |
| **Go** | Safe by default, compile-time checks |
| **Rust** | Memory safety, no runtime exceptions |
| **C#** | .NET sandbox, CAS (Code Access Security) |
| **Prolog** | Dialect capabilities, compilation control |

### Target Security Matrix

```
                  Compile-Time    Runtime       Memory
Target            Safety          Sandbox       Safety
─────────────────────────────────────────────────────
Bash              Low             None          N/A
Python            Medium          Limited       Managed
Go                High            None*         Safe
Rust              Very High       None*         Safe
C#/.NET           High            Strong        Managed
Prolog (SWI)      Medium          Limited       Managed
Prolog (GNU)      Medium          None          Safe

* Compiled binaries can self-sandbox via OS mechanisms
```

## Validation System

UnifyWeaver validates at multiple points:

### Pre-Generation Validation

Before generating code:

```prolog
% Validate predicates for target compatibility
validate_for_target(Predicates, Target, Issues) :-
    findall(Issue,
        (member(P, Predicates),
         incompatible_feature(P, Target, Issue)),
        Issues).
```

### Post-Generation Validation

After code is generated:

```prolog
% Validate generated code
validate_generated(Code, Target, Issues) :-
    check_syntax(Code, Target, SyntaxIssues),
    check_security(Code, Target, SecurityIssues),
    append(SyntaxIssues, SecurityIssues, Issues).
```

### Deployment Validation

Before deployment:

```prolog
% Security validation for remote services
validate_security(Service, Errors) :-
    service_config(Service, Config),
    option(host(Host), Config),
    (Host \== localhost, Host \== '127.0.0.1')
    ->  % Remote service - check encryption
        (option(transport(https), Config)
        ->  Errors = []
        ;   Errors = [remote_requires_encryption])
    ;   Errors = [].
```

## Quick Example

Complete security-aware deployment:

```prolog
% 1. Define firewall policy
:- load_firewall_policy('.firewall').

% 2. Declare service with security
:- declare_service(secure_api, [
    host('api.example.com'),
    port(8443),
    transport(https),         % Required for remote
    target(go)
]).

% 3. Add lifecycle hooks
:- declare_lifecycle_hook(secure_api, pre_deploy, health_check).
:- declare_lifecycle_hook(secure_api, post_deploy, warm_cache).
:- declare_lifecycle_hook(secure_api, pre_shutdown, drain_connections).

% 4. Add error resilience
:- declare_retry_policy(secure_api, [max_retries(3)]).
:- declare_circuit_breaker(secure_api, [failure_threshold(5)]).

% 5. Deploy with full security
deploy_secure :-
    validate_security(secure_api, Errors),
    (Errors == []
    ->  deploy_with_hooks(secure_api, Result),
        format('Deployed: ~w~n', [Result])
    ;   format('Security errors: ~w~n', [Errors]),
        fail
    ).
```

## Book Overview

This book covers:

| Chapter | Topic |
|---------|-------|
| 1 | Introduction (this chapter) |
| 2 | Firewall policies - declaration and enforcement |
| 3 | Lifecycle hooks - runtime security checkpoints |
| 4 | Target security - per-target considerations |
| 5 | Validation systems - pre/post generation checks |
| 6 | Production hardening - deployment security |

## Summary

UnifyWeaver's security architecture provides:

- **Firewall System**: Compile-time policy enforcement
- **Lifecycle Hooks**: Runtime security checkpoints
- **Validation**: Pre/post generation checks
- **Target Awareness**: Per-target security considerations
- **Production Hardening**: Deployment security features

The next chapter covers firewall policies in detail.

---

## Navigation

[📖 Book 8: Security & Firewall](./) | [Next: Chapter 2: Firewall Policies →](02_firewall_policies)
