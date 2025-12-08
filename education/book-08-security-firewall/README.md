<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 8: Security & Firewall

**Cross-Target Security Policies and Production Hardening**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers security considerations for UnifyWeaver deployments, including the firewall system for controlling code generation, validation policies, and production hardening techniques.

> **Note**: This content was extracted from the Prolog target book as it applies to all targets.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - first book in the Integration & Security section
- At least one target book ([Book 2](../book-02-bash-target/README.md), [3](../book-03-csharp-target/README.md), [5](../book-05-python-target/README.md), or [6](../book-06-go-target/README.md))

## What You'll Learn

By completing this book, you will be able to:

- Configure firewall policies for code generation
- Implement security validation across targets
- Harden production deployments
- Audit and monitor generated code
- Implement fallback mechanisms

## Chapters

### [Chapter 1: Introduction](01_introduction.md)
Security architecture overview. Threats, defense in depth, firewall system introduction.

### [Chapter 2: Firewall Policies](02_firewall_policies.md)
Declarative policy system. Allow/deny rules, policy composition, modes (guidance, enforce, audit).

### [Chapter 3: Lifecycle Hooks](03_lifecycle_hooks.md)
Runtime security checkpoints. Pre/post deploy hooks, graceful shutdown, health checks.

### [Chapter 4: Target Security](04_target_security.md)
Per-target security considerations. Bash, Python, Go, Rust, C#, Prolog security matrices.

### [Chapter 5: Validation & Fallback](05_validation_fallback.md)
Pre/post generation validation. Fallback mechanisms, retry policies, circuit breakers.

### [Chapter 6: Production Hardening](06_production_hardening.md)
TLS configuration, secrets management (Vault, AWS, Azure, GCP), monitoring, alerting.

## Related Content

For more detailed coverage, see:
- [Book 11: Prolog Target](../book-11-prolog-target/README.md) - Chapters 7-8 (Prolog-specific firewall, fallback)
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - Chapter 15-16 (Production, Cloud)

## Quick Example

```prolog
% Declare a firewall policy
:- declare_firewall_policy(production, [
    allow_targets([bash, go, rust]),
    deny_targets([python]),  % Python not approved for prod
    require_validation(true),
    max_generated_lines(10000),
    audit(all)
]).

% Apply policy during compilation
?- with_firewall_policy(production,
       compile_predicate(process_data/2, go, Code)).
```

## What's Next?

After completing Book 8, continue to:
- [Book 9: Rust Target](../book-09-rust-target/README.md) - Memory-safe compilation
- [Book 10: SQL Target](../book-10-sql-target/README.md) - Database queries
- [Book 11: Prolog Target](../book-11-prolog-target/README.md) - Meta-programming

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
