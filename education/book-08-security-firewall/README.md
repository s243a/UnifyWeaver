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

## Chapter Overview (Planned)

### Part 1: Firewall System

**Chapter 1: Introduction to Code Generation Security**
- Why firewall code generation?
- Threat models for generated code
- Defense in depth

**Chapter 2: Firewall Policies**
- Policy declaration syntax
- Allow/deny rules
- Target-specific policies
- Default behaviors

**Chapter 3: Validation System**
- Pre-generation validation
- Post-generation checks
- Runtime validation
- Custom validators

### Part 2: Production Hardening

**Chapter 4: Secure Deployment**
- Sandboxing generated code
- Resource limits
- Network isolation
- File system restrictions

**Chapter 5: Transport Security**
- Encryption for remote targets (from Cross-Target Glue)
- TLS configuration
- Certificate management
- Secure service-to-service communication

**Chapter 6: Secrets Management**
- Integration with Vault, AWS, Azure, GCP (from Cross-Target Glue Phase 7b)
- Secret injection patterns
- Rotation strategies

### Part 3: Monitoring and Audit

**Chapter 7: Logging and Audit**
- Structured logging
- Audit trails
- Compliance reporting

**Chapter 8: Alerting**
- Security event detection
- Alert configuration
- Incident response patterns

## Content Status

This book is planned. Some content exists in:
- [Book 11: Prolog Target](../book-11-prolog-target/README.md) - Chapters 7-8 (Firewall, Fallback)
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
