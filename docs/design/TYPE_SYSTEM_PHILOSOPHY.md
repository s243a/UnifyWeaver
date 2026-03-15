# Type Declaration System — Philosophy

## Overview

UnifyWeaver is a declarative code-generation framework that translates Prolog
relations into idiomatic, runnable implementations across many target languages.
This document articulates the *why* behind adding a first-class type declaration
system to that pipeline.

---

## Core Tension

Prolog is untyped. Its terms carry no static type annotation and are unified
purely by structural pattern matching. The languages UnifyWeaver targets,
however, span the full spectrum:

| Category | Languages | Type stance |
|---|---|---|
| Dynamically typed | Python, Ruby, Lua, R | Types optional / inferred at runtime |
| Gradually typed | TypeScript, Python (mypy/mypyc) | Optional static annotations |
| Nominally typed | Java, C#, Kotlin, Scala | Required, explicit |
| Structurally typed | Go | Interface-based |
| Algebraically typed | Haskell, F#, Rust | Types are load-bearing; wrong types = compile error |

Today UnifyWeaver solves this by **hardcoding** the target types inside each
`*_target.pl` or Mustache template. Every Haskell template assumes
`Map.Map String [String]`; every Java target builds `Map<String, List<String>>`
by convention. This works for the current `transitive_closure` use-case (all
nodes are strings) but immediately breaks the moment a user needs:

- Integer node IDs
- Record / struct types as edge labels
- Weighted graphs (`(node, weight)` tuples)
- Domain objects (e.g., `Employee`, `Concept`)

## Philosophy

### 1. The Prolog Layer Owns the Semantics

Prolog predicates are the *semantic source of truth* in UnifyWeaver. If a
predicate describes a relation over `employee_id/1` nodes and `salary/1`
weights, that knowledge belongs at the Prolog level — not buried in a language-
specific template. Type declarations should live alongside the predicates that
give them meaning.

### 2. Types Are Annotations, Not Constraints

The goal is *not* to build a type-checker inside Prolog. Type declarations are
pure annotations — hints that flow downstream into the template context. The
Prolog runtime never validates them; the target language compiler does. This
keeps the Prolog layer lightweight and preserves SWI-Prolog's standard
unification semantics.

### 3. Per-Target Rendering, Not Per-Type Forking

A single set of type annotations should drive all typed targets. The Haskell
target converts `integer` → `Int`, the Java target converts it to `int` or
`Integer`, Rust to `i64`. This mapping lives in the target `.pl` layer and/or
Mustache context-building logic — not in a proliferation of separate type-
specific templates.

### 4. Graceful Degradation

Targets that do not need types (Python without mypy, Lua, R) should silently
ignore type annotations. No annotation = existing behaviour = no breakage.
Type support is strictly additive.

### 5. Composability Over Completeness

Start with the primitives (atom, integer, float, string) and a single
composite form (`pair(A,B)`) for edge labels. More sophisticated forms —
record types, parametric types, dependent types — can be added incrementally.
Resist the temptation to model a full type system upfront.

---

## Design Principles Summary

1. **Declarations live in Prolog** — type hints are Prolog facts/directives.
2. **Rendering is target-specific** — each `*_target.pl` maps abstract types to concrete syntax.
3. **Mustache context carries types** — template variables like `{{node_type}}` are populated by the target layer, not hard-wired in templates.
4. **Backward compatibility is non-negotiable** — existing predicates without type annotations continue to work unchanged.
5. **Progressive enhancement** — typed targets emit richer, more idiomatic code; untyped targets emit the same code as today.
