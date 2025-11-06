<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: Prolog as Target Language

**Status:** 🚧 Partially Implemented (Limited functionality)
**Module:** `src/unifyweaver/targets/prolog_query_target.pl` (if exists)
**Tests:** TBD

---

## Introduction

UnifyWeaver can compile Prolog predicates back to Prolog, generating optimized query code. This serves as:
- A **fallback strategy** when pattern matching fails
- A **debugging tool** to see generated query plans
- A **cross-dialect** compilation path (e.g., SWI-Prolog → Scryer Prolog)

---

## Current Limitations

### What Doesn't Work Yet

1. **No Template Support**
   - Cannot generate full bash scripts with Prolog calls
   - Only query generation, not executable wrappers

2. **Missing unique/unordered**
   - No aggregation predicates
   - No set operations

3. **Limited Optimization**
   - No join reordering
   - No index hints

4. **No I/O Integration**
   - Cannot read from CSV/JSON sources
   - No streaming support

---

## Potential Use Cases

### 1. Compilation Fallback

When bash/C# compilation fails:
```prolog
% If pattern matching fails
compile_predicate(Pred, bash, _) :- fail.

% Fall back to Prolog
compile_predicate(Pred, prolog, Code) :-
    generate_prolog_query(Pred, Code).
```

### 2. Firewall-Driven Selection

When external tools are forbidden:
```prolog
% Firewall denies all external services
:- firewall(deny_all_services).

% Use pure Prolog compilation
preferred_target(prolog).
```

### 3. Cross-Dialect Compilation

Translate SWI-Prolog to other dialects:
```prolog
% Compile for Scryer Prolog
?- compile_predicate(ancestor/2, scryer_prolog, Code).
```

---

## Future Directions

### Needed Infrastructure

1. **Template System Integration**
   - Generate bash wrappers around swipl calls
   - Handle stdin/stdout piping

2. **Aggregation Support**
   - Implement unique via `setof/3`
   - Implement ordered via `sort/2`

3. **Data Source Integration**
   - Read CSV via `csv` library
   - Parse JSON via `json` library

4. **Optimization Passes**
   - Query reordering
   - Index analysis
   - Cut insertion

---

## Related Work

- **Book 2:** C# query runtime - Similar optimization challenges
- **Chapter 4 (this book):** Prolog Bash Service - Alternative approach

---

**Status:** Awaiting design decisions and use case refinement

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**Last Updated:** 2025-11-05

---

## Navigation

**←** [Previous: Chapter 1: Perl Service Infrastructure](01_perl_service_infrastructure) | [📖 Back to Book-Misc](README) | [Next: Chapter 4: Prolog Service →](04_prolog_bash_service)

