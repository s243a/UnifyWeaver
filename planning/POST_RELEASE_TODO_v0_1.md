# Post-Release TODO List (v0.1)

**Created:** 2026-02-17  
**Target Release:** v0.1.x and beyond

This document tracks follow-up work identified after the v0.1.0 release.

---

## Priority 1: C# Target Parity & Runtime Enhancements

- [ ] **Bash/C# feature parity audit**  
  Map existing Bash capabilities (partitioning, constraint handling, recursion patterns, data-source integrations) against the C# query runtime and streaming targets. Produce a checklist and highlight gaps that require runtime or codegen support.

- [ ] **Dynamic facts ingestion**  
  - Detect dynamic predicates in the source program and generate ingestion hooks for the C# runtime.  
  - Design converters that transform piped data (JSON, TSV, null-delimited records, etc.) into in-memory relations (`IEnumerable<(...)>`).  
  - Mirror the Bash behaviour for streaming JSON with null-separated records, extending support to other formats.

- [ ] **Iterable representations**  
  Establish helper classes/utilities in the C# runtime for converting incoming dynamic facts into tuples, arrays, or structs that align with the generated plans.

- [ ] **Pure C# target recursion strategy**  
  Extend `csharp_codegen` to optionally mimic the Bash recursion templates (memoization, BFS loops).  
  Emphasize readability and hackability of generated code, even if the approach is less efficient than the query runtime.

---

## Priority 2: Data Pipeline & Interop

- [ ] **Unified piping conventions**  
  Document and implement consistent conventions for streaming data between components (null separators, field delimiters) so both Bash and C# targets can interoperate.

- [ ] **Type conversion utilities**  
  Provide reusable converters for common data shapes (JSON arrays, CSV rows, key/value streams) targeting both Bash scripts and C# runtime ingestion.

---

## Priority 3: Tooling & Automation

- [ ] **Parity test suite**  
  Create automated tests that compare Bash and C# outputs for the same predicates (including dynamic fact scenarios), ensuring future changes keep the targets aligned.

- [ ] **Generator ergonomics**  
  Explore a configuration switch or preference (`target(csharp_codegen, recursion_strategy(bash_style))`) that chooses the more readable recursion template.

---

Feel free to add new sections as follow-up items are discovered.
