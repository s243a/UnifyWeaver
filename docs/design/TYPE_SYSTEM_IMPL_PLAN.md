# Type Declaration System — Implementation Plan

## Overview

This plan breaks the work into discrete, independently-reviewable phases.
Each phase delivers a self-contained increment; later phases build on earlier
ones but do not require them to be complete before starting.

---

## Phase 0 — Prerequisite Audit (no code changes)

**Goal:** Understand the current type-related code before touching anything.

**Tasks:**
1. Read `src/unifyweaver/targets/haskell_target.pl` and `java_target.pl` in
   full, specifically looking for all places where type strings are hardcoded
   (e.g., `"String"`, `"Map<String, List<String>>"`). Document them in a
   new file `docs/design/TYPE_HARDCODES_AUDIT.md`.
2. Read `templates/targets/haskell/transitive_closure.mustache` and its
   equivalents in `rust/`, `typescript/` — note every hardcoded type token.
3. Enumerate all Mustache keys currently injected into templates by each
   `*_target.pl` (search for `mustache_context` or equivalent dict-building
   predicates in the `.pl` files). Append findings to the audit doc.
4. Identify which targets use the `.pl` Prolog code-generation path vs.
   the Mustache template path (some targets use both).

**Deliverable:** `docs/design/TYPE_HARDCODES_AUDIT.md`

---

## Phase 1 — Core Type Infrastructure in Prolog

**Goal:** Add the `uw_type/3` and `uw_domain_type/2` declaration predicates
and the `resolve_type/3` resolution predicate.

**New files:**
- `src/unifyweaver/targets/type_declarations.pl`
  - Defines `uw_type/3` as a dynamic predicate.
  - Defines `uw_domain_type/2` as a dynamic predicate.
  - Implements `resolve_type(+AbstractType, +TargetLang, -ConcreteString)`
    for all primitive and composite types across all typed targets.
  - Implements `build_type_context(+PredSpec, +TargetLang, -TypeContext)`
    which returns a dict of Mustache key-value pairs for type variables.
  - Implements `uw_typed/2` — succeeds if any `uw_type` fact exists for the
    given predicate.

**Tests:**
- `tests/type_declarations_test.pl`
  - Verify `resolve_type(integer, haskell, "Int")` succeeds.
  - Verify `resolve_type(atom, java, "String")` succeeds.
  - Verify `build_type_context(edge/2, haskell, Ctx)` produces expected dict
    when `uw_type(edge/2, 1, atom)` is asserted.
  - Verify `build_type_context` returns empty / fallback when no `uw_type`
    facts exist.

---

## Phase 2 — Haskell Target Integration

**Goal:** The Haskell target is the simplest strongly-typed target (single
Mustache template, small `.pl` file). Make it the reference implementation.

**Changes to `src/unifyweaver/targets/haskell_target.pl`:**
1. Import `type_declarations.pl`.
2. In the Mustache context-building predicate, call
   `build_type_context(Pred, haskell, TypeCtx)` and merge `TypeCtx` into the
   existing context dict.
3. Add `typed: true` to context when type info is available.

**Changes to `templates/targets/haskell/transitive_closure.mustache`:**
1. Wrap the hardcoded `type Rel = Map.Map String [String]` in `{{^typed}}...{{/typed}}`.
2. Add a `{{#typed}}type Rel = Map.Map {{node_type}} [{{node_type}}]{{/typed}}` block.
3. Replace the hardcoded function signatures with parameterized versions
   guarded by `{{#typed}}`/`{{^typed}}` pairs.

**Tests:**
- Generate Haskell output with no `uw_type` annotations → output identical
  to current baseline.
- Generate Haskell output with `uw_type(edge/2, 1, integer)` → output uses
  `Int` in type signatures.
- Generated code compiles with GHC (integration test, optional/CI only).

---

## Phase 3 — Java Target Integration

**Goal:** Java is the most commonly requested typed target and exercises boxed
vs. unboxed types, import generation, and generic syntax.

**Changes to `src/unifyweaver/targets/java_target.pl`:**
1. Import `type_declarations.pl`.
2. Extend `build_type_context` with Java-specific logic:
   - Boxed vs. unboxed primitive selection (`int` in standalone positions,
     `Integer` in generic positions).
   - Import string generation: `node_type_import` key populated when type
     requires a non-`java.lang` import.
3. Emit a `templates/targets/java/` Mustache template for
   `transitive_closure` (currently Java uses only the `.pl` Prolog path;
   this phase adds a template as an alternative rendering path).

**Note:** The Prolog code-generation path in `java_target.pl` (which emits
code via `format/2` calls rather than Mustache) should also be updated to
respect `uw_type` facts. This is the harder path; it may be deferred to
Phase 5 if the Mustache path is sufficient for initial use.

**Tests:**
- `uw_type(edge/2, 1, integer)` → `Map<Integer, List<Integer>>` in Java output.
- No `uw_type` → unchanged Java output.
- Java output compiles with `javac` (CI integration test).

---

## Phase 4 — TypeScript & Rust Target Integration

**Goal:** TypeScript exercises optional/gradual typing (emit `: string` only
when typed); Rust exercises ownership and struct generation.

**TypeScript changes:**
- `build_type_context` for TypeScript maps abstract types to TS primitive
  names (`string`, `number`, `boolean`).
- Mustache templates gain `{{#typed}}: {{node_type}}{{/typed}}` inline
  type annotations on function parameters and return types.

**Rust changes:**
- `resolve_type` for Rust adds `String` vs `&str` disambiguation rule:
  default to `String` (owned) for node types.
- Struct generation for `record(Name, Fields)` types: emit a
  `#[derive(Debug, Clone, PartialEq, Eq, Hash)]` struct in the preamble
  section of the Mustache template.

---

## Phase 5 — Remaining Typed Targets

**Goal:** Extend to C#, F#, Kotlin, Scala, Go.

Priority order (by type-system complexity / user demand):
1. **Kotlin** — very similar to Java; largely reuses Phase 3 logic.
2. **C#** — similar to Java; `.NET` records use `record` syntax.
3. **F#** — similar to Haskell; discriminated unions for `record` types.
4. **Scala** — case classes for records; similar to Kotlin.
5. **Go** — structural typing; `type NodeType = string` aliases.

Each target follows the same pattern as Phases 2–4:
1. Update `*_target.pl` to import `type_declarations.pl` and build type context.
2. Add `{{#typed}}` guards to Mustache templates.
3. Add tests.

---

## Phase 6 — User-Defined / Domain Types

**Goal:** Support `uw_domain_type/2` and `record/2` composite types.

**Tasks:**
1. Implement `record(Name, Fields)` resolution in `type_declarations.pl`:
   - For each target, generate the appropriate struct/class/data declaration.
   - The declaration is injected into the Mustache context as a
     `type_preamble` string block.
2. Implement `uw_domain_type/2` resolution: check if target supports
   `newtype`-style wrappers (Haskell, Rust) and emit them; otherwise
   resolve to the underlying primitive.
3. Add `newtype` support for Haskell and Rust.

---

## Phase 7 — Documentation & Examples

**Tasks:**
1. Add a worked example to `docs/examples/typed_graph.pl` demonstrating
   `uw_type/3` declarations for a weighted directed graph with integer
   node IDs and float edge weights.
2. Update the main `README.md` with a "Type Annotations" section.
3. Add a `MIGRATION.md` note confirming backward compatibility.

---

## Dependency Graph

```
Phase 0 (Audit)
    │
    ▼
Phase 1 (Core Prolog infrastructure)
    │
    ├──▶ Phase 2 (Haskell)
    ├──▶ Phase 3 (Java)
    └──▶ Phase 4 (TypeScript, Rust)
              │
              ▼
         Phase 5 (C#, F#, Kotlin, Scala, Go)
              │
              ▼
         Phase 6 (Domain types / records)
              │
              ▼
         Phase 7 (Docs & examples)
```

---

## Open Questions for Review

1. **Template vs. Prolog path:** Several targets (Java, C#, Go) emit code
   primarily via `format/2` calls in `.pl` files rather than Mustache templates.
   Should Phase 3+ migrate these to Mustache-first, or extend the `format/2`
   path to also read `uw_type` facts? Recommendation: Mustache-first for new
   work; leave the `format/2` path as a fallback for compatibility.

2. **Type inference vs. declaration:** Should UnifyWeaver attempt to *infer*
   node types from Prolog fact ground terms (e.g., if all `edge/2` facts have
   integer first arguments, infer `integer`)? This would be a Phase 8 item and
   is explicitly out of scope for this plan.

3. **Arity > 2 predicates:** The current `transitive_closure` pattern only
   handles binary relations. How should `uw_type` declarations for ternary or
   higher-arity predicates flow into templates? This needs a concrete use-case
   before specifying.
