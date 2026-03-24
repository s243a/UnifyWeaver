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
5. Verify target interface compatibility in `target_registry.pl`:
   `compile_to_target/4` currently dispatches via `compile_predicate/3`, while
   some targets expose only `compile_predicate_to_<lang>/3`.

**Deliverable:** `docs/design/TYPE_HARDCODES_AUDIT.md`

---

## Phase 1 — Core Type Infrastructure in Prolog

**Goal:** Add the `uw_type/3`, `uw_return_type/2`, and `uw_domain_type/2`
declaration predicates
and the `resolve_type/3` resolution predicate.

**New files:**
- `src/unifyweaver/targets/type_declarations.pl`
  - Defines `uw_type/3` as a dynamic predicate.
  - Defines `uw_return_type/2` as a dynamic predicate.
  - Defines `uw_typed_mode/2` as a dynamic predicate.
  - Defines `uw_domain_type/2` as a dynamic predicate.
  - Implements `resolve_type(+AbstractType, +TargetLang, -ConcreteString)`
    for all primitive and composite types across all typed targets.
  - Implements `resolve_typed_mode(+PredSpec, +Options, +GlobalMode, -Mode)`
    using precedence:
    `uw_typed_mode/2` > per-call option > global setting > target default.
  - Implements `build_type_context(+PredSpec, +TargetLang, -TypeContext)`
    which returns a dict of Mustache key-value pairs for type variables.
  - Resolves predicate return types for targets that care about them.
  - Implements `uw_typed/2` — succeeds if any `uw_type` fact exists for the
    given predicate.

**Tests:**
- `tests/type_declarations_test.pl`
  - Verify `resolve_type(integer, haskell, "Int")` succeeds.
  - Verify `resolve_type(atom, java, "String")` succeeds.
  - Verify per-predicate `uw_typed_mode/2` overrides compile options and
    global defaults.
  - Verify `build_type_context(edge/2, haskell, Ctx)` produces expected dict
    when `uw_type(edge/2, 1, atom)` is asserted.
  - Verify declared `uw_return_type/2` is reflected in type context.
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

## Phase 2.5 — TypR Target (typeR) Pilot

**Goal:** Introduce TypR safely without regressing existing `r` output.

**Status:** Implemented as an initial pilot and validated with the local TypR
CLI. Remaining work is broader generic rule-body lowering, not basic target
bring-up.

**Changes:**
1. Add `src/unifyweaver/targets/typr_target.pl` implementing standard interface:
   - `target_info/1`
   - `compile_predicate/3`
   - optional legacy wrapper `compile_predicate_to_typr/3`
2. Register `typr` in `src/unifyweaver/core/target_registry.pl`:
   - family: `r`
   - capabilities include typed support.
3. Add option handling in TypR target:
   - `typed_mode(off|infer|explicit)`
   - default `typed_mode(infer)`
   - allow global and per-call configuration
   - honor per-predicate `uw_typed_mode/2` override
4. Implement first-pass support for primitive and basic composite types:
   - primitives: `atom`, `integer`, `float`, `boolean`, `string`
   - composites: `list(T)`, `map(K,V)`, `set(T)`, `maybe(T)`
5. Implement `any` policy:
   - missing declaration => omit annotation
   - explicit `any` declaration => emit `Any`
6. Keep initial TypR templates/code paths separate from `r` even if the shared
   type-resolution layer is reused.

**Tests:**
- TypR generation with no `uw_type` declarations in infer mode emits minimal
  annotations.
- Explicit `uw_type(..., any)` emits `Any`.
- `uw_typed_mode(Pred/Arity, infer|explicit|off)` overrides global/per-call mode.
- Existing `r` target output remains unchanged.
- Generated TypR validates through the real TypR CLI in focused smoke tests.

**Implemented pilot scope:**
- real TypR syntax output
- standard target registry integration
- typed-mode precedence support
- simple fact predicate lowering
- transitive-closure generation with compile-time fact seeding
- `uw_return_type/2` support for TypR generic predicates so declared return
  types replace weak `Any` fallbacks where possible
- `r`-target return-type constraints enabled by default when metadata exists,
  with opt-out via `type_constraints(false)`
- optional `type_diagnostics(off|warn|error)` for return-type constraint
  violations, defaulting to `off`
- optional `type_diagnostics_report(Report)` collection for structured
  diagnostics without changing warning/error mode
- improved shallow return-type inference for inferable body shapes, including
  conjunctions whose final goal is `true`
- native TypR lowering for a conservative subset of generic non-recursive rule
  bodies:
  - simple output-producing binding chains
  - guard-style command predicates lowered into clause conditions
  - sequential native control-flow chains where earlier outputs feed later
    guards and outputs
  - simple comparison and boolean guard expressions over already-bound
    intermediates in those chains
  - structured fan-out chains where one earlier bound value feeds multiple
    later derived outputs or conditions
  - structured split-and-recombine chains where those guarded derived values
    later feed a combined output
  - guarded disjunction-style alternative-assignment chains where each
    alternative may introduce different branch-local intermediates before
    binding either the same later intermediate or the final output directly,
    and later native steps continue from the selected result
  - guarded disjunction-style multi-result chains where each alternative binds
    the same later variables after any branch-local intermediates, before
    later native steps continue from those selected results
  - multiple sequential branch/rejoin segments in the same native body,
    including repeated multi-result rejoin chains that feed later native
    steps after each rejoin
  - asymmetric partial-rejoin chains where an earlier rejoin preserves only
    part of the later state, more shared values are derived afterward, and a
    later guarded rejoin still stays native
  - Prolog `if -> then ; else` chains where the branches bind either the same
    later intermediate, the final output directly, the same later result set,
    or guard-only control flow before later native steps continue from the
    selected values, including cases where one branch introduces additional
    branch-local intermediates before producing that shared later result set
  - Prolog `if -> then` chains where the then branch either contributes
    guard-only control flow for later native steps or binds a later
    intermediate, the final output directly, or the later result set needed
    by subsequent native steps
  - accumulator-style tail-recursive predicates lowered to TypR-valid
    functions with raw-expression loop bodies for the currently supported
    tail-recursion shape
  - conservative single-recursive-call numeric linear-recursive predicates
    lowered to TypR-valid functions with raw-expression fold/loop bodies for
    the currently supported single-base shape with one recursion-driving
    argument and invariant context args
  - conservative single-recursive-call list linear-recursive predicates
    lowered to TypR-valid functions with raw-expression fold/loop bodies for
    the currently supported empty-list shape with one recursion-driving list
    argument and invariant context args
  - conservative arity-2 numeric multi-call tree-recursive predicates
    lowered to TypR-valid functions with raw-expression memoized helper
    bodies for the currently supported `fib/2`-style shape
  - conservative arity-1 boolean mutual-recursive predicate groups lowered
    to TypR-valid functions with raw-expression memoized helper bodies for
    the currently supported `is_even/1` / `is_odd/1`-style numeric SCC
    shape, `even_list/1` / `odd_list/1`-style list-structural SCC shape,
    and `even_left_tree/1` / `odd_left_tree/1`-style tree-structural SCC
    shape with one-subtree recursive descent, plus `even_tree/1` /
    `odd_tree/1`-style tree-structural SCC shape with two-subtree boolean
    descent and alias-style prework, guarded branch-local alias selection,
    or direct recursive subtree calls inside supported guarded branch
    bodies with limited branch-local alias or guard state around those
    calls before the shared boolean rejoin after the two recursive subtree
    calls, including one nested branch-local control point around those
    calls, one nested branch-local control point between the two direct
    recursive subtree calls with limited alias or guard state before the
    second call, shared branch-local guard prework before a nested mutual
    branch, and conservative arity-2 tree-structural SCCs with one
    invariant context argument threaded through the shared dual-subtree
    calls, shared computed context updates or guarded shared context
    selection before those calls, and the same conservative guarded
    branch-body family
  - conservative `N`-ary structural tree-recursive predicates lowered to
    TypR-valid functions with raw-expression structural helper bodies for
    the currently supported `[]` / `[V, L, R]` shape with invariant context
    args and limited native guards, local `is` steps, threaded invariant-
    context updates before the two subtree calls, per-subtree invariant-
    context updates for the left and right recursive calls, nested branch-
    local subtree-context selection before shared recursive subtree calls,
    branch-local recursive-call aliases before the two subtree calls,
    guarded pre-recursive branching before the two subtree calls,
    recursive subtree calls inside supported branch bodies, nested
    recursive subtree calls
    inside supported branch bodies, shared pre-recursive local work before
    nested recursive branch bodies, nested branch-local post-recursive
    recombination before a later shared result expression, multiple nested
    branch-local control points around the two subtree calls, or asymmetric
    branch-local prework that is reconciled later by a guarded result
    expression
  - guarded post-recursive recombination inside those same single-recursive-
    call numeric and list linear-recursive shapes when the later result and
    branch-local selected intermediate values are chosen by supported
    `if -> then ; else` choices over TypR-translatable expressions
  - asymmetric post-recursive recombination inside those same single-
    recursive-call numeric and list linear-recursive shapes when different
    branch-local intermediates still feed a shared later TypR-translatable
    result expression
  - two-level nested guarded alternatives inside supported semicolon branches
    where each nested branch still selects the same later result set,
    including nested multi-result selections
  - supported literal-headed multi-clause branch bodies that keep those chains
    native by using `let` for new intermediate locals
  - dataframe helpers such as `filter/3`, `sort_by/3`, and `group_by/3`
  - literal-guarded multi-clause branches built from those chains

**Remaining scope after Phase 2.5:**
- broader lowering for generic non-recursive rule bodies beyond the current
  native TypR subset
- richer typed preamble generation for domain/record-heavy use cases

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
   - This is the point where "typed preamble" becomes a real shared abstraction.
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
3. Add an R/TypR example showing `uw_return_type/2`,
   `type_constraints(false)`, optional `type_diagnostics`, and
   `type_diagnostics_report(Report)`.
4. Add a `MIGRATION.md` note confirming backward compatibility.
5. Add `docs/design/TYPR_TARGET_DESIGN.md` to the type system document index.

**Status note:** The design docs should now describe TypR as an implemented
initial target, not just a proposed rollout item.

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
   is intentionally out of scope for the current TypR merge.
   is explicitly out of scope for this plan.

3. **Arity > 2 predicates:** The current `transitive_closure` pattern only
   handles binary relations. How should `uw_type` declarations for ternary or
   higher-arity predicates flow into templates? This needs a concrete use-case
   before specifying.
