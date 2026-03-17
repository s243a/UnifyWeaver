# TypR (typeR) Target Design

## Purpose

Define how UnifyWeaver should support TypR (a typed superset that transpiles to
R/S3) without breaking the current R target.

This document focuses on architecture and rollout choices specific to TypR.

---

## Current State (Repo Reality)

1. `r_target.pl` is a direct Prolog code emitter (`format/3`-style generation),
   not a pure Mustache target.
2. A Mustache file exists at `templates/targets/r/transitive_closure.mustache`,
   but the main R path today is still direct emission.
3. Type-capable targets are mixed:
   some are direct emitters, some template-driven, and some hybrid.
4. `target_registry:compile_to_target/4` dispatches to `compile_predicate/3`.
5. `typr_target.pl` now exists and emits real TypR syntax validated with the
   local `typr` CLI.
6. The current TypR pilot covers:
   - typed-mode resolution
   - shared type mapping
   - simple fact predicates
   - transitive-closure generation
7. The remaining gap is broader lowering for arbitrary generic rule bodies.

Implication: TypR design must support both generation styles and should not
assume a Mustache-only pipeline.

---

## Design Decisions

### 1) Keep R and TypR as Separate Targets

- Keep `r` unchanged for maximum compatibility.
- Add `typr` as a new target in the `r` family.
- Do not force all R generation through TypR first.
- Keep initial R and TypR templates/code paths separate.
- Revisit template sharing only after TypR output shape stabilizes.

Rationale: clean UX, low migration risk, easy A/B validation.

### 2) Use Shared Type Metadata, Optional Consumption

- Type declarations remain optional in Prolog (`uw_type/3`).
- Return type declarations remain optional in Prolog (`uw_return_type/2`).
- `typr` consumes type metadata when present.
- `r` stays syntactically untyped, but may consume return-type metadata for
  validation and fallback selection.

### 3) Three Annotation Modes for TypR

- `typed_mode(off)`: emit no annotations (closest to plain R style).
- `typed_mode(infer)`: emit minimal annotations; let TypR infer the rest.
- `typed_mode(explicit)`: emit all available annotations.

Default for TypR: `typed_mode(infer)`.

`typed_mode` should be configurable at multiple levels:

- globally for a compilation run
- per target invocation
- per predicate declaration

Precedence order:

1. per-predicate declaration
2. per-call option
3. global compiler setting
4. target default

Recommended declaration form:

```prolog
uw_typed_mode(tc/2, infer).
uw_typed_mode(weighted_edge/3, explicit).
```

This keeps type declarations (`uw_type/3`) separate from emission policy
(`uw_typed_mode/2`).

### 4) Explicit `any` vs Missing Type

- Missing type declaration: omit annotation.
- Explicit `uw_type(..., any)`: emit `Any`.

This preserves meaning: unknown vs intentionally polymorphic.

### 5) TypR Binding Discipline

Generated TypR should follow TypR's stronger binding model:

- use `let` when first introducing a name
- use plain assignment for later updates to the same name

Example:

```typr
let count <- 0;
count <- count + 1;
```

This is the safer codegen rule because TypR distinguishes `let` bindings from
plain assignment in its AST and type-checking path.

---

## IR Contract for TypR

TypR generation should read normalized type metadata from the shared type layer:

- `arg_type(Pred/Arity, Index, TypeTerm)`
- `return_type(Pred/Arity, TypeTerm)`
- `resolved_typed_mode(Pred/Arity, Mode)`
- `has_explicit_any(Pred/Arity, Index)` (or equivalent derivation)

TypR emitter maps abstract terms to TypR syntax, including:

- primitives (`integer`, `float`, `string`, `boolean`)
- containers (`list(T)`, `map(K,V)`, `set(T)`)
- nullable (`maybe(T)`)

Complex records and unions can be staged behind feature flags.

TypR should implement the standard target interface directly:

- `target_info/1`
- `compile_predicate/3`

`compile_predicate/3` should be the public entrypoint used by the registry.
Internally it can delegate to narrower helpers such as
`compile_predicate_to_typr/3`.

This keeps TypR aligned with the existing standard-interface targets and avoids
teaching the registry a TypR-specific exception.

---

## Typed Preamble

"Typed preamble" means declarations that must appear before the main emitted
program body, such as:

- type aliases
- record/struct/class declarations
- imports required only because of those types
- helper wrappers such as `newtype`-style declarations

TypR does not need a shared typed-preamble abstraction in the first pass.
Initial support should inline simple annotations and defer shared preamble
machinery until record/domain-type support lands across multiple targets.

---

## Rollout Plan (TypR-Specific)

Completed:

1. Added `typr_target.pl` with standard interface (`target_info/1`,
   `compile_predicate/3`) and compatibility wrapper.
2. Registered `typr` in `target_registry.pl` as family `r`.
3. Implemented transitive-closure pilot generation in TypR.
4. Implemented option handling:
   - `typed_mode(off|infer|explicit)`
   - default `infer` for target `typr`
   - precedence resolution through shared type declarations
5. Added focused tests for:
   - omitted annotations in infer mode
   - explicit `Any`
   - per-predicate typed-mode override
   - target-registry dispatch
   - real TypR validation through the CLI
6. Added `uw_return_type/2` consumption on wrapped generic TypR paths so
   declared return types replace `Any` where possible.

R-family note:

- `r` does not require any type declarations.
- If `uw_return_type/2` is present, `r` now uses it by default for simple
  compile-time validation and typed fallback/result-shape generation.
- This behavior can be disabled per compile with `type_constraints(false)`.

Current implementation note:

- transitive closure uses valid TypR syntax plus inline raw-R IIFEs where the
  current TypR surface language is too restrictive for the required BFS logic
  inside nested scopes

Follow-on work:

1. Extend TypR beyond simple fact predicates and the transitive-closure pilot.
2. Add broader lowering for generic non-recursive rule bodies.
3. Audit opportunities to share templates or code-generation helpers with `r`
   without making TypR the mandatory path.

---

## Notes for Follow-On Work

1. `r_target.pl` not implementing `compile_predicate/3` remains separate
   technical debt and should be fixed independently of TypR.
2. If TypR and R later converge structurally, template sharing should happen
   after the shared type/context layer is stable, not before.
3. The current TypR backend is mergeable as an initial target, but should still
   be described as an initial/pilot implementation rather than full TypR parity.
