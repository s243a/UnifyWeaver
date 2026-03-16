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
4. `target_registry:compile_to_target/4` dispatches to `compile_predicate/3`,
   but `r_target.pl` currently exports `compile_predicate_to_r/3` only.

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
- `typr` consumes type metadata when present.
- `r` ignores it unless a future explicit typed-R mode is added.

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

---

## IR Contract for TypR

TypR generation should read normalized type metadata from the shared type layer:

- `arg_type(Pred/Arity, Index, TypeTerm)`
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

1. Add `typr_target.pl` with standard interface (`target_info/1`,
   `compile_predicate/3`) and legacy compatibility wrapper if needed.
2. Register `typr` in `target_registry.pl` as family `r`.
3. Implement transitive-closure pilot generation in TypR for parity with
   existing R examples.
4. Add option handling:
   - `typed_mode(off|infer|explicit)`
   - default `infer` for target `typr`
   - resolve mode using the defined precedence order
5. Add golden tests:
   - no types declared -> infer-mode output has omitted annotations
   - explicit `any` -> output contains `Any`
   - explicit scalar/composite declarations -> emitted annotations match
6. Optional validation phase: run generated TypR through TypR transpiler and
   execute resulting R in smoke tests.
7. After the TypR pilot is stable, audit opportunities to share templates or
   code-generation helpers with `r` without making TypR the mandatory path.

---

## Notes for Follow-On Work

1. `r_target.pl` not implementing `compile_predicate/3` remains separate
   technical debt and should be fixed independently of TypR.
2. If TypR and R later converge structurally, template sharing should happen
   after the shared type/context layer is stable, not before.
