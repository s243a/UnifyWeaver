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
   - native lowering for a conservative subset of generic binding-shaped rule
     bodies
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
   - guarded disjunction-style multi-result chains where each alternative
     may introduce different branch-local intermediates before binding the
     same later variables, and later native steps continue from those
     selected results
   - Prolog `if -> then ; else` chains where the branches bind either the
     same later intermediate, the final output directly, the same later
     result set, or guard-only control flow before later native steps
     continue from the selected values, including cases where one branch
     introduces additional branch-local intermediates before producing that
     shared later result set
   - Prolog `if -> then` chains where the then branch either contributes
     guard-only control flow for later native steps or binds a later
     intermediate, the final output directly, or the later result set needed
     by subsequent native steps
   - accumulator-style tail-recursive predicates that match the currently
     supported shared tail-recursion shape, emitted as TypR functions with
     raw-expression loop bodies
   - conservative single-recursive-call numeric linear-recursive predicates
     that match the currently supported single-base fold shape with one
     recursion-driving argument and invariant context args, emitted as TypR
     functions with raw-expression fold/loop bodies
   - conservative single-recursive-call list linear-recursive predicates
     that match the currently supported empty-list fold shape with one
     recursion-driving list argument and invariant context args, emitted as
     TypR functions with raw-expression fold/loop bodies
   - conservative arity-2 numeric multi-call tree-recursive predicates that
     match the currently supported memoized helper shape for `fib/2`-style
     recursion, emitted as TypR functions with raw-expression helper bodies
   - conservative `N`-ary structural tree-recursive predicates that match
     the currently supported `[]` / `[V, L, R]` shape with one tree-driving
     argument, invariant context args, and limited native guards, local
     `is` steps, guarded pre-recursive branching before the two subtree
     calls, or asymmetric branch-local prework that is reconciled later by
     a guarded result expression, emitted as TypR functions with
     raw-expression structural helper bodies
   - guarded post-recursive recombination inside those same single-recursive-
     call numeric and list linear-recursive shapes when the later result and
     branch-local selected intermediate values are chosen by supported
     `if -> then ; else` expressions over TypR-translatable branch results
   - asymmetric post-recursive recombination inside those same single-
     recursive-call numeric and list linear-recursive shapes when different
     branch-local intermediates still feed a shared later TypR-translatable
     result expression
   - multiple sequential branch/rejoin segments in the same native body,
     including repeated multi-result rejoin chains that feed later native
     steps after each rejoin
   - asymmetric partial-rejoin chains where an earlier rejoin preserves only
     part of the later state, more shared values are derived afterward, and
     a later guarded rejoin still stays native
   - two-level nested guarded alternatives inside supported semicolon
     branches where each nested branch still selects the same later result
     set, including nested multi-result selections
   - supported literal-headed branch bodies that keep those chains native by
     using `let` for newly introduced locals inside TypR branches
   - native lowering for the dataframe helpers `filter/3`, `sort_by/3`, and
     `group_by/3`
7. The remaining gap is broader lowering for arbitrary generic rule bodies
   and broader recursive patterns beyond that current subset.

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
6. Added `uw_return_type/2` consumption on generic TypR paths so
   declared return types replace `Any` where possible.
7. Added conservative native TypR lowering for generic rule bodies when the
   body is a supported chain of simple R bindings, including literal-guarded
   multi-clause predicates compiled into TypR `if` / `else if` chains.
8. Extended that native subset to include guard-style command predicates in
   clause conditions and direct lowering for the dataframe helpers
   `filter/3`, `sort_by/3`, and `group_by/3`.
9. Extended the native generic path further so sequential guard/output control
   flow can stay in TypR when a later guard depends on an earlier bound value.
10. Extended native multi-clause TypR lowering so supported branch bodies stay
    native and introduce new branch-local intermediates with `let` rather than
    raw `local({ ... })` wrappers.
11. Extended the native guard path again so simple comparison and boolean
    expressions over already-bound intermediates can stay native in both
    top-level control-flow chains and supported branch bodies.
12. Captured structured native fan-out chains as part of the supported TypR
    subset, where one earlier bound value feeds multiple later outputs or
    conditions without falling back to wrapped R.
13. Captured structured native split-and-recombine chains as part of the
    supported TypR subset, where guarded derived values later feed a combined
    native output without falling back to wrapped R.
14. Added structured diagnostics aggregation on the `r` side via
   `type_diagnostics_report(Report)`, which TypR can pass through on wrapped
   fallbacks and otherwise defaults to `[]` for native-lowered paths.
15. Extended the native generic path again so guarded semicolon alternatives
    can stay in TypR when each alternative binds the same later intermediate
    and later native steps continue from the selected result.
16. Extended that guarded-alternative path further so those semicolon
    alternatives may also bind the final output directly rather than only a
    fresh later intermediate.
17. Extended the same guarded-alternative path so supported semicolon
    alternatives may now preserve the same two later variables and continue
    native TypR lowering from both selected results.
18. Generalized that multi-result guarded-alternative path so supported
    semicolon alternatives may now preserve any shared later-variable set and
    continue native TypR lowering from those selected results.
19. Extended the guarded-alternative path again so a supported semicolon
    branch may itself contain one nested guarded alternative, provided that
    nested branch still selects the same later result natively.
20. Confirmed that the same one-level nested guarded-alternative path also
    covers nested multi-result selection when the nested branch preserves the
    same later result set and later native steps continue from those values.
21. Confirmed that the same guarded-alternative path also covers a second
    nested guarded layer when each nested branch still preserves the same
    later result set and later native steps continue from those values.
22. Confirmed that the same guarded-alternative path also covers supported
    branch-local intermediates inside each alternative, provided the
    alternatives still converge on the same later result variable or result
    set before later native steps continue.
23. Confirmed that the same native guarded-alternative and multi-result path
    also covers multiple sequential branch/rejoin segments in one
    non-recursive body, including repeated multi-result selection followed by
    later native steps after each rejoin.
24. Confirmed that the same native path also covers asymmetric partial
    rejoins, where an earlier guarded rejoin preserves only part of the later
    state and subsequent native steps derive additional shared values before a
    later guarded rejoin.

R-family note:

- `r` does not require any type declarations.
- If `uw_return_type/2` is present, `r` now uses it by default for simple
  compile-time validation and typed fallback/result-shape generation.
- This behavior can be disabled per compile with `type_constraints(false)`.
- Optional diagnostics now exist for that behavior:
  `type_diagnostics(off|warn|error)`, default `off`.
- Structured report collection is also available via
  `type_diagnostics_report(Report)`.

Inference note:

- generic TypR paths now use declared return types first
- if no declaration exists, they may still use shallow inferred return types
  from inferable binding-shaped bodies before falling back to `Any`

Current implementation note:

- transitive closure uses valid TypR syntax plus inline raw-R IIFEs where the
  current TypR surface language is too restrictive for the required BFS logic
  inside nested scopes
- the native generic TypR path is intentionally conservative and currently
  targets simple output-producing binding chains, guard-style command
  predicates, sequential guard/output control-flow chains, simple comparison
  and boolean guard expressions over already-bound intermediates, structured
  fan-out chains where one earlier value feeds multiple later outputs or
  conditions, structured split-and-recombine chains where guarded derived
  values later feed a combined output, guarded disjunction-style
  alternative-assignment chains where each alternative may introduce
  branch-local intermediates before binding either the same later
  intermediate or the final output directly, guarded disjunction-style
  multi-result chains where each alternative may introduce branch-local
  intermediates before binding the same later variables, Prolog
  `if -> then ; else` chains where the branches bind either the same later
  intermediate, the final output directly, the same later result set, or
  guard-only control flow, including cases where one branch introduces
  additional branch-local intermediates before producing that shared later
  result set, Prolog `if -> then` chains where the then branch either
  contributes
  guard-only control flow for later native steps or binds a later
  intermediate, the final output directly, or the later result set needed by
  subsequent native steps, multiple sequential branch/rejoin segments in the
  same body including repeated multi-result rejoin chains, asymmetric
  partial-rejoin chains where an earlier rejoin preserves only part of the
  later state before later native steps expand it, two-level nested guarded
  alternatives inside supported semicolon branches where each nested branch
  still selects the same later result set, accumulator-style tail-recursive
  predicates compiled to raw-expression loop bodies inside TypR functions,
  conservative single-recursive-call numeric linear-recursive predicates
  compiled to raw-expression fold/loop bodies inside TypR functions,
  conservative single-recursive-call list linear-recursive predicates
  compiled to raw-expression fold/loop bodies inside TypR functions,
  conservative arity-2 numeric multi-call tree-recursive predicates
  compiled to raw-expression memoized helper bodies inside TypR functions,
  conservative `N`-ary structural tree-recursive predicates compiled to
  raw-expression structural helper bodies inside TypR functions, including
  limited native guards, local `is` steps, guarded pre-recursive branching
  before the two subtree calls, and asymmetric branch-local prework that is
  reconciled later by a guarded result expression,
  guarded post-recursive recombination inside those same single-recursive-
  call numeric and list linear-recursive shapes, including multi-state
  branch-local recombination after the recursive call,
  asymmetric post-recursive recombination inside those same single-recursive-
  call numeric and list linear-recursive shapes,
  native literal-headed branch bodies built from those chains,
  dataframe helper calls, and literal-guarded branch selection; more complex
  bodies still fall back to wrapped R

Follow-on work:

1. Extend TypR beyond the current conservative native generic subset.
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
