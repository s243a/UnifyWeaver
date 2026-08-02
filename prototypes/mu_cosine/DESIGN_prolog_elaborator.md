# The Prolog-side elaborator — design note

**Status: design note, not a specification and not an authorization.** It maps
[`DESIGN_desugaring_to_prolog_goals.md`](DESIGN_desugaring_to_prolog_goals.md) §§3–6 (mode
split, discharge criterion, the two shipped disciplines) onto the machinery the four
`pattern_stache` consumers have already built, states what can be implemented now against
sealed oracles versus what is fenced behind vNext identity rulings, and ends with the numbered
rulings it cannot make. Registry v0.4, `pec-v3`, the sealed golden bundles, and all shipped
code are untouched. Per the lane's rhythm (philosophy → prototype → spec → production), this is
the design pass that precedes any elaborator prototype.

## Purpose, input, output

**Purpose.** The elaborator turns a surface-desugared term-plus-goal-store into either a
**ground term** (every goal discharged — checked and gone) or a **pattern state** (a term still
carrying the goals that could not yet be discharged). It is the piece that makes "compile time
versus runtime is a mode question" (desugaring §3) executable: ground goals discharge at
elaboration; under-instantiated goals **residuate** — they travel with the term as part of what
it *is*.

**Input.** A goal term under the conventions the consumers established (`pe_emit`'s goal
convention; `pe_where`'s `where/2` binding form) together with its goal store: constraint
goals (`has_type/2`, `non_amplifying/1`, …) and binding goals (`Var = Value`).

**Output.** One of two typed states, never a bare term:

- `ground(Term)` — every goal discharged; `Term` is exactly what `pe_emit` renders today, and
  its canonical bytes are the sealed golden surface;
- `pattern(Term, ResidualStore)` — `Term` still contains variables and `ResidualStore` the
  goals awaiting instantiation. **This state has no digest and no canonical byte form in this
  note** — see "The identity question".

**What it does not do.** It proves nothing (dispatch selects checkers; obligations still
discharge by property test per operator); it assigns no identity to pattern states; and it
does not touch the sealed v0.4 grammar — variables stay out of it (registry v0.4 §10.1).

## The one table: three things that look alike

A discharged side condition, a residual goal, and a binding are all "goals in the store," and
conflating them is the predictable failure. They differ in when they are consumed and in what
they do to identity:

| | discharged side condition | residual goal | binding |
|---|---|---|---|
| example | `non_amplifying(min)` — ground at elaboration | `has_type(X, substrate(C))` with `X` free | `C = simplemind` in a `where` clause |
| consumed | at elaboration, by **checking** | **not consumed** — travels with the term | at elaboration, by **substitution** |
| present in the result? | no — it verified the term, never constituted it | yes — part of what distinguishes this term | no — only the substituted literal remains |
| identity consequence | none: absent from any digest preimage | identity-bearing **as structure** (pattern state; digest deferred, see fences) | identity-bearing **as content**: the substituted literal is in the preimage |
| if conflated | embedding checks bloats identity: two derivations of one process diverge | dropping residuals merges distinct patterns into one false ground term | routing bindings through pins breaks pin transparency or binding determination — refused both directions, ratified in desugaring §12 |

The discharge criterion that separates rows 1 and 2 is desugaring §5, verbatim: *discharged at
elaboration → side condition, not embedded; still residual at ground time → part of the term,
embedded.* Row 3 is §12: a binding vanishes at elaboration and its effect is byte-identical to
the repeated-literal spelling — measured against the sealed bundle by consumer 4.

## 1. Architecture: §§3–6 mapped onto what exists

The four consumers built the elaborator's fragments bottom-up. What remains new is small and
listed explicitly.

### Reused as-is

| desugaring requirement | existing piece | status |
|---|---|---|
| mode split — "dispatch only on goals guaranteed discharged" (§3) | consumer 2's driver: `ground(Goal)` gates dispatch, nonground routes to `residual(Goal)` | the elaborator's outer loop is this, iterated to fixpoint |
| checker selection by shape, policy introspectable (§3) | `constraint_check.stache` + the production `pattern_stache` engine | unchanged; the checker table stays a template, one case per constraint form |
| binding-goal discharge (§12) | `pe_where`: validation (dead/duplicate/illegal/pin-disjoint), substitution on a copy, occurrence walking | unchanged, with one relaxation noted below |
| ground emission under a sealed byte contract | `pe_emit` + the golden bundle | unchanged; the elaborator's `ground(Term)` output feeds it directly |
| closed constraint predicate (§6.1) | consumer 2's `closed_fact/1` table is the *shape*; v0.4's sealed registry is the *authority* | needs promotion — ruling 5 |

### New

1. **The goal store as a first-class object, and a fixpoint loop.** Consumers 2 and 4 each
   process one goal kind in one pass. The elaborator iterates: each pass discharges every goal
   that is currently ground (bindings first — they instantiate; then checks), and a pass that
   discharges nothing terminates the loop. What remains is the residual store. One pass is not
   enough precisely because bindings instantiate other goals' variables: discharging
   `C = simplemind` can turn `has_type(X, substrate(C))` from residual into checkable.
2. **The two-state result type.** `ground/1` versus `pattern/2`, mirroring vNext's
   `PatternAST --ground(bindings)--> GroundAST` state machine (patterns doc §1) as Prolog
   terms. Type-per-state, constructor-checked — the same fail-closed posture as everything
   else in the lane.
3. **Goal origin for typed diagnostics (§7).** Each store goal carries which surface construct
   produced it, so a failed discharge reports "annotation `judge` conflicts with …" rather than
   `goal failed`. Origin is diagnostic metadata, **not** part of the store's canonical content
   — see ruling 3.
4. **Residual store canonicalization (§6.2).** New, measured, and specified below.

### What pe_where's "residuation is the elaborator's future" clause actually commits to

`pe_where` throws `unbound_after_elaboration(Expr)` when bindings leave variables. That error
is the **degenerate case of the elaborator**: `pe_where` is `elaborate/2` with the pattern
state refused. The commitment is exactly this: the elaborator replaces that one throw with a
`pattern(Term, Residual)` return, and **everything else in `pe_where` survives unchanged** —
the validation battery, the pin-channel refusals (ratified both directions), the
copy-don't-mutate discipline, and the walking-not-copying occurrence collection (the
`findall/3` hazard is now recorded in desugaring §12; the elaborator inherits the warning).
`pe_emit`'s own `nonground_dispatch` backstop also survives: the renderer remains ground-only
whatever the elaborator does upstream.

## 2. The identity question — stated, fenced, not solved

A term with residual goals is a **pattern state, not a ground state**. Its canonical form and
digest are vNext territory: the patterns doc versions semantic/deployed identity as `peid-v1`
and assigns identifiers *only when their checked-in content is frozen* (§15), and pattern-state
digests are exactly the content that is not frozen. This note therefore fences:

**Implementable now, against sealed oracles:**

- the full ground path: stores that discharge completely → `ground(Term)` → `pe_emit` → bytes
  equal to the golden bundle (both surfaces), including where-form inputs — consumers 3 and 4
  already prove every piece of this except the fixpoint loop itself;
- the discharge criterion as *behaviour*: which goals discharge, which residuate, in what
  passes — testable structurally (which state comes back, what the store contains) with **no
  digest involved**;
- all fail-closed behaviour: the `pe_where` battery, unknown constraint forms
  (`no_checker`-style refusal), unknown phases, nonground dispatch;
- residual-store *ordering stability* as a property (see §3 below) — stability is testable
  without ever hashing.

**Fenced — requires vNext identity rulings; no answers proposed here:**

- the canonical byte encoding of a pattern state (term + residual store) — `pe-typed-ast-v1` /
  `peid-v1` territory;
- any digest over a pattern state, and any cache or namespace keyed by one (the patterns doc
  already rules diagnostic-pattern namespaces disjoint from deployed caches, §1);
- whether residual goals enter the *encoder's* view as structure or as features — desugaring
  §11 Q3, an open science question this note must not pre-empt;
- `VarId` assignment — vNext's index unification owns variable identity; the prototype-side
  `numbervars` projection below is an ordering device, not a proposed `VarId` scheme.

The fence is the deliverable: an implementation that stays left of it needs no identity ruling
to be correct, because every observable it produces is either a sealed byte string or a typed
Prolog state checked structurally.

## 3. Order-dependence: canonicalizing the residual store

Desugaring §6.2 rules the *what*: the store canonicalizes as a set — sorted, deduplicated —
because goal order must not be able to change a future digest. This section specifies the
*how*, because the obvious key is wrong in a way that was cheap to measure
([`pattern_stache/probe_residual_order.pl`](pattern_stache/probe_residual_order.pl), 5
measurements, all green):

- **Measured:** for goals identical up to their variable (`g(V2,m2)` vs `g(V1,m1)`), raw
  `msort/2` order follows variable **creation order** — flipping which variable was allocated
  first flips the sorted store. Standard order compares unbound variables by age.
- **Measured:** the hazard is **conditional** — when functors differ, the functor comparison
  dominates and the order is accidentally stable. This is the dangerous half: a mixed store
  looks stable in every test that never hits the same-shape case, which is precisely how an
  order bug ships (project principle 6).
- **Measured:** ground stores are stable under raw `msort/2`; the hazard is variables only.
- **Measured:** deduplication by `sort/2` collapses only `==`-identical goals — two
  occurrences of the *same* goal over the *same* store variables — and correctly keeps goals
  that differ only in *which* variable they constrain. (`=@=`-dedup would be wrong: two
  variant goals over different variables are different constraints.)

**Alternatives:**

- **(A) Sort by raw standard order.** Rejected — measured creation-order dependence above.
- **(B) Sort by a numbervars projection: copy `Term-Store`, number variables by first
  occurrence in a fixed traversal (the term first, then the store), sort the store by the
  numbered copies, dedup `==`.** Stable across creation order (measured: the same two runs
  that flip under (A) agree under (B)). Variable order derives from the elaborated term's own
  structure — the only order that is a function of logical content available before vNext
  assigns `VarId`s.
- **(C) Keep insertion order and version it.** Rejected: §6.2 already ruled set semantics, and
  principle 6 rules against order significance inherited by accident.

**Recommendation: (B), stated as an ordering device only.** When `peid-v1` freezes a canonical
pattern encoding, its variable numbering supersedes (B); until then (B) gives the prototype a
stable, content-derived order with no identity claim attached. One boundary case to pin in the
eventual spec: a residual goal whose variables *all* occur only in the store (none in the
term) takes its numbering from the store-traversal phase, which makes the *pre-sort* store
walk order visible for exactly those goals; the cheap fix — number term-occurring variables
first, then number store-only variables in a second pass over the (A)-sorted remainder,
iterating to a fixpoint — is recorded here as the known refinement, not silently assumed away.

## 4. Oracle inventory

Per proposed behaviour, the verification oracle — with "no oracle" stated as the finding it
is, since it gates implementation:

| behaviour | oracle | notes |
|---|---|---|
| ground path end-to-end (store fully discharges → emitted bytes) | **sealed golden bundle** — `canonical_identity_string` / `canonical_full_string`, 25 rows | already exercised by consumers 3–4; the elaborator adds only the loop in front |
| ground path *structure* (the elaborated term itself, pre-rendering) | **sealed golden bundle `resolved_ast`** — each row carries kind/name/args/kwargs/mods/pins as sealed JSON | unused by consumers 3–4 (they compared strings); the natural structural oracle for `ground(Term)` — needs only a documented goal-term ↔ `resolved_ast` correspondence, no Python |
| where/binding discharge (§12 both directions) | sealed bundle bytes | consumer 4's suite, inherited |
| checker selection policy (which checker, which phase) | prototype fixtures only (consumer 2's tests) | **not sealed** — adequate for a prototype, insufficient for production; noted for ruling 5 |
| mode classification (§3's table: which goal discharges when) | **no sealed oracle** — the §3 table is design prose | testable as unit assertions against the table transcribed into tests; becomes trustworthy only when the constraint registry (ruling 5) seals which predicates are closed |
| residual-store ordering stability | property test (stability under construction order), per the probe | stability is testable without a digest; *correctness* of the final key has **no oracle until `peid-v1` freezes** — fenced |
| pattern-state canonical bytes / digests | **no oracle exists** | gates implementation entirely; fenced in §2 |
| typed diagnostics (§7) | **no oracle**; vNext's Python error strings exist but comparing against them would couple to machinery the standing rule forbids importing | recommend own error-fixture goldens, sealed once stable — same lifecycle as every other golden here |
| cross-checks | `process_expression_vnext/testdata/frontend_registry_fixture.json` — registry-shape cross-check **only** | per the standing rule: fixtures as data, never the Python machinery; byte-agreement with vNext `ground()` remains a pleasant fact, not a dependency |

## 5. Rulings needed

Each stated with alternatives and a recommendation, so review is decision-by-decision. None is
assumed in the sections above.

1. **Residual-store sort key** *(AST-lane)*. (A) raw standard order / (B) numbervars
   projection / (C) versioned insertion order — details and measurements in §3.
   **Recommend (B)** as an ordering device, explicitly superseded by `peid-v1`'s numbering
   when frozen.
2. **Where the elaborator lives** *(owner)*. (a) `prototypes/mu_cosine/pattern_stache/` next
   to its fragments; (b) `src/unifyweaver/core/` beside the engine. The `swipl` dependency
   question (desugaring §11 Q2) is contained either way — `mu_cosine`'s Python paths remain
   swipl-free. **Recommend (a)**: the lane's rhythm is prototype → spec → production, and the
   identity fence in §2 means a production elaborator would ship with load-bearing behaviour
   (pattern states) that has no sealed oracle yet.
3. **Goal-origin metadata placement** *(AST-lane)*. Diagnostics need each goal's surface
   origin (§7). (a) wrap goals in the store — `origin(Goal, Construct)` — making origin part
   of store content; (b) carry origin alongside, keyed off the store, leaving store content
   pure. **Recommend (b)**: origin is compile-time diagnostic metadata; putting it inside the
   store would push it toward any future canonical form — the same category error as routing
   bindings through pins, in miniature.
4. **Which surface constructs desugar in the first elaborator** *(AST-lane, informs owner
   review)*. (a) minimal: binding goals + `has_type/2` + the closed constraint checks —
   everything consumers 2/4 already handle; (b) also `interpretation/3` /
   `representation/4`. **Recommend (a)**: (b) is the half of the patterns doc that was
   specified and never built, carries its own receipts machinery, and nothing in §§3–6
   requires it for the discharge/residuate loop to be real.
5. **The closed constraint predicate's shipping form** *(owner, §6.1)*. (a) hand-written
   Prolog facts (consumer 2's current shape); (b) facts generated from the sealed v0.4
   registry with a content-hash check at load, failing closed on mismatch; (c) runtime
   database. (c) is ruled out by §6.1 itself ("anything that can `assert/1` … evaporates
   silently"). **Recommend (b)** — it also subsumes the consumer-3 liberty flagged earlier
   (`pe_where` reading `pe_emit`'s mirror): one generated, hash-checked mirror replaces both
   hand copies.
6. **Residual goals in the encoder: structure or features** *(owner + science, desugaring §11
   Q3)*. Alternatives: structure (tokens in the stream), features (side channel), or both
   behind an ablation. **Recommend: defer, explicitly gated** on encoder experiments that
   cannot begin until pattern states exist at all — this ruling should *follow* the
   elaborator prototype, not precede it, and nothing in this note depends on it.
7. **Pattern-state namespace discipline** *(AST-lane)*. Even without digests, a prototype
   will want to *name* pattern states in test fixtures. (a) forbid any persistent naming until
   `peid-v1`; (b) allow file-local fixture names with a mandatory `no-identity` marker.
   **Recommend (b)** — it keeps fixtures writable while making the fence visible in every
   file that approaches it.

## Acceptance test (project principle 9)

A Haiku-tier reader should be able to answer, from this note alone: *what elaborates* (ground
goals — checks discharge, bindings substitute and vanish), *what residuates*
(under-instantiated goals, which then belong to the term as a pattern state), *what has an
oracle* (every ground-path behaviour, against the sealed bundle's strings and `resolved_ast`;
ordering stability by property test), and *what awaits a ruling* (the seven above — most
consequentially: pattern-state identity, which has no oracle and gates everything right of the
fence in §2).
