# Shared Purity Certificate Module: Proposal

> **Motivation and direction** for extracting UnifyWeaver's existing
> purity / order-independence analyses into a single `core/` module
> with a shared certificate shape. This doc establishes *what* and
> *why*; see `PURITY_CERTIFICATE_SPECIFICATION.md` for the concrete
> shape, and `PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` for the
> phased rollout.

## The problem

Purity-like analyses are scattered across the project. Each consumer
invents its own return shape, its own conservative rules, and its own
opinion on what "safe" means. Nothing crosses target boundaries.

Concrete symptoms:

- **Two purity modules already exist**, overlapping but not unified:
  - `src/unifyweaver/core/clause_body_analysis.pl` — blacklist
    (`is_impure_builtin/1`), `is_pure_goal/1`, `is_order_independent/2`,
    `classify_parallelism/3`, `partition_parallel_goals/5`.
  - `src/unifyweaver/core/advanced/purity_analysis.pl` — a narrower
    whitelist (`pure_builtin/1`: 20 clauses for arithmetic, comparison,
    type checks, list ops), used by the tail-recursion transformation.
- **The Haskell WAM intra-query parallelism spec** (§3.1) already
  *assumes* a centralized purity module exists — Phase 4.1 will need
  it to decide whether to emit `ParTryMeElse` instead of `TryMeElse`.
- **Kernel detection implicitly assumes purity** — FFI-owned kernels
  (`src/unifyweaver/core/recursive_kernel_detection.pl`) are trusted
  without any certificate because they're hand-coded native. There's
  no explicit statement of that trust, and no way to validate it.
- **No serialization.** A verdict can't cross the Prolog↔C# boundary,
  so if the C# pipeline ever *did* produce purity verdicts, we'd have
  nowhere to put them.

## What the Prolog side already knows

**`clause_body_analysis.pl` is doing most of the real work.**

| Predicate | What it decides |
|---|---|
|`is_pure_goal/1`|Single goal side-effect-free — blacklist-based, allows user-declared `parallel_safe/1`.|
|`is_impure_builtin/1`|Hardcoded list: I/O (`write`, `read`, `format`, `get_char`, …), database (`assert*`, `retract*`), globals (`nb_setval`, `b_setval`), and domain-specific impurities.|
|`is_order_independent/2`|Predicate safe to reorder clauses — returns `declared` (user said so) or `proven([pure_goals])` (static check that every goal in every clause is pure).|
|`goals_are_independent/1`|Conjunction safe to parallelize — pure **and** disjoint bindings.|
|`classify_parallelism/3`|High-level verdict: `goal_parallel(...)` / `clause_parallel` / `sequential`.|
|`partition_parallel_goals/5`|Splits a goal list into an independent prefix + dependent tail.|

Sophistication level: **simple syntactic checks**. Functor membership
plus variable-set intersection. No abstract interpretation, no fixpoint
analysis. That's the right call for a conservative analyzer — cheap,
predictable, easy to extend.

**`advanced/purity_analysis.pl`** is the whitelist counterpart: 92 lines
defining `pure_builtin/1` for the tail-recursion transformation. Narrower
(strict whitelist vs permissive blacklist) because of its different
use case — refusing to transform anything not provably pure vs refusing
to parallelize anything provably impure.

**User annotations already feed in:**
- `:- parallel(P/N).`
- `:- order_independent(P/N).`
- `:- parallel_safe(F/A).` (per-functor)

The two forms are correctly recognized by `is_order_independent/2`
but the plumbing isn't exposed as a shared API.

## What the C# target does — and why we're not borrowing from it

The agent investigation surfaced one C# helper that might *look* like
purity analysis but isn't. In `src/unifyweaver/targets/csharp_target.pl`
lines 1981–2006:

```prolog
safe_recursive_clause_for_need(HeadSpec, _-Body) :-
    ...
    body_to_list(Body, Terms),
    findall(Index, (nth0(Index, Terms, Term0), ...), Occs),
    Occs \= [],
    forall(member(Index, Occs), (
        prefix_terms(Index, Terms, Pred, Arity, Prefix),
        \+ (member(T, Prefix), aggregate_goal(T))
    )).
```

This predicate decides whether a recursive clause is **safe for demand
("need") closure materialization** — a query-planning optimization
that pushes demand propagation into a fixpoint computation. It checks:

- The recursive call occurs at least once.
- No `aggregate_*` goal appears in the prefix *before* each recursive call.

That's **a structural property for query-planning safety, not a purity
or order-independence claim**. It doesn't ask "does this goal have side
effects?" — it asks "can I hoist demand analysis through this
recursion without losing aggregate semantics?"

Similarly, `safe_mutual_need_prefixes/1` (line 2055) checks that mutual
recursion has compatible prefixes across clauses. Again structural, not
effect-based.

### Why we're not borrowing the C# logic

1. **Different semantics.** Side-effect freedom ≠ aggregate-compatible
   recursion structure. Borrowing would conflate two distinct concerns.
2. **Narrower scope.** The C# checks are coupled to the need-closure
   optimizer; they wouldn't transfer meaningfully to "should the WAM
   emit `ParTryMeElse`?"
3. **The Prolog side is already richer** than the C# side for purity
   specifically. The C# engine has no `is_pure_goal` equivalent, no
   side-effect blacklist, no I/O detection.

### What (minimally) we *could* borrow later

The C# need-closure safety check is **orthogonal to purity but adjacent
to parallelism safety**: a predicate might be pure *and* unsafe to
parallelize under the need-closure optimization, or vice versa. When
Phase 4.x eventually needs to compose parallelism decisions with
query-plan optimizations, the two analyses might meet at a joint
"safe to parallelize *and* safe to hoist through this fixpoint"
predicate. That joint predicate would call both the purity certificate
and the C# structural check, not merge them.

So the eventual relationship is **cooperation, not absorption**: the
purity module owns effect-freeness; the C# module owns
need-closure-compatibility; a thin integration layer combines them
when a caller needs both.

For Phase 1 of this proposal, we borrow nothing from C#. The C# side
gets the ability to *consume* purity certificates in the future, and
contribute its own structural verdicts through the same certificate
shape if it ever grows side-effect analysis.

## Direction

**Extract, don't invent.** A new module `core/purity_certificate.pl`:

1. Defines a **shared certificate shape** (see the specification doc).
2. **Wraps the existing logic** in `clause_body_analysis.pl` — produces
   certificates from the same blacklist + clause walker.
3. **Absorbs** `advanced/purity_analysis.pl`'s whitelist as an
   alternative strategy (strict mode).
4. **Exposes a registration API** for extra producers (kernel registry,
   user annotations, future C# certificates).
5. **Documents the serialization format** (Prolog term, JSON for
   cross-process) so the Prolog↔C# boundary has an answer when the time
   comes.

The rest of the project gets thin wrappers or migrates gradually.

## Target scope: general, not Haskell-specific

The certificate is a verdict about the **source predicate's semantics**
— does it have side effects? can its clauses be reordered? The verdict
is the same regardless of which backend consumes it. So the module is
**target-agnostic** in principle:

- **Haskell WAM (immutable).** Primary near-term consumer. `ParTryMeElse`
  emission decision. Immutability makes the cost of acting on a
  pure-verdict low — forking a choice point just copies pointers.
- **Rust WAM (mutable).** Can still consume the same verdicts. A pure
  predicate is forkable in Rust too; the target just pays a different
  implementation cost (`Arc`/`Mutex` around shared state, or per-branch
  state cloning via `Clone` derives). The certificate says *whether*
  parallelism is safe; the target decides *whether it's worth the cost*.
- **C# target.** Can consume verdicts for query-plan reordering
  (pure goals can be hoisted past aggregates), join reordering, and —
  eventually — parallel materialization. The structural need-closure
  check it already has is orthogonal; both can be consulted.
- **AWK, Bash, Go, LLVM, WAT** and other targets can reuse the same
  certificate for whatever safety gate they expose. The certificate is
  their oracle, not their implementation strategy.

**What differs across targets is not the verdict — it's the cost/benefit
the target pays to act on it.** Imperative targets may choose to
consume `pure` verdicts more selectively (only when per-branch
cloning is cheap); functional targets can consume them more
aggressively. That decision lives in the target, not in the
certificate.

**Orthogonality to target memory model.** The certificate says nothing
about whether the target can cheaply copy state to fork. That's an
implementation concern for the target's fork primitive. Rust needs
`.clone()` bounds + possibly `Arc` sharing; Haskell needs `rdeepseq`;
C# needs value-type semantics or defensive copies. All three can
consume the same verdict and use it to guide, not dictate, their fork
strategy. A target may even choose to never fork even when the
certificate is `pure` — that's the target's call.

This matters for the implementation plan: we explicitly do *not* put
fork-primitive logic in the certificate module. The module produces
verdicts; targets consume them and decide what to do.

## Non-goals

- **Abstract interpretation.** We stay syntactic. A goal is pure iff
  none of its builtins match the impure blacklist (or all match the
  pure whitelist, in strict mode). No constraint solving, no
  abstract domains, no dataflow analysis.
- **Cross-module call-graph analysis.** The certificate analyzes a
  predicate's clauses directly; it does not chase transitive calls.
  "Calls a user predicate" is treated as `unknown` unless that user
  predicate has its own certificate (declared or analyzed).
- **Proving termination.** Orthogonal concern.
- **Unifying every safety analysis in the project.** The C# need-closure
  check, termination bounds, mode inference — these stay separate.
  This module is specifically about **side-effect freedom and
  order-independence**.

## Why now

Three reasons, in order of weight:

1. **Phase 4.1 of the intra-query parallelism plan will need it.**
   `ParTryMeElse` emission must consult a purity verdict. The spec
   assumes this module exists; the plan slots it in at Phase 4.1. If
   we extract now, Phase 4.1 becomes a simple emission decision
   instead of a full analysis.

2. **The existing logic is fragmenting.** Two modules with subtly
   different rules are easier to reconcile now (a few hundred lines
   total) than after both have grown more callers. Debt is linear in
   caller count.

3. **The certificate shape is a natural place for future producers.**
   Kernel-detection purity (trusted-by-construction), user annotations
   (trusted-by-declaration), static analysis (proven), and eventual
   C#-engine verdicts (certified) all have different provenance and
   different confidence. A shared shape with provenance metadata lets
   consumers make informed fallback decisions (e.g., "this predicate
   has confidence 0.8 — fork, but estimate work conservatively").

## When to stop extracting

If the new module grows past ~400 lines of non-trivial code, we've
probably absorbed something that should have stayed separate.
The target is: thin producers + thin consumers + a stable certificate
shape in the middle. If it becomes a behemoth, we're re-inventing a
static analyzer. That's a different project.

## Related documents

- `docs/design/PURITY_CERTIFICATE_SPECIFICATION.md` — concrete shape
- `docs/design/PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` — phased rollout
- `docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md` §3 — first consumer
- `docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md` §4.1 — emission site
- `src/unifyweaver/core/clause_body_analysis.pl` — existing donor
- `src/unifyweaver/core/advanced/purity_analysis.pl` — alternate whitelist
- `src/unifyweaver/targets/csharp_target.pl` §1975+ — the C# logic we
  are *not* absorbing
