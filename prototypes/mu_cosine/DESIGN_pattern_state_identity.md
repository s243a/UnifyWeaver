# Pattern-state canonical form and identity — design note

**Status: design note, not a specification and not an authorization.** It states what a pattern
state *is* as data, lays out the alternatives for canonicalizing it with their costs, and — its
highest-value section — shows how each possible answer to **ruling 6** constrains the canonical
form, converting that ruling from a preference into a decision with a visible price tag. It
proposes **no answer to ruling 6**; the encoder experiments decide it, and the harness in
[`pattern_stache/pstate_views.pl`](pattern_stache/pstate_views.pl) exists so they can. Sealed
artifacts, registry v0.4, `pec-v3`, and `peid-v1` are untouched.

Companion to [`DESIGN_prolog_elaborator.md`](DESIGN_prolog_elaborator.md), whose §2 fenced this
territory; the elaborator prototype now existing is what lifts ruling 6's own stated gate
("this ruling should *follow* the elaborator prototype").

> **This note reports a defect in merged code.** The lane's stop-clause — *report, do not patch
> around* — fires here for the first time. See **§2, The ordering defect**. It is confined
> inside the identity fence (no sealed oracle, no ground path, no shipped byte contract is
> affected), and it is characterized by tests rather than repaired, because repairing it is a
> ruling, not a patch.

## Purpose, input, output

**Purpose.** Decide what it would mean for two pattern states to *be the same state* — the
prerequisite for any digest, cache key, or tokenization over them. This note designs the
question, not the answer.

**Input.** A pattern state as the elaborator produces it: `pattern(Term, Store)` plus the
origins carried alongside (`elaborate/4`).

**Output (of the eventual scheme, not of this note).** A normal form such that two derivations
of the same state produce identical output and two different states never collide.

**What this note does not do.** It assigns no identity, computes no digest, freezes no
encoding, and does not promote the elaborator's existing ordering device into a canonical form.

## The one table: three things that look alike

The predictable conflation here is between an ordering device, a canonical form, and an
identity scheme. All three "put a state in a definite shape," and treating one as another is
how a prototype convenience becomes a shipped contract by accident:

| | **ordering device** | **canonical form** | **identity scheme** |
|---|---|---|---|
| what it is | a deterministic sequence, for display and comparison within one process | a normal form: the same state always yields the same representation, different states never collide | a name or digest derived from the canonical form |
| example | `pe_elaborate`'s numbervars-by-traversal projection | not built — this note designs the choice | `peid-v1` (fenced) |
| answers | "in what order shall I print these?" | "are these two things the same thing?" | "what is this thing called?" |
| may be wrong without harm? | yes — a bad order is a readability bug | **no** — a wrong canonical form silently merges or splits states | **no** — a wrong identity poisons caches and seals |
| status today | **shipped, and defective** (§2) | **not shipped** | **fenced** |
| promotion rule | must **never** be promoted by default; §2 is what that hazard looks like when it materializes | only by an explicit ruling | only from a frozen canonical form |

The elaborator's own source says its projection is "an ORDERING DEVICE, superseded by
`peid-v1`'s numbering when that freezes — no identity claim attaches to it." That disclaimer is
load-bearing, and §2 shows why: the device does not even satisfy the property an ordering device
is usually assumed to have.

## 1. What a pattern state is, as data

Three components, of which only the first two are candidates for the canonical form:

1. **The term** — a Prolog term containing variables. Ground subterms are already covered by a
   sealed canonical form (the v0.4 surface); the variables are the new content.
2. **The residual store** — a *set* of goals over those same variables (desugaring §6.2 rules
   it a set: "sorted and deduplicated — as a set, not a list"). Crucially, the store and the
   term **share variables**: that sharing is the state's real structure. A store is not a list
   of independent facts; it is a hypergraph whose nodes are variables and whose edges are goals.
3. **Origins** — surface-provenance metadata, carried alongside the store, never inside it
   (ruling 3(b)).

The consequence that shapes everything downstream: **a pattern state is a labelled hypergraph up
to renaming of its variables**, not a string and not a tree. Two states are the same when there
is a bijection of variables carrying one to the other — α-equivalence. Any canonical form is
therefore a *canonical labelling* problem, which is the family of problems where "obvious"
approaches are usually wrong in exactly the way §2 records.

## 2. The ordering defect (stop-clause finding)

**Claim in merged code** (`pe_elaborate.pl`, canonical-store section): *"each step numbering the
projection-least remaining goal, so numbering is a function of logical content, not input
order."* Ruling 1 and desugaring §6.2 rest on that claim.

**Measured: the claim is false.** `least_by_projection_/4` compares with a strict `@<` and keeps
the incumbent on a tie, so when two goals share a projection the tie falls to **input list
order**. That is reachable through `elaborate/3` inside ruling 4(a)'s scope:

```prolog
GJa = has_type(A,  judge(_)),      % A is constrained twice
GJb = has_type(_B, judge(_)),      % B once
GSa = has_type(A,  substrate(_)),
elaborate(fs, [GJa, GJb, GSa], P1),
elaborate(fs, [GJb, GJa, GSa], P2).   % same store, different presentation
```

yields

```text
P1 = pattern(fs, [has_type(A,judge(B)), has_type(A,substrate(C)), has_type(D,judge(E))])
P2 = pattern(fs, [has_type(A,judge(B)), has_type(C,judge(D)), has_type(C,substrate(E))])
```

The two are **not even variants of each other** (`\+ P1 =@= P2`). One store, two presentations,
two different canonical forms.

**Mechanism.** The two `judge` goals project identically (`has_type('$u',judge('$u'))`), and
`judge @< substrate`, so the tied pair is *projection-least* and is numbered **before** the
`substrate` goal — the only goal that distinguishes `A` from `B` — is ever consulted. The
distinguishing information exists; the algorithm commits before reading it.

**Why the merged tests miss it, and why that is the familiar shape.** Every ordering test in the
suites uses goals with *distinct* projections, or a *fully symmetric* pair. Both are stable.
This is the same hazard the lane already measured one level down: `probe_residual_order.pl`
found raw standard order to be *accidentally stable when functors differ*. The numbervars
projection is *accidentally stable when projections differ*. Same bug, one abstraction higher —
which is itself the argument for canonical labelling done properly rather than by refinement of
a sort key.

**Blast radius, stated precisely.** Zero on anything sealed: the ground path has no residual
store, so all 360 merged tests and both sealed oracles remain valid. The defect lives entirely
inside the §2 fence of the elaborator note — the territory that by construction has no oracle.
That is the fence working as designed: the unsound thing was quarantined before it was trusted.

**Not repaired here, deliberately.** A fix is a choice among the schemes in §3, i.e. a ruling.
The defect is instead *characterized* by tests
([`test_pstate_views.pl`](pattern_stache/test_pstate_views.pl),
`ordering_defect_characterization`) in the lane's recorded-divergence style: they assert
today's behaviour, so the day the ordering is fixed they fail and force the fix, the tests, and
this note to be reconciled in one review.

## 3. The VarId question

A canonical form needs variables to have *derived* identities. The elaborator's numbering is not
one, and must not be quietly promoted into one. The alternatives:

**(A) Keep the numbervars-by-traversal projection.** Cheapest; already written. Costs: it is
defective (§2); and even repaired, position-derived numbering is *non-local* — adding one goal
can renumber every variable, so nothing incremental can be cached against it.

**(B) Canonical labelling by refinement *plus individualization*.** Colour each variable by its
local signature (the multiset of goal-shapes and argument positions it occurs in), refine to a
fixpoint (Weisfeiler–Leman style) — and then, **if a non-singleton colour class remains,
individualize**: pick the smallest such class, branch on each member, recurse, and take the
lexicographic minimum over branches (the standard nauty/bliss shape). Costs: a real algorithm
with a real correctness burden, and worst-case exponential branching on highly symmetric states.
Benefits: it is the *correct* answer for a hypergraph up to renaming, and it fixes §2 by
construction — the distinguishing `substrate` goal enters `A`'s colour in the first refinement
round, before any tie is broken.

> **Refinement alone is not enough, and getting this wrong would be the third instance of the
> same bug.** Colour refinement is sound but **incomplete**: it never gives distinguishable
> variables the same colour, but it can *fail to separate* them, stabilizing with a non-singleton
> class on symmetric or regular structures. At that point something must break the tie — and if
> that something is input order, the defect is rebuilt one layer deeper: *"accidentally stable
> when refinement separates."* Individualization is what makes the scheme a genuine canonical
> form rather than merely an invariant. Ruled explicitly (ruling 1) so it is specified before it
> is built, not discovered afterwards.

Two riders on (B), both ruled:

- **Cap the branching, fail closed.** Individualization–refinement is exponential in the worst
  case. At pattern-state sizes (a handful of goals) that is irrelevant, but a pathological store
  must **refuse** rather than hang or silently degrade to a weaker labelling — this lane's
  standing discipline. A hard branch-count cap with a named error, not a timeout and not a
  fallback.
- **The correctness property is permutation stability.** For any store, *every* input
  permutation must yield `=@=` outputs. It is checkable **without a digest**, so the acceptance
  test for the scheme sits entirely left of the identity fence. Already transcribed as runnable
  code — `permutation_stable/2` in
  [`test_pstate_views.pl`](pattern_stache/test_pstate_views.pl) — so the implementer inherits
  the criterion rather than re-deriving it. Today it passes for distinct-projection, fully
  symmetric, and shared-term-variable stores, and fails for the §2 shape; that failure is
  asserted, so the scheme landing will fire it.

**(C) Adopt vNext's `VarId` directly.** vNext's index unification already mints variable
identities on the pattern surface. Costs: a cross-stack dependency on the path that currently
has none, and a category mismatch — see the boundary below.

**(D) Content-addressed variable names** (name each variable by a digest of its occurrence
signature). Costs: this *is* (B) with a hash bolted on, and it crosses into identity, which is
fenced.

**Recommendation: (B), and specifically not (A) or (C).** (A) is the thing that just failed;
(C) is the wrong layer.

**Where the boundary with vNext sits** — the important distinction, because "vNext owns
variables" is true but not in the way it first reads. vNext's `VarId` answers **"are these two
occurrences the same variable?"** — an *authoring/scope* question, settled by index unification
when the pattern is written, and already ruled: two occurrences of `C` in a `where`-form are one
variable *by construction* (§12). A canonical form asks a different question: **"given that
sharing structure, in what order do the variables appear?"** — a *content* question with no
authored answer. The two compose rather than compete: vNext establishes the sharing; the
canonical labelling derives an order from it. So the boundary is not "vNext owns this territory,
stay out" — it is *vNext fixes the graph; the canonical form labels it*. What must not happen is
the reverse: a labelling scheme inventing sharing that the surface did not author.

## 4. Store membership: do origins enter the canonical form?

**Ruling 3(b) holds, and this note argues it more strongly than before.** Origins stay out.

The reason is the one §12 already ruled for bindings. A `where`-form and its repeated-literal
spelling must produce the *same* state — that is measured, byte-exact, for the ground path. If
origins entered the canonical form, the same state authored two ways would carry two different
origin sets and therefore become two different states: **identity would fragment by spelling**,
which is precisely the property the lane refuses for pins and for bindings. Origins are
compile-time diagnostic metadata about *how the author got here*, not about *what this is* — the
same category as a pin, and excluded for the same reason.

One consequence worth stating so it is not discovered later: this makes origins **lossy across
canonicalization**. Two goals that dedup to one (`==`-identical) may carry different origins, and
the surviving origin is arbitrary. That is acceptable for diagnostics (either origin is a true
statement about where a copy came from) and would be unacceptable if origins were identity — a
second, independent reason they must not be.

## 5. The flagged boundary case: decidable, and hiding a worse one

The elaborator suite flags a symmetric case as "fenced behind `peid-v1`"
(`test_pe_elaborate.pl`, `same_shape_goals_both_kept`): two same-shape goals over distinct
store-only variables, where which live variable lands first is symmetric.

**Measured answer: it is decidable, and not actually ambiguous.** Swapping those two goals
produces a store that is a **variant** of the original (`[G1,G2] =@= [G2,G1]`). The two
presentations therefore denote *one* state up to renaming, and any canonical form defined up to
α-equivalence is well-defined on it — the "ambiguity" is entirely about which live Prolog
variable is which, which is exactly what α-equivalence quotients away. The tie is an
**automorphism**, and on an automorphism every choice is the same choice.

So the flagged case dissolves. What the investigation found instead is §2: a *near*-symmetric
case, where the goals are indistinguishable **locally** but distinguishable **globally**, and the
current device commits before it looks. That is the more dangerous shape, and it was hiding
behind the one that was flagged. Recording the distinction, because it generalizes:

| shape | swap is an automorphism? | decidable? | current behaviour |
|---|---|---|---|
| distinct projections | no | yes | correct |
| fully symmetric (tie, no global distinguisher) | **yes** | yes — any choice is right | correct, and correct *for a reason* |
| **near-symmetric** (local tie, global distinguisher) | **no** | yes — refinement finds it | **defective (§2)** |

Scheme (B) handles all three rows uniformly; (A) handles rows one and two and fails row three.

## 6. How ruling 6 constrains the canonical form

**This is the section the note exists for.** Ruling 6 — do residual goals reach the encoder as
*structure* or as *features*? — is usually discussed as a modelling preference. It is not. It
decides whether a canonical-labelling problem enters this lane's critical path, and the harness
makes the difference measurable rather than arguable.

| if ruling 6 answers… | the canonical form is load-bearing for… | does §2 block it? | what must be built first |
|---|---|---|---|
| **structure** (tokens in the stream) | **tokenization** — goal order *is* token order, so the canonical form determines what the encoder sees on every example; two presentations of one state would train as two different examples | **yes, blocking** | scheme (B): a real canonical labelling, α-equivalence proven, §2 fixed |
| **features** (side channel) | **caching only** — a feature vector is a sorted bag of counts; goal order cannot reach it | **no** | nothing: a features-only cache key derives from sorted counts without solving variable ordering at all |
| **both, behind an ablation** | tokenization for the structure arm; caching for the features arm | yes, but **only for the structure arm** | (B) before the structure arm can be trusted; the features arm can run immediately |

**This is measured, not asserted.** The harness emits both views from the same elaboration, and
the characterization suite proves the asymmetry on the §2 counterexample:

- `features_view_immune_to_the_ordering_defect` — the two presentations produce `==`-identical
  feature bags;
- `structure_view_inherits_the_defect_KNOWN` — the two presentations produce different token
  streams.

So the price tag is explicit: **answering "structure" buys a graph-canonicalization problem;
answering "features" does not.** That is a fact about the two answers, not an argument for
either — a structure encoding may well be worth its price, and this note takes no position. What
it removes is the possibility of choosing "structure" without noticing the bill.

A second-order consequence worth recording: because the features view is order-immune, the
features arm of an ablation can proceed **today**, on the current defective ordering, without
contaminating its results. The structure arm cannot. If the encoder lane wants evidence before
the ruling, that is the cheap half to run first.

## 7. Oracle inventory

| behaviour | oracle | note |
|---|---|---|
| pattern-state generation from real elaborations | the elaborator's own suites (103 tests) | states are generated, never hand-written |
| structure-view token shape | **own fixtures only**, probe-versioned | deliberately **not sealed**: sealing a candidate view would freeze a canonical form by the back door, the exact hazard this note exists to prevent |
| features-view counts | own fixtures only | same reasoning |
| ordering stability, distinct projections | property test | green, and still true |
| ordering stability, fully symmetric ties | property test | green, and green *for the right reason* (automorphism, §5) |
| ordering stability, near-symmetric ties | property test | **characterized as a defect** (§2) — the honest state is "known-wrong, pinned" |
| α-equivalence correctness of any candidate scheme | **permutation-stability property test** — `permutation_stable/2` | *this row changed with ruling 1.* It was recorded as "no oracle exists"; the ruling supplied one, and it is a real oracle rather than a fixture: every input permutation of a store must yield `=@=` output. Digest-free, so it stays left of the identity fence |
| pattern-state digests / canonical bytes | **no oracle exists** | fenced, `peid-v1` |
| cross-check against vNext `PatternAST` | vNext testdata as **data only** — and no fixture there covers pattern states today | so: **no oracle**, and the standing rule forbids reaching past the fixtures into the machinery |

**Two** "no oracle" rows remain, both gating, both stated as findings rather than gaps to be
filled by opinion. The third was closed by ruling 1 supplying permutation stability — worth
noting as a pattern: an oracle-less row is not permanently oracle-less, and naming it is what
lets a later ruling close it deliberately.

## 8. Rulings

*Rulings 1, 2, 3 and 6 were decided on the PR that carried this note, after an independent
reproduction of §2; the decisions are folded into the sections above and recorded here with
their alternatives so the reasoning survives. Rulings 4 and 5 stand as recommendations.*

**Decided:**

- **Ruling 1 — the canonical-labelling scheme: ACCEPTED as (B), *amended*.** Refinement **plus
  individualization**, not refinement alone; branch-count capped with fail-closed refusal;
  acceptance criterion is permutation stability. The amendment matters and is recorded in §3:
  refinement alone is incomplete, and an input-order tie-break after it would rebuild §2's
  defect one layer deeper.
- **Ruling 2 — repairing §2: after ruling 6's sequencing, via ruling 1's scheme.** Not a
  minimal tie-break patch — that is another sort-key refinement, the move that has now failed
  twice at two levels. The defect stays characterized in the meantime.
- **Ruling 3 — origins: OUT, confirmed.** Ruling 3(b) reaffirmed on §4's argument (including
  them would fragment identity by spelling, contradicting §12's measured where-form
  equivalence).
- **Ruling 6 — sequencing, not an answer: the features arm runs NOW; the structure arm runs
  after canonical labelling lands.** Licensed by §6's measurement rather than by preference:
  `features_view_immune_to_the_ordering_defect` is what makes the features arm uncontaminated
  by the open defect, so running it first is not a compromise — it is the only arm that is
  currently sound. **Ruling 6 itself remains genuinely open** as a science question; this
  sequences the ablation, it does not pre-decide it.

**Still open:**

- **Ruling 4 — may a candidate view ever be sealed before ruling 6 is answered?** *(owner)*.
  (a) no view is sealed until the ruling lands; (b) seal the features view early, since it is
  order-immune. **Recommend (a)**: (b) is exactly how a probe becomes a contract, and the
  features view's immunity is a reason to *trust its ablation results*, not a reason to freeze
  its shape.

*(Ruling 5 as originally posed — whether the features arm may run before the defect is
repaired — was absorbed into ruling 6's sequencing: yes, and that is now the ruled plan.)*

Out of scope, unchanged: ruling 4(b) (`interpretation/3`/`representation/4`), registry-mirror
derivation from the component vocabulary (parked on #4064, with the `non_amplifying/1`
divergence test as its tripwire), and any answer to ruling 6 itself.

## Acceptance test (project principle 9)

A fresh reader should be able to answer from this note alone: *what a pattern state is* (a
term plus a set of goals sharing its variables — a hypergraph up to renaming), *why ordering it
is hard* (canonical labelling, not sorting — with a measured counterexample), *what the current
code gets wrong and how badly* (§2: near-symmetric ties follow input order; blast radius zero
outside the fence), *what ruling 6 costs* (§6: "structure" buys a canonical-labelling problem,
"features" does not), and *what has no oracle* (α-equivalence correctness, digests, and any
cross-check against vNext pattern states).
