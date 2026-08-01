# Registry v0.4 — design

**Status: design note. Not implemented, and not an authorization to implement.** It records the
rulings issued on [issue #4013](https://github.com/s243a/UnifyWeaver/issues/4013) so they survive
outside GitHub comments, states what a `v0.4` registry must therefore change, and lists what is
deferred or still undecided.

Registry `v0.3`, `pec-v2`, `tok-v1`, and both sealed golden bundles remain frozen until a `v0.4`
implementation lands and is sealed through the procedure in
[`DESIGN_process_expression_patterns.md`](DESIGN_process_expression_patterns.md) §15.

## 1. Why `v0.4` exists

Issue #4013 recorded ten findings: the grammar cannot express the project's own lineage/μ
semantics, and one landed artifact records a process expression that does not parse. None is a
runtime bug; all are provenance-integrity and expressiveness gaps.

They matter because corpus enumeration was about to freeze a support against `v0.3`. **13 of the
15** currently enumerable `lineage(...)` argument forms are semantically empty and would become
unenumerable under the ruled split, so sealing first would have frozen a support that `v0.4`
immediately invalidates.

## 1.1 The model in plain terms

Every process expression answers six questions in six separate places, and no slot may borrow
another's answer:

| question | where | example |
|---|---|---|
| What relation are we estimating? | `estimand=` | `ancestry`, `path`, `assoc` |
| How is it computed (the procedure)? | the function term itself | `max(floor, gamma^hops * lca_frac)` |
| Which implementation should run? | `impl=` (a request) | graph-walk vs materialized table |
| Which code actually ran? | factory fingerprint | content-bound realization record |
| Audit/provenance annotations? | `@` pins — their own channel | `@impl_hash/4e13e0d` |
| Who supplies the μ labels, over what structure? | `mu=` + the substrate argument | `mu=haiku`, `principal_tree(pearltrees)` |

On the last row: **the judge is not a third methodological axis — it is a role the function
plays.** For an LLM judge the label-generating function is `prompt(Model, PromptText, Harness)`;
for graph-derived labels it is the registered function over the substrate (e.g.
`max(floor, gamma^hops * lca_frac)` over `principal_tree(pearltrees)`). So the methodology is
defined by **function + substrate (+ estimand)**, and `mu=haiku` is an *abbreviated reference to
the label-generating function* — abbreviated because R10 keeps judges as bare atoms in v0.4,
with prompt text and model revision carried by factory binding rather than the grammar.

## 2. The rulings

Ten rulings, all issued by the owner — R1–R6 on #4013, R7–R10 in the consolidated stage-1 ruling
on #4055 (which is final where anything overlaps). Reasoning is reproduced because a decision
without its reasoning gets re-litigated.

### R1 — Transitive μ is the quantity; hop decay is one estimator of it

*(finding 4 — the ruling the issue principally asked for)*

Transitive μ is graded element/subcategory relatedness computed over several hops. **Hop decay is
one estimator of it; an LLM judge is another.** The substrate, judge, and relation type jointly
define the methodology.

*(Refinement, ruled after this on #4055: the "judge" in that triple is not an independent axis —
it reduces to the function. An LLM judge's function is `prompt(Model, PromptText, Harness)`; a
graph judge reduces to substrate + function. The doctrine restated without the redundancy:
**function + substrate (+ estimand) define the methodology**, and "judge" names the
μ-label-generating role a function plays. See §1.1 and R10.)*

`sm_fs_freeze.py`'s targets are therefore graph-judge *estimates*, not assertions of semantic μ,
and do not contradict [`DESIGN_transitive_relations.md`](DESIGN_transitive_relations.md). That
document's objection is to committing to a composed point value; its ordinal constraint
`μ(A→C) ≤ min(links)` is the rule for **fusing** per-link estimates, while hop decay is a rule for
**generating** one estimator's per-link values. Different stages, no conflict.

Two riders from the same ruling:

- Hop decay alone is an incomplete *closeness* measure. `gamma^hops` tends to zero where two
  arbitrary nodes tend to the corpus mean. `lca_frac` is the pure-closeness component;
  the cumulative-walk model in [`DECISIONS_graph_geometry.md`](DECISIONS_graph_geometry.md)
  (2026-07-12) is the one with the right asymptote.
- `decay` specifically models **semantic drift**, expected to be larger where a node has many
  parents (`full_dag`) than where there is a principal parent (`principal_tree`).

### R2 — Split `source`; register the corpora

*(findings 2/3)*

`source` currently covers two roles and is therefore split into **substrate** / **μ-source** /
**scorer**. The corpora are registered as substrates.

The ruling's own reasoning: *judge, function, and corpus jointly define the methodology, so they
cannot share one type.*

### R3 — `decay` is a retention factor; document the direction

*(finding 6)*

`decay=0.9` multiplies by 0.9 per hop. Reading it as "decays by 0.9" is a natural and wrong
inference. Renaming to `gamma`/`retention` is **optional and permitted**; the registry must at
minimum state the direction.

### R4 — Add the `mu=` slot

*(finding 7)*

`lineage(X, mu=haiku)` and `lineage(X, mu=graph, decay=0.85)` spell the two estimators properly.
R1 *is* that slot.

### R5 — Split `impl` into `estimand=` and `impl=`

*(ruling request 7, finding 10)*

- **`estimand=`** — which quantity is computed. `hop_targets` is an estimand, not an
  implementation.
- **`impl=`** — which implementation of a *named* estimand. `structural` and `attention` are
  `impl` values under one estimand.

A single `impl` would have recreated the `source` conflation one level deeper: methodology has
two axes — *what quantity* and *how computed* — and they must not share a slot. **Only
implementations that actually exist may be registered.**

### R6 — Do not demote `lineage`; the pairwise-atoms direction is deferred, not forbidden

*(ruling request 5, findings 8/9)*

`hops` and `lca_frac` are not registered as atoms this cycle. `blend(hops(gamma=0.9), lca_frac)`
reads as a weighted sum, while `prototype_graph_judge.py` computes
`max(floor, gamma^hops * lca_frac)` — a product under a floor — so adopting it would replace a
visible omission with an invisible inaccuracy.

The direction is sound and the form is not ready. It may be prototyped in the experimental vNext
package, where the registry is a test fixture and nothing seals. `v0.4` does not wait for it.

### R7 — `estimand=` names the relation, not the procedure

*(#4055 findings 11/12; consolidated ruling item 1)*

The term *is* the procedure, so there is no `method=` field. The enumerated set is the
**relation** vocabulary, reconciling `DESIGN_transitive_relations.md` with the labels live in
`score_inferred_tail.py`: `subcategory`, `super_category`, `element_of`, `subtopic`, `see_also`,
`assoc`, `bridge`. `see_also` and `assoc` are two distinct relations in one symmetric family —
judges already emit separate μ for each — and both are non-transitive.

**The three hierarchical labels are one family with positional roles, and the label is secondary
to the graph relation.** `subtopic` and `subcategory` compose interchangeably, in any order — the
distinction (categories more general, often pluralized) is curational, and a subtopic may be
promoted to a subcategory without the graph changing. `element_of` **terminates** a chain at the
item end: membership is not transitive through membership (`element_of ∘ element_of` is
deliberately absent), but survives descent (`element_of ∘ subcategory ⇒ element_of`) — i.e. it may appear only as the item-end step of a chain, which then continues upward through descent; it never passes through a second membership.

Plus two **derived** estimands, with a typing rule: a composition of descent primitives in one
direction types as **`ancestry`**; any mixed-direction composition types as **`path`**, of which
ancestry is the monotone special case. `sibling` (up then down through a common ancestor,
generalizing to kinship up^m ∘ down^n) is a named *shape* of path, not a separate estimand, and
is label-insensitive among the descent family. Ancestry chains: at most one `element_of` at the
item end, then any mix of `subtopic`/`subcategory`; `bridge` transparent; `assoc`/`see_also`
excluded from chains, checkable by construction. Both derived estimands are registered as
*defined by rule*, not opaque — their meaning is computed from the composition rules rather than
stipulated as an eighth primitive. The lineage **object** (a path value) witnesses an `ancestry`
or `path` claim; it is not itself an estimand.

μ over derived composites is fuzzy and weakens with length toward the corpus mean — bounded by
the ordinal constraint for monotone chains, modeled by the random walk (drift-away vs
regression-to-the-mean per hop; cumulative walk in `DECISIONS_graph_geometry.md`, depth-drift
analysis in `DESIGN_bidirectional_walk.md`). Estimator-side; not registry surface.

Naming: `lineage_op` stays the vNext coarse operator; `lineage_at` is superseded by `ancestry`
and omitted per the patterns doc's own escape clause. Migration note for the vNext lane: §15.1
will need rows mapping coarse `lineage_op(...)` requests to `ancestry`/`path` resolutions.

### R8 — Add `max` and product; the graph judge registers under `path`

*(#4055 finding 11; consolidated ruling item 2)*

`max` and multiplication join the **function vocabulary** — they are operators of the function
term (the procedure), not estimand values. The graph judge registers as
`max(floor, gamma^hops * lca_frac)` under **estimand `path`**, not under any lineage/ancestry
name and not under primitive `assoc`: measured against `prototype_graph_judge.py`, it walks an
*undirected* adjacency with ancestors and descendants excluded from the candidate pool, so it
computes derived kinship-shaped relatedness (the up-down-through-LCA shape that `lca_frac`
measures) rather than either a hierarchical relation or the curated lateral edge `assoc` names.

### R9 — Pins are outside semantic identity; v0.4 must make that true

*(#4055 finding 13; consolidated ruling item 3)*

Audit identifiers (`impl_hash`, commit/line) are **pins** — a third channel (`@kind/value`),
outside both the estimand slot and the function term. They are excluded from **semantic
identity**: per §16.25/26 two derivations of the same precise AST share semantic identity, so a
regeneration of realizing code must not mint a new process. The realization that ran is
identified by the factory fingerprint; `impl=` remains a selection constraint only.

**Stage-1 obligation.** v0.3's canonical digest currently **includes** pins (*measured*:
`ast_sha("lineage(graph,decay=0.85)@run/2026-07-25")` ≠ `ast_sha("lineage(graph,decay=0.85)")`),
so v0.4 must split identity — a pin-free **semantic** digest for seals, cards, and cache keys,
with pins + factory fingerprint layered outside it. Enforced **by construction, not by flag**:
the transpiler emits two artifacts from one parse — the semantic AST (pin-free, the only form it
can take) and a provenance envelope (pins with their §16.27 target-role-path + node-digest
attachments). No configuration can produce a pin-bearing digest.

**Model-input visibility is a separate, training-time choice.** Renderings fed to the encoder may
include the pin channel as auxiliary conditioning (the V3 card already renders pins; V1/V2 elide
them), letting the model learn where implementations diverge without identity forking. Because
hash tokens are opaque and can only be memorized, pin-visible input enters as an **ablation arm**
(V2 vs V3 cards), not the default. Encoder-side consequences (per-channel edge-role namespaces,
pin positions as target-path ⊗ pin-role, coordinates materialized while encodings are computed
from shared tables, pin-channel dropout instead of format mixing) are recorded in
`DESIGN_tree_position_encoding_theory.md` §11.

### R10 — Judges stay atoms in v0.4

*(consolidated ruling item 4)*

A bare `haiku` hides `prompt(Model, PromptText, Harness)`, so two judges differing only in prompt
text share an expression — finding 8's under-determination in the judge namespace. The limitation
is accepted and recorded rather than fixed this cycle. Revisit when a prompt slot can bind
through the factory mechanism — prompt text and model revisions belong in factory binding, which
already versions realizing code, rather than in the grammar.

Under the §1.1 reduction (judge = the function's role), the revisit condition has a sharper
meaning: it is the moment the judge **atom dissolves entirely into an ordinary function term** —
`prompt(Model, PromptText)` becomes expressible, factory binding versions the model revision and
harness, and judge registry entries become function terms like any other. The atom is
scaffolding; the reduction is the building.

The judge as a *conditioning channel* in model input remains legitimate and unchanged — it tells
the encoder whether labels came from a model or from graph structure, which affects how
supervision is interpreted. Verified against landed code: that channel already embeds
**descriptive function cards, not names** — `judge_cards.py` supplies e.g. `"deterministic
structural judge, graph walk, no language model"`, consumed as `W·e5(card) + residual`
(`mu_attention.py` NameFunctionCond), with the stated rationale "e5 can't place opaque tokens,
give it words." The remaining refinement path is card → actual prompt text for LLM judges, which
would make the e5 channel a genuine low-fidelity *function* channel, redundant with the future
custom encoder — redundancy that licenses aggressive, **asymmetric** channel dropout (drop the
e5 judge channel more often than the function channel, so the from-scratch encoder gets gradient
pressure) and gives legacy models a bridge input. Interface constraint to state now: the future
function channel's output vector should be drop-in compatible with the judge channel's slot, so
the substitution is a swap rather than a rewire. These are encoder-side notes for
`DESIGN_expression_encoder_future.md`; none of it is registry surface.

## 3. What `v0.4` must change

Current state, *measured* against the landed `v0.3` registry:

| quantity | value |
|---|---:|
| `REGISTRY_VERSION` | `v0.3` |
| registry entries | 18 |
| entries typed `source` | 9 |
| registered processes | 9 |
| registered processes naming a `source`-typed entry | **8** |

The changes implied by §2:

1. **`REGISTRY_VERSION` → `v0.4`.**
2. **Retire the `source` type**, replacing it with substrate / μ-source / scorer (R2). Nine
   entries are re-typed; `graph`'s two roles separate.
3. **Register the corpora** as substrates — `pearltrees`, `simplemind`, a Wikipedia corpus, and
   `fs` (R2). This is what makes finding 1's expression parseable.
4. **Add `mu=`** to lineage-shaped operators (R4).
5. **Add `estimand=` and `impl=`** as separate fields, registering only implementations that
   exist (R5). The `estimand` enumeration is the relation vocabulary of R7 — seven primitives
   plus rule-defined `ancestry` and `path`.
6. **Document `decay` as a retention factor**; optionally rename (R3).
7. **Add `max` and multiplication** to the function vocabulary; re-register the graph judge as
   `max(floor, gamma^hops * lca_frac)` under estimand `path` (R8).
8. **Split identity**: pin-free semantic digest, provenance envelope outside it, emitted as two
   artifacts by construction (R9).
9. **Update `DESIGN_transitive_relations.md`'s composition table** with the mixed-chain and
   terminator semantics of R7, so the table states what the ruling extended.

Item 2 is the one with reach: eight of nine registered processes name a `source`-typed entry, so
the split touches nearly the whole registry rather than a corner of it.

### 3.1 Two couplings that make this larger than a registry edit

Both were verified against the landed code and neither is optional.

**Both sealed golden bundles stop loading.** They pin the registry version, and the loader
*validates* it:

```python
# process_expression_contract.py
if document.get("registry_version") != REGISTRY_VERSION:
    raise ...
```

`PROCESS_EXPRESSION_GOLDEN_v1.json` and `_v2.json` both record `registry_version: v0.3`. A bump to
`v0.4` therefore makes both fail to load, so a **v3 bundle is mandatory**, produced through the
supersession procedure in `DESIGN_process_expression_generator.md` §0. This is not a tidy-up that
can follow later; nothing that reads a bundle works until it exists.

**The tokenizer vocabulary moves, so `tok-v1` becomes `tok-v2`.** The vocabulary is *derived from
the registry* rather than fixed:

```python
terms += [f"<NAME:{name}>"    for name in sorted(REGISTRY)]
terms += [f"<OUTPUT:{output}>" for output in sorted({s.output for s in REGISTRY.values()})]
keys   = sorted({key for s in REGISTRY.values() for key in s.kwargs})
```

All three inputs change under `v0.4`: new corpus names, retired and added output types, and the
new `mu=` / `estimand=` / `impl=` kwarg keys. `<OUTPUT:source>` is presently a literal token in
the 376-term `tok-v1` vocabulary, and it ceases to exist. Since `VOCAB_VERSION` "moves when the ID
assignment changes," `v0.4` forces `tok-v2` — superseding the tokenizer merged in #4012.

The consequence for planning: **`v0.4` is a staged project, not a single change.** Registry,
bundle, tokenizer, migration manifest and their test suites move together, and the order matters
because each stage's tests depend on the previous stage's artifacts.

## 4. What this fixes

| finding | fixed by | how |
|---|---|---|
| 1 — `lineage(fs,…)` does not parse | R2 | `fs` becomes a registered substrate |
| 2 — `source` conflates roles | R2 | `lineage(haiku, …)` fails type checking |
| 3 — `graph` overloaded | R2 | its substrate and judge roles get distinct types |
| 4 — point target vs ordinal constraint | R1 | estimator, not assertion; fusion vs generation |
| 5 — "decay" means two things | R1 | it is a parameter of the graph-judge method |
| 6 — direction misleading | R3 | documented as retention |
| 7 — scalar cannot express composition | R4 | `mu=` names the estimator |
| 8, 10 — expression under-determines the process | R5, R8 | `estimand`/`impl` name the axes; `max`/product make the graph judge's computation writable |
| 9 — pairwise measures inexpressible | — | **not fixed**; deferred by R6 |
| 11 (#4055) — graph judge measures lateral kinship, not hierarchy | R8 | registered under estimand `path`, not `lineage`/`assoc` |
| 12 (#4055) — "estimand" conflated relation with procedure | R7 | estimand = relation; the term is the procedure |
| 13 (#4055) — `impl` conflated request with result | R9 | `impl=` requests; factory fingerprint records; audit ids are pins |

## 5. Identity and migration consequences

**Every process digest changes.** `REGISTRY_VERSION` is in the identity preimage —
`sha256(REGISTRY_VERSION + "|" + canonical_identity_string)` — so a version bump moves every
digest even where the spelling is unchanged. §15.1 says as much: *"every inventory item requires a
migration row even if its display spelling appears unchanged."*

**A migration manifest is required**, per §15.1, over the frozen `LegacyIdentityInventory`. Two
specifics already settled there:

- `lineage(graph,…)` *"cannot auto-map without a substrate and estimand ruling"* — R1 and R2
  supply exactly that ruling, so these rows become mappable.
- `lineage(fs,…)` artifacts are **regenerated rather than migrated**, and are out of migration
  scope entirely. Regeneration is exact: `sm_fs_freeze.py` computes path-component depth on
  `dest` with a constant retention, with no graph or LCA involvement, so re-running under the
  `v0.4` spelling reproduces byte-identical targets.

**The conditioning-card cache key must include `registry_version`.** Cards are e5 embeddings *of
canonical strings*, and canonical strings are registry-dependent. Without the version in the key,
a post-`v0.4` lookup returns a vector embedded from a string that no longer exists — silently.
*Owner: ranking lane. Trigger: `v0.4` ship.*

**Corpus enumeration stays gated on the `v0.4` ship**, not on the rulings. A ruling unblocks
designing the registry; it does not unblock running the generator. See
[`DESIGN_process_expression_generator.md`](DESIGN_process_expression_generator.md) §2.

## 6. Deferred, with reasons

| item | reason | revisit when |
|---|---|---|
| Pairwise-measure atoms (`hops`, `lca_frac`) — R6 | proposed form mis-describes the judge; needs a type and combinator the registry lacks | vNext's `entity[S]` ownership makes the two-lineage domain explicit |
| `superposition` as an `impl` value — R5 | no ancestor-superposition implementation exists | an implementation plus tests land |
| Renaming `decay` → `gamma`/`retention` — R3 | permitted but not required | anyone touching those signatures anyway |

## 7. Still undecided

**The `decay` estimand fork.** `DESIGN_process_expression_generator.md` §5.5 names two estimands
and does not choose between them:

- fit `decay` on **observed** titles — a drift-plus-typo-noise composite;
- fit on **canonicalized** titles — drift on the intended hierarchy.

Title typos split what should be one parent into two and inflate apparent drift, and no holdout
reveals the error because the holdout carries the same typos. The choice decides what `decay`
*means* as a quantity and belongs in the registry as methodology.

## 8. Test obligations

A `v0.4` implementation is not done until:

1. `lineage(haiku, …)` **fails** type checking, and `lineage(<substrate>, …)` passes;
2. every registered corpus parses in a substrate position, including `fs`;
3. `estimand` and `impl` are independently settable, and an unregistered `impl` value is refused
   rather than defaulted;
4. absence of `estimand`/`impl` blocks deployment while leaving ordinary registered defaults such
   as `decay=0.85` working;
5. the registry states `decay`'s direction;
6. the migration manifest covers every item in the legacy inventory, using only `mapped`,
   `ambiguous`, or `tombstoned`, with `lineage(fs,…)` rows out of scope and regenerated;
7. regenerated `sm_fs_freeze.py` targets are byte-identical to the originals;
8. the enumerable `lineage(...)` argument forms drop from 15 to the substrate-only set;
9. every committed artifact field named `process_expression` parses under `v0.4`
   (`DESIGN_process_expression_patterns.md` §16.34);
10. **adding a pin does not move the v0.4 semantic digest** — checkable as: stripping the fenced
    `<PINS>` sections is a no-op on the digest preimage (R9);
11. the graph judge's registered form is `max(floor, gamma^hops * lca_frac)` with estimand
    `path`, and it round-trips (R8);
12. `estimand` values are exactly the R7 enumeration; an `ancestry` or `path` value is accepted
    only where its composition rule derives it, and `assoc`/`see_also` inside a chain is a type
    error;
13. `subtopic` and `subcategory` are interchangeable inside a derived chain, and `element_of`
    is rejected anywhere but the item end.

Per the project convention, these land as tests rather than as prose.

## 9. Sequencing

```text
rulings issued (#4013)          done
this design note                done
v0.4 implementation + seal      not started  <- gates everything below
corpus enumeration              gated
encoder step 2 part 2           gated
```

Within the implementation, §3.1's couplings force an order:

| stage | contents | gated on |
|---|---|---|
| 1 | registry types and entries; `REGISTRY_VERSION` → `v0.4`; signatures gain `mu=`, `estimand=`, `impl=`; `max`/product operators; identity split (R9); `decay` direction documented; transitive-relations table updated (R7) | nothing — all rulings issued |
| 2 | golden bundle **v3** via the §0 supersession procedure — both current bundles stop loading at stage 1 | stage 1 |
| 3 | tokenizer **`tok-v2`**; vocabulary re-derived, round-trip re-proved over the v3 bundle | stage 2 |
| 4 | migration manifest over the frozen legacy inventory; `lineage(fs,…)` rows regenerated rather than migrated | stages 1–3 |
| 5 | conditioning-card cache key gains `registry_version` — *ranking lane* | stage 1 shipping |

Stages 1–3 cannot be split across releases: between them the repository has a registry whose
bundles do not load and a tokenizer whose vocabulary does not match. They land together or not at
all.

Issue #4013 is closed; its purpose — *"rulings needed before registry v0.4"* — is discharged.
Implementation is tracked in #4055, whose consolidated stage-1 ruling this document reproduces as
R7–R10. Stage 1 is unblocked.

## 10. Stage-1 concrete registrations (implementation blueprint)

Written before the code, so the decisions are reviewable as decisions. Three of them are
implementation calls not explicitly ruled; each is flagged.

### 10.1 The writable form of the graph judge

R8's mathematical form `max(floor, gamma^hops * lca_frac)` is not directly writable in v0.3:
positional numeric arguments are **rejected** (*measured*: `kalman(0.02, luna.S)` →
`ParseError: unregistered name`), there is no exponentiation operator, and there are no
variables. Resolution:

- **Grammar extension (flagged decision 1):** v0.4 accepts numeric literals in positional
  argument positions, typed by the signature's declared arg kind — the same
  declared-type-wins rule kwarg values already follow. `tok-v2` serializes them as value
  nodes inside `<ARG>` fences.
- **No exponentiation operator.** `gamma^hops` is the *parameterized interior* of a registered
  call — `hop_decay(<substrate>, gamma=…)` — because `hops` is supplied by the walk at run
  time, not by the expression. The term describes the procedure; the exponent is the
  procedure's definition.
- **Registered spelling:**

```text
max(0.02, product(hop_decay(simplemind, gamma=0.6), lca_frac(simplemind)))
```

**The repeated literal is the deliberate ground form.** Writing `simplemind` twice hides the
functional connection (both components must walk one substrate), and the connected form exists —
it is vNext's pattern surface: `product(hop_decay(C, gamma=0.6), lca_frac(C))` with a binding
`C=simplemind` is a `PatternAST` plus `ground(bindings)`, already implemented, where the two `C`
occurrences are provably one variable and grounding produces a term byte-identical to the
repeated-literal spelling. A future `where` clause is sugar over that API, not a v0.4 feature —
variables stay out of the sealed grammar. Known v0.4 limitation: the flat signatures cannot
*check* that `hop_decay` and `lca_frac` received the same substrate; a mismatched pair parses
and is legal-but-odd. vNext's index unification checks it for free.

with `estimand='path'`. Parameter values are the prototype's measured defaults
(`prototype_graph_judge.py`: `--gamma 0.6`, `--floor 0.02`); the old `lineage-graph`
process (`decay=0.85`) is retired and handled by the stage-4 migration manifest, not
re-registered.

**R6/R8 reconciliation (flagged decision 2).** R6 declined registering `hops`/`lca_frac` via
the `blend` restructuring; R8 requires the composite to be writable, which requires its
components. What survives of R6: `lineage` is *not* demoted (it remains, gaining `mu=`), and
`blend` is *not* widened (it stays a weighted sum over judges). The components arrive as
`hop_decay/1+gamma` and `lca_frac/1` — substrate-taking calls, not the bare pairwise atoms R6
declined.

### 10.2 Entry table

**Output types.** `source` splits: **`substrate`** (walkable structure) and **`judge`**
(μ-source). `score`, `target-set`, `pick` are unchanged; R2's "scorer" is the existing
`score`-producing family.

**Substrate atoms (flagged decision 3 — the name set):** `pearltrees`, `simplemind`,
`simplewiki`, `fs`. These are the corpora the landed code actually names; `enwiki` waits
until something uses it.

**Judge atoms:** `haiku`, `luna`, `sonnet`, `llm`, `opus`, `gemini`, `human`, `gpt-5.5-low`,
and `graph` — which becomes **judge-only**, its former substrate role taken by the real
corpora. Existing modifier sets carry over (`graph.discrim`, `llm.element`, …).

**Operators:**

| entry | signature | output |
|---|---|---|
| `product` | `(score, score, …)` variadic ≥2 | `score` |
| `max` | `(number\|score, score, …)` variadic ≥2 | `score` |
| `hop_decay` | `(substrate, gamma=number)` — gamma is retention: multiplies per hop | `score` |
| `lca_frac` | `(substrate)` | `score` |
| `lineage` | `(substrate, mu=judge, decay=number, depth=int)` — decay documented as retention (R3) | `target-set` |
| `blend`, `kalman`, `e5`, `routing`, `margin`, `distill` | as v0.3, `source` arg types re-typed `judge` | unchanged |

**New kwargs on mechanism-bearing operators:** `estimand=` and `impl=`, optional, enumerated,
fail-closed on unknown values (§16.15 posture):

```text
ESTIMANDS = { subcategory, super_category, element_of, subtopic,
              see_also, assoc, bridge, ancestry, path }
IMPLS     = { structural, attention }        # only implementations that exist (R5)
```

`hop_targets` appears in neither: it names a procedure, and the term is the procedure (R7).

### 10.3 Identity split (R9)

`canonical()` gains a semantic/full split: **`canonical_semantic`** strips pins and is the
digest preimage for `ast_sha`, `full_ast_digest`, seals, and cache keys;
**`canonical_full`** retains pins for provenance and round-trip. Test obligation 10 checks
that the split is real: adding a pin moves `canonical_full` and leaves the semantic digest
byte-identical.

### 10.4 Registered processes under v0.4

The eight surviving processes re-spell unchanged (their judge arguments simply re-type).
`lineage-graph` is retired per §10.1. Two additions exercise the new vocabulary:
`graph-judge` (the §10.1 spelling, estimand `path`) and `lineage-haiku` =
`lineage(pearltrees, mu=haiku)` (R4's own example, estimand `ancestry`).
