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

## 2. The rulings

Six rulings, all issued by the owner on #4013. Reasoning is reproduced because a decision without
its reasoning gets re-litigated.

### R1 — Transitive μ is the quantity; hop decay is one estimator of it

*(finding 4 — the ruling the issue principally asked for)*

Transitive μ is graded element/subcategory relatedness computed over several hops. **Hop decay is
one estimator of it; an LLM judge is another.** The substrate, judge, and relation type jointly
define the methodology.

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
5. **Add `estimand=` and `impl=`** as separate fields with enumerated value sets, registering only
   implementations that exist (R5).
6. **Document `decay` as a retention factor**; optionally rename (R3).

Item 2 is the one with reach: eight of nine registered processes name a `source`-typed entry, so
the split touches nearly the whole registry rather than a corner of it.

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
| 8, 10 — expression under-determines the process | R5 | `estimand`/`impl` make the mechanism expressible |
| 9 — pairwise measures inexpressible | — | **not fixed**; deferred by R6 |

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

**`max` and product operators.** R6 declines the pairwise-atoms restructuring, which leaves the
graph judge's actual computation — `max(floor, gamma^hops * lca_frac)` — inexpressible. Adding
`max` and multiplication would let it be registered as *what it computes* rather than mislabelled,
closing finding 8 by description rather than by label. **This is an addition to `v0.4`'s scope
that no #4013 ruling requested, and deserves its own decision rather than riding one.**

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
   (`DESIGN_process_expression_patterns.md` §16.34).

Per the project convention, these land as tests rather than as prose.

## 9. Sequencing

```text
rulings issued (#4013)          done
this design note                done
v0.4 implementation + seal      not started  <- gates everything below
corpus enumeration              gated
encoder step 2 part 2           gated
```

Issue #4013's stated purpose — *"rulings needed before registry v0.4"* — is discharged. The
findings it recorded are not all fixed, so a separate implementation issue should carry findings
1, 8, 9 and 10 forward before it is closed.
