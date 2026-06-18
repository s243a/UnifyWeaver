# WAM-Rust caret / hub / similarity — real-data run (roadmap 3e), 2026-06-18

The end-to-end composition (§8 increment 3e of `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`) run on
**real Wikipedia category graphs**, and the lesson that you must **scope to a single bounded
subtree** before the read-outs mean anything.

## Setup

- **Data:** `data/benchmark/{dev,300,10k,10x}/category_parent.tsv` — `child<TAB>parent` category
  edges, nominally **Physics-rooted**, offline (live `en.wikipedia.org` egress is blocked in this
  environment). *Provenance / exact correctness not guaranteed — a demonstration on real-shaped
  data, not an authoritative measurement.*
- **Harness:** `boundary_cache::tests::wikipedia_category_subtree_end_to_end_3e`, env-var gated on
  `UW_CATEGORY_TSV` (skips in CI). Knobs `UW_CATEGORY_ROOT` (default `Physics`) and
  `UW_CATEGORY_MAXDEPTH` scope the graph to the **bounded-depth descendant cone of the root with
  induced edges** — a single coherent subtree.

## The data is not wrong — Wikipedia categories are not a taxonomy

A natural first reaction to "Physics's cone contains `States_of_the_United_States`" is *the data
must be corrupted*. It is not. Every step is a real row in the file and an individually-plausible
Wikipedia subcategory link; the absurdity is **transitive**:

```
Physics → Matter → Physical_objects → Organisms → … → Humans → Human_activities → … → Movies
Physics → Matter → Physical_objects → Astronomical_objects → … → Earth → … → States_of_the_United_States
Physics → Time → History
```

The load-bearing edge is `Organisms → Physical_objects` (Wikipedia really does file living things
under "physical objects"); once crossed, Physics's *subcategory* cone bleeds into all of biology →
humans → everything. This is the documented property that **Wikipedia's category graph is
associative, not is-a** — following "subcategory" links transitively, almost everything is a
"subcategory" of almost everything within a handful of hops. A live fetch would show the same
leaks (and is blocked here anyway). So a clean *downward* "subtree of Physics" **cannot be cut
structurally** — only with semantic filtering.

## The resolution: the per-pair bidirectional bridge needs no curated cone

The leak is a property of the *downward* cone. The per-pair caret/bridge metrics are
**bidirectional and bounded** — they explore only the *up-cones* of the two chosen nodes and find
where *their* lineages mix — so the downward leak never bites. Pick two nodes that *should* be
related, bound the ancestor space (`caret_optimal_bridge(u, v, budget=10)`), and read off their
real lowest common ancestor — **directly on the raw, uncurated, cyclic graph**:

| pair | optimal bridge (raw graph, budget 10) | caret |
|------|---------------------------------------|-------|
| `Classical_mechanics` – `Electromagnetism` | **`Subfields_of_physics`** | 2 |
| `Thermodynamics` – `Optics` | **`Subfields_of_physics`** | 3 |
| `Quantum_mechanics` – `Classical_mechanics` | **`Subfields_of_physics`** | 2 |
| `Electromagnetism` – `Optics` | **`Electromagnetism`** | 1 |

The bridges are semantically correct and **stable across all scales** (dev / 300 / 10k give the
same bridges), because the bounded up-search from two physics topics recovers their genuine common
ancestor regardless of the downward mess. `Electromagnetism`–`Optics` meeting at `Electromagnetism`
itself (caret 1) is exactly right — optics is filed under electromagnetism. **This is the honest
way to use the metric on real Wikipedia: per-pair, bidirectional, bounded — not a curated cone.**

## Scoping a downward cone — still useful for the DAG-only payloads, but only a band-aid

The *global* / *downward* read-outs (fan-in hubs, the descendant-sketch IC) still need an acyclic,
semantically-coherent graph, and for those a bounded-depth cone helps — with the caveat that it
only *partially* cleans the leak (it stays physics-ish to depth ~3, then the cross-listings take
over).

The *unbounded* "Physics-rooted" graph is not a clean subtree at all: Physics's descendant cone
reaches **most of the encyclopedia** within a few hops (at `10k`, **7811 of 8247** nodes "reach
Physics"). Two symptoms on the raw graph:

- **It is cyclic** (≈5 back-edges at `dev`, ≈20 at `10k`), so `descendant_minhash` returns `None`
  and IC similarity is unavailable without SCC-condensation.
- **Fan-in hub selection is dominated by *maintenance* categories.** Top fan-in at `10k`:
  `Container_categories` (**1778** children), `CatAutoTOC_generates_no_TOC` (1219),
  `Navseasoncats_year_and_decade` (691) — bookkeeping, not topics. Routing carets through them
  inflates the hub-quantized caret far above exact (`Electromagnetism`–`Optics`: exact **1**,
  hub-quantized **7**).

**Scoping to a bounded-depth subtree fixes both at once.** Depth ≤ 3 from `Physics` keeps the
genuine physics neighbourhood (deeper, the cross-listings into temporal/organizational categories
take over — `Categories_by_decade` etc.), and that subtree turns out **acyclic**, so IC runs:

| scale | raw nodes | scoped (depth≤3) | scoped edges | back-edges |
|-------|-----------|------------------|--------------|-----------|
| dev   | 121       | 29               | 35           | 0         |
| 10k   | 8247      | 74               | 88           | 0         |
| 300   | 2276      | 107              | 130          | 0         |

## Invariants confirmed (scoped and raw, all scales)

- `caret_distance_lca_boundary == caret_distance_lca`
- `caret_min_over_cached_bridges == caret_min_over_hubs` (3f)
- `min_distance_closure` terminates, roots at 0 — the 2a/2b cycle-correctness earns its keep on
  the cyclic raw graph.

## Read-outs on the clean (scoped) subtree

**Hubs become semantic.** Depth ≤ 3 top fan-in: `Subfields_of_physics`, `Matter`, `Mechanics`,
`Energy` (dev); `Physicists_by_nationality`, `Subfields_of_physics`, `Mechanics` (300). The
quantization gap closes (hub-quantized caret == exact for the close physics pairs), because the
hubs are now the real convergence points.

**IC similarity runs and is physically sensible** (`k = 64`, scoped subtree acyclic). At `10k`,
depth ≤ 3:

| pair | Resnik | Lin | FaITH |
|------|--------|-----|-------|
| `Electromagnetism` – `Optics` | 3.21 | **0.68** | **0.52** |
| `Classical_mechanics` – `Electromagnetism` | 1.82 | 0.46 | 0.30 |
| `Thermodynamics` – `Optics` | 1.82 | 0.36 | 0.22 |

The ordering is meaningful: `Electromagnetism`–`Optics` (optics *is* a part of electromagnetism)
scores far higher than `Thermodynamics`–`Optics` (weakly related). The exact per-pair carets agree
(`Electromagnetism`–`Optics` = 1, the closest; `Thermodynamics`–`Optics` = 3). So on real data the
relatedness read-outs track genuine physics structure.

## LLM-curated test set: the external semantic signal — and the metric as its audit

The graph alone can't tell physics from not-physics (that's the whole leak). So bring the semantic
signal from *outside*: random walks down the category graph surface candidate nodes
(`scripts/physics_random_walk_candidates.py`, seeded), and a **Haiku subagent** classifies which
are genuine physics topics — saved as the reusable fixture
`tests/fixtures/wikipedia_physics_curated_nodes.txt` (46 nodes). Then
`wikipedia_physics_curated_set_bridges` computes the per-pair `caret_optimal_bridge` across the
whole curated set on the **raw 10k graph** (1035 pairs, all related within budget 10). Two results:

- **The recurring bridges are the *real* semantic hubs** — `Physics` (215 pairs),
  `Subfields_of_physics` (169), `Natural_sciences` (160), `Matter`, `Energy`. This is what fan-in
  *failed* to find (it surfaced `Container_categories`): curate the **nodes** semantically, let the
  bidirectional metric find their bridges, and the genuine hubs emerge. The only structural-noise
  leftover (`CatAutoTOC_generates_no_TOC`) ranks below all the physics hubs.
- **The metric audits the classifier.** Mean optimal-bridge caret to the rest of the set separates
  the physics *centre* (`Subfields_of_physics` 3.27, `Matter` 3.47, `Energy` 3.64,
  `Electromagnetism`, `Classical_mechanics`) from the *outliers* — `Nitrogen` (6.78),
  `Hydrogen_compounds`, `Hydrogen`, `Chalcogens`, `Oxygen`: exactly the **chemistry** nodes Haiku
  over-included. The LLM supplies the semantic signal the graph lacks; the graph distance in turn
  flags the LLM's borderline calls. They are complementary — neither alone is enough, together they
  are a clean, self-checking pipeline for building a semantic test set on a non-taxonomic graph.

## Fuzzy / graded membership (implemented)

Binary keep/discard is lossy at the boundary — `Fire` (combustion → thermodynamics) and `Nitrogen`
(physical chemistry) are not physics *topics* yet are not unrelated either. So the classifier now
returns a **graded physics-relevance score** `μ ∈ [0,1]` per node (a fuzzy-set membership,
Zadeh 1965) instead of a bit — `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (Haiku-scored),
exercised by `wikipedia_fuzzy_membership_threshold_and_fusion`.

- **The threshold knob works.** On the 90-node fixture: `|μ≥0.8| = 25` (core) → `|μ≥0.5| = 42` →
  `|μ≥0.4| = 48` — a strict cut keeps the core, a loose cut pulls in the chemistry/combustion
  boundary. `Fire` lands at `μ = 0.5` (loose-in, strict-out), exactly the graded boundary case.
  This is the same tightness-vs-reuse knob as §5c (budget, quantization) and §7 (hard vs soft
  distance), now on the *membership* side.
- **The two signals fuse, and they agree.** The LLM `μ` is the *semantic* prior; the graph's
  **mean optimal-bridge caret to the physics core** (a few canonical `μ=1.0` anchors) is the
  *structural* signal. They correlate: high-`μ` (≥0.8) nodes sit at mean caret-to-core **3.02**,
  low-`μ` (≤0.3) at **4.77** — so each can audit the other. The disagreements are the interesting
  cases the fusion surfaces (e.g. `Sound`, `Waves`, `Cold` read `μ=1.0` but sit slightly farther
  structurally — genuine physics the graph places at a remove).

Still future work: **membership-weighted read-outs** — carry `μ` as a per-node weight into the
functionals (`μ`-weighted Resnik/Lin that down-weights borderline ancestors in the MICA search, or
a `μ`-thresholded boundary so a query "stays in physics with tolerance `τ`"). That is the
graph-functional-semiring move applied to a *soft* node set rather than a hard one.

## Takeaways

1. **The data is real, not wrong — Wikipedia categories are associative, not is-a**, so a clean
   *downward* "subtree of X" cannot be cut structurally; the transitive leak (`Organisms →
   Physical_objects → …`) is genuine Wikipedia.
2. **The per-pair bidirectional bridge is the robust answer** — `caret_optimal_bridge` (bounded
   up-search, no curated cone) recovers semantically-correct lowest common ancestors
   (`Subfields_of_physics`, `Electromagnetism`) for related pairs, **directly on the raw, cyclic
   graph**, stable across all scales. This is how the metric should be used on real Wikipedia:
   pick two nodes that should be related, bound the ancestor space, read the bridge.
3. **Downward/global read-outs (fan-in hubs, descendant-sketch IC) need an acyclic, coherent
   graph**, for which a bounded-depth cone is a useful *band-aid* (acyclic + physics-ish to depth
   ~3), but only that — it cannot fully clean the associative leak. On the scoped subtree the IC
   numbers still track real physics (`Electromagnetism`–`Optics` Lin 0.68 ≫ `Thermodynamics`–
   `Optics` 0.36).
4. **The machinery is correct on real cyclic data** — all algebraic invariants held across four
   scales (boundary == full caret, cached-landmark == per-query, cycle-correct termination).
5. **Global hub selection still needs semantics.** Even scoped, fan-in surfaces
   `Physicists_by_nationality` over `Subfields_of_physics`; the associative-leak finding is the
   deeper reason structure alone can't pick good *global* bridges — the deferred diversity-based
   selection (§5d) is what would. Per-pair bridges, by contrast, are already good *without* it.
