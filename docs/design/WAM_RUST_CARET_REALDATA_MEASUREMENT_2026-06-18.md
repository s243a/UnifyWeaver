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
- **The single-anchor version agrees too: depth-to-root vs `μ`.** Distance to the `Physics` root is
  the *one-legged caret* `caret(u, Physics) = d(u→Physics)` (the root is a universal ancestor, so one
  leg is zero). Statistically it anti-correlates with `μ`: there are more deep nodes than shallow,
  and each downward hop has a chance to leak out of the semantic space, so deeper ⇒ more likely
  non-physics (a few deep-but-genuine physics nodes buck it). Measured: high-`μ`(≥0.8) depth **2.24**,
  low-`μ`(≤0.3) depth **3.86**, **`corr(μ, depth) = −0.48`**. So the cheap single-anchor signal
  agrees with the multi-anchor mean-caret and with the LLM — "either way, similar results." (Note
  this is the *robust* use of depth: the per-pair caret to a chosen anchor, not the raw root-depth
  ranking that the associative leak makes unreliable — a leaked chemistry node is deep via a long
  associative path, which is exactly why deep ⇒ less-physics holds *statistically* here.)
- **Generating `μ` from semantic vectors** (category embeddings) instead of an LLM is the natural
  alternative — same fuzzy membership, a different prior source; the structural agreement above
  predicts it would land in the same place. (Future: needs an embedding source.)
- **Gated hybrid `μ` — cheap prior, LLM only on the close band** (`wikipedia_fuzzy_gated_hybrid_
  membership`). The cost-optimal design: a *cheap* prior (a batched embedding-similarity `μ`, or —
  as a stand-in here — the depth-to-root score) decides the confident nodes outright; the expensive
  LLM is consulted **only in the "just-missed-but-close" band** `[τ_lo, τ_hi)` below the prior's
  threshold, and fused there by the **geometric mean** `√(prior·μ_llm)` (which hard-vetoes if either
  signal is ~0). Two thresholds: one on the prior, one on the geo-mean. Measured with the *weak*
  depth stand-in prior on the 90-node fixture: **29 accept-on-prior, 23 reject-on-prior, only 38
  consult the LLM — 58% of the LLM calls saved**; a stronger embedding prior would widen the
  confident bands and shrink the consulted middle further. This is the same "cheap signal broadly,
  expensive signal only where uncertain" pattern as the reconstruction gate's Monte-Carlo fallback.
- **The gate is tier-agnostic — it can be a *model cascade*** (`wikipedia_model_cascade_haiku_then_
  sonnet`). The two stages need not be embedding+LLM: a cheap *model* (Haiku) handles the bulk and a
  strong *model* (Sonnet) is invoked **only on the cheap model's uncertain band** (the CLOSED band
  `μ∈[0.3,0.7]`, endpoints escalated — this matches the 26-node Sonnet fixture exactly; the open band
  would drop the `μ=0.3/0.7` endpoints and under-count to 19) — an `n`-tier cascade in the limit.
  Measured: Haiku decides **64/90 (71%)** outright; the **26** escalated nodes go to Sonnet, which is
  markedly more discriminating (band score spread `σ`: Haiku `0.117` → Sonnet `0.184`) and **resolves
  16 of the 26** decisively — un-clustering Haiku's `0.5`
  pile (pure chemistry `Arsenic 0.1`, `Pnictogens 0.05`; engineering `Electric_vehicles 0.1`; vs
  physics-adjacent `Astronomical_objects 0.7`, `Electronics 0.55`).
- **Batch-size caveat on the savings.** The escalation only pays off if the escalated band is large
  enough to amortize the *sunk cost* of a call — the system prompt + invocation overhead. Rule of
  thumb: a batch should be **at least as large as the fixed (system-prompt) cost**, so the fixed
  overhead is ≤ half the call; ideally `band ≫ sunk_cost` so the marginal per-item cost dominates.
  A gate that escalates *too few* items per batch spends mostly fixed cost on each escalation and
  erodes its own win — so the band, the batch size, and the tier-cost ratio should be sized together.
- **What `μ` *means*: disciplinary vs foundational membership.** The physics/chemistry boundary is
  not just fuzzy, it is *hierarchical* — chemistry is **founded on** physics (electron bands → band
  theory; p-bonding → molecular orbitals → quantum mechanics), so `Noble_gas_compounds` is
  *chemistry by discipline* yet *physics by foundation*. Those are two distinct membership notions:
  **disciplinary** ("is this studied as / a physics topic") and **foundational** ("does this rest on
  physics principles"). The rubric here asked for the *disciplinary* sense, so the strong tier scored
  chemistry low (`Arsenic 0.1`); a *foundational* rubric would score the same nodes higher. Crucially
  **neither current signal captures the foundational dimension**: the LLM (as prompted) measured
  discipline, and the graph *depth* measures taxonomic distance (chemistry sits deep in the cone, so
  it reads "far"), which also misses the foundation. This is a strong argument for the **embedding-`μ`
  ** — semantic embeddings reflect *conceptual* similarity (`electron bands` ≈ band theory ≈ physics),
  capturing the foundational relationship that both the disciplinary-LLM *and* the taxonomic-depth
  miss. So embeddings are not merely a cheaper classifier; they measure a foundationally-relevant
  facet the other two signals cannot. The fuzzy framework is dimension-agnostic: *which* `μ` you want
  is a choice of prompt (or reference vector) for the application. So the **discrimination context is
  a first-class, user-supplied prompt parameter** — "discriminate by discipline / by foundation /
  by …" — chosen per the user's preference and need; the graph algorithms downstream are
  dimension-neutral (they consume a `μ` fixture without caring what `μ` means). The one practical
  consequence: a fixture should **record the discrimination context it was scored under** (a header
  field), so a `μ` value is interpretable — the existing fixtures are disciplinary-leaning.
- **When the discrimination actually matters: per-pair vs global.** How strictly we draw the
  physics/chemistry line is *not critical* for the **per-pair** algorithms — the bidirectional caret
  and the IC similarity operate on specific chosen node pairs and work whether or not the membership
  is clean (the same robustness as the 3e "no curated cone needed" finding). The discrimination
  becomes **load-bearing only for *global* graph properties** — fan-in vs fan-out hub selection,
  where which nodes/edges count as in-region decides the convergence structure (recall raw fan-in on
  the full graph surfaced `Container_categories`; a `μ`-weighted fan-in would down-weight non-physics
  and surface the real physics hubs). So the fuzzy `μ` and the deferred **membership-weighted
  read-outs** (`μ`-weighted fan-in/fan-out, Resnik/Lin) are precisely the tool the *global*
  hub-selection problem needs — and the reason careful discrimination is worth the effort there but
  not in the per-pair path.

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
