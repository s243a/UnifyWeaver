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

## The scoping lesson (why the raw graph is not a subtree)

The *unbounded* "Physics-rooted" graph is not a clean subtree at all: in real Wikipedia, Physics's
descendant cone reaches **most of the encyclopedia** within a few hops via cross-listings (at
`10k`, **7811 of 8247** nodes "reach Physics"). Two symptoms on the raw graph:

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

## Takeaways

1. **Scope first.** A nominal "X-rooted" Wikipedia crawl is not a subtree of X — its cone is the
   whole encyclopedia. A bounded-depth descendant cone with induced edges is the actual single
   subtree, and it is the precondition for *any* of the read-outs (hubs, IC) to be meaningful.
2. **Bounded scoping also restores acyclicity**, so the DAG-only IC payloads (`descendant_minhash`
   → Resnik/Lin/FaITH) run without SCC-condensation. The depth knob is the §5c tightness lever made
   concrete: small depth = coherent + acyclic; large depth = the cross-listed, cyclic soup.
3. **The machinery is correct on real cyclic data** — all algebraic invariants held across four
   scales — and on the scoped subtree the similarity numbers track real physics relatedness.
4. **Global hub selection still needs semantics.** Even scoped, fan-in surfaces
   `Physicists_by_nationality` over `Subfields_of_physics` at `300`; the deferred diversity-based
   selection (§5d) is what would prefer the latter. Concrete motivation, now from clean data.
