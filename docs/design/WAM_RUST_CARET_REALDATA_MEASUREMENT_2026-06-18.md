# WAM-Rust caret / hub / similarity — real-data run (roadmap 3e), 2026-06-18

The end-to-end composition (§8 increment 3e of `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`) run on
**real, cyclic, cross-listed Wikipedia category graphs**, at four scales.

## What was run

- **Data:** `data/benchmark/{dev,300,10k,10x}/category_parent.tsv` — `child<TAB>parent` category
  edges, all **Physics-rooted**, offline (no network). Genuinely **cyclic** (≈5 back-edges at
  `dev`, ≈20 at `10k`) and cross-listed (e.g. `Systems` under 4 parents). *Provenance / exact
  correctness is not guaranteed by the maintainer — this is a demonstration on real-shaped data,
  not an authoritative measurement.*
- **Harness:** `boundary_cache::tests::wikipedia_category_subtree_end_to_end_3e` (env-var gated on
  `UW_CATEGORY_TSV`, so it skips in CI and runs on demand). It interns the category names, builds
  the parent map, and exercises the full stack: `min_distance_closure` (to root),
  `caret_distance_lca` vs `caret_distance_lca_boundary`, `convergence_fanin` (hub ranking),
  `bridge_distance_fields` + `caret_min_over_cached_bridges` (3f landmarks), and
  `descendant_minhash` / `lin_similarity`.

| scale | nodes | edges | reach root | back-edges |
|-------|-------|-------|-----------|-----------|
| dev   | 121   | 198   | 108       | ~5        |
| 10x   | 1593  | 3932  | 1512      | —         |
| 300   | 2276  | 6008  | 2165      | —         |
| 10k   | 8247  | 25227 | 7811      | ~20       |

## Invariants confirmed on real cyclic data (the correctness story)

Across **all four scales**, with the graph cyclic:

- **`caret_distance_lca_boundary` == `caret_distance_lca`** for every query pair — the
  boundary-restricted search gives the same answer as full-cone on real data.
- **`caret_min_over_cached_bridges` == `caret_min_over_hubs`** — the 3f landmark cache equals the
  per-query path on real data.
- **`min_distance_closure` terminates** and roots at distance 0 — the cycle-correctness (2a/2b)
  earns its keep: the BFS-fixpoint / joint-up-BFS functions all terminate where a naive DFS
  recurrence would loop.

## Finding 1 — the cycle/DAG split shows up immediately

`descendant_minhash` returned **`None` at every scale** — the real category graph is cyclic, so
the *sketch-based* payloads (descendant sketch, `descendant_weight`, `convergence_jump`, hence IC
/ Resnik / Lin / FaITH) are unavailable on the raw graph and need **SCC-condensation first**, the
documented prescription. Meanwhile the **cycle-robust** payloads — `convergence_fanin`, the caret
family, `bridge_distance_fields` — ran directly. This is exactly the cycle-robust-vs-DAG-only
distinction the code carries, now observed on real data rather than a synthetic cycle.

## Finding 2 (headline) — naive fan-in hub selection is dominated by *maintenance* categories at scale

The top fan-in "hubs" diverge sharply by scale:

- **dev (clean, 121 nodes):** `Subfields_of_physics` (10), `Subfields_by_academic_discipline` (7),
  `Scientific_disciplines` (6), … — *semantic* categories, genuinely good bridges. Here the
  **hub-quantized caret equals the exact caret** (e.g. `caret(Classical_mechanics,
  Electromagnetism) = 2`, hub-quantized `= 2`): the high-fan-in nodes *are* the real
  convergence points.
- **10k (8247 nodes):** `Container_categories` (**1778** children), `CatAutoTOC_generates_no_TOC`
  (1219), `Navseasoncats_year_and_decade` (691), `Navseasoncats_decade_and_century`, … — these
  are Wikipedia **bookkeeping / navigation** categories, not topics. They have enormous fan-in but
  are **meaningless bridges**. Routing carets through them inflates the hub-quantized caret far
  above the exact one (`caret(Electromagnetism, Optics) = 1` but hub-quantized `= 7`;
  `caret(Classical_mechanics, Electromagnetism) = 2` vs `6`).

The exact per-pair carets stayed small and sensible at every scale (`Electromagnetism`–`Optics`
= 1; `Classical_mechanics`–`Electromagnetism` = 2; `Thermodynamics`–`Optics` = 3) — the
**per-pair boundary caret is unaffected**; only the *globally-selected-hub* approximation
degrades.

### Why this matters

This is concrete real-data evidence for the still-open **global hub-selection** problem
(§5d, §8). Pure structural **fan-in is fooled by administrative categories** — `Container_categories`
maximizes fan-in while carrying no semantic relatedness — which is precisely the failure the
**semantic-diversity** signal (the geometric-mean-of-singular-values conjecture, §5d) or even a
simple maintenance-category filter is meant to fix. The §5c tightness-vs-reuse tradeoff is no
longer abstract: at scale, the cheapest selector (fan-in) picks bridges so general (or so
administrative) that the quantization gap `2·d(LCA→nearest hub)` swamps the signal. The per-pair
mixing-boundary search remains the reliable answer; the global hub set needs a semantic filter to
be useful.

## Takeaways

1. The caret / landmark / cycle-correct-distance machinery is **correct on real cyclic data** —
   all algebraic invariants held across 4 scales up to 25k edges.
2. The **cycle-robust vs DAG-only** payload split is real and load-bearing: IC similarity needs
   SCC-condensation on the (cyclic) category graph; the caret stack does not.
3. **Global hub selection needs semantics, not just structure** — naive fan-in surfaces
   `Container_categories`, a vivid argument for the deferred diversity-based selection. The
   immediate cheap fix (filter maintenance categories) and the principled one (semantic diversity)
   are both now motivated by data.
