# WAM-Rust Bridge Detector — Implementation Plan

*Build order for the detector specified in `WAM_RUST_BRIDGE_DETECTOR_SPECIFICATION.md` (philosophy:
`..._PHILOSOPHY.md`; theory: `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §5f–§5g). Each increment is its own
small PR, **additive** (no change to existing functions), `cargo test`-verified by rendering the
`boundary_cache.rs.mustache` template (only `{{date}}` is a real tag; render at **edition 2021**).*

## Where it goes

`templates/targets/rust_wam/boundary_cache.rs.mustache`, next to `descendant_mu_mass_gated` /
`gated_ic_node_filtered` / `cone_purity` / the sketch functions it reuses. No ML, no HF, no torch — the μ
map is an input, produced separately (the directional model #3302, or any dense `name→μ` map).

## Increment 1 — `fanout_diversity` (the foundational piece)

The `n_eff` core, exact first.

- Compute `U_P` from `descendant_mu_mass_gated(P)` (the path-gated reachable set).
- For each in-domain child `c_i` (`μ(c_i) ≥ θ`): `E_i^P = {nodes of U_P under c_i}`; `μ(E_i^P)` = its
  in-domain mass; accumulate `Σ μ(E_i^P)` and the union set `⋃ E_i^P`.
- `diversity = μ(⋃) / Σ μ(E_i^P)`; `n_eff = n · diversity`. `None` if `< 2` in-domain children.

**Tests** (exact, hand DAGs):
- disjoint branches (a tree fan-out) → `diversity = 1`, `n_eff = n`;
- `n` children all pointing at the *same* shared descendant set → `diversity ≈ 1/n`, `n_eff ≈ 1`;
- partial overlap → strictly between;
- `< 2` children / leaf → `None`.

Runnable and self-contained — this is the increment to do first.

## Increment 2 — purity + `bridge_classify`

Combine `n_eff` with `cone_purity` into the 2-D classification (`BridgeScore` / `BridgeClass`).

- `bridge_classify(P, params)` → `{n_eff, diversity, purity, fan_out, class}`.
- **Tests**: three synthetic nodes — a **bridge** (distinct *in-domain* branches: high `n_eff`, high
  purity), a **leak conduit** (distinct *out-of-domain* branches: high `n_eff`, low purity — give the
  branch subtrees `μ < θ`), and a **redundant hub** (overlapping branches: low `n_eff`). Assert each lands
  in the right quadrant.

## Increment 3 — candidate generation + `rank_bridges`

- Fan-out filter (`φ ≥ φ_min`) as the default candidate set; `rank_bridges` returns sorted
  `(node, BridgeScore)`.
- *(Optional, layered)* MICA-frequency candidates via `resnik_from_ic` over sampled distant pairs, if the
  fan-out filter over-generates.
- **Test**: ordering on a graph with a planted bridge, leak conduit, and hub — the bridge ranks above the
  hub; the leak conduit is labeled `LeakConduit` not `Bridge`.

## Increment 4 — scale path (sketches) + live wiring

- Swap the exact cone walk for the **sketch union read**: per-child bottom-`k` sketches restricted to
  `U_P`, merged, mass via `sketch_mu_mass`; cross-parent ranking via `sketch_mu_overlap_lift`. Assert the
  sketch `n_eff` matches the exact `n_eff` within the KMV tolerance on the test graphs (the same
  exact-vs-sketch assertion pattern the `gated_ic`/`resnik` tests already use).
- Live path: `WamState::build_bridge_scores` (precompute) + `category_bridge_score(p)` (per-query),
  mirroring `build_descendant_sketches`.

## Increment 5 — real-data validation (measurement doc)

Run on the physics category graph with a real μ map (the directional `SYM`/`LLM` map from #3302, or the
e5 dense map). Confirm the qualitative predictions and write them up as an addendum to
`WAM_RUST_CARET_REALDATA_MEASUREMENT_*.md`:

- known generic apexes (`Main_topic_classifications`, `Categories`) classify as **leak conduits** (high
  `n_eff`, low purity) — recovering the earlier leak-conduit finding from the *positive* side;
- genuine in-domain merge points score as **bridges**;
- redundant `X_by_country`-style fan-outs score as **redundant hubs**.

## Conventions

- Additive only — never modify `descendant_mu_mass_gated`, `gated_ic`, or the similarity functions.
- Exact functions for correctness tests, sketches for scale (and assert they agree).
- Edition 2021 when rendering for `cargo test` (the template hits an edition-2024 match-ergonomics error
  in a *pre-existing* test otherwise — not your code).
- Keep the μ map an **input**; do not couple the detector to any particular μ producer.
