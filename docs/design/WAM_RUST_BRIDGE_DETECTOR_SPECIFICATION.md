# WAM-Rust Bridge Detector — Specification

*Precise definitions, the API, and the invariants. The **why** is in
`WAM_RUST_BRIDGE_DETECTOR_PHILOSOPHY.md`; the **theory** in `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`
§5f–§5g; the **build order** in `WAM_RUST_BRIDGE_DETECTOR_IMPLEMENTATION_PLAN.md`.*

## Inputs

- `parents: HashMap<u32, Vec<u32>>` (child → parents) and the derived `children` map — the DAG.
- `mu: HashMap<u32, f64>` — membership, `μ ≥ 0`. The dense μ map: directional `μ(·|root)` from the
  attention model (#3302), the symmetric map, or any `name→μ` map fed through the usual loaders.
- `threshold θ` — the gating threshold (default `0.3`).
- For scale: the descendant sketches (`descendant_minhash_weighted`). For correctness tests on small
  graphs: the exact `descendant_mu_mass*` functions.

## Definitions

For a candidate node `P` with in-domain children `c_1 … c_n` (children with `μ(c_i) ≥ θ`):

- **Universe** `U_P` = `P`'s **path-gated** cone — the reachable set of `descendant_mu_mass_gated(P)` (the
  membership frontier: "what `P` organizes"). *Path-gated on purpose* — the universe is what `P` reaches
  while staying in-domain. (Contrast the *similarity* IC of §5f, which uses the node-gated cone.)
- **Child region** `E_i^P` = `desc(c_i) ∩ U_P` — child `c_i`'s contribution inside `P`'s frame. Concretely
  the nodes of `U_P` that lie under `c_i`. A child whose subtree leaves the domain contributes only its
  in-domain part; a low-μ child contributes `∅`.
- **Diversity** `diversity(P)` = `μ(⋃_i E_i^P) / Σ_i μ(E_i^P)` ∈ `[1/n, 1]`. By inclusion–exclusion the
  union ≤ the sum, with equality iff the child regions are disjoint; the shortfall is exactly the
  **reconvergence** (mass shared across sibling subtrees).
- **Effective branches** `n_eff(P)` = `n · diversity(P)` — equals `n` for disjoint branches, collapses to
  `1` for `n` identical ones. *(Mass-weighted variant: replace the count `n` by the in-domain child-mass
  share if branch* importance *matters more than branch* count.)*
- **Purity** `purity(P)` — domain coherence of `P`'s cone, measured **in the same frame as `n_eff`** (the
  in-domain cone), **not** the raw cone.
  - *The original raw-cone read* `cone_purity(P) = m_μ(desc(P)) / |desc(P)|` *conflates leaky with
    big/general* and must **not** be used as the classifier's purity: #3306's real-data run showed it
    mislabels genuine in-domain hubs (`Subfields_of_physics` 0.019, `Energy` 0.014) as `LeakConduit`,
    because a large raw cone accumulates out-of-domain nodes that dilute the average regardless of
    coherence.
  - *Use an in-domain-frame read.* Primary candidate — the **in-domain fraction**
    `purity(P) = |gated_desc(P)| / |desc(P)|`: high when most of the cone is in-domain (a coherent hub),
    low for a leak conduit whose cone reaches out-of-domain junk. Alternative — the gated average-μ
    `m_μ(gated_desc(P)) / |gated_desc(P)|`. **Pick the cleaner separator by the increment-5 validation**
    (the requirement: `Subfields_of_physics`/`Energy` → high, `Matter` → low). `descendant_mu_mass_gated`
    supplies the gated cone; `cone_purity` (raw) remains available but is *not* the classifier input.
- **Fan-out** `φ(P)` = the in-domain child count `n`.

## Classification

Parameters `φ_min` (min fan-out), `τ_div` (diversity / `n_eff`), `τ_pure` (purity). Report the continuous
`(n_eff, purity)` for ranking; the labels are a convenience cut:

- `NotCandidate` — `φ(P) < φ_min` or `< 2` in-domain children (not a branch point).
- `RedundantHub` — `n_eff(P)` low (children overlap) — candidate for *collapse*.
- `Bridge` — `n_eff(P)` high **and** `purity(P) ≥ τ_pure` — organizes distinct in-domain subfields.
- `LeakConduit` — `n_eff(P)` high **and** `purity(P) < τ_pure` — distinct but *unrelated* regions → prune /
  down-weight.

**`τ_pure` is self-calibrating, not a fixed constant.** The in-domain-fraction purity has no universal
scale: on a clean graph genuine hubs sit near `1`, but on an associative graph (the physics-exploded 10k
slice) *every* cone leaks, so even real hubs read `~0.01–0.04` and leak conduits `~0.002`. A hard-coded
`τ_pure` therefore mislabels on a graph of a different scale. Instead `rank_bridges` **derives `τ_pure`
per graph** from the candidate purity distribution: the purities are log-bimodal — a low **leak cluster**
and a higher **hub cluster** — so it takes the **largest log-gap whose low side is a minority** (leaks are
the minority low tail; the minority constraint rejects a high-outlier gap above the hub cluster that would
otherwise label most candidates as leaks) and sets `τ_pure` to the **geometric mean** of the two
bracketing purities. A bimodality test (the split gap must clearly exceed the typical gap) falls back to a
low **percentile** for the rare unimodal (no-leak-cluster) case. The classifier thus ports across graphs
of any absolute purity scale with no constant baked in. `BridgeParams.tau_pure` is `Option<f64>`: `None`
(default) self-calibrates; `Some(x)` pins an explicit cut (deterministic — for tests or a hand-tuned
override). Single-node `bridge_classify` cannot self-calibrate (it sees no distribution) and uses `0.5`
when `None`; use `rank_bridges` for the calibrated split.

## Candidate generation (the MICA / upward step)

Two options, cheap-first:

- **Fan-out filter** (default): every node with `φ(P) ≥ φ_min`. Cheap, a good first cut.
- **MICA-frequency** (principled, optional): sample distant pairs and count how often `P` is their MICA
  (`resnik_from_ic`); a frequent merge point is a stronger candidate than raw degree. Heavier; layer it on
  only if the fan-out filter over-generates.

## API (Rust, additive — no change to existing functions)

```rust
pub enum BridgeClass { Bridge, LeakConduit, RedundantHub, NotCandidate }

pub struct BridgeScore { pub n_eff: f64, pub diversity: f64, pub purity: f64,
                         pub fan_out: u32, pub class: BridgeClass }

/// φ_min, τ_div, and τ_pure: Option<f64> — None = self-calibrate (default), Some(x) = explicit cut.
pub struct BridgeParams { pub phi_min: u32, pub tau_div: f64, pub tau_pure: Option<f64> }

/// (diversity, n_eff) for P over its in-domain children; None if < 2 in-domain children or U_P empty.
pub fn fanout_diversity(p: u32, parents: &.., mu: &.., threshold: f64 /*, sketches or exact*/)
    -> Option<(f64, f64)>;

pub fn bridge_classify(p: u32, parents: &.., mu: &.., threshold: f64, params: &BridgeParams)
    -> BridgeScore;

/// Derive τ_pure from a candidate purity distribution (largest minority-low log-gap; geometric-mean
/// boundary; low-percentile fallback when unimodal). The self-calibration `rank_bridges` uses.
pub fn calibrate_tau_pure(purities: &[f64]) -> f64;

/// All candidates, sorted (e.g. by n_eff·purity, or n_eff within the Bridge class). With
/// `params.tau_pure = None` it self-calibrates τ_pure from the diverse candidates' purities.
pub fn rank_bridges(parents: &.., mu: &.., threshold: f64, params: &BridgeParams)
    -> Vec<(u32, BridgeScore)>;
```

Live path (mirroring `build_descendant_sketches` / `build_boundary_distances`):
`WamState::build_bridge_scores` precomputes once; `category_bridge_score(p)` answers per-node; `None`
until built, eager-edge only.

**Cross-parent ranking note.** Raw `diversity`/`n_eff` are computed inside each `P`'s own universe `U_P`,
so they are comparable *as ratios* but their universes differ in size. To rank bridges *across* parents on
an absolute footing, use the configuration-model **lift** (`sketch_mu_overlap_lift`, shared mass vs the
`m_u·m_v/|U_P|` null) — the same normalization the fan-in hub work uses — rather than comparing raw union
masses.

## Invariants & edge cases

- **Tree-triviality.** In a pure tree, siblings have disjoint subtrees, so `diversity ≡ 1` and
  `n_eff ≡ n`: *every* node looks like a perfect fan-out. The measure only discriminates on a **DAG**
  (where it detects reconvergence). Document it; do not error. (Wikipedia categories are a DAG, so this is
  fine here.)
- **Leaves / `< 2` children** → `NotCandidate` (`fanout_diversity` returns `None`).
- **`P` out-of-domain** (empty `U_P`) → `None`.
- **Determinism / approximation.** With sketches, fix the sketch seed; `diversity`/`n_eff` carry the KMV
  error bounds of `sketch_mu_mass` (exact `descendant_mu_mass*` for the correctness tests, sketches for
  scale).
- **Guards** (the usual): `μ ≥ 0` (it is summed as mass); category names must resolve verbatim or they
  silently become `μ = 0`.

## Reuses (no new math)

`descendant_minhash_weighted`, `sketch_mu_mass`, `sketch_mu_overlap`, `sketch_mu_overlap_lift`
(cross-parent ranking), `descendant_mu_mass_gated` (`U_P`), `cone_purity`, `resnik_from_ic` (MICA
candidates). The one new sketch *operation* is a **union read** — merge the per-child bottom-`k` sketches
(restricted to `U_P`), take the bottom-`k` of the union, read its mass with `sketch_mu_mass`. KMV union is
standard (bottom-`k` of a union of bottom-`k` sketches is exact for the union's bottom-`k`).
