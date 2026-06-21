# WAM-Rust bridge-informed clustering + declaration — design

Two goals, from the request: **(1)** use the now-complete bridge detector to drive graph **clustering**,
and **(2)** improve the Prolog **declaration** so transpiled Rust does bridge-detection + clustering
together. Theory context: `WAM_RUST_BRIDGE_DETECTOR_*.md` and `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md`
§5g.

## A. How bridges aid clustering (the algorithm)

The bridge classifier's three classes **are** partition decisions:

- **LeakConduit** = inter-cluster **glue** (it connects genuinely *unrelated* regions). **Cut it** — and
  the graph falls apart into its real clusters.
- **RedundantHub** = intra-cluster (its children *overlap*/reconverge) → **keep its region whole** (one
  cluster).
- **Bridge** = intra-cluster **organizer** (distinct in-domain sub-branches) → a **sub-cluster boundary**:
  split per child-branch.

So **bridge-informed clustering**:
1. `build_bridge_scores` (already exists, merged).
2. **Top-level clusters** = connected components of the graph with `LeakConduit` nodes (and their edges)
   removed — the glue is cut, the components are the domains.
3. **Hierarchy within a cluster** = recursively split at `Bridge` nodes (each bridge's child-branches are
   sub-clusters), using `n_eff` as the split-quality / stop criterion.

**This is the structural twin of the `J = D/(1+H)` hierarchy objective** (the Python
`mst_folder_grouping.py` / `skill_hierarchy_objective.md`): splitting at high-`n_eff` bridges = high-`H`
*informative* splits; keeping redundant-hub regions together = low-`D` *tight* clusters. So `n_eff` is the
cheap **structural estimate of the objective's `H`** (§5g), and the partition is a structural minimizer of
`J` — bridge detection and the hierarchy objective are computing the same "is this a good split?" signal,
one via cone overlap, one via semantic entropy. That makes the WAM-Rust clustering a fast structural
*prior* the Python entropy-based tools can refine or cross-validate against.

## B. Current state (the gap — verified)

Mapping the WAM-Rust target found:
- The bridge detector + similarity (`build_bridge_scores`, `category_bridge_score`, `rank_bridges`,
  `category_resnik`/`lin`/`faith`, `build_descendant_sketches`) exist as **Rust `WamState` methods**
  (`templates/targets/rust_wam/state.rs.mustache`) but are **NOT exposed to Prolog** — there are *no*
  `foreign_predicate` declarations for them. They're internal-only (e.g. `build_caret_landmarks` already
  consumes `build_bridge_scores` internally — bridges → caret landmarks, §5c use case 2).
- **No clustering** exists on the WAM-Rust side at all (the Python MST tool is not mirrored).
- The exposure mechanism is **4 sites per predicate**:
  (A) the `WamState` method (`state.rs.mustache`);
  (B) a `foreign_predicate(Pred/Arity, [setup_ops], _)` fact (declares `native_kind` + string/usize/f64
  configs);
  (C) a `"native_kind" =>` dispatch arm in `execute_foreign_predicate` (`wam_rust_target.pl`) that pulls
  args from registers, reads configs, calls the method, packs the result via `finish_foreign_results`;
  (D) auto-registration.
  Exposed graph predicates today: `category_ancestor`, `category_ancestor_boundary`,
  `bidirectional_ancestor`.

So "declare bridge + clustering" = **expose** the existing detector + **add** a clustering primitive +
**(C)** make the declaration ergonomic.

## C. Improving the declaration (the build-dependency pipeline)

The friction: graph-analysis predicates have a **build → query dependency**
(`edge graph + μ → descendant sketches → bridges → clusters`), and today an author manually orders the
build calls *and* hand-wires 4 sites per query predicate. The fix is a **single declarative pipeline** that
captures the dependency and generates the wiring. The author writes one declaration:

```prolog
:- graph_analysis(edge_parent,                       % the edge predicate (parents)
     [ mu(category_mu), threshold(0.3) ],            % inputs
     [ sketches(k=32),                               % build stages, ordered by dependency:
       bridges,                                       %   needs sketches + mu
       clusters(method=bridge_partition) ],          %   needs bridges
     [ category_bridge_score/2,                      % exposed QUERY predicates:
       category_cluster/2,                           %   node → cluster id
       cluster_members/2,                            %   cluster id → member nodes
       category_lin/3 ]).
```

The transpiler then:
- emits the `WamState` build calls **in dependency order** at setup
  (`build_descendant_sketches` → `build_bridge_scores` → `build_clusters`),
- generates the `foreign_predicate` facts + dispatch arms for each listed query predicate, collapsing the
  4-site boilerplate into the one declaration,
- threads the `edge_pred` / `mu` / `threshold` / `k` configs through.

That turns "4 sites × N predicates + manual build-ordering" into one statement — the real *improve how we
declare*: you declare the **analysis pipeline**, not the plumbing. (It's the same spirit as the existing
`foreign_predicate` facts, one level up — a dependency-aware bundle.)

## Implementation plan (increments — additive to the WAM-Rust target)

1. **Expose the bridge detector** (no new Rust logic): add the `foreign_predicate` + dispatch arm for
   `category_bridge_score/2` (and a `rank_bridges` → `bridge/3`: node, class atom, `n_eff`). Test: a
   Prolog program classifies a node on the 10k fixture. *Establishes the graph-analysis exposure pattern.*
2. **Bridge-partition clustering primitive** (new Rust, reuses `condense_scc` / a connected-components
   pass): `cluster_by_bridges(parents, bridge_scores) -> HashMap<u32, ClusterId>` (leak-removal →
   components; then the hierarchical bridge-split). `WamState::build_clusters` + `category_cluster(node)`.
   Expose `category_cluster/2` + `cluster_members/2`. Test on the 10k graph: do the physics sub-domains
   come out as clusters, with leak conduits (`Matter`, `Physical_objects`) as the cut points?
3. **The `graph_analysis/4` pipeline declaration**: the Prolog macro that captures the build-dependency
   and generates the wiring; validate it reproduces the hand-wired increments 1–2.
4. **(Later) close the loop to the Python tools**: a `hierarchy_objective`-style read-out from `n_eff`,
   so the WAM-Rust structural clustering and the `J = D/(1+H)` tool agree / cross-check.

**Reuses:** the merged bridge detector, the descendant sketches, `condense_scc` (Tarjan) for the
component/partition pass. Each increment renders `boundary_cache.rs.mustache` + `state.rs.mustache` and
`cargo test`s at **edition 2021** (the established harness). Bridge detection stays additive; nothing in
the merged similarity/detector changes.

## Increment 1 — implemented: bridge detector exposed as a foreign predicate

The detector is now Prolog-callable through the established 4-site foreign-predicate mechanism
(mirroring `category_ancestor` / `category_ancestor_boundary`), additive throughout.

**How μ is supplied (the decision).** The bridge detector consumes a dense `name→μ` membership map as an
**input** (the directional model, the symmetric map, or any score). On the Prolog side that map is a
**fact predicate** — `category_mu(Node, Score)` — exactly analogous to the edge predicate
`category_parent(Child, Parent)`. It is materialised into a `HashMap<u32, f64>` by `register_ffi_mu`
(the score-fact twin of `register_ffi_fact_pairs`) and stored in `WamState.mu_facts`, keyed by the μ
predicate name. A node absent from the facts defaults to `μ = 0` (out of domain), the same convention as
the rest of the detector. The μ predicate name is threaded as the `mu_pred` string config, so a project
can carry several μ maps (e.g. `category_mu_sym`, `category_mu_llm`) and select one per declaration.

**The four sites.**
- **(A) `WamState` methods** (`state.rs.mustache`): the merged `build_bridge_scores` / `category_bridge_score`
  plus new additive glue — `register_ffi_mu`, `build_bridge_scores_named(edge_pred, mu_pred, threshold,
  params)`, `ensure_bridge_scores` (build-on-first-use), `category_bridge_class(node) -> (class atom,
  n_eff)`, `bridge_candidates()` (ranked enumeration), and `atom_name` (id→atom for streaming).
- **(B) the declaration** — a `foreign_predicate/3` spec passed as the `foreign_lowering(...)` option (the
  same shape `category_ancestor` uses), now with a `register_foreign_f64_config` setup line (added) for
  `threshold` and a `register_ffi_mu` setup line (added) for the μ facts.
- **(C) dispatch arms** in `execute_foreign_predicate` (`wam_rust_target.pl`): `"category_bridge_score"`
  (Node atom in → Class atom out, build-on-first-use) and `"bridge"` (enumerate ranked candidates as
  Node, Class atom, `n_eff` float). The class atom is `boundary_cache::bridge_class_atom(class)` —
  `bridge` / `leak_conduit` / `redundant_hub` / `not_candidate` — defined once next to `BridgeClass`.
- **(D) registration** via the `foreign_lowering` setup ops (`register_foreign_native_kind` +
  `register_foreign_result_mode(deterministic)` + `register_foreign_result_layout(tuple(1))` + the
  configs).

**The declaration an author writes** (lowering `category_bridge_score/2`):

```prolog
:- Mu = [ 'Subfields_of_physics'-1.0, 'Matter'-1.0, 'Energy'-1.0 /* … category_mu facts … */ ],
   write_wam_rust_project([user:my_query/2],
     [ module_name(physics),
       foreign_lowering(
         foreign_predicate(category_bridge_score/2,
           [ register_foreign_native_kind(category_bridge_score/2, category_bridge_score),
             register_foreign_result_mode(category_bridge_score/2, deterministic),
             register_foreign_result_layout(category_bridge_score/2, tuple(1)),
             register_foreign_string_config(category_bridge_score/2, edge_pred, category_parent),
             register_foreign_string_config(category_bridge_score/2, mu_pred, category_mu),
             register_foreign_f64_config(category_bridge_score/2, threshold, 0.3),
             register_ffi_mu(category_bridge_score/2, category_mu, Mu) ],
           [])) ],
     'output/physics').
```

The author then queries `category_bridge_score(Node, Class)` — `Class` unifies with the bridge atom — or
`bridge(Node, Class, Neff)` to enumerate the ranked candidates. `threshold` defaults to `0.3` and
`τ_pure` is self-calibrating (the merged auto-calibration), so the cut needs no tuning. (Increment 3 will
fold this boilerplate into the single `graph_analysis/4` macro.)

**Tests.** `boundary_cache.rs.mustache`: `bridge_class_atom_wire_names` (the wire atoms) and the gated
`wikipedia_bridge_foreign_predicate_atoms` (the 10k fixture through the same rank_bridges → class-atom
path: `Subfields_of_physics ⇒ bridge`, `Matter ⇒ leak_conduit`) — both render + `cargo test` at edition
2021 with no swipl. `tests/test_wam_rust_bridge_foreign_dispatch.pl` exercises the full
transpile→dispatch path (swipl + cargo, CI-gated).

## Increment 2 — implemented: bridge-informed clustering, exposed

The clustering primitive and its two query predicates, wired the same 4 sites as increment 1, additive.

- **(A) core** (`boundary_cache.rs`): `cluster_by_bridges(parents, bridge_scores) -> HashMap<u32,
  ClusterId>`. Stage 1 — **top-level clusters** = weakly-connected components of the graph with every
  `LeakConduit` node and its edges removed (BFS over the undirected non-leak adjacency; each leak is its
  own singleton cut point). Stage 2 — **bridge split (one level, v1)** = within a component each `Bridge`
  node's child-branches become distinct sub-clusters, bridges processed by `n_eff` descending (strongest
  wins; a higher bridge claims shared nodes first), the bridge node itself staying in its component as
  the boundary. RedundantHubs are neither cut nor split ⇒ kept whole. Cycle-safe (all BFS dedup).
- **(A) glue** (`state.rs`): `clusters` (node→id) + `cluster_index` (id→members) fields;
  `build_clusters_named` (ensures bridge scores, then `cluster_by_bridges`), `ensure_clusters`
  (build-on-first-use), `category_cluster(node) -> ClusterId`, `cluster_members(id) -> Vec<node>`.
- **(C) dispatch arms**: `"category_cluster"` (Node atom → cluster id `Integer`) and `"cluster_members"`
  (cluster id `Integer` → member atoms, streamed). Build-on-first-use; both honor an optional `tau_pure`
  f64 config (pins the leak/bridge purity cut; absent ⇒ self-calibrating).
- **(B)/(D)** as increment 1, with the extra `tau_pure` config.

**The declaration an author writes** (clustering):

```prolog
:- write_wam_rust_project([user:my_query/2],
     [ module_name(physics),
       foreign_lowering(
         foreign_predicate(category_cluster/2,
           [ register_foreign_native_kind(category_cluster/2, category_cluster),
             register_foreign_result_mode(category_cluster/2, deterministic),
             register_foreign_result_layout(category_cluster/2, tuple(1)),
             register_foreign_string_config(category_cluster/2, edge_pred, category_parent),
             register_foreign_string_config(category_cluster/2, mu_pred, category_mu),
             register_foreign_f64_config(category_cluster/2, threshold, 0.3),
             register_ffi_mu(category_cluster/2, category_mu, Mu) ],
           [])),
       foreign_lowering(
         foreign_predicate(cluster_members/2,
           [ register_foreign_native_kind(cluster_members/2, cluster_members),
             register_foreign_result_mode(cluster_members/2, stream),
             register_foreign_result_layout(cluster_members/2, tuple(1)),
             register_foreign_string_config(cluster_members/2, edge_pred, category_parent),
             register_foreign_string_config(cluster_members/2, mu_pred, category_mu),
             register_foreign_f64_config(cluster_members/2, threshold, 0.3),
             register_ffi_mu(cluster_members/2, category_mu, Mu) ],
           [])) ],
     'output/physics').
```

The author then queries `category_cluster(Node, ClusterId)` (the node's cluster) and
`cluster_members(ClusterId, Member)` (enumerate its members). Add
`register_foreign_f64_config(.../2, tau_pure, 0.3)` to pin the cut on a small/sparse graph. (Increment 3
folds this into the single `graph_analysis/4` macro.)

**Real-data finding (honest — the widening motivation).** On the 10k physics slice
(`wikipedia_bridge_clustering_10k`, gated): the detector flags exactly **two** leak conduits, `Matter`
and `Physical_objects`. Removing them does **not** disconnect the slice — `desc(Physics)` is densely
cross-linked, so the largest non-leak component is **8244 of 8247** nodes (top-level leak-cut leaves one
giant blob). The **bridge-split** is what provides structure: `cluster_by_bridges` yields **12 clusters**
with sizes `[6828, 1142, 235, 18, 11, 4, 3, 2, 1, 1, …]` — the strongest bridges carve sizable
sub-clusters out of the blob. So on this slice the *hierarchy within* (bridge-split) is the usable
signal, not the *top-level separation* (leak-cut) — a concrete argument for a less-exploded edge set
(scoped cone / denser μ on the connectors) if clean top-level domains are wanted.

**Tests.** `boundary_cache.rs.mustache`: `cluster_by_bridges_leak_removal_separates_domains`,
`cluster_by_bridges_bridge_splits_into_subclusters`, `cluster_by_bridges_keeps_unmarked_graph_whole`
(synthetic, render + `cargo test`), and the gated `wikipedia_bridge_clustering_10k` (the finding above).
`state.rs`: `cluster_foreign_glue_partitions_via_named_facts`.
`tests/test_wam_rust_cluster_foreign_dispatch.pl` exercises the full transpile→dispatch path (swipl +
cargo): `category_cluster(d1)=category_cluster(x1) ≠ category_cluster(d2)` once the leak conduit is cut.

## Increment 3 — implemented: the `graph_analysis/4` pipeline declaration

Increments 1+2 worked, but the author had to repeat a full `foreign_lowering(foreign_predicate(P/A,
[...], []))` block — native_kind + result mode/layout + the *same* `edge_pred` / `mu_pred` / `threshold`
configs + `register_ffi_mu(MuPred, Mu)` — once **per** query predicate. `graph_analysis/4` collapses that
into one statement: declare the shared inputs and the query list **once**, and the transpiler expands to
**exactly** the per-predicate `foreign_lowering` terms (term-identical, byte-identical generated Rust
setup), threading the shared `edge_pred` / `mu_pred` / `threshold` / sketch `k` and the right native_kind
/ layout / `register_ffi_mu` into each from a small query→kind table. The build-dependency
(`clusters → bridges → sketches`) needs no ordering here — it is already handled by the build-on-first-use
`ensure_*` chain in `state.rs` — so this is pure **declaration bundling**.

**BEFORE** (increments 1+2 — one verbose block *per predicate*; ~4 blocks, the shared
`edge_pred`/`mu_pred`/`threshold`/`Mu` repeated in every one):

```prolog
:- write_wam_rust_project([user:my_query/2],
     [ module_name(physics),
       foreign_lowering(foreign_predicate(category_bridge_score/2,
         [ register_foreign_native_kind(category_bridge_score/2, category_bridge_score),
           register_foreign_result_mode(category_bridge_score/2, deterministic),
           register_foreign_result_layout(category_bridge_score/2, tuple(1)),
           register_foreign_string_config(category_bridge_score/2, edge_pred, category_parent),
           register_foreign_string_config(category_bridge_score/2, mu_pred, category_mu),
           register_foreign_f64_config(category_bridge_score/2, threshold, 0.3),
           register_ffi_mu(category_bridge_score/2, category_mu, Mu) ], [])),
       foreign_lowering(foreign_predicate(bridge/3,            [ /* …the same 7 lines, native_kind=bridge, layout tuple(3), stream… */ ], [])),
       foreign_lowering(foreign_predicate(category_cluster/2,  [ /* …the same 7 lines, native_kind=category_cluster… */ ], [])),
       foreign_lowering(foreign_predicate(cluster_members/2,   [ /* …the same 7 lines, native_kind=cluster_members, stream… */ ], [])) ],
     'output/physics').
```

**AFTER** (increment 3 — the shared inputs + query list declared once):

```prolog
:- graph_analysis(category_parent,                   % edge predicate
     [ mu(category_mu), threshold(0.3) ],            % shared inputs (declared ONCE)
     [ sketches(k=32), bridges, clusters ],          % build stages present
     [ category_bridge_score/2, bridge/3,            % exposed query predicates
       category_cluster/2, cluster_members/2 ]).

main :- graph_analysis_options(Opts, QueryPreds),    % expand to the 4 foreign_lowering specs
        write_wam_rust_project(QueryPreds, [module_name(physics) | Opts], 'output/physics').
```

`graph_analysis_expand/5` is the pure expander (used by the directive and directly testable);
`graph_analysis_options/2` collects all asserted `graph_analysis/4` declarations into the
`foreign_lowering(...)` list + the flat query-predicate list. The query→kind table
(`bridge_query_kind/4`) maps each known predicate to its `native_kind` / result mode / layout; the mu
facts are harvested from `MuPred/2` into `register_ffi_mu`. Adding a new exposed predicate is one table
row, not another block at every call site.

**Validation.** `tests/test_graph_analysis_declaration.pl` (swipl): the expansion is **term-identical** to
the hand-written increment-1/2 blocks (`expansion_is_term_identical_to_handwritten`) and generates
**byte-identical** Rust setup per predicate (`generated_rust_setup_is_identical`, via
`rust_foreign_setup_code`); the directive bundles all four queries into one options list
(`directive_bundles_all_queries`). Because `graph_analysis/4` adds Prolog only (no template change), the
generated crate is unchanged — the **134** generated-crate lib tests, both
`test_wam_rust_{bridge,cluster}_foreign_dispatch.pl` e2e suites, and the `boundary_cache` render
(`cargo test`, edition 2021) all still pass, with the same `"category_bridge_score"` / `"category_cluster"`
dispatch arms and the same dispatch results as increments 1+2.
