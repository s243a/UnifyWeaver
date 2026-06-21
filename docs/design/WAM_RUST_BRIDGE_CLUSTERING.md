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
