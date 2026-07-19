# Prospective protocol: bounded grounded-diffusion fidelity

**Status:** prospective and frozen before any real-corpus fidelity result is
computed. Synthetic unit tests and implementation smoke tests are permitted in
the protocol PR. If the real graph, embeddings, budgets, selectors, protected
set, leakage rule, metric, or decision threshold must change, amend this file
and its machine-readable manifest before running the affected comparison.

This is an outcome-blind numerical study. Placement labels, judge outputs,
filing ranks, and downstream MRR are prohibited inputs. A selector that wins
here is only a better approximation to a larger grounded-diffusion system at a
matched resource budget; it is not thereby a better filing ranker or a learned
covariance model.

## 1. Decision and estimand

The practical question is which bounded domain best preserves the local
grounded-diffusion geometry that would be obtained from a larger retained
domain. The first decision compares the topology-first candidate skeleton with
the ordinary hop-prefix baseline. A revision-pinned semantic-resistance
selector is a gatekept secondary comparison. Sparse graph-derived boundary
closure is a final, explicitly experimental comparison against the same selected
domain with its exact Dirichlet boundary. For closure, semantic distance may
filter expensive graph calculations, but never licenses a pair or sets bridge
strength. This restriction does not apply to the separate semantic-operator
and RESISTANCE-selector sensitivity in Section 2.

For anchor `s`, candidate domain `D`, common protected nodes `P`, and the
larger reference `R`, let

    g_D^s = J_D^-1 e_s,
    h_D^s(i) = g_D^s(i) / g_D^s(s).

The primary per-anchor error is the relative raw-response error

    E_g(s,D) = ||g_D^s[P] - g_R^s[P]||_2
               / max(||g_R^s[P]||_2, tiny).

Raw response is primary because it preserves the circuit units and the
Dirichlet monotonicity interpretation. Normalized screening, selected grounded
effective resistance, and rank fidelity are required noninferiority checks;
none may be used post hoc to replace a failed primary endpoint.

An honest outcome may be: a smaller adequate domain; better fidelity at the
same budget; no material difference, retaining the simpler hop default; no
adequate bounded domain; or an inadequate outer reference. All are complete
results.

## 2. Frozen graph and conductance regimes

The first real run uses one content-hashed Pearltrees snapshot. Freeze before
anchor selection:

- the typed node universe and complete undirected incident adjacency;
- the directed principal-parent relation used only by the skeleton selector;
- canonical node IDs, titles/text construction, and all exclusion reasons;
- the largest eligible positive-edge connected component;
- software commit, float64 backend, thread settings, and resource ceiling; and
- every embedding model ID, exact revision, normalization, prefix, node order,
  array hash, and missing-node manifest.

Run two operator regimes without mixing their conclusions:

1. **Topology-only primary:** unit base conductance on every graph edge. Compare
   hop and candidate-skeleton selectors. This isolates domain construction from
   embedding choice.
2. **Semantic sensitivity:** use the frozen RBF-with-floor conductance in
   `docs/design/LEAKY_GRAPH_DIFFUSION.md`. Set `epsilon=0.05`; set `ell` to the
   median finite positive within-edge embedding distance on the frozen graph,
   computed once before anchor sampling. Run Nomic only when its exact revision
   and `clustering: ` text contract are available. MiniLM is descriptive
   sensitivity. Do not substitute e5 silently.

The semantic arm tests a selector for a frozen semantic operator. It cannot
establish that Nomic is a better embedding model, semantic truth, or
statistically independent of e5.

## 3. Anchor batches and protected nodes

Eligible anchors are non-isolated folder nodes in the frozen component. Their
selection cannot use bookmark destinations, placement counts, judge outcomes,
existing ranker scores, or previous fidelity results.

Stratify eligible nodes by undirected degree quartile and principal-depth
quartile, with missing depth as its own level. Within each stratum order nodes
by SHA-256 of `(seed, typed_node_id)` using seed `3882001`. Select 128 anchors
as evenly as possible across nonempty strata; redistribute deficits in
lexicographic stratum order. Persist the ordered manifest and every stratum
count. Form 32 consecutive batches of four anchors. Reserve the first eight batches (32
anchors) for leakage calibration and implementation configuration; their fidelity
metrics cannot enter a selector or closure claim. The remaining 24 batches (96
anchors) form the untouched audit. The batch, not the anchor, is the resampling
unit because four anchors share one domain and factorization.

For each anchor independently, freeze its source-relative ordinary-hop order
with the canonical tie key. Its protected set is the anchor plus its first 16
reachable non-anchor nodes. The batch protected set is the union across its
four anchors. This set is fixed before any selector is scored. A family/budget
that does not retain the complete protected set fails that batch; do not form a
post-hoc intersection and do not rank missing nodes last.

Report source-to-source hop distances within each batch. The primary result
averages the prespecified 24 audit batches rather than selecting only
convenient local or dispersed batches. Degree/depth and distance-stratified
summaries are descriptive.

## 4. Domain families and matched budgets

Freeze one maximum-budget ordering per family and take deterministic prefixes,
so within a family

    S_256 subset S_512 subset S_1024.

Every domain is one union-of-anchors system with one precision matrix and one
factorization. Its budget includes anchors, retained connecting paths, and all
other retained nodes.

1. **HOP:** deterministic multi-source hop ordering with the existing stable
   final-shell tie rule.
2. **SKELETON:** retain anchors, principal ancestors, and shared/common parents
   with their connecting paths first; use deterministic multi-source local
   expansion to fill the remaining budget. The exact priority and tie order are
   part of the fingerprint. Missing or cyclic parent data fail this family
   rather than falling back silently.
3. **RESISTANCE:** secondary only. Run truncated Dijkstra on existing graph
   edges with positive length `c_ref/c_ij`. Embeddings may change resistance on
   an existing edge but may not create a shortcut edge in this selector.

Use nominal retained budgets `K={256,512,1024}`. A required protected set or
connecting closure that exceeds `K` is a recorded coverage failure. Complete
final-shell runs may be reported descriptively, but the decision-bearing
comparison uses the strict matched budgets.

Exact one-port Kron reductions are a future family until an implementation has
an exact small-graph oracle and a provenance-preserving expansion map.
Multi-port elimination may create fill and must never be mislabeled a scalar
tree shunt.

## 5. Common larger-domain reference and leakage

For each batch and conductance regime, form `U` as the union of every available
family's frozen `S_1024`. Build an exact-Dirichlet model on `U`. Then expand
from `U` by the frozen hop ordering to at most 4,096 retained nodes, preserving
all of `U`, and build the outer reference `R`. If `U` itself exceeds 4,096 or
the resource contract cannot build `R`, the batch is reference-inadequate and
cannot promote a selector. If the connected component is exhausted, the whole
component is an exact admissible reference.

`R` is a bounded reference, not whole-graph truth. It is adequate only if the
model on `U` versus `R`, on the frozen protected set, has:

- 90th-percentile `E_g <= 0.01`;
- 90th-percentile maximum absolute `h` error `<=0.005`; and
- 10th-percentile top-8 overlap `>=0.98`.

Failure is reported as right-censored domain convergence. Do not choose a
different outer size after inspecting which family wins.

On the first eight calibration batches, find the anchor-specific uniform leakage
needed on each `R` for the graph-hop radius-3 shell and target `exp(-1)`, as
specified by `LOCAL_GROUNDED_DIFFUSION.md`. Every shell node must be reachable
and strictly interior (`beta=0`) in its reference. Freeze the single largest
required value as the study-wide `alpha`, then apply it unchanged to every
calibration and audit reference, `U`, and candidate domain. Recalibrating each candidate
would confound boundary approximation with a different physical model and is
not allowed in the primary comparison. Per-domain recalibration and censoring
may be reported as a secondary operational diagnostic.

## 6. Fidelity and boundary metrics

Compute per anchor on the complete frozen protected set:

- relative L2 raw-response error `E_g`;
- maximum absolute error in `h_s`;
- relative error of selected grounded effective-resistance distances from the
  anchor to protected nodes, using factor coordinates or selected solves;
- deterministic pairwise rank-inversion fraction and top-8 overlap for `h_s`;
- source diagonal Green-response error; and
- solve residual and sign/maximum-principle violations.

Effective resistance may be omitted only when the run manifest declared the
selected-solve resource arm unavailable before any family result. Record an
explicit omitted flag; the reduced run cannot resolve the resistance part of
`OPENQ-004`.

For every batch/family/budget also report:

- retained/boundary node counts, cut-edge count and cut mass;
- per-anchor cut-current fraction and the protected-set maximum of
  `p=J_D^-1 beta`;
- calibrated and numerical-minimum `alpha`, realized e-fold radius and censor
  flag for every anchor;
- reciprocal condition number, M-matrix sign check, Cholesky/solve residuals;
- adjacency calls or touched weighted degree where observable;
- selection, assembly, factorization, metric, and total wall times; and
- projected dense bytes, peak RSS, backend identity, and thread settings.

An individual family/budget is **adequate** only with 100% protected coverage,
all numerical checks passing, and across anchors:

- 90th-percentile `E_g <= 0.05`;
- 90th-percentile maximum absolute `h` error `<=0.025`;
- 90th-percentile rank-inversion fraction `<=0.05`;
- 10th-percentile top-8 overlap `>=0.90`; and
- 90th-percentile protected boundary-harmonic maximum `<=0.10`.

When resistance is evaluated, its 90th-percentile relative error must also be
`<=0.05`. These are deliberately conservative engineering tolerances on a
unit-normalized screening field, not claims about downstream filing loss.

## 7. Experimental graph-derived boundary closure

Plain exact Dirichlet remains the baseline: every omitted edge contributes to
the boundary shunt `beta`. For an omitted exterior `E`, the exact multi-port
target is its Dirichlet-to-Neumann/Schur response, not a collection of semantic
similarities. If `B` denotes the nonnegative Schur term subtracted from the
Dirichlet principal block, write its retained sparse approximation as transfer
conductances `kappa_ij` and per-port self-return `sigma_i`. Require

    beta_i = beta_res_i + sigma_i + sum_j kappa_ij,
    beta_res_i >= 0,
    sigma_i >= 0,
    kappa_ij = kappa_ji >= 0.

The closed model is

    J_closed = L_ind + L(kappa) + diag(alpha + beta_res)
             = J_D - B_hat,

where `B_hat` has off-diagonal transfer `kappa_ij` and diagonal self-return
`sigma_i`. Thus bridge degree and self-return are both removed from the full
Dirichlet shunt; bridges may never be added on top of `beta`. Verify the ledger
to float64 tolerance and recheck component grounding, M-matrix signs, SPD, and
the maximum principle.

For a genuinely two-terminal ungrounded exterior, `kappa=1/R_eff` is the exact
replacement conductance. More generally, compute the full two-port response:
for boundary branches `c_i`, `c_j` meeting at one exterior node with leakage
`a` and `d=c_i+c_j+a`,

    kappa_ij = c_i c_j / d,
    sigma_i = c_i^2 / d,
    sigma_j = c_j^2 / d,
    beta_res_i = c_i a / d,
    beta_res_j = c_j a / d.

Pairwise effective resistances do not uniquely determine a multi-terminal or
grounded exterior. Independently adding many `1/R_eff` bridges can double-count
shared paths. A component touching more than two retained ports therefore
requires an audited sparse approximation to its joint Schur map, or remains
grounded under the plain Dirichlet baseline.

Only cut branches represented inside the frozen reference exterior `E=R\D`
may fund a graph-derived bridge. Candidate cut edges from `D` directly beyond
`R` remain in `beta_res`; they cannot be credited to an unobserved path.
Discovery traverses only the induced `E`. Edges from `E` beyond `R` become
zero-bath shunts in the exterior precision before `E` is eliminated. Thus an
exact component is exact relative to the frozen bounded reference `R`, not a
claim about the unbounded full graph.

Exterior discovery uses one deterministic component traversal with a memoized
visited set. Memorization terminates loops and deduplicates work only. A search
frontier colliding with an already visited exterior node means path convergence
or a cycle in the same exterior component; it is not a cut point and must not
create grounding. Dirichlet cuts are defined only by actual edges crossing from
the retained set to the omitted set. Articulation status, if ever needed for a
selector, requires a separate explicit topology calculation. Collect every
retained boundary port incident to each exterior component once; a two-port
component may be reduced exactly, while a multi-port component is handled
jointly or left grounded rather than decomposed at traversal collisions.

Repository precedents constrain the implementation. Reuse the deterministic
BFS and actual cut-edge accounting in
[`local_diffusion.py`](../../src/unifyweaver/graph/local_diffusion.py). The
cycle-correct minimum-distance and caret mixing-boundary routines in the
[Rust WAM boundary cache](../../templates/targets/rust_wam/boundary_cache.rs.mustache)
are examples for bounded search and common-parent stopping, not physical Schur
oracles. [Tarjan SCC
condensation](../../src/unifyweaver/core/advanced/scc_detection.pl) is
available if a future directed variant needs cycle contraction. [Node-only
memoization](../../scripts/parent_histogram_recurrence.py) is not licensed for
simple-path counts or other path-state-dependent transfer quantities; this PR
uses global component memoization only for connectivity. Semantic
[k-nearest-neighbour code](../../scripts/mindmap/mst_folder_grouping.py) may
propose searches, but an embedding MST or kNN cut is never a cut in the
original graph.

Freeze one primary closure rule before the real run:

- identify cut exterior components and their retained boundary ports from the
  graph; exact two-port reductions have first priority;
- for a resource-bounded multi-port component, semantics may retain at most
  `q=2` plausible non-adjacent port pairs per boundary node before graph solves,
  but may not determine `kappa` or `sigma`;
- derive `kappa` and `sigma` from topology-only `c0` graph resistance or
  selected Schur solves;
- apply the exact ledger of a fully traversed two-port component without `rho`
  truncation or an approximate bridge cap; a simple series path is naturally
  weaker than each constituent branch, but parallel exterior paths can be
  stronger and must not be clipped if the reduction is called exact;
- for an approximate multi-port reduction only, cap each bridge at one half of
  the minimum positive ordinary retained graph-edge conductance and consume at
  most `rho=0.25` of any `beta_i` through
  `sigma_i + sum_j kappa_ij`, leaving the rest grounded; and
- apply deterministic pair ordering and reject shared-path pairwise composition
  unless it is represented by one joint sparse-DtN calculation.

Exact graph two-port reductions do not require embeddings. If no
revision-pinned semantic embedding is available, only the optional multi-port
pair filter is unavailable; do not substitute e5 after results. Other
`q/rho/cap` values are exploratory sensitivities with family-wise labels.

The SKELETON selector is the primary way to avoid boundary error: it retains
common parents and other high-value junctions before spending budget on local
fill. Closure is considered only for residual boundary ports joined by an
actual path through one represented omitted component. It must never force a
bridge merely because two ports are semantically close. Consequently, an
empty closure is a valid and scientifically useful result. In particular, the
primary hypothesis is that preserving common parents may leave no eligible
cross-boundary bridge, or may leave so little Schur transfer mass that closure
cannot materially improve the selected-domain solution.

Report, before any response comparison, the number of exterior components with
two or more retained ports, the number of graph-connected port pairs proposed
and realized, total transfer degree relative to original cut mass, and
`||B_hat||_F / ||J_D||_F`. For the optional semantic filter, also report the
fraction of graph-derived transfer mass captured by its top-`q` proposals.
That recall diagnostic evaluates search efficiency only: missing a pair leaves
its mass grounded, while semantic proximity alone never licenses a bridge.
If the realized closure is empty, record a no-op and do not manufacture an
experimental arm. If it is nonempty but small, still report the ledger and
paired response comparison; retain plain Dirichlet unless the fixed Section 8
promotion gate is met.

Before the corpus run, exact small-graph tests must include equal and unequal
exterior series paths, exterior leakage, and a multi-terminal shared-path case,
plus oversubscribed, asymmetric, self-edge, already-adjacent, ungrounded, and
double-counted-beta failures. On the corpus, compare closure with plain
Dirichlet on the identical selected domain, protected set, `alpha`, and
reference.

## 8. Decision hierarchy and paired uncertainty

Use 9,999 deterministic percentile bootstrap resamples of the 24 audit anchor
batches with seed `3882002`. Within a resample, apply identical batch weights
to both methods. The interval describes variation over the frozen structural
anchor-sampling design, not solver randomness.

Apply this gatekeeping hierarchy:

1. Verify outer-reference adequacy. If it fails, make no selector claim.
2. Find each topology family's smallest adequate `K`. Prefer the family that
   is adequate at a strictly smaller `K`. If both first pass at the same `K`,
   compare batch-mean `log(E_g + 1e-15)`. Promote SKELETON over HOP only when
   its geometric-mean error is at least 10% lower and the upper endpoint of the
   paired 95% interval for the log-error difference is below zero. Otherwise
   retain HOP.
3. In the semantic regime only, compare RESISTANCE with the selected topology
   family by the same rule. A win is scoped to that frozen semantic operator.
4. Compare graph-derived experimental closure with plain Dirichlet on the
   selected topology-only domain and topology-only `c0` operator. Semantic
   filtering may reduce which graph solves are attempted, but there is no
   semantic-conductance closure arm. Enable closure only when it reduces
   geometric-mean `E_g` by at least
   10%, the paired upper interval is below zero, and every noninferiority and
   safety gate still passes.

At every stage, `h`, resistance, rank inversion, top-8 overlap, and resources
are noninferiority gates: the candidate may worsen no metric by more than 10%
relative or 0.01 absolute for unit-interval metrics, and may not increase peak
memory or total wall time by more than 25% at the same `K` unless it achieves a
smaller adequate `K`. Report all tested families, budgets, batches, and losses.

The hierarchy supplies one decision-bearing comparison at a time. MiniLM,
complete-shell domains, alternate anchor strata, alternate closure strengths,
per-domain `alpha`, and degree/depth subgroups remain descriptive and cannot
rescue a failed primary comparison.

## 9. Reproducibility and fail-closed rules

The run fingerprint covers content rather than machine-local paths:

- source snapshot, canonical graph/parent manifests, and all byte hashes;
- anchor, batch, protected-node, family-ordering, domain, boundary, and
  reference manifests;
- embedding identity/revision/text contract, arrays, `ell`, `epsilon`, and
  missing-node manifest;
- `K`, shell, `alpha`, selector priorities, stable tie keys, and closure ledger;
- metric definitions, tolerances, bootstrap seeds/resamples, and software SHA;
- float dtype, numerical thresholds, backend/thread identity without absolute
  library paths; and
- per-phase timings, peak RSS, cache keys, and deterministic-rerun hashes.

Fail closed on incomplete or asymmetric adjacency, missing required parent or
embedding data, duplicate/unknown IDs, non-nested family prefixes, missing
protected nodes, stale caches, changed edge weights between candidate and
reference, nonfinite values, closure-ledger error, an ungrounded component,
M-matrix sign failure, insufficient reciprocal condition, solve residual above
the recorded tolerance, or reference/calibration inadequacy. Numerical jitter,
zero cut conductance, insulating truncation, and hidden embedding fallback are
not repairs.

The final report must separate synthetic correctness, topology-only primary,
semantic sensitivity, closure experiment, reference failures, and resource
failures. No geometry result may be described as a filing improvement,
covariance estimate, CUDA crossover, or Stage-A dependence promotion.
