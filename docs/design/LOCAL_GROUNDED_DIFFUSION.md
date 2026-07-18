# Local grounded diffusion on massive graphs

## Status and scope

This document extends
[LEAKY_GRAPH_DIFFUSION.md](LEAKY_GRAPH_DIFFUSION.md) from a dense,
whole-graph correctness reference to a bounded local problem. It specifies the
neighborhood, boundary condition, screening calibration, and diagnostics that
must remain invariant across dense, sparse, CPU, and possible future GPU
backends.

The construction is general graph infrastructure. It does not promote a
Green kernel to learned measurement covariance, relax the failed Stage-A
source-power gates, or make a CUDA performance claim.

Separate outcome-blind application uses are authorized now: frozen-geometry
candidate generation, per-anchor `h_s` screening values as ranking features
(not calibrated probabilities or covariance entries), and routing. Selection
of `D`, `K`, `ell`, `epsilon`, `alpha`, and screening thresholds may use
topology, frozen embeddings, resource limits, and numerical diagnostics, but
no placement or judge outcomes. Any learned downstream consumer still requires
train-only selection and held evaluation. This does not reopen the failed
Stage-A dependence campaign or authorize judge covariance, independent
batching, QR, sparse/CUDA, or performance claims.

For scale orientation, the legacy 5,004-tree RDF slice documented in
[pearltrees_data_completion.md](../pearltrees_data_completion.md) needs
200,320,128 bytes (191.04 MiB) per dense float64 square array. The current
reference retains four square arrays, about 764.16 MiB before construction and
linear-algebra workspaces. The current assembled Pearltrees DAG has 14,709
nodes rather than 5,004: one array is about 1.61 GiB and four are about
6.45 GiB before workspaces. Thus the dense path is plausible today for a
measured export-slice or bounded candidate-domain study on a sufficiently
provisioned host, not yet a whole-current-corpus deployment or latency claim.
Record the exact node universe, peak RSS, and wall time.

## Why locality is necessary

A Wikipedia-scale graph can have millions of nodes. A dense float64 matrix on
one million nodes would require about 8 TB before factorization, and dense
Cholesky would require cubic work. More accelerator memory does not make that
whole-graph dense representation a viable default.

Most uses in this project are anchored queries or batches: only the response
near one or more source nodes is decision relevant. We therefore choose an
outcome-blind local node set `S`, impose an explicit zero-temperature/grounded
bath outside it, and solve one positive-definite system on `S`. Locality is a
modelled boundary condition, not an excuse to silently delete edges.

## Choosing the local domain

Let `A` be the set of query, source, or measurement anchors. Rank nodes by a
distance `D(A,i)` and retain a deterministic prefix `S_K` containing every
anchor.

### Primary: graph hop distance

The default is ordinary multi-source shortest hop distance on the fixed graph
support, with every anchor explicitly assigned distance zero. A truncated BFS
can stop after the requested budget rather than scanning the whole corpus.
For non-anchor nodes, this is the same minimum edge-count geometry documented
by [WAM_TRANSITIVE_DISTANCE3_CONTRACT.md](WAM_TRANSITIVE_DISTANCE3_CONTRACT.md);
that relation itself intentionally exposes positive distances rather than the
selector's seeded zero.

Equal-distance nodes need a revision-stable tie rule. If memory permits, keep
the complete final hop shell; this avoids an arbitrary directional cut around
a high-degree anchor. A strict `K` cap may split the shell, but must record the
tie rule and realized boundary. Nested convergence runs must use prefixes of
one frozen ordering so that

    S_K subset S_2K subset S_4K.

The reference tie key assumes value-like node identifiers with revision-stable
representations, such as strings and integers. Custom objects must first be
mapped to stable string or scalar identifiers.

### Secondary: Nomic-resistance weighted shortest path

When a hop shell is too large or heterogeneous, use a weighted shortest-path
ordering on the same topological edges. For the conductance `c_ij` defined in
the global diffusion design, set a positive dimensionless edge length such as

    r_ij = c_ref / c_ij

and define `D` by the minimum sum of `r_ij` along a path. The reference scale
`c_ref` changes units, not ordering. This makes a semantically discordant
existing edge more resistant; it never creates an embedding-only shortcut.
Truncated Dijkstra supplies the top prefix. The repository's weighted
shortest-path machinery is illustrated by
`examples/benchmark/min_semantic_distance.pl`, although this design uses the
strictly positive resistance derived from the diffusion conductance rather
than a raw cosine-distance edge weight.

A revision-pinned Nomic embedding is the preferred secondary geometry because
e5 is already used by the filing ranker. MiniLM remains a sensitivity check.
The embedding model, revision, conductance transform, length scale, floor, and
tie rule are provenance, not tuning details.

The dense reference in `src/unifyweaver/graph/local_diffusion.py` currently
implements the primary hop selector. The Nomic-resistance Dijkstra ordering is
a specified adapter/follow-up, not an implemented performance claim in this
change.

The selector is outcome blind. Effective resistance and the fitted Green
kernel are not allowed to choose their own local domain: that would require
solving the object being approximated and would make the approximation
circular.

## The exact Dirichlet cut

Let `W_SS` contain conductances whose endpoints are both retained, and let

    L_ind = diag(W_SS 1) - W_SS

be the Laplacian of the induced subgraph. For every retained node, aggregate
all severed conductance into

    beta_i = sum over j not in S of c_ij.

Fixing every exterior node at bath potential zero makes the local Kirchhoff
equation exactly

    J_S = L_ind + diag(alpha + beta),
    J_S u_S = q_S.

Equivalently, `J_S` is the principal block of the full grounded precision.
The `beta` term is the conductance of the cut to the common bath. It is not
optional bookkeeping: dropping it turns the cut into an insulating Neumann
boundary, traps flux inside `S`, and generally exaggerates long-range
influence.
The optional `bath_temperature` API is a coordinate shift for one common
bath connected through both `alpha` and `beta`. In absolute coordinates,

    J_S T_S = q_S + (alpha + beta) T_b,
    T_S = T_b 1 + J_S^-1 q_S.

It does not represent an exterior-only nonzero bath with intrinsic leakage
held at a different ground; that model would require a separate boundary
right-hand side.


This is exact for the stated Dirichlet boundary, not an exact marginalization
of unknown exterior values. A Schur complement of the exterior would instead
create additional, usually dense, couplings among boundary nodes.

The implementation need not enumerate the entire exterior. It only needs the
incident edges of retained nodes. If total weighted degree

    d_i^c = sum over all j of c_ij

is precomputed, then `beta_i = d_i^c - sum_{j in S} c_ij`; otherwise a boundary
pass reads the outside endpoints of cut edges. Semantic conductance on a cut
edge may require the outside endpoint's embedding, but that endpoint does not
enter the factorization.

Under the bounded conductance transform, `0 <= c_ij <= c0_ij`. If a cut edge's
exterior embedding is unavailable, the dense correctness reference continues
to fail closed. A scale adapter may explicitly substitute `c0_ij` for that cut
edge only: this upper-bounds `beta` and, by grounded M-matrix inverse
monotonicity, lowers nonnegative raw responses, so the degradation over-grounds
rather than exaggerates influence. Record the fallback edge count and
substituted cut mass. Substituting zero, applying this fallback to retained
internal edges, or claiming that normalized `h_s` values or their ranking are
conservative is not allowed.

### Nonzero grounding remains mandatory

Every positive-conductance component of the induced graph must reach a
positive entry of `alpha + beta`. A completely retained connected component
has `beta=0`, so it still needs model leakage `alpha>0`. An ungrounded
component fails closed; numerical jitter may not substitute for a bath.

## Screening and leakage calibration

For source `s`, let

    G_alpha = J_S(alpha)^-1,
    h_s(i; alpha) = G_alpha(i,s) / G_alpha(s,s).

The normalized column `h_s` has `h_s(s)=1`. On nodes other than `s` it solves
the killed harmonic equation. Probabilistically, it is the chance that a
continuous-time graph walk reaches `s` before distributed leakage or the
Dirichlet cut absorbs it. Electrically, it is the voltage relative to the
source voltage. This gives an attenuation target without correlation scaling.

With a uniform `alpha`, increasing `alpha` couples the walk to an earlier
killing clock. Consequently `h_s(i;alpha)` is nonincreasing for every
`i != s`, and so is its maximum on any frozen calibration shell. Bracketed
bisection can therefore choose the smallest `alpha` satisfying

    max over s in A, i in F_e(s) of h_s(i;alpha) <= exp(-1),

where `F_e` is the preregistered e-fold shell and lies strictly inside the
truncation boundary. Since `R_leak=1/alpha`, this selects the largest leakage
resistance that still meets the attenuation target, avoiding unnecessary
over-grounding. If the upper bracket violates the float64 conditioning or
physical-scale contract, calibration fails rather than adding hidden jitter.
Calibration uses the raw Green response ratio; unit-diagonal correlation
normalization does not have the required monotone interpretation.
The API therefore requires `shell_nodes` explicitly and never substitutes the
hard truncation frontier by default. The caller must preregister an interior
shell; the implementation validates membership and reachability but cannot
infer scientific interiority from node identifiers alone.

### Per-anchor realized screening radius

One uniform `alpha` is set by the tightest anchor and can over-ground the
others. Preserve that heterogeneity instead of reporting only the joint
maximum. For a frozen outcome-blind source-relative distance `d_s`, define the
radial tail envelope

    A_s(r) = max over reachable i with d_s(i) >= r of h_s(i).

Unlike individual shell values, `A_s(r)` is nonincreasing by construction even
when an irregular graph's exact-distance shell responses rebound. For target
`q`, define

    R_s(q) = min {r > 0 : A_s(r) <= q}.

At `q=exp(-1)` this is the realized e-fold radius. For any other target it is
the realized `q`-screening radius; do not infer or interpolate an exponential
length from one shell value. Report each anchor's calibration-shell
attenuation, threshold, distance metric, and discrete crossing bracket
`(r_previous, r_crossing]`. If no crossing occurs inside the retained
positive-conductance component, right-censor the result beyond the maximum
observed radius.

The current hop implementation recomputes source-relative hops inside each
anchor's realized positive-conductance retained component. It excludes
disconnected components rather than treating their zero Green response as
instant attenuation, and it fails closed unless every calibration anchor has
a reachable non-source shell node. A future weighted selector must supply its
own frozen source-relative distances rather than reuse the multi-source
minimum distance.

### Chain initializer

On an infinite regular chain with edge conductance `c`, uniform leakage
`alpha`, and hop radius `R_e`, the response has form `exp(-kappa r)` with

    cosh(kappa) = 1 + alpha/(2c).

Setting `kappa=1/R_e` gives the exact one-dimensional initializer

    alpha_0 = 2c [cosh(1/R_e) - 1]
            approximately c/R_e^2.

Using a robust typical internal conductance for `c` supplies a bracket seed,
not a final calibration on an irregular graph.

Tree-like outward branching can attenuate faster than a chain at the same
`alpha`, so the chain seed can be conservatively high and over-ground.
Cycles, bottlenecks, and degree heterogeneity can shift it in either direction.
The frozen-shell bisection, not `alpha_0`, determines the final model.

### Screening radius is not truncation radius

`R_e` is the distance at which influence falls to `exp(-1)`, about 0.368. A
hard zero bath at that same distance can still be a substantial perturbation.
When accuracy rather than only a fixed memory cap determines the boundary,
put the hard bath farther away:

- `3 R_e` leaves the chain envelope at `exp(-3)`, about 0.050;
- `4.6 R_e` leaves it near one percent.

Thus use the e-fold shell to calibrate leakage and a farther shell or larger
`K` to control truncation. A strict resource budget may force a nearer bath;
the diagnostics below then quantify, rather than conceal, its influence.

## Boundary-error diagnostics

Diagnostics are evaluated per anchor and jointly over a batch. They are part
of the result provenance.

### Maximum-principle envelope

For nonnegative sources, the Dirichlet local solution is a lower bound on the
restriction of the full positive solution. If all omitted boundary voltages
are at most `M`, the error `e` satisfies

    0 <= e <= M p,
    p = J_S^-1 beta.

Here `p_i` is the killed-walk harmonic measure of the artificial cut: the
probability of reaching the cut before model leakage, under the local
continuous-time interpretation. Because `J_S` is a grounded M-matrix,
`0 <= p_i <= 1`. Report `p` at every anchor and its maximum on the scored
region. It is an outcome-independent sensitivity envelope even when `M` is
not known tightly.

### Cut current

For a maintained nonnegative source and zero exterior bath, report

    I_cut = sum_i beta_i u_i,
    I_leak = sum_i alpha_i u_i.

Kirchhoff balance gives `sum_i q_i = I_cut + I_leak`, up to numerical error.
The fraction `I_cut / sum(q)` says how much injected flux is absorbed by the
artificial hard boundary rather than the intended distributed leakage. Large
cut fraction requests a larger domain; it is not repaired by relabelling the
boundary as harmless.

### Nested-domain convergence

Calibrate `alpha` once on the frozen largest reference domain, then hold it
fixed while solving the nested domains `K`, `2K`, and `4K` with the same
conductance model. For nonnegative sources, raw Dirichlet voltages
on common nodes obey domain monotonicity:

    u_K <= u_2K <= u_4K <= u_full.

Report relative changes on anchors and the scored inner region. The ordering
is a useful implementation invariant. It need not hold after dividing each
column by its changing diagonal, so normalized Green correlations and
effective resistances receive separate convergence summaries rather than a
false monotonicity claim.

## Multiple anchors and positive semidefiniteness

For a batch of anchors, form one shared domain from the union of their frozen
neighborhoods, or use one multi-source top-`K` ordering. Build one `J_S`, one
factor, and one Green kernel on that domain. This preserves symmetry and
positive definiteness for all cross-anchor entries.

Factoring a different local system for each anchor and splicing the resulting
columns together does not, in general, produce a symmetric or PSD matrix. A
per-anchor solve may be used for independent routing scores, but not to claim
one joint covariance or precision root.

Record requested `K`, realized union size, per-anchor coverage, cut size, and
whether a complete distance shell was retained. Batch construction must be
independent of downstream labels.

## Computational contract and scale path

For a retained domain with `K` nodes and `E_S` internal edges:

- truncated multi-source BFS costs `O(K + E_touched)`;
- weighted selection uses truncated Dijkstra and a priority queue;
- boundary aggregation touches incident cut edges, or uses precomputed
  weighted degrees;
- sparse assembly stores `O(K + E_S)` entries;
- the current dense factor remains a small-`K` correctness oracle.

A strict node cap bounds the dense algebra, not necessarily
`E_touched`. The current correctness selector materializes the complete
incident list of each retained node, so a million-neighbor hub can still
require universe-scale preprocessing. Likewise, requesting a complete final
shell can realize far more than `K` nodes. Production use on such graphs
requires an indexed/streaming top-`K` neighbor adapter and precomputed
weighted-degree or cut-conductance aggregation. Until that path exists, report
maximum touched degree and do not call preprocessing `O(K)`.

The included CPU microbenchmark intentionally uses a constant-degree implicit
million-node universe; it proves absence of a global node scan in that regime,
not a hub-safe or deployment-latency result.

Sparse Cholesky is the first direct-solver candidate, but its cost depends on
fill-in and elimination order rather than only `K`. Selected equilibrium
columns can instead use preconditioned conjugate gradients; heat action can
use Lanczos or Chebyshev methods. Hierarchical domains, graph partitions,
multigrid, and spectral sparsifiers are later options if one local domain is
still too large.

A GPU is not automatically faster. Truncated graph traversal is irregular,
sparse factorization can be fill-bound, and host/device transfer matters. Any
CUDA specialization requires a matched full-cost benchmark including
neighborhood selection, embedding access, boundary aggregation, assembly,
factorization or iteration, diagnostics, and transfers. Until then the API
records backend facts and makes no CUDA crossover claim.

## Implementation invariants

An implementation of local grounded diffusion must:

1. accept an explicit, deterministic retained node ordering or domain;
2. compute and expose nonnegative `beta` rather than discard cut edges;
3. build `J_S = L_ind + diag(alpha + beta)` and verify the grounded-component
   and reciprocal-condition contracts from the global design;
4. retain the Cholesky/solve-first interface and never require an explicit
   inverse for primary computation;
5. expose killed-response, per-anchor realized screening-radius,
   harmonic-measure, cut-current, and nested-domain diagnostics with
   finite-value and Kirchhoff-residual checks;
6. keep model leakage, Dirichlet cut conductance, and any future numerical
   jitter as three separately named quantities;
7. preserve the node map, selector provenance, requested and realized `K`,
   and boundary statistics; and
8. compare small local systems against the whole-graph dense reference.

## Rejected and deferred alternatives

- **Delete cut edges:** rejected because it creates an insulating boundary and
  overstates retained influence. Aggregate them into `beta`.
- **Use the whole graph densely:** rejected by quadratic memory and cubic
  factorization before any million-node experiment begins.
- **Embedding-only nearest neighbors:** rejected as the default domain because
  they can jump across graph topology. Semantics may weight existing paths.
- **Let Green distance choose `S`:** rejected as circular and expensive. Use a
  cheaper frozen `D` before solving.
- **Hard-boundary calibration at `exp(-1)` alone:** acceptable only as an
  explicit resource compromise. Prefer a farther `exp(-3)` or one-percent
  truncation boundary and verify nested convergence.
- **Drop distributed leakage after adding the cut bath:** rejected. `beta`
  represents truncation; `alpha` represents the intended finite correlation
  range and also grounds a fully retained component.
- **Correlation-normalized calibration:** rejected because its changing
  diagonal obscures the killed-walk monotonicity used by bisection.
- **One local matrix per anchor for joint inference:** rejected because the
  assembled cross-anchor object need not be symmetric or PSD.
- **Exact exterior Schur complement by default:** deferred because it destroys
  strict locality through fill and models an integrated exterior, not the
  requested fixed-temperature bath.
- **Tune `D`, `K`, or `alpha` against held outcomes:** rejected. Geometry and
  numerical accuracy are selected outcome blind; any later statistical use
  still requires train-only fitting and held, dependence-aware validation.
- **Immediate sparse/GPU or covariance promotion:** deferred pending separate
  correctness, full-cost crossover, and statistical gates.

## Cross-references

- [LEAKY_GRAPH_DIFFUSION.md](LEAKY_GRAPH_DIFFUSION.md) — whole-graph physical
  operator, Green kernel, effective resistance, and precision-root algebra.
- [COST_FUNCTION_PHILOSOPHY.md](COST_FUNCTION_PHILOSOPHY.md) — hop, semantic,
  and local-exact graph scoring in the broader cost-function design space.
- [WAM_TRANSITIVE_DISTANCE3_CONTRACT.md](WAM_TRANSITIVE_DISTANCE3_CONTRACT.md)
  — fleet-wide shortest positive hop-distance semantics.
- `examples/benchmark/min_semantic_distance.pl` — existing weighted
  shortest-path/Dijkstra specification surface.
- `prototypes/mu_cosine/benchmark_local_grounded_diffusion.py` —
  compute-only CPU scaling and memory estimates on an implicit million-node
  universe.
- `prototypes/mu_cosine/DECISIONS_graph_geometry.md` — experiment-specific
  authorization boundaries and geometry decisions.
