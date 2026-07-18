# Graph-geometry covariance decision log

This is the append-only companion to `DESIGN_graph_geometry_confirmatory.md`.  Record a decision when it is
made, the evidence available at that time, alternatives rejected or deferred, and what result would reverse it.
Do not silently rewrite rationale after outcomes are known.

## 2026-07-12 — separate distance, kernel, and covariance

**Decision:** use scalar graph distance for sampling/interpretation, but admit a covariance geometry only with
an explicit PSD certificate.

**Reason:** an arbitrary graph shortest-path RBF need not be PSD.  The conditioner needs a valid covariance,
not merely a plausible similarity score.

**Accepted:** feature Grams, symmetric-Laplacian spectral kernels, nonnegative sums, and Schur products.

**Rejected:** clipping negative eigenvalues of an otherwise indefinite candidate as the primary construction.
Clipping hides a model error and changes the intended geometry.

**Reconsider if:** a proposed graph metric is proved negative type on the exact graph family and its kernel
passes the same held gates.

## 2026-07-12 — scalable graph family

**Decision:** make finite-hop walk-feature diffusion the primary scalable family; retain exact heat and
regularized-Laplacian kernels as small-graph references.

**Reason:** `Phi Phi.T` is PSD by construction, nonnegative hop weights are easy to constrain, and local sparse
walk features avoid whole-graph eigendecomposition.  Heat/resolvent references test whether the finite feature
family loses important spectral behavior.

**Rejected for now:** immediate CUDA work on exact full-graph diagonalization.

**Reconsider if:** a statistically validated component routinely exceeds CPU/sparse-feature budgets and matched
timing shows the exact spectral form improves held predictions enough to justify it.

## 2026-07-12 — independent semantic geometry

**Decision:** preregister Nomic `nomic-embed-text-v1.5` with the `clustering:` prefix as the primary frozen
external semantic candidate and MiniLM `all-MiniLM-L6-v2` as the lower-capacity comparator.  Pin exact model
revisions and cache content.

**Reason:** current mu readouts consume e5, so using e5 again would make redundancy likely and attribution weak.
Nomic and MiniLM provide a representation channel not used by the operating pipeline.  Nomic's clustering
prefix matches symmetric topic geometry; query/document prefixes would inject an unnecessary retrieval role.

**Not claimed:** either embedder is statistically independent of all language-model priors or superior on this
corpus.  Independence is an empirical correlation/ablation question.

**Rejected as primary:** e5 semantic geometry.  Keep it as a shared-input control because it is scientifically
useful to measure how much apparent benefit comes from reusing the model's own representation.

**Reconsider if:** an independent embedder has inadequate title coverage or fails held likelihood while e5 adds
incremental benefit under an honest redundancy ablation.

## 2026-07-12 — judge-derived distance

**Decision:** exclude distances computed from the same observed judge calls.  Allow cross-fitted `mu_hat`
distance only as a future leave-one-judge-out diagnostic.

**Reason:** same-call noise otherwise enters both the geometry and the residual, exaggerating correlation.  A
cross-fitted prediction can be a valid conditional covariate, but it is not independent evidence.

**Reconsider if:** a complete generative model jointly specifies the observation-dependent covariance and wins
outer-held likelihood/calibration without leakage.

## 2026-07-12 — graph plus embedding combination

**Decision:** nonnegative convex sums are primary; Schur products are secondary.

**Reason:** a sum lets either geometry explain an independent covariance mode.  A product is conservative but
can erase real graph-local structure whenever a general-purpose embedder misses domain-specific similarity.

**Rejected:** unconstrained signed kernel weights and post-hoc choice of a large mixture grid.

**Reconsider if:** a preregistered multiple-kernel learner with nonnegative weights passes full-procedure null
calibration and outer-held gates.

## 2026-07-12 — evidence order

**Decision:** mechanism audit, then outcome-blind campaign construction, then repeated-judge confirmation, then
QR integration/optimization.

**Reason:** the merged pilot showed that stable covariance-looking structure can coexist with inadequate power
and identification.  Numerical sophistication cannot repair weak covariance evidence.

**Rejected:** lowering the 95% deployment bound to use more apparent correlation.

**Reconsider if:** additional independent components and repeated calls tighten the simultaneous bound while
held posterior/calibration metrics remain favorable.

## 2026-07-12 — v1 mechanism audit failed: common alpha was not matched effect

**Observed:** the preregistered v1 family-wise selector controlled the block-null nonzero rate, but selected held
gain was usually negative.  Common `alpha` planted 4.25 times more off-diagonal RMS energy in the closed kernel
than the walk kernel.  Heat and resolvent were 0.999 correlated on the fixed graph, so exact family recovery was
also unidentifiable.

**Decision:** preserve v1 as failed.  Freeze v2 around matched maximum off-item coupling `rho`, which is the
operational batching quantity, and score predictive equivalence classes rather than requiring arbitrary labels
for nearly identical kernels.

**Rejected:** deleting weak families, shrinking the candidate grid after seeing v1, or calling larger training
field counts a successful primary result.

**Reconsider if:** a different outcome-blind graph makes the spectral kernels distinguishable; equivalence
classes must then be recomputed before outcomes on that graph.

## 2026-07-12 — same-hop walk concatenation rejected by outcome-blind topology inventory

**Observed without residual outcomes:** the separate-hop feature Gram was almost identity and did not make
directly adjacent campaign roots closer than nonadjacent roots.  It only compares equal-hop landing
distributions and misses the important `p_0(a)` versus `p_1(b)` overlap.

**Decision:** retain it as a negative/control geometry and add a cumulative-walk Gram that sums hop
distributions in one feature space.  This preserves PSD and sparse execution while admitting cross-hop overlap.

**Rejected:** tuning hop weights against residuals to rescue the original construction.

**Reconsider if:** a different estimand specifically requires equal diffusion time rather than local adjacency;
the same-hop version can then be preregistered for that estimand.

## 2026-07-12 — independent embeddings are distinct models but not independent geometry

**Observed without residual outcomes:** Nomic versus shared-e5 distance Spearman correlation was 0.776
exploratory and 0.838 fresh; MiniLM versus e5 was 0.810 and 0.868.  Nomic and MiniLM were themselves 0.894 and
0.907 correlated.  All three semantic models made adjacent roots closer on average.

**Decision:** keep Nomic as the primary external semantic candidate because it is modestly less redundant with
e5; keep MiniLM as a lower-cost sensitivity comparator, not a separate high-capacity selector branch.  Sample
the graph/semantic disagreement cells deliberately (about 4% exploratory and 10% fresh in the first inventory).

**Rejected:** describing either external embedding as independent ground truth, or treating Nomic plus MiniLM
as two independent votes.

**Reconsider if:** residual-held ablation demonstrates incremental benefit from both after full search/null
calibration.

## 2026-07-12 — cumulative walk accepted as the scalable local representative

**Observed without residual outcomes:** cumulative walk restores the intended adjacency ordering: mean distance
adjacent/nonadjacent was `0.857/0.996` exploratory and `0.770/0.982` fresh.  Its distance correlation with the
closed-neighborhood baseline was 0.730 / 0.885 Spearman.  On the fixed synthetic graph it was 0.962--0.972
correlated with closed, heat, and resolvent kernels, placing it in the same local/spectral predictive class.

**Decision:** use closed neighborhood as the cheapest fixed baseline and cumulative walk as the scalable
learnable representative.  Keep exact heat/resolvent as mathematical references, not three extra selector
branches unless a future outcome-blind graph makes them distinguishable.  Do not create a v3 selector merely
to add a nearly equivalent kernel; that would increase null multiplicity without adding identified geometry.

**Independent-embedding update:** relative to cumulative walk, graph/semantic disagreement cells are only
`13/764` exploratory and `22/777` fresh for Nomic (1.7% / 2.8%).  Existing data are thin for interaction
estimation, so the new campaign must oversample disagreement rather than rely on natural frequency.

**Rejected:** interpreting a larger candidate count as broader evidence, and treating heat/resolvent/cumulative
as independent votes on this topology.

**Reconsider if:** outcome-blind kernel correlation falls below the preregistered equivalence threshold on a new
corpus or graph representation.

## 2026-07-12 — prefer one selected item kernel before numerical diagonalization

**Decision:** the first structured covariance promoted to the conditioner should have one selected item kernel,

```text
R = I_n tensor B0 + K_theta tensor Bg.
```

If `K_theta=U Lambda U.T`, the item transform `U tensor I_m` reduces `R` to `n` independent channel blocks
`B0 + lambda_i Bg`.  For the proposed `n=128,m=32` regime, this exposes one 128-dimensional eigendecomposition
plus 128 parallel 32-dimensional factorizations, which is a plausible CPU/GPU path after statistical validation.

**Rejected for the first optimized path:** several independently weighted, noncommuting item kernels
`sum_g K_g tensor Bg`.  They generally cannot be simultaneously diagonalized.  Select a nonnegative convex
item-kernel mixture first; retain dense LMC as a statistical reference.

**Not yet authorized:** CUDA performance claims or a specialized inverse-root implementation.  The existing
dense QR conditioner remains the correctness baseline until real repeated-judge covariance passes its gates.

**Reconsider if:** multiple item kernels show separately identified held benefit large enough to justify a
joint/block diagonalization or iterative solver.

## 2026-07-12 — distinguish rank deficiency from positive-spectrum conditioning

**Decision:** report both exact numerical rank and `positive_spectrum_condition_number`.  The latter describes
the ratio among nonzero eigenvalues only; it is not the ordinary condition number of a singular matrix.
Closed-neighborhood Gram kernels can be rank deficient by construction, so a finite value must never imply
that the full kernel is invertible.

**Rejected:** silently dropping zero eigenvalues while labelling the remaining ratio `condition_number`.

**Operational consequence:** downstream covariance construction still adds the separately selected nugget or
block-noise term before whitening.  Kernel diagnostics do not authorize directly inverting a singular kernel.

## 2026-07-13 — replace hard source-region separation with an explicit exposure bridge

**Decision:** keep the exclusive connected source regions, discard the failed three-hop-core requirement, and
represent cross-region exposure by the normalized Gram of region-average cumulative-walk features.  Lift that
PSD Gram through a cap-constrained greedy allocation minimizing each exact incremental exposure and report the
exact stipulated-correlation design effect over the frozen `rho` grid.

**Reason:** full regions have usable raw endpoint capacity, while hard cores discarded 67--98% of nodes.  A
feature Gram preserves PSD and makes the dependence assumption inspectable.  The stipulated `rho` path
separates topology exposure from unknown residual amplitude and includes within-region correlation explicitly.

**Authorization boundary:** topology alone cannot identify an effect size, residual covariance, exact
candidate exposure, or end-to-end power.  A narrow pass unlocks nothing; it supplies fixed matrices to the
next full-procedure null/power extension.  Only a passing power gate may unlock attempted-input identities.
This audit does not select `K` or authorize candidates, Nomic, judges, covariance promotion, independent
batching, QR, or CUDA.

**Rejected:** calling regions independent; treating effective rank, Kish ESS, or a mean-test proxy as the
complete campaign power calculation; tuning thresholds from the resulting matrices; and using entrywise lower
covariance bounds for inference.

**Reconsider:** candidate selection becomes eligible only after Stage A source-atomic full-procedure power with
two-way prompt/source inference, followed by history exclusions, exact structural enumeration, and realized-
design revalidation.  The ordering is Stage A power on the registered diagnostic allocation, then identity-
only history if it passes, then Stage B exact candidate enumeration and revalidation.

## 2026-07-14 — calibrate source power over the complete dependence family

**Decision:** treat each corpus-specific `eta_source,c={0,.025,.05,.10,.20}` as five nuisance environments,
not an ordered covariance confidence band, and cover their full 25-pair Cartesian product.  Use source-atomic
folds, split-contained source-spreading prompts, a component-marginal quasi-NLL, and one conservative
graph-aware prompt-plus-source multiplier spanning both endpoints, both corpora, and all five values per
corpus.  Calibrate the selector against both block and source-smooth-mean nulls in every pair.

**Reason:** `H E H.T-I` has positive and negative eigenvalues.  Increasing `eta_source,c` raises the variance of
the equal-weight mean in these audited allocations, but it does not Loewner-dominate every nonlinear selector
or endpoint contrast.  Calling `.20` a universal upper dependence envelope would therefore be
anti-conservative for some directions.  Nor must the exploratory and fresh corpora share one coupling
strength, so five equal-strength worlds do not establish joint power.  The marginal score keeps the
within-component covariance question identified; source and prompt dependence are handled by atomic fitting,
robust inference, and the complete finite-null procedure.

**Selection and authorization:** a complete discovery run may choose the smallest passing `G`, then the
coarsest passing `K`, but unlocks nothing.  Only a seed-disjoint confirmation at that fixed pair, using
simultaneous Bonferroni-adjusted exact binomial bounds, may unlock attempted-input identities.  Stage B must
rerun the fixed-design procedure for the realized candidate exposure and prompt incidence.  A failed
confirmation is terminal absent a prospective cross-attempt alpha-spending amendment; new seeds alone do not
reset multiplicity.

Stage B is ordered: identity-only history, topology-only enumeration and capacity/provenance gates, a
revision-pinned Nomic cache first used only for agreement-cell quotas, immutable exact packing at the selected
`(K,G,R=3)`, then the Nomic Gram on those fixed components and fixed-design realized revalidation.  Continuous
Nomic values cannot trigger repacking or candidate optimization, and judge calls stay locked throughout.

**Rejected:** `.20`-only error control; equal-strength-only corpus worlds; prompt-only clustering;
source/prompt inclusion-exclusion with an indefinite subtraction; inverse-variance component weights; observed
power of exactly `.80` without an exact lower bound; automatic fallback after failed confirmation; and scalar
ESS/row-sum certification of the realized design.

## 2026-07-17 — general leaky semantic diffusion, without covariance promotion

**Observed:** the exact immutable Stage-A source-power run completed all 18
configurations. All 12 evaluable R=3 discovery designs failed their
preregistered gates, the six R=4 diagnostics were non-evaluable by design, no
confirmation design was selected, and every downstream authorization remained
false.

**Decision:** implement semantically weighted, shunt-grounded diffusion as a
general graph/numerical primitive rather than as a promoted residual-covariance
model. Topology defines the edge support. Frozen external embedding distance
may modulate conductance only on those edges. Uniform or component-covering
shunt conductance makes the combinatorial Laplacian positive definite and
gives the regularized-Laplacian diagonal term a physical leakage-to-bath
meaning.

**Numerical consequence:** for grounded precision J=L+diag(alpha)=U.T U, U is
directly the inverse-covariance root of the Green kernel. The model diagonal
alpha is recorded separately from any future floating-point jitter. Dense
equilibrium and heat kernels are correctness references; primary solves use
the root and do not form an inverse.

**Accepted:** analytic source response, Green correlation normalization,
grounded effective-resistance distance, sparse boundary grounding when every
component reaches a shunt, and Nomic as the preferred future semantic
conductance candidate because it is not the deployed e5 input.

**Rejected for this PR:** semantic shortcut edges, arbitrary shortest-path RBF
covariance, asymmetric transition matrices, calling leakage numerical jitter,
interpreting the resulting Green kernel as empirically validated judge
covariance, lowering the failed Stage-A gates, or making a CUDA performance
claim.

**Reconsider:** sparse and CUDA specialization requires a matched full-cost
crossover benchmark. Statistical use as cross-item measurement covariance
still requires new prospective data or a separately authorized design with
train-only fitting and dependence-aware held validation.

## 2026-07-17 — local Dirichlet grounding for million-node graphs

**Decision:** bound anchored diffusion with an outcome-blind top-`K` graph
domain. Use multi-source hop distance as the primary ordering and a
revision-pinned Nomic-resistance weighted shortest path as the secondary
ordering. For retained set `S`, preserve every severed edge through
`beta_i=sum_{j not in S} c_ij` and solve
`J_S=L_ind+diag(alpha+beta)`. The exterior is one fixed zero-temperature bath;
it is not silently deleted.

**Calibration:** choose uniform model leakage from the killed-response ratio
`G_alpha(i,s)/G_alpha(s,s)`, targeting at most `exp(-1)` on a frozen e-fold
shell. Treat that as a screening length, not automatically an accurate hard
truncation: prefer a boundary near `exp(-3)` or one-percent attenuation when
the memory budget permits. Verify `K`, `2K`, and `4K` nested-domain
stability, harmonic measure of the artificial cut, and the fraction of
injected current absorbed by the cut.

**Batch consequence:** construct one shared union or multi-source domain and
one grounded operator for all anchors whose cross-relations are used jointly.
Splicing columns from separate anchor-specific operators is not guaranteed
symmetric or PSD.

**Reason:** a million-node dense float64 matrix is already about 8 TB before
factorization. The local Dirichlet system preserves the physical meaning of
the omitted adjacency while making work depend on the touched neighborhood.
It also supplies monotone, outcome-blind diagnostics for deciding whether the
boundary is far enough away.

**Rejected:** dropping cut edges and thereby imposing an insulating boundary;
embedding-only nearest neighbors that create topological shortcuts; selecting
`S` with the fitted Green distance; using correlation-normalized response to
calibrate leakage; setting `alpha=0` merely because a cut bath exists;
per-anchor matrices presented as one joint kernel; or tuning `D`, `K`, or
`alpha` against held residual outcomes.

**Implementation status:** this change ships the strict-`K` multi-source hop
selector, exact cut-shunt assembly, common-bath dense factor, leakage
calibration, and nested-domain diagnostics. It also ships a constant-degree
CPU microbenchmark. Nomic-resistance Dijkstra, hub-streaming adjacency,
precomputed weighted cut degrees, sparse solvers, and CUDA remain follow-ups.

**Not authorized:** learned judge covariance, relaxed source-power gates,
sparse/CUDA performance claims, or hidden numerical jitter. Sparse direct and
iterative backends remain engineering candidates only after parity with the
dense local reference and a matched end-to-end crossover benchmark.
