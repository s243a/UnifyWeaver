# Confirmatory graph geometry for cross-item residual covariance

## Status and scope

**Frozen before running the graph-geometry synthetic benchmark or any new geometry comparison on real
residuals.**  This design follows the merged adjacent-residual pilot.  That pilot found a stable descriptive
adjacency-plus-hop signal but could not identify stochastic covariance separately from persistent item effects
and failed its synthetic power gate.

This workstream asks which outcome-blind item geometry, if any, can predict transferable cross-item conditional
residual covariance.  It does not change the square-root/QR conditioner: that implementation already accepts a
dense covariance.  It does not promote `JointPosterior`, which remains a learned decision comparator rather
than a covariance factorization.

The first PR is infrastructure and mechanism evidence only:

1. specify PSD-safe graph and independent-embedding kernel families;
2. implement their deterministic feature maps and small-graph references;
3. audit planted-geometry recovery, misspecification, and covariance sensitivity synthetically;
4. preregister the repeated-judge campaign that would be required for a deployment claim.

No kernel may enter production QR, certify independent batching, or relax the 95% simultaneous safety rule in
this PR.

## Statistical object

For measurement item `i`, after train-only calibration, prior/measurement conditionalization, and mean removal,
let the conditional residual be

```text
q_i = m(x_i) + epsilon_i,
E[epsilon_i | x_i] = 0,
Cov(epsilon_i, epsilon_j | z_i, z_j) = alpha K_theta(i,j) B_cross.
```

`B_cross` is a PSD judge/channel covariance contribution.  The exact within-item marginal `B` is preserved by
the correlation path used by the conditioner.  In the first mechanism study, `B_cross=B` to isolate item
geometry; the confirmatory study must estimate judge-axis structure from independent repeated calls.

The crucial ordering is:

1. fit the mean on outer-training endpoint components;
2. compute residuals on disjoint components;
3. learn kernel scales/weights and covariance amplitudes inside training components only;
4. score the entire frozen procedure on outer-held components.

A geometry that only absorbs a missed regional mean is not residual covariance evidence.

## Distance is not automatically a covariance kernel

A scalar distance `d(i,j)` is useful for sampling and interpretation, but `exp(-d^2/(2 ell^2))` is PSD for
Euclidean/negative-type distances, not for every graph shortest-path metric.  The implementation therefore
constructs kernels from one of three PSD certificates:

- an explicit feature Gram `Phi Phi.T`;
- a nonnegative spectral function of a symmetric graph Laplacian;
- a nonnegative sum or Schur product of already-PSD kernels.

Every materialized item kernel must be symmetric, unit-diagonal, finite, and have minimum eigenvalue at least
`-1e-10` in float64.  Numerical clipping may remove roundoff below that tolerance; it may not repair a
structurally indefinite candidate.

## Frozen graph candidate family

All graph topology is outcome-blind.  Candidate hyperparameters are selected within outer-training components.

### G0 — independent block

`K=I`.  This is the production baseline and is always an eligible fallback.

### G1 — closed-neighborhood Gram

The merged pilot's same-descendant cosine between binary closed one-hop root neighborhoods.  It is the fixed,
low-capacity baseline and must reproduce the existing implementation.

### G2 — finite-hop walk-feature diffusion (primary scalable family)

For root `r`, let `p_h(r)` be its outcome-blind `h`-step random-walk landing distribution on the undirected
graph, with `p_0(r)` a point mass.  For nonnegative hop weights `w_h`, define

```text
Phi(r) = concat_h sqrt(w_h) p_h(r),
K_walk(r,s) = <Phi(r),Phi(s)> / (||Phi(r)|| ||Phi(s)||).
```

This is PSD by construction for any nonnegative weights.  It is local and sparse for small maximum hop, avoids
whole-graph diagonalization, and generalizes the one-hop overlap idea.  The frozen mechanism grids are:

```text
local      w = (1, 1)
two_hop    w = (1, 1, 1)
decay      w = (1, 1/2, 1/4, 1/8)
```

Future learning may use a softmax over hop logits, preserving nonnegativity.  Unconstrained signed hop weights
are not allowed.

### G3 — heat/diffusion kernel (small-graph mathematical reference)

For symmetric normalized Laplacian `L`,

```text
K_heat(t) = exp(-t L),       t in {0.25, 0.5, 1, 2, 4}.
```

The diagonal-normalized form is PSD.  Exact eigendecomposition is a reference for small components and
synthetic checks, not the large-graph execution plan.  This follows the diffusion-kernel construction of
[Kondor and Lafferty (2002)](https://people.cs.uchicago.edu/~risi/papers/diffusion-kernels.pdf).

### G4 — regularized-Laplacian/resolvent kernel (small-graph reference)

```text
K_resolvent(tau) = (I + tau L)^-1,  tau in {0.25, 0.5, 1, 2, 4}.
```

The diagonal-normalized inverse is PSD for `tau>0`.  It decays spectrally more slowly than heat and is a useful
longer-range comparator; see [Smola and Kondor (2003)](https://people.cs.uchicago.edu/~risi/papers/SmolaKondor.pdf).

### Role gating

The first confirmatory estimand retains the pilot's same-descendant restriction:

```text
K_item((left_i,root_i),(left_j,root_j))
  = 1[left_i=left_j] K_root(root_i,root_j).
```

The indicator is itself a one-hot Gram, so the Schur product remains PSD.  Cross-descendant covariance is a
later extension requiring its own repeated-call evidence.

## Independent embedding geometry

An embedding candidate is acceptable only when it is frozen before the judge campaign, available at inference,
not trained on the campaign labels, revision-pinned, and stored with content-addressed provenance.  “Not called
a judge” is insufficient if it shares the same outcome labels or representation stack.

The current mu readouts consume e5, so e5 geometry is a **shared-input control**, not the preferred independent
semantic geometry.  Two locally available candidates are frozen:

- `sentence-transformers/all-MiniLM-L6-v2`, revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, symmetric title encoding;
- `nomic-ai/nomic-embed-text-v1.5`, revision
  `e9b6763023c676ca8431644204f50c2b100d9aab`, using the model-card `clustering:` prefix because the task is
  symmetric semantic grouping, not query/document retrieval.  Nomic's open embedding design is described in
  [Nussbaum et al. (2024)](https://arxiv.org/abs/2402.01613).

Nomic is the primary independent candidate; MiniLM is the lower-capacity comparator.  This is a preregistered
ordering, not a claim that Nomic will win.

For normalized node vectors `u(node)`, form the role-aware item feature

```text
f_i = normalize(concat(u(left_i), u(root_i))).
```

Use a Euclidean RBF with median bandwidth fit on outer-training components, then apply the same-descendant Gram
gate.  Raw cosine is retained only as a diagnostic.  The cache builder must never download in tests and must
record model id, exact revision, task prefix, normalization, node ordering, and a SHA-256 over the emitted
arrays/metadata.

## Combining graph and embedding geometry

The primary combination is a nonnegative convex sum,

```text
K_mix = gamma K_graph + (1-gamma) K_embed,
gamma in {0.25, 0.5, 0.75}.
```

This asks whether either geometry explains an independent residual mode.  The Schur product
`K_graph * K_embed` is PSD and is retained as a secondary “both must agree” comparator, but it can erase a real
graph correlation when the external embedder misses domain-specific proximity.  No candidate count may be
expanded after held results are inspected; any expansion requires an amended preregistration and repeated null
calibration.

## Judge-derived mu distance

Distance from the same observed judge measurements is excluded: measurement noise would enter both the
covariance predictor and the residual being explained.  A distance from a cross-fitted `mu_hat` is a valid
future conditional-covariance covariate only when it is produced on disjoint endpoint components and preferably
leave-one-judge-family-out.  Even then its interpretation is “covariance conditional on `mu_hat`,” not
independent structural evidence.  It is preregistered as a diagnostic ablation after repeated calls exist, not
as a primary geometry.

## Synthetic mechanism audit

Before any new real-residual comparison, generate independent graph components and repeated residual fields
under block, walk-feature, heat, resolvent, and deranged-kernel truths.  Use known zero mean and known channel
covariance first; label this mechanism-only.

For each candidate, select covariance amplitude from

```text
alpha in {0, .025, .05, .10, .20, .35, .50}
```

on training fields, then score untouched held fields by Gaussian NLL per scalar.  Report:

- block-null nonzero-selection rate and held harm;
- correct-geometry selection and NLL recovery;
- wrong-geometry/derangement rejection;
- sensitivity when the assumed alpha is below/equal/above truth;
- kernel eigenvalue, condition-number, and diagonal-loading diagnostics.

Frozen mechanism gates:

1. every kernel construction is PSD and unit-diagonal without structural repair;
2. block-null nonzero selection is at most 10%;
3. at planted `alpha>=0.10`, the correct family has positive mean held gain and beats the deranged candidate in
   at least 80% of replicates;
4. an over-coupled candidate may not improve mean held NLL merely by numerical loading;
5. all deployment flags remain false regardless of these outcomes.

Failure narrows or redesigns the candidate family; it does not prove accurate covariance would be useless.

## Confirmatory repeated-judge campaign

The mechanism audit does not identify real stochastic covariance.  The next data campaign must:

1. sample balanced `(anchor, adjacent/local positive, matched distant or hard negative)` triples;
2. balance hop, campaign tag, degree bin, and graph component by construction;
3. stratify outcome-blind graph-kernel similarity and independent-embedding similarity so their disagreement is
   observed, not extrapolated;
4. obtain at least three independent calls per judge family and preserve call identity;
5. use more than 400 independent endpoint components, with the final count chosen by a full-procedure power
   simulation rather than computational convenience;
6. split whole endpoint components and refit mean, marginal covariance, kernel weights, amplitude, and any
   selector inside each outer fold;
7. include a different judge family or non-LLM target for generality claims.

Primary gates are outer-held joint residual NLL, posterior NLL/risk, calibration, and numerical loading.  A
geometry earns deployment consideration only if the 95% simultaneous lower bound on held benefit is positive.
Distance-separated batching additionally requires a 95% simultaneous **upper** bound on whitened cross-block
coupling below a preregistered `epsilon_batch`.

## Rejected or deferred alternatives

The rationale and reconsideration conditions are maintained in `DECISIONS_graph_geometry.md`.  At freeze time:

- shortest-path Gaussian covariance: rejected as a direct kernel because PSD is not guaranteed on an arbitrary
  graph; retain shortest path for sampling strata;
- naive random-walk/PPR matrix: rejected because it is generally asymmetric on irregular graphs; use a feature
  Gram or symmetric Laplacian construction;
- unconstrained learned dense covariance/kernel: rejected for low-sample overfit, weak identifiability, and PSD
  risk;
- e5 as the primary semantic distance: rejected because it is a shared input to the current mu readouts;
- same-judge observed-mu distance: rejected for endogeneity;
- effective-resistance geometry: deferred because disconnected components, hub/global-edge sensitivity, and
  pseudoinverse cost complicate the first confirmatory cut;
- whole-graph exact heat/resolvent diagonalization and CUDA optimization: deferred until a geometry passes the
  statistical gates;
- lowering the 95% deployment confidence level: rejected because it changes error tolerance rather than adding
  information.

## Outcome-blind topology amendment — cumulative walk features

This amendment was frozen after the first campaign **geometry-only** inventory and before any comparison with
judge residuals.  The original G2 concatenation places each `p_h` in a separate feature block.  On the campaign
roots its kernel was nearly identity: mean distance was `0.9991` adjacent versus `0.9982` nonadjacent
exploratory, and `0.9973` versus `0.9919` fresh.  Separate blocks omit cross-hop overlap, so `p_0(a)` cannot
match `p_1(b)` even when `a` and `b` are directly connected.

Retain that same-hop construction as a negative/control geometry.  Add the primary scalable variant

```text
Phi_cumulative(r) = sum_h sqrt(w_h) p_h(r),
K_cumulative(r,s) = <Phi_cumulative(r), Phi_cumulative(s)>
                      / (||Phi_cumulative(r)|| ||Phi_cumulative(s)||).
```

It is an explicit Gram and therefore PSD.  Nonnegative weights keep its diffusion interpretation.  The frozen
topology grids remain `local=(1,1)`, `two_hop=(1,1,1)`, and `decay=(1,1/2,1/4,1/8)`.  Adding it to a covariance
selector requires a separately calibrated v3 mechanism audit; v1/v2 outputs are not rewritten.
