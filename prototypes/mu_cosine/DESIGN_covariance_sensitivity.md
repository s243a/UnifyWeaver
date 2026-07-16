# Covariance misspecification sensitivity — nested shrinkage protocol

## Status and question

This is a post-hoc follow-up to the merged structured-residual evidence gate (#3671).  That gate established
that the fitted semantic/graph covariance models did not improve held estimates when their plug-in covariance
was treated as exact.  It did **not** establish that accurate correlation information is useless.

The question here is narrower:

> Did #3671 fail because the tested correlation geometry had no useful held signal, or because a real but
> uncertain covariance estimate was too aggressive when inverted?

No new judge calls are used.  Results remain GPT-5.5 operating-judge fidelity rather than independent truth.
Because this hypothesis was formulated after inspecting #3671, this study is explicitly post-hoc and uses new
outer partition seeds rather than retroactively changing the earlier gate.

### Synthetic-control erratum (recorded before the corrected v2 rerun)

The first v1 synthetic output is retained only as an invalidated diagnostic.  It scored the residual after a
train-fitted KRR mean, `c = e_H - W e_T`, but used the pre-KRR held covariance `R_HH` as its “known truth.”  Even
for fixed `W`, the correct covariance is

```text
R_HH + W R_TT W.T - R_HT W.T - W R_TH,
```

and selecting the KRR kernel/ridge from the same outcome field makes `W` random.  Consequently the original
known-truth NLL/posterior recovery fractions are invalid (the spuriously positive block-null denominator is a
direct symptom).  Corrected v2 separates an end-to-end KRR selection/harm track, which makes no such recovery
claim, from an oracle-zero-mean/known-block mechanism track where `R_HH` really is the scored covariance.

The first wrong-geometry control was also underpowered by construction: its narrow unrelated RBF had roughly
32 times less RMS off-item coupling.  Corrected v2 uses a fixed derangement congruence `P K P.T`, preserving
PSD, spectrum, maximum coupling, and off-diagonal energy while assigning covariance to the wrong item pairs.
Because this also preserves the dense uncentered RBF's positive/common mode, the deranged result is a
common-mode diagnostic rather than an orthogonal block-selection negative control.

## Why covariance error can matter

For the prior-centered conditional update,

```text
A(R) = P^-1 + J.T R^-1 J
b(R) = J.T R^-1 r
x(R) = A(R)^-1 b(R).
```

For a symmetric perturbation `dR`, the first-order posterior-mean response is

```text
dx = -A^-1 J.T R^-1 dR R^-1 (r - J x).
```

Square-root QR stabilizes the computation for a supplied `R`; it cannot make a misspecified `R` statistically
correct.  This protocol therefore measures both predictive performance and the response of the estimate to
PSD-preserving covariance perturbations.

## Gaussian distance uncertainty and the RBF kernel

The covariance indices are kept distinct:

```text
Cov(error[item i, channel a], error[item j, channel b])
    = K_item(i,j) * Sigma_channel(a,b).
```

The product is ordinary scalar multiplication.  If `Sigma_channel` is fixed, then
`E[K_item * Sigma_channel] = E[K_item] * Sigma_channel`.

If a signed scalar separation `D ~ Normal(mu, tau^2)` is inserted into an RBF,

```text
E exp(-D^2 / (2 ell^2))
  = ell / sqrt(ell^2 + tau^2)
    * exp(-mu^2 / (2 (ell^2 + tau^2))).
```

For isotropic Gaussian uncertainty on two feature vectors, the normalized expected kernel is an RBF with a
broader effective length scale.  Thus homogeneous Gaussian distance uncertainty is already a length-scale
misspecification problem.  Merely drawing arbitrary covariance entries from a Gaussian distribution would
not guarantee a PSD matrix and is excluded.

Because the source predictions live nominally in mu-space with neutral point `0.5`, also record the distinct
signed linear geometry

```text
z_i = calibrated_source_vector_i - 0.5
K_mu(i,j) = z_i.T z_j / (norm(z_i) norm(z_j)).
```

This is a normalized Gram matrix and therefore PSD; positive similarity means that the available sources put
two items on broadly the same side of neutral.  It is **not** an RBF correction: subtracting the same `0.5`
from both arguments leaves every RBF distance unchanged.  The six-vector is the train-fitted calibrated
`[prior_D, prior_S, graph_D, graph_S, luna_D, luna_S]` representation, evaluated on held rows without held
outcomes.  Affine calibration may extrapolate slightly outside `[0,1]`; values are not clipped after looking
at held data.

The centered-mu kernel replaces the graph-feature RBF in a separately labelled secondary geometry family.
It uses semantic multipliers `{0.5, 1, 2}`, the same `alpha` and `beta` grids, and no graph bandwidth.  It gets
its own nested selection and null-max calibration and cannot make the primary graph/semantic gate pass.
Normalization makes this a direction/sign kernel and deliberately removes the literal product's amplitude
with distance from `0.5`.  The unnormalized product is implemented and tested as a PSD Gram, but is excluded
from this grid because its item-varying diagonal would change the within-item marginal; testing it fairly
requires a later per-item marginal-whitening extension rather than pretending it is the normalized kernel.

### Distance-separated batching implication

For a defensible RBF geometry, sufficiently separated items have negligible off-item covariance.  The
operational criterion is not raw distance alone but the whitened cross-block coupling

```text
norm(B_item^-1/2 R_ij B_item^-1/2, 2) <= epsilon_batch.
```

For `K(d)=exp(-d^2/(2 ell^2))`, the scalar kernel is about `0.011` at `d=3 ell` and `0.00034` at `d=4 ell`.
This motivates a future distance-separated batching policy: retain the full within-item judge block, put
items with negligible predicted coupling in the same independent/block update, and keep nearby connected
components separate or condition them jointly.  Distance uncertainty broadens the effective kernel, so the
separation threshold must use the uncertainty-inflated length scale.  This is a sufficient engineering rule
when the geometry is validated; failure of the tested geometry does not prove nearby residuals independent.

## Data and nested split contract

Use the same immutable campaign, Luna, checkpoint, graph, and E5 artifacts recorded by #3671.

For each corpus and each **outer seed 10 through 19**:

1. Create the same strict node-disjoint 40% held-node split with 64 outcome-blind candidates.  The retained
   outer-held rows are never used for calibration, covariance fitting, or hyperparameter selection.
2. Create three inner node-disjoint partitions with seeds `10000 + 3*outer_seed + k`, `k=0,1,2`, 35% held
   nodes, 64 candidates, and campaign-neighborhood stratification.  These are repeated inner partitions, not
   independent population folds.
3. On each inner-fit partition only, refit source calibration, `P/C/R`, graph feature scaling, regional-mean
   selection, RBF base bandwidths, and covariance components.
4. Select covariance trust/geometry by macro inner-held joint conditional-residual NLL per scalar, subject to
   the conservative stability rule below.
5. Refit every selected statistical object on all outer-train rows, relative to newly computed outer-train
   median bandwidths, then evaluate once on outer-held.

The outer seeds are new partitions but not new labels; mean and SD remain partition-stability descriptions,
not population confidence intervals.

## PSD-preserving sensitivity family

The independent baseline is `R_block = I kron B_block`.  The structured endpoint is the separable model from
#3671, refit at each semantic/graph kernel-scale pair:

```text
R_struct = I kron B0
         + (w_sem K_sem + w_graph K_graph) kron B_shared.
```

Before varying correlation, match the structured endpoint to the block marginal.  If `A_struct` is the
constant within-item diagonal block of `R_struct`, form the correlation-like matrix

```text
C_struct = (I kron A_struct^-1/2) R_struct (I kron A_struct^-1/2).T
```

using Cholesky solves, not an explicit inverse.  Then, with `B_block = L_block L_block.T`, define the primary
path

```text
R(alpha) = (I kron L_block)
           [(1-alpha) I + alpha C_struct]
           (I kron L_block).T.
```

Every point is PSD and every within-item block remains exactly `B_block`; only cross-item correlation changes.
The direct convex blend of the two raw covariances is retained as a secondary diagnostic and cannot be used
to claim a correlation-only effect.

Three uncertainty axes are varied:

1. **Correlation strength / trust**

   ```text
   alpha in {0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0}.
   ```

   `alpha=0` means do not trust cross-item correlation; `alpha=1` uses the full fitted correlation pattern
   while retaining block-model marginals.

2. **Item-distance scale**

   ```text
   semantic multiplier in {0.5, 1.0, 2.0};
   graph-feature multiplier in {0.5, 1.0, 2.0}.
   ```

   `1.0` is each fit partition's train-median RBF bandwidth.  Values below/above it test correlations that
   decay faster/slower.  The broader side also covers homogeneous Gaussian input-distance uncertainty after
   kernel normalization.  The component factors are refit for every one of the nine bandwidth pairs.

3. **Channel-covariance shape**

   ```text
   B_shared(beta) = (1-beta) B_shared + beta diag(B_shared),
   beta in {0, 0.5, 1.0}.
   ```

   This is also PSD.  It tests whether uncertain cross-channel interaction terms, rather than item correlation
   itself, destabilize the estimate.

Equivalent `alpha=0` copies count once.  A nonzero candidate must beat block on at least two of the three inner
partitions.  Among eligible candidates, find the best macro inner NLL, then select the smallest `alpha` within
`0.001` NLL/scalar of it; remaining ties prefer larger `beta`, then semantic/graph multipliers closest to 1.0.
If no eligible candidate has positive macro gain, select block.  This conservative rule is fixed before outer
results exist.

## Outer-held records

For every outer split report:

- regional block (`alpha=0`);
- the nested-selected covariance;
- the full plug-in covariance at the nested-selected geometry (`alpha=1`);
- the original canonical plug-in endpoint (`alpha=1`, multiplier 1, `beta=0`);
- the best point on the **outer-held** perturbation grid, clearly labelled a non-deployable oracle diagnostic.

The oracle is upward-biased by grid search and is not evidence for a usable estimator.  On each exact held
geometry, draw 200 residual fields from the fitted block null, repeat the identical oracle grid search, and
record the 95th percentile of the null maximum gain.  A real oracle gain is called `above_null_max95` only
when it exceeds that split-specific threshold.  Even then it remains diagnostic.  It separates two possible
failure modes:

- null-calibrated oracle headroom but nested selection fails: possible covariance value with inadequate
  estimation/selection;
- even oracle does not help: magnitude/length/channel-shape errors within this family do not explain #3671.

## Metrics and sensitivity diagnostics

Primary:

- outer-held full joint conditional-residual Gaussian NLL per scalar.

Posterior/decision guardrails:

- bivariate posterior NLL, MSE, Mahalanobis/dimension, and 95% ellipse coverage;
- accuracy, log-loss, ECE-10, and margin AURC from an outer-train-only bridge;
- prior/noise diagonal loading.

Sensitivity:

- normalized analytic directional derivative of the posterior mean along `R_struct - R_block`;
- mean per-item norm of `dx/dalpha`;
- posterior-mean and NLL ranges along the nested-selected correlation path, plus NLL across the full outer
  perturbation grid (conditioning every oracle candidate would turn a diagnostic grid into an avoidable
  cubic-time model-selection workload);
- selected `alpha`, kernel multiplier, and `beta` stability across partitions.

For covariance-estimation stability, use 100 deterministic 80%-node induced subsamples of each outer-train
set at fixed selected **primary-family** hyperparameters.  Refit calibration, regional mean, scaling, and
covariance on each subsample; this is a stability analysis, not a bootstrap confidence interval.  Report the
whitened covariance error radius, posterior-sigma-normalized mean displacement, mean marginal symmetric
Gaussian KL, log-standard-deviation change, and decision-flip rate.  Individual rows are never bootstrapped
as though independent.  The later-added centered-mu geometry is an endogenous secondary diagnostic and does
not double this 2,000-refit stability budget; it cannot make the primary gate pass.

Before fitting an outer split, deterministically enumerate all requested induced-node subsets.  Every subset
must retain at least 20 rows and non-degenerate semantic and graph-feature geometry; otherwise abort that
split before its expensive fits rather than silently dropping, replacing, or redefining a replicate.  The
composite-likelihood pair-sampling seed is held fixed across the 100 subsets so the stability distribution
reflects changed nodes/data rather than an avoidable second source of optimizer randomness.

## Synthetic truth control

Run 100 **selection-procedure controls with known covariance endpoints** per frozen scenario on deterministic
RBF item geometry with known block and Kronecker channel covariance.  Each replicate refits the regional mean
and block marginal and runs the same three-fold conservative `alpha`/length/channel selector, but deliberately
does not re-run the real-data LMC optimizer: these controls isolate selector power and covariance
misspecification from component-estimation quality.  Scenarios are block-null, regional-mean-only, wrong item
geometry, and in-family maximum whitened coupling 0.04, 0.10, and 0.20.  Compare residual risk using:

- the true covariance;
- the independent block covariance;
- under- and over-correlated strength;
- incorrect shorter/longer RBF scales;
- incorrect channel interaction shrinkage.

For null, mean-only, and wrong-geometry scenarios, require mean held harm no greater than 0.001 NLL/scalar and
block/conservative selection in at least 80% of replicates.  At coupling 0.10 and 0.20, require nonzero-alpha
selection in at least 80% and recovery of at least 50% of the true-covariance oracle NLL/posterior-risk gain.
Coupling 0.04 is a measured-power scenario with no required success.  These are mechanism checks, not evidence
that the campaign has that covariance.

## Historical v1 interpretation gate — disabled after synthetic controls

The text below records the originally frozen v1 gate.  It must **not** be applied: v1 failed family-wise
synthetic calibration, and its conditional fixed-path null omits mean/marginal fitting uncertainty.  No v1
real-data result can be called robustly useful or oracle-headroom evidence under this rule.  The versioned v2
requirements are in `DESIGN_covariance_sensitivity_v2_null_calibration.md`; a real-data full-procedure v2
selector is not implemented in this PR.

The nested-selected covariance is considered robustly useful only if it has positive mean outer-held residual
NLL gain over block on both corpora, is positive on at least 8/10 seeds per corpus, does not worsen posterior
NLL, and keeps decision log-loss/AURC degradation within 0.01.

If that gate fails but the real oracle gain exceeds its null-max95 threshold and is positive on at least 8/10
splits in both corpora, report **oracle headroom consistent with an estimation limitation** and prioritize
better covariance data (repeated judges/independent targets).  This is not a deployable estimator claim.
If both nested and null-calibrated oracle diagnostics fail, retain the block model for these kernels; this
still does not rule out correlation supplied by a different, better-measured geometry.

The inherited graph-feature geometry includes `hit_prob`, which is also the graph-D measurement.  It is
available at inference and is not target leakage, but it makes the covariance explicitly conditional on an
observed measurement rather than an exogenous noise law.  Results must be described that way; a topology-only
kernel excluding `hit_prob` remains a future confirmatory geometry.

No inverse-root optimization or CUDA claim is in scope.
