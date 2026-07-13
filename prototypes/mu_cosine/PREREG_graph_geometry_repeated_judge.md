# Repeated-judge graph-covariance campaign — preregistration

## Status and authorization boundary

**Frozen before running the new power harness, selecting any new endpoint, or making any fresh judge call.**
This PR is deliberately no-spend: it freezes the candidate capacity, outcome-blind sampler, repeat-aware data
contract, simulation, estimands, and decision rules.  It creates no labels, authorizes no API/judge calls, and
cannot unlock covariance deployment, independent batching, QR specialization, or CUDA claims.

If a named judge or prompt is unavailable, this document must be amended before any fresh response exists.
Operational dry runs may use synthetic responses only.  A 64-component live engineering pilot is permitted
only under a separate amendment and must be permanently excluded from confirmation.

## Question and experimental unit

The question is whether an outcome-blind graph/semantic geometry predicts transferable cross-item conditional
measurement error after calibration and mean removal.  The independent unit is an endpoint-disjoint component
containing three rows with a shared descendant `x`:

```text
anchor:             (x, a)
adjacent positive:  (x, b), where a--b is a direct undirected graph edge
matched negative:   (x, c), where distance(a,c) >= 3 or c is disconnected
```

The positive and negative roots must match on shortest descendant-to-root hop, campaign tag, and root-degree
quartile.  The sampler balances anchor-to-comparator hop transition, anchor-degree quartile, corpus, and frozen
graph/Nomic agreement class.  No graph endpoint ID or normalized endpoint title may occur in two selected
components or in the historical scored campaigns.  Any graph connected component contributes at most 10% of
a corpus sample where topology permits; otherwise the shortfall is reported and the campaign does not silently
weaken this rule.

Graph/Nomic thresholds are computed and recorded on the frozen structural candidate universe before endpoint-
consuming selection.  Structural near/far defines the sampling contrast; cumulative-walk and Nomic similarities
are retained continuously.  The target count is determined by the power procedure below, separately for every
corpus required to pass.

## Repeated-call estimand

For component `g`, row `i`, judge family `f`, and independent call wave `r`, the conditional residual model is

```text
q_gifr = m(x_gi, f) + u_gif + w_fr + epsilon_gifr.
```

`u` is persistent item-by-judge error, `w` is an optional recorded repeat-wave effect, and `epsilon` is fresh
call noise.  Every selected row is crossed with at least three fresh, stateless requests per judge family.
Four waves are a frozen sensitivity design.  Wave, request, batch, model revision, prompt hash, settings/seed,
timestamp, raw response, retry, and failure identity are retained.  Model or prompt changes create a new
stratum and are never pooled silently.

Repeats are not averaged before variance decomposition.  The analysis reports:

1. within-call D/S covariance and judge-specific sampling variance from repeat deviations;
2. persistent item/judge covariance from repeat means with the `W_call/R` diagonal correction;
3. cross-repeat U-statistic covariance;
4. same-wave covariance after fitted wave-effect removal; and
5. the same-wave-minus-cross-wave contrast.

Arbitrarily paired independent requests cannot identify stochastic cross-judge covariance.  It is zero by
design unless a recorded shared wave/session supports that estimand.  Persistent cross-judge covariance is
estimable.  A stochastic graph-local covariance claim additionally requires a positive simultaneous contrast
in the repeat-specific component; a persistent-only result remains useful prediction error but is not relabelled
fresh-call correlation.

## Frozen judges and claim boundary

The intended measurement families are:

- operating judge `gpt-5.5-low` (model `gpt-5.5`, low reasoning), using the established campaign prompt;
- cheap comparison judge `gpt-5.6-luna` (low reasoning), with train-only global affine calibration first.

Each produces the frozen D/S response schema.  Luna is a second measurement family, not independent truth.
An operating-judge fidelity claim needs repeated operating-judge targets.  A generality/truth claim additionally
requires a preregistered blinded third-family, human, graph-structural, or downstream target subset.

For `G` selected components, three rows, `F` judge families, and `R` repeats, the planned evaluation count is

```text
calls = 3 * G * F * R.
```

The sampler emits this count but never performs a call.

## One deployment-capable geometry family

The production baseline is block independence (`rho_max=0`).  The only nonzero family eligible to win is

```text
K_gamma = gamma K_cumulative + (1-gamma) K_Nomic,
gamma in {0, .25, .50, .75, 1},
rho_max in {.025, .05, .10, .20}.
```

`K_cumulative` uses walk weights `(1,.5,.25,.125)`.  Both endpoints and every convex mixture are PSD and unit
diagonal.  For each non-block kernel, convert the common maximum off-item coupling to its path coefficient via

```text
alpha(K, rho) = rho / max_{i != j} abs(K_ij),
C(K, rho) = (1-alpha) I + alpha K.
```

Ineligible cells with `alpha>=.95` are declared before outcomes.  This avoids the v1 error of comparing a common
path coefficient across kernels with different off-diagonal energy.  The entire `gamma x rho_max` search is one
familywise maximum, not five independent votes.

Closed-neighborhood, same-hop walk, exact heat/resolvent, MiniLM, shared e5, Schur graph-by-Nomic, and equal-
energy topology derangements are diagnostics or negative controls and cannot win deployment.  A broader kernel
zoo requires a new preregistration and recalibrated null before outcomes.

## Component-disjoint fitting contract

Each corpus uses five deterministic outer folds and three inner component folds.  Every row, repeat, and judge
from one endpoint component remains in one fold.  Endpoint IDs and normalized titles are disjoint across folds.
All outcome-dependent objects are refit inside the appropriate training components:

- judge calibration and conditionalization;
- regional/role mean and its ridge choice;
- repeat-wave effects and missing-call policy;
- `W_call` and persistent marginal/channel covariance `B`;
- Nomic bandwidth and any permitted kernel normalization;
- `gamma`, `rho_max`, path coefficient, and statistical loading;
- JointPosterior comparison/calibration and its margin gate.

If the graph judge/model is trained on the adjacent-positive/distant-negative curriculum, training is restricted
to outer-training components and repeated inside each outer fold.  Held components never teach the mean model.

## Frozen full-procedure simulation

The primary sample-size grid per required corpus is

```text
G in {160, 320, 512, 800}, R=3;
R=4 sensitivity at G in {320, 800}.
```

Use 1,999 block-null calibration draws and 200 independent evaluation replicates per scenario/cell.  Synthetic
smoke tests may use smaller explicit CLI values but cannot set the reported recommendation.

The repeat-aware generator uses two judge families by D/S channels (`m=4`) and three rows per component:

```text
q_gir = X_gi beta + u_gi + epsilon_gir,
u_g ~ Normal(0, C_theta,g tensor B_persistent),
epsilon_gir ~ Normal(0, I_item tensor W_call).
```

Candidate item covariances are explicit feature Grams.  Frozen scenarios are block null, smooth-mean only,
cumulative truth, Nomic truth, convex-mixture truth, and equal-energy derangement at `rho_max=.04,.10,.20`.
An optional shared-wave scenario is labelled separately.  Nulls preserve component topology, regional mean,
repeat heteroskedasticity, within-call D/S covariance, missingness, and any wave strata while removing cross-item
coupling.

Every null and alternative replicate repeats component splitting, mean/ridge selection, repeat decomposition,
`W_call`, `B`, kernel/mixture/coupling selection, loading, and endpoint scoring.  Cache only outcome-blind feature
Grams, folds, eigensystems, and linear-system structure.  Seeds derive from `(G,R,scenario,replicate)`; scientific
JSON excludes wall time and output paths.

The finite familywise threshold is the upper-95% null order statistic at one-based rank

```text
ceil(.95 * (K_null + 1)).
```

with strict `observed > threshold` promotion.  Calibration and evaluation null seeds are disjoint.

## Simulation sizing gate

The recommended count is the smallest `G` for which, at `rho_max=.10` and `.20`:

1. independent block-null false deployment is consistent with the nominal 5% familywise rule and at most 10%;
2. full deployment-event power is at least 80% for cumulative, Nomic, and mixture truths;
3. mean selected outer-held residual NLL gain is positive;
4. mean harm is nonpositive under block-null and smooth-mean controls;
5. topology truth beats the equal-energy derangement in at least 80% of replicates; and
6. the procedure does not pass with an incomplete scenario grid.

If no grid value passes, increase independent components.  Do not reduce confidence, drop a difficult truth,
or use additional repeats as a substitute for components.

## Real-data endpoints and simultaneous inference

Primary equal-component paired endpoints are:

```text
d_residual,g  = NLL_block,g - NLL_structured,g
d_posterior,g = posterior_NLL_block,g - posterior_NLL_structured,g.
```

Use a preregistered one-sided component multiplier/max-statistic construction for simultaneous 95% lower bounds
across both primary endpoints and every corpus required to pass.  The familywise selector must reject block and
every primary lower bound must exceed zero.  Secondary requirements are:

- posterior calibration/coverage and Mahalanobis diagnostics do not worsen;
- decision log-loss and margin-gated AURC degradation are each at most `.01`;
- ECE uses ten fixed equal-width bins and AURC uses component bootstrap intervals;
- graph/Nomic source correlations and same-split ablations are reported;
- topology benefit exceeds derangement/hard negatives;
- statistical and numerical loading remain below frozen report-only limits.

JointPosterior is a learned decision comparator; it is not the covariance factorization.  Factored independent
PoE is retained only as a control.

## PSD-safe deployment adapter and batching

No elementwise covariance lower bound is called conservative.  Select `alpha_safe` inside outer training as
the largest frozen-grid value whose multiplicity-adjusted held-benefit lower bound is positive; otherwise use
zero.  In whitened coordinates,

```text
C_alpha = (1-alpha_safe) I + alpha_safe C_hat,
R_safe = (I tensor L_B) [C_alpha + delta_95 I] (I tensor L_B)^T.
```

`delta_95` is the full-procedure 95th percentile of maximum missing spectral mass.  Statistical `delta_95` and
numerical Cholesky loading are distinct and reported separately.  Distance-separated batching additionally
requires a simultaneous 95% upper bound on every proposed cross-batch whitened block norm below a separately
frozen `epsilon_batch`; neither small `alpha` nor added loading establishes independence.

## Decision outcomes

This no-spend PR can conclude only that the protocol and tooling are valid or that they require redesign.  The
later campaign can conclude:

- **promote one item kernel to dense-QR validation**, if every simultaneous/statistical/repeat gate passes;
- **retain block covariance**, if the gates fail; or
- **not identified**, if repeat, power, calibration, or data-quality requirements fail.

Only the first outcome unlocks the single-kernel eigenmode/QR/CUDA comparison at
`N_item=128, M_channel=32, P_state=2`.  Multiple noncommuting kernels, explicit covariance inversion,
eigenvalue clipping as structural repair, randomized eigensolvers, and automatic block discovery remain
deferred.
