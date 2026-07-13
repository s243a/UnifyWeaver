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
measurement error after calibration and mean removal.  The structural sampling unit is an endpoint-disjoint
component containing three rows with a shared descendant `x`:

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

`G` is the component count **per required corpus**.  Each corpus has 32 joint cells (four hop transitions by
four degree quartiles by two agreement classes), so every frozen `G` supports exact equal cell counts.  The
selector rejects a requested per-corpus total that cannot preserve those frozen quotas.

## Repeated-call estimand

For component `g`, row `i`, judge family `f`, independent call wave `r`, and prompt block `p(g)`, the
conditional residual model is

```text
q_gifr = m(x_gi, f) + u_gif + w_fr + b_f,r,role(i),p(g) + epsilon_gifr.
```

`u` is persistent item-by-judge error, `w` is an optional recorded repeat-wave effect, `b` is a fresh shared
prompt-request effect, and `epsilon` is row-specific call noise.  Every selected row is crossed with at least
three fresh, stateless request waves per judge family.
Four waves are a frozen sensitivity design.  Wave, request, batch, model revision, prompt hash, settings/seed,
timestamp, raw response, retry, and failure identity are retained.  Model or prompt changes create a new
stratum and are never pooled silently.

To amortize the system prompt, one confirmatory request contains up to ten rows from distinct endpoint
components and a single row role.  Prompt-block membership is stable across roles, judges, and repeats, so the
request-connected dependence clusters remain bounded.  A block and all of its rows remain wholly inside one
corpus, outer fold, and global inner-fold label; no request crosses an analysis boundary.  The prompt block,
not an individual component, is the conservative inference/resampling cluster.  Request and block effects are
refit inside training folds.  The later `N_item=128,M_channel=32` numerical batch is a different conditioning
layout over already collected measurements.

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

For `G` selected components, three rows, `F` judge families, and `R` repeats, the planned item-evaluation count is

```text
item_evaluations = 3 * G * F * R.
```

The sampler emits this count and the smaller prompt-request count implied by the split-contained blocks, but
never performs a call.

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

Each corpus uses five deterministic outer folds and three inner component folds.  Fold totals differ by at most
one, and each feasible stratum margin is distributed as evenly as integer counts allow; equal per-cell counts
inside every fold are not required.  Every row, repeat, and judge from one endpoint component remains in one
fold.  Endpoint IDs and normalized titles are disjoint across folds.  For every outer-held fold the selector
also materializes the three inner-fold assignments of its outer-training components, before outcomes exist.
Prompt blocks are then formed only among components with the same corpus/outer/inner signature, and whole
prompt blocks—not component rows—are the resampling units for uncertainty intervals.
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
q_gir = X_gi beta + u_gi + w_r + b_r,role(i),p(g) + epsilon_gir,
u_g ~ Normal(0, C_theta,g tensor B_persistent),
epsilon_gir ~ Normal(0, I_item tensor W_call).
```

`w`, `b`, and request-level missingness are generated and refit; uncertainty resamples stable prompt blocks.
Candidate item covariances are explicit feature Grams.  Frozen scenarios are block null, smooth-mean only,
cumulative truth, Nomic truth, convex-mixture truth, and equal-energy derangement at `rho_max=.04,.10,.20`.
An optional shared-wave scenario is labelled separately.  Nulls preserve component topology, regional mean,
repeat heteroskedasticity, within-call D/S covariance, missingness, and any wave strata while removing cross-item
coupling.

Every null and alternative replicate repeats component splitting, mean/ridge selection, repeat decomposition,
`W_call`, `B`, kernel/mixture/coupling selection, loading, and endpoint scoring.  Cache only outcome-blind feature
Grams, folds, eigensystems, and linear-system structure.  Seeds derive from `(G,R,scenario,replicate)`; scientific
JSON excludes wall time and output paths.

The frozen implementation details are: prompt-block capacity 10; five outer and three inner folds; mean-ridge
grid `{0,.01,.1,1,10}`; 5% shrinkage toward channel-diagonal targets for `W_call` and `B`; and a relative
numerical SPD floor of `1e-8`.  A non-block candidate is inner-eligible only when its macro held gain is positive
and at least two of three inner folds are positive.  Ties prefer larger gain, then smaller `rho_max`, then
`gamma` closest to `.5`, then smaller `gamma`.  The committed simulation module freezes the numeric generative
`W_call`, `B`, mean, wave, request, heteroskedasticity, and missingness constants; its content hash is recorded
before a reported run and changing it requires a new preregistered run.

The finite familywise threshold is the upper-95% null order statistic at one-based rank

```text
ceil(.95 * (K_null + 1)).
```

with strict `observed > threshold` promotion.  Calibration and evaluation null seeds are disjoint.

## Simulation sizing gate

The recommended count is the smallest `G` for which, at `rho_max=.10` and `.20`:

1. independent block-null false deployment does not reject `p<=.05` in a one-sided exact binomial test at 5%,
   and its observed rate is at most 10%;
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

Use a preregistered one-sided prompt-block multiplier/max-statistic construction for simultaneous 95% lower bounds
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

No elementwise covariance lower bound is called conservative.  Select the shrinkage multiplier
`s_safe` from `{0,.25,.50,.75,1}` inside outer training, using only its inner-held components, as the
largest value whose multiplicity-adjusted benefit lower bound is positive; otherwise use zero.  It multiplies
the already selected, rho-matched correlation path and is not a second unconstrained covariance fit.  In
whitened coordinates,

```text
C_safe = (1-s_safe) I + s_safe C_hat,
R_safe = (I_N tensor L_B)
         [C_safe tensor I_m + delta_95 I_(N*m)]
         (I_N tensor L_B)^T
         + I_N tensor W_call/R_eff.
```

`C_hat` is the already rho-matched selected `C(K_gamma,rho_max)`, `B=L_B L_B^T` is the persistent channel
covariance, and `R_eff` is the number of independent calls averaged for that deployed measurement.  Thus
`s_safe` is distinguishable from the kernel path coefficient `alpha(K,rho)` and call noise is not silently
absorbed into persistent covariance.

For each inner split, whiten with training-only `B`, form the inner-held repeat-mean residual second moment
`S_held`, include the correspondingly whitened `W_call/R_eff` term in `T_model`, and record the positive
missing-mass statistic

```text
e_delta = max(0, lambda_max(S_held - T_model)).
```

`delta_95` is the simultaneous one-sided 95% upper prompt-block-bootstrap bound on the maximum `e_delta` across
inner splits and required corpora, with the whole mean/covariance/selector refit inside each resample.  If that
bound is not finite and identified, deployment fails closed.  Statistical `delta_95` and numerical Cholesky
loading are distinct and reported separately.  Distance-separated batching additionally requires a
simultaneous 95% upper bound on every proposed cross-batch whitened block norm below
`epsilon_batch=.025`, the smallest nonzero coupling resolved by the frozen selector grid; neither small
`alpha`, large distance, nor added loading establishes independence.

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
