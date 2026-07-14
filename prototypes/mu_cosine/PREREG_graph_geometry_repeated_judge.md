# Repeated-judge graph-covariance campaign — preregistration

## Status and authorization boundary

**Frozen before running the new power harness, selecting any new endpoint, or making any fresh judge call.**
This PR is deliberately no-spend: it freezes the **planned** candidate-capacity grid, outcome-blind selector
contract, repeat-aware data contract, simulation, estimands, and decision rules.  It creates no labels,
authorizes no API/judge calls, and cannot unlock covariance deployment, independent batching, QR
specialization, or CUDA claims.  The candidate builder and approved live request contract are not included;
caller-supplied hashes make a dry-run reproducible but do not verify those missing artifacts.

If a named judge or prompt is unavailable, this document must be amended before any fresh response exists.
Operational dry runs may use synthetic responses only.  A 64-component live engineering pilot is permitted
only under a separate amendment and must be permanently excluded from confirmation.

**2026-07-13 structural preflight:** the literal 10% undirected-connected-component source cap makes every
registered `G` infeasible on both frozen corpora.  The primary optimistic proof charges only one endpoint to
each declared source before cells, history, or Nomic; the sharper four-endpoint sensitivity also fails every
registered `G`, so the conclusion does not depend on how the permitted disconnected negative is interpreted.
Candidate enumeration is therefore blocked.  Source-component identities remain frozen through later endpoint
exclusions; they are not recomputed to create more cap slots.  A separate outcome-blind amendment must
define and power a feasible dependency/source partition; graph branches are not silently relabelled connected
components.  See `REPORT_repeated_judge_candidate_capacity.md`.

**2026-07-13 source-region amendment and audit:** replace the infeasible weak-component concentration unit in
the *planned v3 candidate contract* with an explicit `source_region`, while retaining `weak_component_id` as
a separate immutable graph diagnostic.  Starting with one region per true weak component, allocate each next
region to the eligible component maximizing `node_count/current_region_count`, with canonical-component ties,
then recursively cut a graph-distance-rooted spanning tree into exactly `K in {64,96,128}` exclusive
induced-connected regions.
For support radius three, a node belongs to a region core exactly when its complete undirected three-hop ball
stays in that region.  All four candidate endpoints must later belong to one core; consequently the distant
root must have a finite canonical distance at least three and `anchor_distant_disconnected=false`.  Every
registered `G` must satisfy the 10% source-region cap's optimistic four-endpoint bound, at least 50% of graph
nodes must remain in cores, and at least 20 regions must have four core nodes, in both corpora.  The coarsest
jointly passing `K` would be selected without fallback outside the grid.

The audit found no passing `K`: exploratory core retention is below 3.0% and its four-endpoint capacity fails
at the larger registered sizes; fresh capacity passes but core retention remains below 32.7%.  Therefore the
v3 selector migration, historical inventory, candidate enumeration, Nomic cache, builder, and every live or
numerical-deployment authorization remain blocked.  Three-hop core separation proves only disjoint support
for the frozen radius-three graph feature.  It does **not** make source regions independent: Nomic, global,
prompt, weak-component, and residual dependence may cross their boundaries.  See
`REPORT_repeated_judge_source_regions.md`.

## Question and experimental unit

The question is whether an outcome-blind graph/semantic geometry predicts transferable cross-item conditional
measurement error after calibration and mean removal.  The structural sampling unit is an endpoint-disjoint
component containing three rows with a shared descendant `x`:

```text
anchor:             (x, a)
adjacent positive:  (x, b), where a--b is a direct undirected graph edge
matched negative:   (x, c), where finite distance(a,c) >= 3
```

The positive and negative roots must match on shortest descendant-to-root hop, campaign tag, and root-degree
quartile.  The sampler balances anchor-to-comparator hop transition, anchor-degree quartile, corpus, and frozen
graph/Nomic agreement class.  No graph endpoint ID or normalized endpoint title may occur in two selected
components or in the historical scored campaigns.  Under the attempted v3 contract, all four endpoints must
belong to one three-hop-safe `source_region` core and each source region contributes at most 10% of a corpus
sample.  `weak_component_id` remains a distinct diagnostic and is never used as an alias for that cap or fold
unit.  The source-region audit found no jointly feasible frozen partition, so this remains a blocking
amendment point rather than an executable selection rule; current v2 `source_component` selector artifacts
are historical and do not satisfy the proposed contract.

Graph/Nomic thresholds are computed and recorded on the frozen structural candidate universe before endpoint-
consuming selection.  Structural near/far defines the sampling contrast; cumulative-walk and Nomic similarities
are retained continuously.  The target count is determined by the power procedure below, separately for every
corpus required to pass.

`G` is the component count **per required corpus**.  Each corpus has 32 joint cells (four hop transitions by
four degree quartiles by two agreement classes), so every frozen `G` supports exact equal cell counts.  The
selector rejects a requested per-corpus total that cannot preserve those frozen quotas.

## Repeated-call estimand

For component `g`, row `i`, judge family `f`, independent call wave `r`, and prompt block `p(g)`, the D/S
conditional residual vector `q_gifr in R^2` follows

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
request-connected dependence clusters remain bounded.  Each role-by-judge measurement gets an independent
deterministic base order and evenly spaced repeat rotations.  Exact serialized request-input bytes are hashed
into request identity, and provider call identities may not be reused across logical attempts.  A block and all
of its rows remain wholly inside one
corpus, outer fold, and global inner-fold label; no request crosses an analysis boundary.  The prompt block,
not an individual component, is the conservative inference/resampling cluster.  Request and block effects are
integrated as held random effects using covariance fit inside training folds; their realized held values are
not estimated from training data.  The later `N_item=128,M_channel=32` numerical batch is a different
conditioning layout over already collected measurements.

Every submitted score row includes a stable `row_id`; the eventual approved prompt/parser must return that key.
Responses join on `(request_id,row_id)` and never infer identity from list position.

Repeats are not averaged before variance decomposition.  The analysis reports:

1. within-call D/S covariance and judge-specific sampling variance from repeat deviations;
2. persistent item/judge covariance from repeat means after subtracting the full fitted repeat-mean sampling
   covariance (`R_call + R_prompt + R_wave`), not only a diagonal `W_call/R` term;
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

Each corpus uses five deterministic outer folds and three inner component folds.  **Outer** fold totals differ
by at most one, and each feasible stratum margin is distributed as evenly as integer counts allow; equal
per-cell counts inside every fold are not required.  A single stable global inner label keeps prompt requests
from crossing any nested split.  In the synthetic sizing model, where components are the dependency atoms,
each leave-one-outer training set's inner totals differ by at most two; at `G=160` a 44/42/42 split is the
unavoidable integer optimum, so a false at-most-one rule is not imposed.  In the planned v3 real sampler,
`(corpus, source_region)` groups are indivisible and their realized leave-one-outer imbalance is recorded;
it is not silently described as the synthetic optimum.  Every row, repeat, and judge from one endpoint
component remains in one fold.  Endpoint IDs and
normalized titles are disjoint across folds.  For every outer-held fold the selector also materializes the
three inner-fold assignments of its outer-training components, before outcomes exist.
Prompt blocks are then formed only among components with the same corpus/outer/inner signature, and whole
prompt blocks—not component rows—are the resampling units for uncertainty intervals.
Every `(corpus, source_region)` group is also wholly contained in one outer and global inner fold.  Primary
intervals use prompt blocks; a two-way prompt-block/source-region multiplier sensitivity must also pass,
unless source-region dependence is proved absent from training-only residual diagnostics.  True weak
components are reported but are not required to be fold-atomic because one frozen corpus has a single weak
component.  No selector migration is authorized until a source-region construction passes its upstream gate.
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

## Frozen synthetic primary-event simulation

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
The synthetic sizing model does **not** generate arbitrary list-position effects.  Counterbalanced scheduling
mitigates but does not identify that nuisance: live approval requires an engineering pilot, a frozen
position-by-role-by-judge train-only adjustment, and a position-effect power sensitivity in a later amendment.
The **block null** sets graph-local persistent item coupling to zero while retaining fitted prompt, wave, and
row-call incidence covariance; it is not full independence of every row in `R`.  Candidate item covariances
are explicit feature Grams.  Frozen scenarios are block null, smooth-mean only,
cumulative truth, Nomic truth, convex-mixture truth, and equal-energy derangement at `rho_max=.04,.10,.20`.
An optional shared-wave scenario is labelled separately.  Nulls preserve component topology, regional mean,
repeat heteroskedasticity, within-call D/S covariance, missingness, and any wave/prompt strata while removing
only graph-local persistent coupling.

Every null and alternative replicate repeats component splitting, mean/ridge selection, repeat decomposition,
`W_call`, `B`, kernel/mixture/coupling selection, loading, and endpoint scoring.  Cache only outcome-blind feature
Grams, folds, eigensystems, and linear-system structure across replicates.  Ephemeral factor memoization is
permitted only within one fitted nuisance object and may not cross a fit, fold, replicate, scenario, or corpus.
Seeds derive from `(G,R,scenario,replicate)`; scientific
JSON excludes wall time and output paths.

The frozen implementation details are: prompt-block capacity 10; five outer and three inner folds; mean-ridge
grid `{0,.01,.1,1,10}`; 5% shrinkage toward channel-diagonal targets for `W_call` and `B`; and a relative
numerical SPD floor of `1e-8`.  A non-block candidate is inner-eligible only when its macro held gain is positive
and at least two of three inner folds are positive.  Ties prefer larger gain, then smaller `rho_max`, then
`gamma` closest to `.5`, then smaller `gamma`.  The committed simulation module freezes the numeric generative
`W_call`, `B`, mean, wave, request, heteroskedasticity, and missingness constants; its content hash is recorded
before a reported run and changing it requires a new preregistered run.

For synthetic sizing, the finite familywise threshold is the upper-95% null order statistic at one-based rank

```text
ceil(.95 * (K_null + 1)).
```

with strict `observed > threshold` promotion.  Calibration and evaluation null seeds are disjoint.  This
synthetic threshold is never transferred to real residuals.  Inside every real outer fold, recalibrate the same
complete selector from outer-training prompt blocks under a graph-local block null that retains fitted
mean/call/prompt/wave/missingness/source structure; rerun every nuisance fit and inner selection in each null
draw, and apply the resulting threshold only to that fold's untouched held blocks.

## Synthetic primary-event sizing gate

The reported primary-event count is the smallest `G` for which the joint two-corpus simulation, at
`rho_max=.10` and `.20`, satisfies:

1. for **both** the graph-local block null and zero-coupling smooth-mean-only control, false primary promotion
   does not reject `p<=.05` in a one-sided exact binomial test at 5%, and the observed rate is at most 10%;
2. simultaneous residual/posterior primary-event power is at least 80% for cumulative, Nomic, and mixture
   truths in both corpora;
3. mean selected outer-held residual NLL gain is positive;
4. mean harm is nonpositive under block-null and smooth-mean controls;
5. topology truth beats the equal-energy derangement in at least 80% of replicates; and
6. the procedure does not pass with an incomplete scenario grid.

This is not yet the final campaign `G`: decision calibration, margin-AURC/noninferiority, source-region
sensitivity, `s_safe/delta_95`, and cross-batch safety are unsimulated secondary gates and therefore fail
closed in this PR.  A final count requires their own power/feasibility bridge and is at least the primary-event
count.  If no grid value passes, increase prompt-block/source-region information.  Do not reduce confidence,
drop a difficult truth, or use additional repeats as a substitute for independent structure.

## Real-data endpoints and simultaneous inference

Primary equal-component paired endpoints are:

```text
d_residual,g  = NLL_block,g - NLL_structured,g
d_posterior,g = posterior_NLL_block,g - posterior_NLL_structured,g.
```

Use a preregistered one-sided prompt-block multiplier/max-statistic construction for simultaneous 95% lower bounds
across both primary endpoints and every corpus required to pass.  The familywise selector must reject block and
every primary lower bound must exceed zero both before and after the complete `R_safe` adapter (including
`delta_95` and numerical loading).  Secondary requirements are:

- posterior calibration/coverage and Mahalanobis diagnostics do not worsen;
- decision log-loss and margin-gated AURC degradation are each at most `.01`;
- ECE uses ten fixed equal-width bins and AURC uses prompt-block bootstrap intervals;
- graph/Nomic source correlations and same-split ablations are reported;
- topology benefit exceeds derangement/hard negatives;
- the two-way prompt-block/source-region sensitivity also passes;
- statistical loading is reported, and relative numerical loading
  `max(load_i / max_j R_jj)` is at most `1e-6`.

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
L = I_N tensor L_B,
R_safe = L [C_safe tensor I_m
            + T_call + T_prompt + T_wave
            + delta_95 I_(N*m)] L^T.
```

`C_hat` is the already rho-matched selected `C(K_gamma,rho_max)` and `B=L_B L_B^T` is the persistent channel
covariance.  In the same whitened coordinates,

```text
R_call   = sum_t A_t W_call,f(t) A_t^T,
T_call   = L^-1 R_call L^-T,
R_prompt = sum_q Z_q V_prompt,f(q) Z_q^T,
T_prompt = L^-1 R_prompt L^-T.
```

Here `t` indexes an observed row-specific call and `q` an actual family/role/wave prompt request.  `A_t` and
`Z_q` have shape `(N*m, m_f)`, map the affected family channels into the full observation vector, and contain
the actual repeat-mean aggregation weights; they are zero for unaffected observations.  `T_wave` is defined
analogously from retained wave effects and is zero only when their fitted removal plus uncertainty envelope is
demonstrated.  Missing calls change the recorded `A_t` and `Z_q` weights rather than being replaced by a nominal
common `R`.  Thus `s_safe` is
distinguishable from the kernel path coefficient `alpha(K,rho)`, and neither row-specific call noise nor
shared prompt/wave covariance is silently absorbed into persistent `B`.

For each inner split/corpus index `j`, stack each held component's three row residuals into a `3m` vector and
form the equal-component pooled `3m x 3m` second moment `S_held,j`; the block bootstrap resamples whole prompt
blocks containing those vectors.  Whiten with training-only `B`, include the matching local restrictions of
`T_call`, `T_prompt`, and `T_wave` in `T_model,j`, and record

```text
T_model,j = C_safe,j tensor I_m + T_call,j + T_prompt,j + T_wave,j,
theta_hat_j = lambda_max(S_held,j - T_model,j).
```

Under exchangeable prompt blocks, refit the whole mean/covariance/selector in every block-bootstrap resample
and compute

```text
c_95 = quantile_.95(max_j(theta_hat_j - theta_hat_star_j)),
delta_95 = max(0, max_j(theta_hat_j + c_95)).
```

This is the centered simultaneous one-sided missing-mass bound; an uncentered percentile of `theta_hat` is not
substituted.  If prompt blocks are not exchangeable or the bound is not finite and identified, deployment fails
closed.  Statistical `delta_95` and numerical Cholesky loading are distinct and reported separately.
Distance-separated batching additionally requires a
simultaneous 95% upper bound on every proposed cross-batch whitened block norm below
`epsilon_batch=.025`, the smallest nonzero coupling resolved by the frozen selector grid; neither small
`alpha`, large distance, nor added loading establishes independence.

The schedule-dependent prompt term need not commute with `C_safe tensor B`.  Dense joint QR remains valid;
the single-item-kernel eigenmode shortcut is eligible only when the extra terms are block separable or their
commutation is proved for the deployed schedule.

This campaign's local covariance blocks contain three rows.  Its `delta_95` is not extrapolated to a coherent
128-item block: omitted spectral mass can grow with block size.  An `N_item=128` conditioner may be tested for
numerical equivalence against stipulated covariance, but statistical deployment at that scale additionally
requires an `N=128` full-procedure simulation/validation or an explicit N-aware matrix-concentration/row-sum
envelope.

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
