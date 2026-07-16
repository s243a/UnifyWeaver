# Repeated-judge source dependence — Stage-A no-history power design

## Purpose and authorization boundary

The topology bridge in `DESIGN_repeated_judge_source_dependence.md` showed that the complete exclusive source
regions can support the registered component counts when cross-region exposure is represented explicitly.
It did **not** establish a residual covariance, a valid source-dependent inference procedure, candidate
packability, or power.  This Stage-A design supplies the next no-spend statistical bridge.

Stage A reads only the frozen topology artifact and synthetic outcomes.  It does not read attempted-input
history, candidate identities, Nomic embeddings, labels, judge responses, or model outputs.  It makes no API
call.  A passing **seed-disjoint synthetic confirmation run**, not a sizing run, may unlock only the immutable
attempted-input-identity inventory needed by Stage B.  It does not unlock candidate selection, Nomic,
scoring, a live campaign, covariance promotion, independent batching, QR specialization, or CUDA.

`JointPosterior` remains the separately learned and calibrated decision comparator.  It is not a source
kernel, a confidence-weighted covariance estimator, or a square-root factorization.  Stage A neither trains
nor evaluates it.

## Frozen topology inputs and notation

For each required corpus `c`, region count `K in {64,96,128}`, and component count
`G in {160,320,512,800}`, Stage A consumes the complete source-dependence payload whose SHA-256 and size are
bound by `repro/repeated_judge_source_dependence/summary.json`.  The tracked projection alone is insufficient:
it intentionally replaces exact matrices and allocations with content records.  The runner must either read
the byte-identical complete payload or deterministically reconstruct every omitted object and match all
content records before simulation.

The canonical compact bundle is
`repro/repeated_judge_source_power/source_design.json`: 969,914 bytes with SHA-256
`da7c2ec6d003150aeb0465eb099508aea9918b495ff00ae25ea3f6e44cfe5fb9`.  Its reviewed parent full payload is
2,767,735 bytes with SHA-256
`bf9a09c35e54bd36c2e7efea19c432ccf1e9105ff67c4154cfc1c6e744a843b2`.  Loading revalidates every compacted
matrix and allocation against the tracked projection and the parent identity.

The frozen objects are:

- the ordered `K x K` PSD, unit-diagonal exposure matrix `E_cK`;
- the registered length-`G` source-region assignment;
- its one-hot incidence matrix `H_cKG in {0,1}^{G x K}`; and
- the exact per-region component counts and stable region IDs.

Mathematical prose uses `eta_source,c` for corpus-specific cross-component source dependence.  Python
arguments use `source_eta_by_corpus`, serialized generator records use `generator_source_eta_by_corpus`, and
the compact bundle exposes `source_eta_grid`.  Continue to use `rho_item` for the within-component three-row
covariance candidate/truth.  The legacy parent bridge field `rho_grid` is accepted only while validating the
frozen parent payload and is not propagated into Stage-A records.  No new Stage-A serialized checkpoint or
result field uses bare `eta` or `rho`; the inherited internal `Candidate.rho` and `rhos` adapter are interpreted
only as `rho_item`.  For each corpus `c`, the source sensitivity family is

```text
eta_source,c in {0, .025, .05, .10, .20},
S_cKG = H_cKG E_cK H_cKG.T,
C_source,c(eta_source,c) = (1-eta_source,c) I_G + eta_source,c S_cKG.
```

The joint two-corpus nuisance family is the full Cartesian product of 25 ordered
`(eta_source,exploratory, eta_source,fresh)` pairs; the five equal-strength diagonal pairs are not a substitute.
Every member, not only `eta_source,c=.20`, is part of the primary family.  In every frozen corpus cell `G>K` and
`rank(S)<=K`, so `S` has a zero eigenvalue and `S-I` has eigenvalue `-1`.  Also `diag(S)=1`, hence
`trace(S)=G`; its at most `K` nonzero eigenvalues include one at least `G/K>1`, so `S-I` also has a positive
eigenvalue.  It is therefore indefinite in every frozen cell.  Although the variance of an equal-weight mean
increases along this nonnegative path, `C_source,c(.20)-C_source,c(eta_source,c)` is not a positive-semidefinite
upper bound for arbitrary selector or endpoint contrasts.  No interpolation or continuum-wide claim is made
between the five frozen values.

## Exact source-dependent synthetic field

Retain the baseline repeated-judge generator's three rows, four channels, persistent channel covariance `B`,
call covariance, prompt-request covariance, repeat-wave effects, request-level missingness, mean/ridge grid,
5% covariance shrinkage, and numerical SPD checks.  Replace independent persistent components by

```text
u_c = sqrt(1-eta_source,c) epsilon_c + sqrt(eta_source,c) H_c z_c,
epsilon ~ Normal(0, I_G tensor C_item tensor B),
z       ~ Normal(0, E   tensor C_item tensor B).
```

Therefore

```text
Cov(u_c) = C_source,c(eta_source,c) tensor C_item tensor B.
```

This draw uses a region-factor representation without a dense `G x G` factor: Cholesky-factor the audited
`E`, draw one region field, map it through `H`, and add the independent component field.  Every reviewed `E`
is strictly positive definite and must pass the recorded finite/symmetry/positive-definiteness/unit-diagonal
checks again.  A Cholesky failure is not repaired by clipping or diagonal loading; it fails closed.  A future
merely PSD exposure would require a prospective amendment freezing an eigendecomposition/SVD square root and
rank tolerance before outcomes exist.

The within-component truth is unchanged:

- block null: `C_item=I_3`;
- cumulative, Nomic, and 50/50 mixture truths at `rho_item in {.04,.10,.20}`;
- one cumulative-kernel deranged-DGP control at each of those three `rho_item` values.

Here “Nomic truth” names the baseline harness's fixed explicit synthetic semantic kernel; Stage A reads no
real Nomic embedding, model, or cache.  Separately, inside every nonderanged truth replicate, the truth kernel
is compared with its own deterministic row-permuted equal-energy derangement.  Those paired comparisons supply
the topology-over-derangement gates; they are not nine additional deranged DGP scenarios.  Thus the frozen
scenario grid has exactly three deranged-DGP controls total.

The primary power truths are cumulative, Nomic, and mixture at `rho_item in {.10,.20}`.  The `.04` truths are
reported sensitivity cells and cannot rescue a failed primary truth.

Every null retains source dependence, prompt effects, wave effects, call covariance, heteroskedasticity,
missingness, and the complete nuisance-refitting procedure.  The block null removes only within-component
off-row persistent coupling.  The second null is a **source-smooth-mean null**, not merely a larger generic
mean: within each world draw a scalar region field with covariance `E`, map it through `H`, subtract its
equal-component mean, normalize its centered component RMS, and multiply it by the already frozen nonlinear
role loading
`(1,.65,-.35)`, channel loading `(.70,-.45,.35,-.25)`, and scale `.10`.  Add that fixed-within-world term to
the baseline conditional mean while keeping `C_item=I_3`.  It is refit through the same train-only mean/ridge
path.  This control tests whether topology-smooth mean misspecification masquerades as covariance.

The source field, source-smooth mean field, prompt field, call field, wave field, missingness, latent posterior
states, and fold construction have separately derived seeds.  Calibration, sizing evaluation, and the
seed-disjoint synthetic confirmation use disjoint seed namespaces.

Within one `(phase,K,G,R,scenario,replicate,corpus)`, the five `eta_source,c` worlds reuse the same underlying
component, region, prompt, call, wave, missingness, and latent-state streams; only the frozen source-mixture
coefficient changes.  Exploratory and fresh use independent corpus streams.  The runner computes each of the
ten corpus worlds once and forms the 25 ordered two-corpus pairs deterministically.  This common-random-number
reuse is exactly the prescribed joint construction, not an independence claim: each pair still has 200
independent replicate indices, while dependence between different pairwise gates is permitted by the
Bonferroni confirmation rule.  Pairing is always by the same replicate index, and the full joint
20-coordinate max-t statistic (two corpora by two endpoints by five inference values) is recomputed for each
pair.  Computationally, each corpus world prepares its point estimates, five source-eta standard errors, and
per-draw standardized negative deviations once; each ordered pair then recomputes the draw-wise joint maximum,
finite order statistic, and lower-bound arrays from its two prepared contributions.  This is an exact
factorization of the joint multiplier, not a combination of finalized corpuswise bounds.  A zero-scale
prepared contribution is cached as a nonidentified sentinel, so every pair containing it produces the same
explicit nonpromotion without repeating failed multiplier work.  Null calibration uses the analogous
per-corpus reuse within each draw.

## Source-atomic outer and inner folds

Source regions are concentration and fold units, not independent clusters.  Every component assigned to one
source region receives one outer label and one stable global-inner label; a region is never divided.

For each `(c,K,G)`, freeze the labels before generating outcomes:

1. Discard zero-count regions for scheduling only; retain them in the recorded `E` and metadata.
2. Sort active regions by decreasing registered component count, then stable region ID.
3. Assign each region to the outer bin minimizing
   `(component_total_after_assignment, region_count_after_assignment, outer_id)`.
4. Within each outer bin, repeat the same sorted list-scheduling operation over the three global-inner labels,
   minimizing `(component_total_after_assignment, region_count_after_assignment, inner_id)`.
5. For each leave-one-outer training set, inner fold `j` holds every training region with global-inner label
   `j`.  Labels are never recomputed for a particular outer split.

The runner records outer and leave-one-outer inner component totals, source-region totals, and imbalance.  It
fails unless:

- all components occur in exactly one outer fold and one global-inner label;
- every region is atomic in both labelings;
- every outer-held set contains at least five active source regions and four prompt blocks;
- every leave-one-outer inner-held set contains at least five active source regions and four prompt blocks;
- the largest-minus-smallest outer component total is no larger than the largest active-region count; and
- within each outer bin, the largest-minus-smallest inner total is no larger than that bin's largest region
  count.

These are grouped-data bounds, not the component-atomic `44/42/42` synthetic optimum used by the earlier
independent-component harness.  No result may describe them as that old optimum.

Every registered `G` is arithmetically divisible across the intended 32 campaign cells.  Stage A does not
materialize those labels or simulate their empirical distribution, so they are neither fold inputs nor mean
covariates.  Real cell construction and balance remain Stage-B candidate-packing questions.

## Split-contained source-round-robin prompt incidence

Prompt blocks are formed only after outer and global-inner labels.  Within each
`(corpus,outer_label,inner_label)` signature:

1. Let `B=ceil(N_signature/10)` and create `B` empty blocks.
2. Process source regions by decreasing component count and stable region ID.
3. Process each region's registered component indices in ascending order.  Put the next component in the
   nonfull block minimizing
   `(that_region_count_in_block, total_block_count, block_id)`.
4. Reject any block with more than ten components or any omitted/duplicated component.

This spreads a repeated source across prompt requests before placing it twice in the same request whenever
capacity permits.  It does not declare prompt blocks or regions independent.  Block membership is stable
across row roles, judge families, and repeat waves.  Each actual synthetic request contains one role from each
component in its block, matching the intended amortized live layout.

The runner records the prompt-by-source incidence table, its bipartite connected components and rank, the
maximum source share of a prompt, and prompt/source cluster counts in every outer and inner analysis set.  A
rank or crossing diagnostic is descriptive; no post-result prompt rearrangement is allowed.  Failure of the
minimum cluster gates above stops that `(c,K,G)` design.

## Component-marginal quasi-likelihood

Cross-source and shared-prompt dependence affect uncertainty, but a joint source-plus-prompt likelihood would
couple all held components, remove the additive per-component endpoint needed by the frozen multiplier, and
make the prompt and source covariance terms noncommuting.  Stage A therefore freezes a robust marginal target.

For each inner or outer fit:

1. Fit calibration, mean/ridge, wave removal, call covariance, persistent channel covariance, prompt-request
   covariance, missingness handling, and numerical checks using training regions only.
2. For each held component, form its `3 x 4` repeat-mean residual and its **marginal** covariance.  Include its
   persistent `C_item tensor B` term and the matching diagonal/marginal call, prompt, and retained-wave terms
   induced by its observed repeat schedule.
3. Score block and structured candidates with component Gaussian NLL.  Do not insert cross-component source or
   prompt off-diagonals in this point score.

This is explicitly a component-marginal quasi-NLL, not the joint likelihood of the campaign.  Dependence is
carried by source-atomic fitting, the graph-aware two-way inference below, and the full null calibration.

The deployment-capable candidate family remains the single frozen
`gamma x rho_item` grid.  In each outer-training set, a nonblock candidate is inner-eligible only when its
equal-component macro gain is strictly positive and at least two of three region-atomic inner folds are
strictly positive.  Ties retain the existing order: larger gain, smaller `rho_item`, `gamma` closest to `.5`,
then smaller `gamma`.  The familywise threshold below is applied strictly; a value equal to the threshold does
not reject block.

Outer-held component endpoints remain

```text
d_residual,g  = marginal_NLL_block,g - marginal_NLL_selected,g,
d_posterior,g = marginal_posterior_NLL_block,g - marginal_posterior_NLL_selected,g.
```

The point estimate is the equal-component mean.  Repeated calls, source-region sizes, prompt-block sizes,
confidence scores, and estimated inverse variances do not create observation weights.

## Conservative graph-aware prompt-plus-source multiplier

For corpus `c`, stack its two cross-fitted endpoint gains in `D_c in R^(G x 2)` and define

```text
dbar_c = column_mean(D_c),
Psi_c  = D_c - 1 dbar_c,
P_c    = one-hot component-by-prompt incidence.
```

For multiplier draw `b`, generate independent standard-normal component and region vectors `e_cb,z_cb` and
independent Rademacher prompt signs `v_cb`.  Within each corpus reuse the same base `e_cb,z_cb,v_cb` across
the five `eta_source,c` values and set

```text
xi_cb(eta_source,c) = sqrt(1-eta_source,c) e_cb + sqrt(eta_source,c) H_c L_E,c z_cb,
Delta_cb(eta_source,c) = Psi_c.T [xi_cb(eta_source,c) + P_c v_cb] / G,
```

where `L_E L_E.T=E`.  Corpora use independent multipliers.  The source and prompt perturbations are **added**.
There is no source/prompt intersection subtraction and no `sqrt(2)` renormalization.  The resulting inflated
working multiplier covariance is PSD and avoids relying on independent source clusters when `E` explicitly
says otherwise.  It is not a universal Loewner upper bound on the unknown data covariance.  Its admissibility
comes only from the frozen complete-procedure null calibration; it is not presented as an exact finite-sample
cluster theorem.

Corpus independence is a frozen synthetic assumption, not a transferable fact.  Before any Stage-B realized
rerun, audit revision drift, wave, request/session, calibration-fit uncertainty, endpoint/source overlap, and
every other random or estimated nuisance across corpora.  A common revision-pinned model and prompt are fixed
conditioned-on strata and do not by themselves violate independence.  If any stochastic or fitted nuisance
does span both corpora, the assumption fails: add the corresponding shared DGP and multiplier term
prospectively and rerun Stage A.  Relabelling the same source as two corpora is forbidden.

Use 999 multiplier draws.  For each corpus, endpoint, and `eta_source,c`, compute the draw standard deviation
`se_cj(eta_source,c)`.  A nonfinite or numerically zero standard error fails closed.  For each draw form the
maximum negative standardized deviation over both corpora, both endpoints, and all five corpus-specific
values.
The critical value is the order statistic at one-based position

```text
ceil(.95 * (999 + 1)) = 950.
```

The simultaneous lower bound for endpoint `(c,j)` is

```text
lower_cj = min_eta_source,c [dbar_cj - critical * se_cj(eta_source,c)].
```

Thus the reported endpoint claim covers the complete frozen source family, not whichever `eta_source,c`
happens to be most favorable.  Prompt-only bounds, independent-source bounds, and
`eta_source,c=.20`-only bounds may be
reported as diagnostics but cannot pass the primary event.  Position 950 is the frozen empirical multiplier
critical-value rule; unlike the exchangeable null-calibration position below, it is not claimed to give an
exact finite-sample coverage theorem.

If an identically zero all-block gain makes the multiplier scale zero, record that replicate explicitly as
`inference_identified=false` and `promoted=false`; this is an observed nonpromotion, not a missing replicate or
a global configuration failure.  Its frequency is reported and already lowers the applicable power/event
count.  Shape, nonfinite-input, factorization, or unrelated numerical errors remain fatal and are not converted
to nonpromotion.

## Uniform finite-null selector calibration

For each `(K,G,R)` separately, perform 1,999 calibration draws for every

```text
null_type in {block_null, source_smooth_mean_null}
    x eta_source,exploratory in {0,.025,.05,.10,.20}
    x eta_source,fresh in {0,.025,.05,.10,.20}.
```

One draw's selector statistic is the maximum eligible inner gain over both required corpora, all five outer
folds, and the complete deployment-capable `gamma x rho_item` family.  Every nuisance fit, split, synthetic
prompt schedule, missingness pattern, and selector decision is rerun.  For each null cell, take the finite
95% threshold at one-based position

```text
ceil(.95 * (1999 + 1)) = 1900.
```

The operational selector threshold is the maximum of the 50 cell thresholds.  Promotion requires a strict
`observed > threshold`.  Under exchangeability with any one frozen null cell, the strict finite-position rule has
Monte Carlo size at most `100/2000=.05`; taking the maximum threshold preserves that bound.  The synthetic
threshold is never transferred to real responses.

The primary event is a conjunction and therefore a subset of this calibrated selector rejection.  A
replicate passes only when:

1. the calibrated selector rejects block in **both** required corpora; and
2. all four graph-aware simultaneous lower bounds—two endpoints by two corpora—are strictly positive.

There are not separate votes for endpoints, corpora, `eta_source,c`, `K`, or kernels.

## Control, power, and Monte Carlo gates

The primary `R=3` sizing grid is the complete Cartesian product of both corpora,
`K={64,96,128}`, `G={160,320,512,800}`, all 25 ordered corpus-specific `eta_source,c` pairs, and every frozen
null/truth/control scenario.  Each evaluation cell has 200 independent replicates after null calibration.

For a binomial success count `x` in `n` replicates, use exact one-sided Clopper-Pearson bounds.  A lower power
claim uses `Beta^-1(alpha; x, n-x+1)` and an upper false-promotion claim uses
`Beta^-1(1-alpha; x+1, n-x)`, with the conventional zero/one endpoints.  The sizing run records ordinary 95%
bounds and uses them—not raw observed rates—for provisional passage.

A `(K,G)` sizing cell passes only when all of the following hold:

- for both null types at every `eta_source,c` pair, the primary-promotion upper bound is at most `.10`, the finite
  selector calibration is complete, and neither corpus has negative mean selected residual-NLL gain;
- for cumulative, Nomic, and mixture truths at `rho_item in {.10,.20}` and every `eta_source,c` pair, the one-sided
  95% lower bound on the joint primary-event probability is at least `.80`;
- for the same truth cells, the one-sided 95% lower bound on the probability that the true topology's mean
  residual NLL beats its equal-energy derangement in both corpora is at least `.80`;
- deranged-truth primary promotion has a one-sided 95% upper bound at most `.10` at each registered
  `rho_item` and `eta_source,c` pair;
- all four corpus-by-endpoint mean gains are positive for every primary truth cell; and
- the complete scenario, source, candidate, corpus, fold, and endpoint grids are present.

The raw mean-sign conditions are conservative descriptive fail-closed guards inside the conjunction.  They do
not make a standalone population-mean or confidence-coverage claim and are not additional members of the
binomial confirmation family.

The exact finite selector threshold is the familywise-null argument.  The evaluation-null bounds are
additional implementation and model-adequacy gates; failure to reject an excess-error test is not substituted
for an upper confidence bound.

`R=4` is a separately labelled sensitivity run only at `G={320,800}` and every registered `K`.  There is no
`R=4,G={160,512}` cell and no rescue or interpolation.  It otherwise includes all 25 source-strength pairs and
the complete frozen scenario grid.  It cannot select a smaller `G`, repair an `R=3` null failure, or unlock
Stage B.

## Provisional sizing, fixed selection, and seed-disjoint confirmation

The complete sizing run chooses a design prospectively:

1. choose the smallest numeric `G` having at least one passing `K`;
2. at that `G`, choose the smallest numeric `K` (the coarsest passing partition); and
3. if no pair passes, stop without fallback outside the frozen grid.

The selected pair is only **provisional**.  It cannot unlock history.  Run a seed-disjoint synthetic
confirmation at exactly that `(K,G,R=3)` with disjoint calibration, field, multiplier, latent-state, and
missingness seeds: 1,999 new draws per null-by-`eta_source,c`-pair calibration cell and 200 new evaluation replicates
per frozen scenario-by-`eta_source,c`-pair cell.
Do not reselect `K` or `G` from confirmation.

For confirmation, compute simultaneous one-sided Clopper-Pearson bounds by Bonferroni-adjusting `alpha=.05`
over exactly `M=425` binomial gates: six primary truths by 25 pairs (`150` primary-power plus `150` paired
topology-over-derangement gates), two nulls by 25 pairs (`50`), and three cumulative-kernel deranged-DGP
controls by 25 pairs (`75`).  Each one-sided confirmation bound therefore uses `alpha=.05/425`.  Record the family
size and each adjusted bound.  Every lower power/topology bound
must remain at least `.80`; every null/deranged upper bound must remain at most `.10`; every mean and
completeness gate above must also pass.  If confirmation fails, Stage A fails.  Trying the next `K` or `G`
is forbidden; that failure is terminal under this preregistration.  A later prospective amendment may retry
only if it freezes a
cross-attempt alpha-spending or closed-testing rule that accounts for the failed attempt before any new seed is
used; seed disjointness alone does not reset multiplicity.

Only this passing confirmation may set `attempted_input_identity_inventory_unlocked=true`.  Every other
authorization remains false.

## Compute pilots, complete runs, and provenance

Timing pilots and smoke runs may reduce `K/G` cells, null draws, evaluation replicates, multiplier draws, or
scenarios explicitly.  They validate shape, checkpointing, numerical equivalence, and projected cost only.
They hard-code every authorization and sizing/confirmation decision to false/null even if their observed
numbers look favorable.

With corpus-world reuse, each exact discovery configuration still requires `39,980` expensive null corpus
worlds (`2*1,999*2*5`) and `28,000` expensive evaluation corpus worlds (`14*200*2*5`).  The 18 registered
`R=3/R=4` configurations therefore require `1,223,640` corpus worlds; a selected-pair confirmation adds
`67,980`, for at most `1,291,620`.  Deterministic construction of the 25 pairwise gates from same-index corpus
worlds does not lower any draw or replicate count.  Review and timing pilots precede this large immutable
run; a tooling PR cannot report or infer its power result.

A sizing result is evaluable only for the exact complete frozen grid and counts.  A confirmation result is
evaluable only for the one pair selected by a complete sizing payload, the exact seed-disjoint counts above, and
provably disjoint seed namespaces.  Customized or incomplete runs cannot impersonate either mode.

The runner inherits atomic indexed checkpoints, one BLAS thread per worker, path-free runtime identity,
content-addressed scientific inputs, deterministic seed derivation, start/end provenance checks, and rejection
of mismatched checkpoints.  Wall time, worker count, checkpoint location, and output path are operational;
they do not change scientific JSON.  Any change to this design, the source payload, simulation constants,
folding/packing algorithms, scoring path, or decision logic changes the scientific fingerprint and requires a
new full run.

## Mandatory Stage-B realized-design rerun

A Stage-A pass applies only to its registered diagnostic assignment, exposure matrix, and synthetic prompt
incidence.  A pass unlocks only the immutable attempted-input identity inventory.  Stage B must proceed in
this order: apply history exclusions and enumerate the topology-only structural universe; audit its capacity
and provenance; if and only if those gates pass, build a revision-pinned Nomic cache solely to assign the
frozen graph/Nomic agreement cells; freeze the exact 32-cell endpoint-disjoint packing at the already selected
`(K,G,R=3)`; only then compute the Nomic Gram on those immutable components, without repacking or optimizing
against continuous Nomic values; rebuild realized `H`, lifted exposure, source-atomic folds, and prompt
incidence; and rerun the complete 1,999-draw-per-null-cell calibration and fixed-design 200-replicate
evaluation, followed by a second
seed-disjoint 1,999-draw calibration namespace and 200-replicate synthetic confirmation.  Stage B does not
reselect across the `K/G` grid.  Judge calls remain locked throughout this sequence.  No earlier stage may
inspect Nomic values or optimize candidate selection against them.

There is no scalar shortcut.  Better equal-mean ESS, a smaller maximum exposure entry, a smaller row sum, or a
smaller nominal `eta_source,c` does not prove PSD/Loewner dominance for the nonlinear selector and all endpoint
contrasts.  Unless a separate prospective amendment supplies and verifies a matrix-order certificate, the
realized full-procedure rerun is mandatory.

## Rejected and deferred alternatives

| Alternative | Disposition | Reason |
|---|---|---|
| use only `eta_source=.20` as the dependence envelope | reject | larger `eta_source` is not a Loewner increase because `S-I` is indefinite |
| test only equal `eta_source` in both corpora | reject | the corpora need not share a coupling strength, and five diagonal worlds do not establish power over the 25 ordered pairs |
| call source regions independent | reject | regions are concentration/fold labels; `E` encodes cross-region exposure |
| split a large source region across folds | reject | leaks one persistent source field into training and held data |
| form prompts before folds | reject | a shared request could cross an analysis boundary |
| random prompt packing without source spreading | reject | can needlessly alias prompt and source effects |
| joint source-plus-prompt NLL as the Stage-A endpoint | defer | it destroys the frozen additive component endpoint and introduces a large noncommuting joint covariance; the marginal quasi-score plus robust inference is the preregistered target |
| prompt-only cluster bootstrap | reject | ignores the dependence path that motivated Stage A |
| ordinary two-way source/prompt subtraction | reject | source regions are mutually correlated through `E`, and the subtraction can be indefinite; the added PSD perturbations are deliberately conservative |
| normalize the added source and prompt multipliers by `sqrt(2)` | reject | would change the frozen inflated working covariance without null calibration |
| hand-set confidence or inverse-variance weights | reject | correlated evidence requires later held-out JointPosterior calibration; Stage A uses equal components |
| select `K` from ESS or exposure alone | reject | those diagnostics omit the selector, effect, prompt dependence, and multiplicity |
| declare 80% power from an observed rate of `.80` | reject | Monte Carlo uncertainty must be reflected in a one-sided exact lower bound |
| let the sizing grid itself unlock history | reject | selecting a favorable `K/G` needs a seed-disjoint confirmation |
| fall through to another `K/G` after confirmation fails | reject | that would add an uncalibrated second selection step |
| let four repeats rescue three-repeat power | reject | repeats estimate call noise and do not replace independent source information |
| accept the tracked summary without exact matrices/allocations | reject | its omissions are content-addressed review projections, not simulation inputs |
| certify a realized Stage-B design from mean ESS or row sums | reject | scalar summaries do not order every covariance contrast or the complete selector |
| covariance deployment, independent batching, QR/CUDA work | defer | those require real repeated residuals and all later calibration/safety gates |
