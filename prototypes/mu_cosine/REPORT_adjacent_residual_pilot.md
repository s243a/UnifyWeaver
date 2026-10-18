# Adjacent conditional-residual pilot — strong descriptive signal, no covariance deployment

## Bottom line

Adjacent roots under the same descendant have more similar out-of-fold, conditionally whitened judge residuals
than anchor-matched nonadjacent roots.  The component-macro contrast was positive for every one of ten
component-fold assignments in both corpora: `0.2799 +/- 0.0074` exploratory and `0.1697 +/- 0.0111` fresh.
Every leave-one-positive-component-out estimate was also positive.

This is promising evidence that graph-local measurements share predictable structure.  It is **not** yet a
validated covariance estimate.  Existing campaign construction confounds direct adjacency with local hop/tag
position, provides only one residual field per corpus, and supplies just 28 / 40 eligible positive endpoint
components.  A low-capacity adjacency Gram improves held residual likelihood more than a topology-deranged
Gram, but the comparison is diagnostic and its best tested coupling lies at the edge of the frozen grid.

Under the frozen two-outcome decision rule, the formal result is **not resolved by existing data**: the
observed direction and component stability are strong, but the synthetic mechanism does not meet the required
80% power gate.  “Promising” below describes the observed signal only; it is not a passed advancement gate.

The operational decision therefore remains unchanged:

- keep independent item blocks in the square-root/QR conditioner;
- do not certify distance-separated batching from this pilot;
- run a purpose-built, component-balanced campaign with repeated judge calls before estimating cross-item
  covariance;
- retain a 95% simultaneous safety bound for deployment.  An 80--90% band may be used only as a labelled
  discovery screen or shadow-mode trigger for collecting more data.

## What was measured

The protocol in `DESIGN_adjacent_residual_pilot.md` was frozen after an outcome-blind graph/campaign inventory
and before any adjacency-stratified residual statistic was computed.  Five-fold cross-fitting holds out whole
components of the exact endpoint-incidence graph.  For each held fold, calibration, prior/measurement
conditionalization, the regional mean, and the within-item covariance `B` are refit on the other components.

For held row `i`, the analyzed residual is

```text
q_i = v_i - C.T P^-1 e_i
z_i = B^-1/2 (q_i - m_hat_i).
```

Each positive record is a pair whose roots share a direct edge and whose rows share a descendant.  Up to three
nonadjacent controls share that descendant and one anchor row, and are ranked without outcomes.  The primary
contrast is the equal-component average of

```text
trace[ G_adjacent - mean(G_anchor-control) ] / 4,
G_ij = 0.5 (z_i z_j.T + z_j z_i.T).
```

Seeds 20--29 vary the whole-component fold assignment.  These are stability probes over the same two label
fields, not ten independent experiments.  Rademacher component-multiplier bands are therefore conditional
stability summaries, not population confidence intervals.

## Inventory and matching

| corpus | matched rows | adjacent pairs | eligible pairs | eligible positive components | positive folds |
|---|---:|---:|---:|---:|---:|
| exploratory | 1,000 | 84 | 83 | 28 | 41 / 50 |
| fresh | 700 | 206 | 206 | 40 | 46 / 50 |

One exploratory pair had no anchor-sharing nonadjacent control.  The existing campaign offers no exact
tag-pair-matched nonadjacent controls: adjacency is concentrated among consecutive hop tags.  The identified
contrast is therefore **adjacency plus local-hop predictiveness**, not a pure causal adjacency effect.

## Primary stability result

| corpus | mean primary contrast | assignment SD | assignment range | positive assignments | minimum positive LOCO fraction |
|---|---:|---:|---:|---:|---:|
| exploratory | +0.279877 | 0.007426 | +0.265592 to +0.291888 | 10 / 10 | 100% |
| fresh | +0.169740 | 0.011065 | +0.154176 to +0.191264 | 10 / 10 | 100% |

The diagnostic 95% pointwise lower band for the primary contrast was positive in 10/10 assignments in both
corpora.  The max-|t| simultaneous lower band was positive in 0/10 exploratory assignments and 6/10 fresh
assignments.  The exploratory CI-like gate is disabled because it has fewer than 30 eligible components.
Even for fresh, the band describes conditional component stability only; repeated measurement fields are
needed to separate persistent item/mean structure from stochastic judge covariance.

## Low-capacity likelihood diagnostic

The PSD comparator uses a same-descendant, closed-one-hop-neighborhood Gram and tests only

```text
C_alpha = ((1-alpha) I + alpha K_adj) tensor I_4,
alpha in {0, .025, .05, .10, .20, .35, .50}.
```

The table reports held component-macro NLL gain per scalar over the independent block at the best tested
`alpha=0.50`, averaged over assignments.  Alpha was inspected on held outcomes, so this is an oracle diagnostic,
not a fitted or deployable selector.

| corpus | adjacency Gram gain | deranged Gram gain | topology-specific excess |
|---|---:|---:|---:|
| exploratory | +0.014673 | +0.003216 | +0.011457 |
| fresh | +0.021167 | +0.010030 | +0.011137 |

Both adjacency curves increase through the largest tested alpha.  That supports continued study, but it also
means the pilot does not locate an interior coupling.  The deranged Gram's positive gain shows that common-mode
structure remains useful even after topology is permuted; the excess for the true Gram is the more relevant
topology-specific diagnostic.

## Synthetic mechanism audit and sizing

The implemented synthetic control isolates the statistic: the mean and `B` are known, and each independent
component contributes one equal-weight adjacent-minus-two-controls contrast.  It does **not** rerun the real
calibration, KRR mean selection, unequal component sizes, or `B` estimation.  Consequently it is a mechanism
and rough sizing audit, not end-to-end power.

At the real eligible-component counts:

| components | true coupling | pointwise 95% lower band positive | simultaneous 95% lower band positive |
|---:|---:|---:|---:|
| 28 | 0.00 | 3.5% | 1.0% |
| 28 | 0.10 | 18.0% | 5.5% |
| 28 | 0.20 | 45.0% | 23.5% |
| 40 | 0.00 | 0.5% | 0.5% |
| 40 | 0.10 | 19.5% | 6.5% |
| 40 | 0.20 | 61.0% | 29.5% |

The frozen 80% power target fails.  A diagnostic extension with 200 replicates per cell gives:

| independent components | coupling 0.10 pointwise / simultaneous | coupling 0.20 pointwise / simultaneous |
|---:|---:|---:|
| 160 | 58.0% / 26.5% | 99.0% / 91.0% |
| 240 | 75.5% / 42.5% | 100% / 97.5% |
| 320 | 81.0% / 54.0% | 100% / 100% |
| 400 | 91.5% / 65.0% | 100% / 100% |

Thus roughly 320 independent components are needed to reach 80% **pointwise** mechanism detection for a 0.10
coupling under these favorable assumptions.  That is not adequate for a covariance-wide 95% gate: more than
400 would be needed for an 80% simultaneous target; the exact number was not estimated.  A full-procedure
campaign should be sized more conservatively because it
must also estimate calibration, mean, marginal covariance, and any coupling selector.

## Why not lower the confidence level?

The 95% quantity is a coverage/safety level, not a minimum covariance magnitude.  Lowering it does not reveal
more information; it moves the decision boundary so that noisier apparent correlations are admitted.  That is
reasonable for generating hypotheses, but unsafe for a production covariance because an overstated coupling
can make the posterior overconfident and can place supposedly independent measurements in separate batches.

For deployment, retain the 95% simultaneous rule and address low power with more independent components and
repeated judge calls.  For an explicitly non-operational discovery dashboard, an 80--90% pointwise band can
rank candidates for follow-up, provided it cannot modify `R`, QR block membership, or a production decision.

Also, “err on the low side of covariance entries” is not generally matrix-conservative: reducing an
off-diagonal correlation increases one eigenvalue and decreases another, and elementwise lower bounds need not
be PSD.  The future safe adapter should shrink on a PSD path and add a simultaneous spectral envelope:

```text
R_hat(alpha) = L_B [(1-alpha) I + alpha C_hat] L_B.T
R_safe       = R_hat(alpha_safe) + delta_95 I.
```

`alpha_safe` should be chosen by a lower bound on held-out benefit or worst-case validation.  `delta_95` must
come from a confirmatory full-procedure resampling audit.  A batching separation requires a high-confidence
**upper** bound on whitened cross-block coupling, not a lower bound.

## Recommended confirmatory campaign

1. Sample balanced `(anchor, adjacent positive, matched distant/hard negative)` triples and balance hop/tag
   strata by construction.  Adjacent samples test local smoothing; distant/hard negatives identify where it
   should stop and may become independent-batch candidates.
2. Obtain at least three independent calls per judge family for each sampled row.  This separates repeatable
   item effects from within-judge stochastic covariance and permits judge-specific and cross-judge estimates.
3. Split and resample whole endpoint components.  Size for at least 320 independent positive components as a
   favorable lower bound for 0.10 pointwise detection, then increase the target for simultaneous and
   full-procedure uncertainty.
4. Refit calibration, conditionalization, regional mean, `B`, covariance, and coupling selection inside each
   confirmatory null/power replicate.  Gate on held residual NLL, posterior calibration/risk, decision metrics,
   numerical loading, and a simultaneous PSD-safe error envelope.
5. Only after those gates pass, feed the validated dense block covariance to the existing square-root/QR
   conditioner.  Keep distant blocks independent only when the simultaneous upper coupling bound is below a
   preregistered batching tolerance.

This follows the calibrated joint-posterior and component-disjoint validation policy: covariance is admitted
because it improves held probabilistic predictions under its full uncertainty, not because a hand-set
correlation weight looks plausible.

## Reproduction

Real pilot:

```bash
python3 prototypes/mu_cosine/run_adjacent_residual_pilot.py \
  --artifact-repo /home/s243a/Projects/UnifyWeaver \
  --ckpt /home/s243a/Projects/UnifyWeaver/prototypes/mu_cosine/model_prod_namecond.pt \
  --assignments 10 --multiplier-draws 9999 --lmdb-no-lock \
  --out /tmp/adjacent_residual_pilot_final.json
```

Known-mean/known-`B` mechanism audit:

```bash
python3 prototypes/mu_cosine/run_adjacent_residual_synthetic.py \
  --out /tmp/adjacent_residual_synthetic_final.json
```

Focused tests:

```bash
python3 -m pytest -q \
  prototypes/mu_cosine/test_adjacent_residual_pilot.py \
  prototypes/mu_cosine/test_run_adjacent_residual_pilot.py
```

The original real run completed in 119.0 seconds; the portable-provenance rerun completed in 113.1 seconds.
Runtime and runtime paths are printed only to stdout and excluded from the deterministic scientific JSON.
Payload schema v2 stores role-keyed byte sizes and content hashes, not checkout/input/output locators.  Two
complete synthetic runs written to different output paths were byte-identical, and the schema-v2 real run
reproduced every prior scientific field exactly.

Thirteen focused tests pass; including the inherited structured-covariance, covariance-sensitivity, and
NumPy/Torch square-root-conditioner suites gives `123 passed, 9 skipped`.  Portable full-output SHA-256 values
are:

- real pilot: `407d52d8fe64df3bf9220da95976bee5c046c5d806d676aa38aafa4a475e100a`;
- synthetic mechanism: `62695568b1eae247ece2d3e37a18250c756b57536bdf6fe7e0ec7a91f8b8c18e`.

The compact tracked record is `repro/adjacent_residual_pilot/summary.json`.  Relocated inputs with identical
content now produce identical provenance.  Across different numerical stacks, still compare scientific fields
within tolerance rather than assuming byte-identical floating-point output.
