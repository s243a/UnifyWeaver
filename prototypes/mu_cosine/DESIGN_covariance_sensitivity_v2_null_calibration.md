# Covariance sensitivity v2 — family-wise nested-selection calibration

## Status

This is a **post-v1 addendum**, frozen after the v1 synthetic controls failed and before running v2.  It does
not replace or reinterpret `DESIGN_covariance_sensitivity.md`: v1 remains the originally preregistered
analysis.  In 100 block-null controls, v1 selected a nonzero covariance 90 times.  The 2-of-3 sign rule did
not protect a search over 216 nonzero candidates, and KRR mean estimation may itself induce correlated held
prediction error.  Therefore a nonzero v1 selection cannot be treated as evidence without family-wise null
calibration.

### Blocking correction before interpreting v2

An independent audit invalidated the first completed v2 output before it was interpreted.  The end-to-end
residual after a fitted regional mean is `c=e_H-W e_T`, not `e_H`; using `R_HH` as its recovery denominator is
wrong.  For fixed `W`, use

```text
Cov(c) = R_HH + W R_TT W.T - R_HT W.T - W R_TH.
```

Here kernel/ridge selection makes `W` outcome-dependent, so corrected v2 does not call the pre-KRR covariance
the truth.  It has two separately reported tracks:

1. **End-to-end KRR:** rerun KRR selection and block fitting; evaluate only selection rate, harm/gain versus the
   fitted block comparator, and the held-grid oracle.  It has no known-truth recovery denominator.
2. **Oracle-mean/known-B mechanism:** set the regional mean to known zero and supply the true block marginal.
   Then the held residual is exactly `e_H`, `R_HH` is the correct covariance, and residual/posterior recovery is
   a valid power gate.  This track uses its matching fixed-path/known-B null threshold.

The corrected wrong-geometry control is a fixed seed-33517 **derangement** congruence `P K P.T` (the first
cyclic shift of the seeded permutation with no fixed points).  It preserves the canonical kernel's complete
spectrum, maximum coupling, and off-diagonal RMS energy.  The earlier narrow-RBF control and both earlier v1
and pre-correction v2 recovery outputs are invalidated diagnostics, not results.

The energy-matched derangement is retained without redesign after its targeted result, but its block-selection
gate is relabelled diagnostic.  A dense uncentered RBF has a large positive/common correlation mode, and
`P K P.T` deliberately preserves that mode; therefore the candidate and permuted truth can remain useful
covariance approximations even though their strongest pair identities differ.  It is not an orthogonal
wrong-geometry null.  This common-mode contamination is itself reported and is not grounds for constructing a
friendlier control after seeing the result.

## Frozen v2 rule

Keep the v1 grid, three repeated partitions, eligibility rule, `0.001` near-best tolerance, and tie-breaking
unchanged.  For candidate `c`, define its observed macro inner gain

```text
g(c) = mean_fold [NLL_block(fold) - NLL_c(fold)].
```

Candidate `c` is eligible only when `g(c) > 0` and its gain is positive on at least two of three partitions.
The observed family-wise statistic is

```text
M_observed = max(0, max_eligible_c g(c)).
```

Generate 2,000 deterministic null fields and repeat the identical eligibility/grid search for each, yielding
`M_null[draw]`.  The threshold is the finite-simulation upper-95% order statistic at one-based rank
`ceil(0.95 * (draws + 1))`, capped at `draws`.  V2 may retain the candidate selected by the v1 conservative
tie-break only when `M_observed` is **strictly greater** than this threshold; otherwise it selects block.

Two null calibrations are recorded separately:

1. **A — conditional fixed-path/shared-z null.**  Hold all candidate correlation eigensystems fixed.  Draw one
   global field of iid standardized channel residuals per simulation, map the same item residual to every
   repeated held partition in which that item occurs, and run the candidate grid.  This preserves partition
   overlap but conditions away mean and marginal estimation.
2. **B — full block-null procedure.**  Draw one global field from the known block covariance.  On each of the
   same three partitions, rerun regional KRR mean selection, exact LOO residual construction, block-marginal
   fitting, and the complete candidate grid.  This includes mean-estimation error and partition overlap.
   Calibration seed is `991000`, disjoint from evaluation seeds.

The real-data implementation must use an analogous full-procedure callback or precomputed null distribution
on its exact item identities and partitions.  Unlike these synthetic controls, whose covariance endpoints are
known and outcome-blind, real endpoints are estimated by `fit_lmc_model`.  Therefore every real null draw must
refit calibration, conditional residuals, KRR mean, block marginal, bandwidths, and **all LMC endpoints** before
repeating selection.  Holding fitted real paths fixed omits endpoint-estimation overfit and must be labelled a
conditional/non-deployable diagnostic; it cannot silently stand in for B.

Two explicitly exploratory deployment capacities are compared:

- **v2A, full-grid calibrated:** retain all 216 nonzero length/channel/alpha candidates and gate the v1
  tie-break with the full-procedure family-wise threshold from that same grid.
- **v2B, canonical-alpha calibrated:** deployment selection contains only the eight nonzero alpha values at
  semantic/graph multipliers `(1,1)` and `beta=0`.  Its threshold comes from the full-procedure null maximum
  over those eight candidates.  The length/beta grid remains available only as an outer sensitivity/oracle
  diagnostic and cannot determine the deployed v2B covariance.

Thus “null A/B” below refers to conditional versus full-procedure calibration, while “selector v2A/v2B”
refers to full-grid versus canonical-alpha deployment capacity.

## Frozen attribution audit

Use 1,000 additional block-null fields with common random numbers.  Cross three mean treatments:

- full regional KRR selection/refit;
- fitted intercept only, with exact leave-one-out intercept residuals;
- oracle zero mean, with no mean estimation.

with three search sizes:

- all 216 nonzero candidates (`3 x 3 x 3 x 8`);
- the eight nonzero alpha values at canonical length/channel geometry;
- one canonical full-strength candidate.

Report the v1 nonzero-selection rate and the 50th/95th percentiles of `M_null`.  This audit attributes the v1
failure to grid multiplicity versus mean estimation; it is diagnostic and does not change threshold B.

## Corrected evaluation and gates

Rerun the same 100 evaluation replicates per v1 scenario for both v2A and v2B, using the same seeds and common `0.20` candidate
endpoint (true coupling `0.04`, `0.10`, and `0.20` corresponds to canonical `alpha=0.20`, `0.50`, and `1.0`).
Write a separate v2 JSON; do not overwrite the v1 JSON.

On the end-to-end KRR track, block-null and mean-only require mean harm no greater than `0.001` NLL/scalar and
block selection in at least 80%; deranged wrong-geometry is the common-mode diagnostic above.  In-family selection/gain is descriptive because no valid
closed-form truth denominator is asserted after outcome-selected KRR.

On the oracle-mean/known-B mechanism track, block-null uses the same harm/80%-block gate and deranged
wrong-geometry remains diagnostic.  Coupling `0.10` and `0.20` require nonzero selection in at least 80% and recovery of at least 50% of the
now-valid known-truth residual-NLL and posterior-risk gain.  Coupling `0.04` remains measured power only.
Failure after family-wise calibration means the procedure lacks power at this sample size; it is not
permission to lower the threshold after inspecting corrected v2.
