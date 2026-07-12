# Synthetic covariance sensitivity v2 — corrected family-wise controls

## Bottom line

The original 2-of-3 nested selector is not calibrated for its search.  Under a block-null field it selected a
nonzero covariance about 85% of the time when KRR mean and block fitting were rerun.  A full-procedure
family-wise null threshold fixes the false-selection problem, but the corrected selector does not reach its
80% power gate at this sample size.

The oracle-mean/known-block mechanism control answers the scientific question more positively: at true maximum
whitened coupling 0.10 and 0.20, the selected procedure recovers 67–84% of the available residual-NLL gain and
74–96% of the posterior-risk gain, depending on selector capacity.  It detects the effect in only 31–46% of
replicates.  Thus accurate correlation **can** improve the estimate; the current estimation/selection procedure
is underpowered.

## Blocking corrections

The first v1 and first v2 outputs are invalidated diagnostics.  They scored the post-KRR residual
`c=e_H-W e_T` against the pre-KRR covariance `R_HH`.  For fixed `W`, the correct covariance is

```text
R_HH + W R_TT W.T - R_HT W.T - W R_TH,
```

and here KRR kernel/ridge selection makes `W` outcome-dependent.  Corrected v2 therefore separates:

- **end-to-end KRR**, which reports selection and gain/harm versus its fitted block comparator but no
  “known-truth recovery”; and
- **oracle-zero-mean/known-B**, where the scored held residual is exactly `e_H`, making `R_HH` and recovery
  mathematically valid.

The original wrong-geometry RBF also had roughly 32 times less RMS coupling than the in-family kernel.  It was
replaced with a fixed seed-33517 derangement congruence `P K P.T`, preserving PSD, spectrum, maximum coupling,
and off-diagonal energy.  This preserves the dense RBF's positive/common mode too, so its result is retained as
a common-mode-contaminated diagnostic, not an orthogonal negative-control gate.

The invalidated records are
`/tmp/covariance_sensitivity_synthetic_v1_INVALID_pre_krr_denominator_and_weak_wrong_geometry.json` and
`/tmp/covariance_sensitivity_synthetic_v2_INVALID_pre_krr_denominator.json`.  The shared v1 runner was amended
to the corrected derangement and therefore does not reproduce the original weak-geometry diagnostic.

## Null calibration and failure attribution

The 2,000-draw upper-95% family-wise thresholds were:

| candidate search | fixed-path/shared-z threshold | full KRR/block-procedure threshold | ratio | v1 nonzero rate, fixed / full |
|---|---:|---:|---:|---:|
| one canonical `alpha=1` | 0.003663 | 0.008390 | 2.29x | 10.6% / 18.9% |
| canonical eight-alpha grid | 0.004656 | 0.009416 | 2.02x | 28.2% / 43.2% |
| full 216-candidate grid | 0.008437 | 0.017601 | 2.09x | 60.7% / 84.7% |

The full-procedure null is about twice the fixed-path threshold; the conditional fixed-path null would
undercalibrate real selection after mean/marginal estimation.

The separate 1,000-field audit gives v1 nonzero-selection rates:

| mean treatment | one candidate | canonical alpha grid | full 216 grid |
|---|---:|---:|---:|
| oracle zero mean | 12.1% | 30.1% | 64.9% |
| fitted intercept only | 21.4% | 43.3% | 81.5% |
| selected regional KRR | 21.9% | 44.4% | 84.5% |

The largest contribution is search multiplicity: under KRR, full versus canonical capacity adds 40.1
percentage points, and canonical versus one candidate adds 22.5 points.  Mean estimation also matters:
oracle-zero to fitted-intercept adds 16.6 points on the full grid; the extra flexible-KRR step adds another
3.0 points.  The failure is therefore neither “just KRR” nor “just a bad covariance”: the grid is dominant,
with shared fitted-mean error materially raising the null maximum.

## Corrected end-to-end KRR controls

V2A searches all 216 candidates behind its full-procedure threshold.  V2B deploys only canonical lengths,
`beta=0`, and the alpha grid; the larger grid remains an outer oracle diagnostic.

| scenario | v2A nonzero | v2A NLL gain | v2B nonzero | v2B NLL gain | status |
|---|---:|---:|---:|---:|---|
| block null | 5% | -0.000171 | 7% | -0.000493 | both pass null gate |
| regional mean only | 19% | +0.001351 | 22% | +0.000524 | v2A passes (81% block); v2B misses by 2 points |
| deranged wrong geometry | 4% | +0.000076 | 4% | -0.000005 | common-mode diagnostic |
| coupling 0.04 | 2% | +0.000395 | 5% | +0.000537 | descriptive |
| coupling 0.10 | 3% | +0.000158 | 6% | +0.000174 | descriptive |
| coupling 0.20 | 6% | -0.000091 | 14% | +0.000169 | descriptive |

The in-family end-to-end rows deliberately have no known-truth recovery fraction.  Selection remains close to
the calibrated null rate even at coupling 0.20.  This is an end-to-end power limitation; it does not negate the
separate known-covariance mechanism result below.

## Oracle-mean/known-B mechanism and power

| scenario | v2A detect | v2A NLL / posterior recovery | v2B detect | v2B NLL / posterior recovery | gate |
|---|---:|---:|---:|---:|---|
| block null | 3% | n/a | 6% | n/a | both pass |
| coupling 0.04 | 16% | 46.9% / 12.8% | 18% | 65.8% / 45.9% | measured power only |
| coupling 0.10 | 31% | 67.3% / 73.5% | 35% | 69.5% / 85.6% | recovery passes; detection fails |
| coupling 0.20 | 46% | 71.1% / 89.2% | 46% | 83.5% / 95.9% | recovery passes; detection fails |
| deranged wrong geometry | 41% | 57.9% / 66.7% | 43% | 57.2% / 66.6% | common-mode diagnostic |

V2B's lower capacity gives the best in-family recovery, but neither capacity meets the frozen 80% detection
requirement.  The deranged kernel remains useful because permutation preserves the large positive/common mode
of this dense uncentered RBF; “wrong strongest pairs” does not imply covariance orthogonality.

## Kernel, adjacency, and batching implications

Homogeneous Gaussian uncertainty in item-feature distance does not define a new covariance family after
normalization; it broadens the RBF length scale.  For scalar `D ~ Normal(mu, tau^2)`,

```text
E exp(-D^2/(2 ell^2))
  = ell / sqrt(ell^2 + tau^2)
    * exp(-mu^2 / (2 (ell^2 + tau^2))).
```

Centering bounded predictions at `0.5` is a different idea.  The raw product
`(mu_i-0.5)(mu_j-0.5)` is a PSD linear Gram and preserves signed distance-from-neutral amplitude; subtracting
the same `0.5` inside an RBF distance cancels exactly.  The implemented secondary normalized Gram keeps only
centered direction/sign and is not presented as the literal raw product.

An outcome-blind inventory found that adjacent measurement rows already occur in the #3671 campaign, but
only incidentally: 876 exploratory and 688 fresh row-pairs have some endpoints one graph edge apart.  The
clean same-descendant/adjacent-root subset has 84 and 206 pairs respectively; the reverse same-root/adjacent-
descendant subset has none.  #3671 used per-row graph features, not an explicit cross-row adjacency kernel, so
these counts do not constitute an adjacency-covariance test.

The next sampling design treats adjacent rows as positive smoothing examples and matched distant rows as
contrastive negatives.  Adjacent positives teach dataset topology; distant/hard negatives show where
smoothing stops and provide candidate near-independent items.  Mean/representation learning remains separate
from covariance estimation on cross-fitted residuals.

For a validated geometry, distance-separated batching can use the operational condition

```text
norm(B^-1/2 R_ij B^-1/2, 2) <= epsilon_batch.
```

With an RBF, scalar coupling is about `0.011` at `3 ell` and `0.00034` at `4 ell`; distance uncertainty requires
using the broader effective length.  Items beyond a fresh-validated threshold can use independent/block
updates, while adjacent connected components remain joint QR blocks.  This is a sufficient batching rule,
not evidence that untested nearby rows are independent.

## Decision

- Retain the block covariance for deployment under the tested sample size/procedure.
- Do not interpret a v1 nonzero selection; a real family-wise null must repeat the exact partitions and refit
  every outcome-dependent candidate endpoint.
- Treat v2B as the more promising future capacity, not as validated: its mechanism recovery is stronger, but
  it fails the end-to-end mean-only gate (78% block versus required 80%) and remains underpowered.
- Better covariance data—repeated independent judge calls, more independent item fields, or a jointly modeled
  mean/covariance procedure—is higher leverage than expanding the kernel grid.
- Run the explicit adjacent-positive / matched-distant-negative study in
  `DESIGN_adjacent_row_residual_correlation.md` before claiming graph distance licenses sparse batching.
- These controls support the narrow claim that good correlation information can help.  They do not show that
  the current semantic/graph distances estimate that information reliably.

A real-data v2 null would also have to refit every outcome-dependent LMC endpoint on every null draw.  Holding
the real fitted paths fixed is only a conditional diagnostic and is one reason no real v2 gate is claimed here.

## Reproduction

```bash
python3 prototypes/mu_cosine/run_covariance_sensitivity_synthetic_v2.py \
  --replicates 100 --calibration-draws 2000 --audit-draws 1000 \
  --summary-only --out /tmp/covariance_sensitivity_synthetic_v2_corrected_final.json
```

The corrected run took 170.1 seconds.  The four focused covariance-sensitivity test files have 34 passing
tests (55 when the two inherited structured-covariance suites are included).  Real v1
multi-seed inference is explicitly blocked unless requested as a descriptive legacy run; no real-data v2
full-procedure selector has been implemented.

Corrected summary-only output SHA-256:
`6bbe4fdd35bc70fae71dcb4dfade8bbaed143d310dd26a75469350c2f670997f`.  The compact tracked record is
`repro/covariance_sensitivity/synthetic_v2_summary.json`; the full `/tmp` JSON also contains the audit tables
and exact configuration but is not a durable repository artifact.

```bash
python3 -m pytest -q \
  prototypes/mu_cosine/test_covariance_sensitivity.py \
  prototypes/mu_cosine/test_run_covariance_sensitivity.py \
  prototypes/mu_cosine/test_run_covariance_sensitivity_synthetic.py \
  prototypes/mu_cosine/test_run_covariance_sensitivity_synthetic_v2.py
```
