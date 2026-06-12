# LMDB Depth Prior Calibration Sweep

This follow-up turns the depth-conditioned prior probe into a small calibration
tool for materialization planning.  The target question is no longer "does the
prior match the node-local histogram shape?"  The primary question is:

```text
does the prior predict support/storage risk well enough to decide whether exact
recurrence materialization is worth attempting?
```

Shape error remains in the output as a diagnostic, but planning metrics now get
their own table.

## Script Updates

`scripts/lmdb_depth_planning_prior_probe.py` now reports:

- empirical size-biased excess prior effective bins and bytes;
- binomial prior effective bins and bytes;
- Gamma prior effective bins and bytes;
- storage prediction ratios against realized recurrence histograms;
- under-prediction counts by prior family;
- a configurable empirical-prior safety factor; and
- a coarse admission decision per comparable target.

The default safety factor is `1.25`.

The calibration uses three distinct statistical objects:

- **parent-count distribution**: the distribution of root-reaching parent counts
  for nodes in an `L_max` bucket;
- **path-length distribution**: the node-local histogram of root-reaching parent
  paths by finite length; and
- **planning prior**: a depth-conditioned forecast of path-length support and
  storage cost derived from parent-count statistics.

This distinction matters for enwiki.  Gamma-like or heavier-tailed families may
be useful for parent-count variation, because the size-biased branching signal
`E[P^2]/E[P]` can be far above one.  The path-length histogram is a different
object: it is built by repeated shifted mixtures of parent states, so
finite-variance cases should drift toward binomial/normal-like behavior as more
layers are composed.  Skew can persist for shallow depth, dependent parent cones,
or heavy parent-count tails.  A binomial approximation is still useful in the
small-`n` regime because it can represent skew, for example at `n=10` with
small `p`.

## Smoke Command

```bash
python3 scripts/lmdb_depth_planning_prior_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_depth_prior_calibration_smoke \
  --target-depths 1,2,3,4 \
  --children-per-node 24 \
  --frontier-limit 180 \
  --targets-per-depth 4 \
  --max-parent-depth 24 \
  --max-prior-depth 24 \
  --tail-epsilon 0.001 \
  --safety-factor 1.25 \
  --path-cap 50000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-depth-prior-calibration-v1 \
  --output-dir /mnt/c/Users/johnc/Scratch/depth-planning-prior-calibration
```

Outputs:

```text
/mnt/c/Users/johnc/Scratch/depth-planning-prior-calibration/enwiki_mtc_depth_prior_calibration_smoke_depth_planning_prior_summary.json
/mnt/c/Users/johnc/Scratch/depth-planning-prior-calibration/enwiki_mtc_depth_prior_calibration_smoke_depth_planning_prior_summary.md
```

## Results

Selection:

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 24 |
| 2 | 180 |
| 3 | 180 |
| 4 | 180 |

Prior buckets:

| L_max | targets | mean_root_p | b_root | mean_excess | empirical_eff_bins | binomial_eff_bins | gamma_eff_bins | binom_p | gamma_shape | empirical_bytes | binomial_bytes | gamma_bytes |
|------:|--------:|------------:|-------:|------------:|-------------------:|------------------:|---------------:|--------:|------------:|----------------:|---------------:|------------:|
| 12 | 1 | 3.000 | 3.000000 | 2.000000 | 25 | 13 | 25 | 1.000000 | n/a | 400 | 208 | 400 |
| 15 | 1 | 2.000 | 2.000000 | 1.000000 | 16 | 16 | 16 | 1.000000 | n/a | 256 | 256 | 256 |
| 24 | 14 | 2.643 | 3.000000 | 2.000000 | 65 | 25 | 66 | 1.000000 | 88.800000 | 1040 | 400 | 1056 |

Planning calibration:

| L_max | rows | mean_realized_bins | empirical_eff_bins | mean_emp_ratio | mean_safety_ratio | safety_under | mean_binom_ratio | binom_under | mean_gamma_ratio | gamma_under | capped | cycle_approx |
|------:|-----:|-------------------:|-------------------:|---------------:|------------------:|-------------:|-----------------:|------------:|-----------------:|------------:|-------:|-------------:|
| 12 | 1 | 8.000 | 25.000 | 6.250 | 7.812 | 0 | 3.250 | 0 | 6.250 | 0 | 0 | 1 |
| 15 | 1 | 13.000 | 16.000 | 1.455 | 1.818 | 0 | 1.455 | 0 | 1.455 | 0 | 0 | 1 |
| 24 | 14 | 21.000 | 65.000 | 3.222 | 4.027 | 0 | 1.239 | 0 | 3.271 | 0 | 5 | 14 |

Admission decisions:

| decision | rows |
|----------|-----:|
| risky_try_capped_or_approx | 16 |

## Interpretation

The planning priors were conservative on this sample.  None of the empirical,
binomial, or Gamma storage estimates under-predicted realized sparse histogram
storage in the sampled buckets.  The empirical prior with a `1.25` safety factor
gave mean storage ratios of `7.812`, `1.818`, and `4.027` for `L_max` buckets
`12`, `15`, and `24`.

The binomial prior remained the cheapest storage predictor in these buckets, but
its `p` parameter saturated at `1.0` everywhere.  That makes it useful as a
compact planning proxy, not as evidence that a binomial family captures the
measured path-length shape in this capped sample.  The Gamma prior tracked the
empirical support more closely in the `L_max=24` bucket, but that should be read
as a storage-width diagnostic, not as a claim that Gamma is the right
path-length family.

All comparable rows were marked `risky_try_capped_or_approx` because all rows
were cycle-approximate, and five of the fourteen `L_max=24` rows still hit the
recurrence cap.  That is the correct policy outcome for this smoke: the
planning prior says exact recurrence may be expensive, so downstream search
should either try a capped recurrence or start from a closed-form approximation.

Shape diagnostics are intentionally secondary here.  The mean L1/CDF shape
errors were maximal in all buckets, which is expected when the recurrence rows
are cycle-approximate and sometimes capped.  The useful result is the storage
admission signal, not a distribution-fit claim.

## Validation

```bash
python3 -m unittest tests.test_lmdb_depth_planning_prior_probe tests.test_parent_histogram_recurrence tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_depth_planning_prior_probe.py tests/test_lmdb_depth_planning_prior_probe.py
python3 scripts/lmdb_depth_planning_prior_probe.py ... --graph-name enwiki_mtc_depth_prior_calibration_smoke
```
