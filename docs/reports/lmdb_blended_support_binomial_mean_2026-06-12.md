# LMDB Blended Support-Binomial Mean

This pass separates the finite-support binomial shape from the rule used to
choose its mean.  The previous support-binomial variants were hard-coded:

- `support-binomial` used the prior mean clipped to the support interval; and
- `support-binomial-midpoint` used the midpoint of the support interval.

The prior-clipped rule often saturated at `L_max`, while midpoint worked better
as a support-placement baseline.  This pass adds an explicit mean rule so the
benchmark can test whether small amounts of prior skew help.

## Script Changes

`scripts/lmdb_parent_boundary_cache_benchmark.py` now accepts:

```text
--parametric-mean-model prior-clipped|midpoint|blend
--parametric-mean-blend <alpha>
```

For `support-binomial`, the mean rules are:

```text
prior-clipped: mean = clamp(prior_mean, 0, width)
midpoint:      mean = width / 2
blend:         mean = clamp(alpha * prior_mean + (1 - alpha) * midpoint, 0, width)
```

The older `support-binomial-midpoint` shape remains as a compatibility alias for
the midpoint rule.

Boundary rows now carry the selected mean model, blend alpha, chosen binomial
mean excess, and chosen binomial probability.

## Smoke Commands

All runs used the same d1 enwiki MTC setup:

```bash
python3 scripts/lmdb_parent_boundary_cache_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --boundary-depths 1 \
  --target-depths 3 \
  --children-per-node 64 \
  --frontier-limit 600 \
  --boundaries-per-depth 24 \
  --targets-per-depth 8 \
  --boundary-budget 6 \
  --budgets 6,8 \
  --path-cap 50000 \
  --expansion-cap 100000 \
  --seed enwiki-mtc-parametric-boundary-v1 \
  --admission-policy depth-prior \
  --safety-factor 1.25 \
  --max-histogram-bytes 64 \
  --parametric-bytes 64 \
  --parametric-shape-model support-binomial \
  --parametric-mass-model oracle \
  --parametric-mass-cap 100000 \
  --tail-epsilon 0.001 \
  --max-parent-depth 24 \
  --output-dir /mnt/c/Users/johnc/Scratch/blended-support-binomial-mean
```

The compared mean models were:

```text
prior-clipped
midpoint
blend alpha = 0.05
blend alpha = 0.10
blend alpha = 0.25
blend alpha = 0.50
blend alpha = 0.75
```

All runs selected:

| boundary_nodes | histogram_cached | parametric_cached | targets | boundary_budget |
|---------------:|-----------------:|------------------:|--------:|----------------:|
| 24 | 6 | 18 | 8 | 6 |

## Results

Budget 6:

| mean_model | alpha | mean_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_param_bins_spliced |
|------------|------:|--------:|---------:|-------------------------------:|--------------------:|------------------------:|
| prior-clipped | n/a | 1.139211 | 0.694605 | 0.562386 | 12.250 | 0.000 |
| midpoint | n/a | 1.048619 | 0.524309 | 0.258724 | 5.750 | 7.375 |
| blend | 0.05 | 1.064968 | 0.529012 | 0.204151 | 3.750 | 5.500 |
| blend | 0.10 | 1.238579 | 0.619289 | 0.301660 | 6.750 | 3.250 |
| blend | 0.25 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 0.000 |
| blend | 0.50 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 0.000 |
| blend | 0.75 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 0.000 |

Budget 8:

| mean_model | alpha | mean_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_param_bins_spliced |
|------------|------:|--------:|---------:|-------------------------------:|--------------------:|------------------------:|
| prior-clipped | n/a | 1.002190 | 0.501026 | 0.290814 | 16.000 | 0.250 |
| midpoint | n/a | 0.957091 | 0.477349 | 0.159200 | 6.000 | 8.625 |
| blend | 0.05 | 1.004725 | 0.498352 | 0.184026 | 7.000 | 7.125 |
| blend | 0.10 | 1.040822 | 0.520411 | 0.232611 | 11.000 | 3.750 |
| blend | 0.25 | 1.002576 | 0.501026 | 0.286303 | 15.625 | 0.750 |
| blend | 0.50 | 1.002190 | 0.501026 | 0.290814 | 16.000 | 0.250 |
| blend | 0.75 | 1.002190 | 0.501026 | 0.290814 | 16.000 | 0.250 |

## Interpretation

The blend knob works, but this sample says the raw prior mean is too large to
mix aggressively.  At alpha `0.25` and above, the blended mean is effectively
back near the saturated `L_max` regime.  Very small prior weights are usable,
but midpoint remains the strongest simple baseline overall:

- midpoint gives the best normalized L1/CDF on both budgets;
- alpha `0.05` gives the best budget-6 path-count relative error, but worse
  normalized shape than midpoint; and
- alpha `0.10` already loses much of the splice benefit.

The next refinement should not simply tune alpha globally.  A better rule would
normalize the prior mean relative to support width first, or learn a bucketed
alpha from boundary histograms.  For now, `midpoint` is the conservative default
candidate for finite-support binomial comparisons, and `blend` is useful as a
sweep knob.

## Validation

```bash
python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_depth_planning_prior_probe tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_parent_boundary_cache_benchmark.py tests/test_lmdb_parent_boundary_cache_benchmark.py
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model prior-clipped
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model midpoint
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model blend --parametric-mean-blend 0.05
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model blend --parametric-mean-blend 0.10
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model blend --parametric-mean-blend 0.25
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model blend --parametric-mean-blend 0.50
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mean-model blend --parametric-mean-blend 0.75
```
