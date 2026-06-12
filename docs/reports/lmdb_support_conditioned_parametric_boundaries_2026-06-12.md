# LMDB Support-Conditioned Parametric Boundaries

This pass adds finite-support parametric boundary shapes to the LMDB parent
boundary-cache benchmark.  The previous empirical-prior shape could be hit many
times without contributing any suffix bins, because its support was shifted
outside the remaining target path budget.

## Script Changes

`scripts/lmdb_parent_boundary_cache_benchmark.py` now accepts:

```text
--parametric-shape-model empirical-prior|support-binomial|support-binomial-midpoint
```

The shape models are:

- `empirical-prior`: previous behavior, using the depth-conditioned empirical
  prior distribution and aligning it to the boundary `L_min`;
- `support-binomial`: finite-support binomial over
  `[histogram_L_min, histogram_L_max]`, with `p` derived from the prior mean;
  and
- `support-binomial-midpoint`: finite-support binomial over the same interval,
  with midpoint mean `p=0.5`.

The midpoint variant is intentionally simple.  It is a support-placement
baseline that answers whether target-level mass comparisons become meaningful
once approximate suffix bins actually fall inside the remaining path budget.

## Smoke Commands

All runs used:

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
  --parametric-mass-cap 100000 \
  --tail-epsilon 0.001 \
  --max-parent-depth 24 \
  --output-dir /mnt/c/Users/johnc/Scratch/support-conditioned-parametric-boundaries
```

The compared modes were:

```text
--parametric-shape-model empirical-prior --parametric-mass-model oracle
--parametric-shape-model support-binomial --parametric-mass-model oracle
--parametric-shape-model support-binomial-midpoint --parametric-mass-model oracle
--parametric-shape-model support-binomial-midpoint --parametric-mass-model unit
--parametric-shape-model support-binomial-midpoint --parametric-mass-model depth-prior
```

All runs selected the same cache shape:

| boundary_nodes | histogram_cached | parametric_cached | targets | boundary_budget |
|---------------:|-----------------:|------------------:|--------:|----------------:|
| 24 | 6 | 18 | 8 | 6 |

## Results

### Oracle Mass Shape Comparison

| shape_model | budget | mean_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_param_hits | mean_param_bins_spliced |
|-------------|-------:|--------:|---------:|-------------------------------:|--------------------:|----------------:|------------------------:|
| empirical-prior | 6 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 17.625 | 0.000 |
| empirical-prior | 8 | 1.070372 | 0.535117 | 0.293339 | 16.125 | 35.125 | 0.000 |
| support-binomial | 6 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 17.625 | 0.000 |
| support-binomial | 8 | 1.002190 | 0.501026 | 0.290814 | 16.000 | 35.125 | 0.250 |
| support-binomial-midpoint | 6 | 1.048619 | 0.524309 | 0.258724 | 5.750 | 17.625 | 7.375 |
| support-binomial-midpoint | 8 | 0.957091 | 0.477349 | 0.159200 | 6.000 | 35.125 | 8.625 |

The empirical-prior baseline reproduces the previous failure mode: many
parametric hits, but zero parametric bins spliced.  The prior-derived
support-binomial keeps bins inside the boundary support, but its prior mean
saturates at the support maximum on this sample, so it only weakly improves
splicing.  The midpoint support-binomial is the first useful target-level
baseline: it turns parametric hits into actual suffix bins and reduces
path-count error.

### Midpoint Shape Mass Comparison

| mass_model | budget | mean_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_param_bins_spliced | mean_parametric_mass_ratio |
|------------|-------:|--------:|---------:|-------------------------------:|--------------------:|------------------------:|----------------------------:|
| oracle | 6 | 1.048619 | 0.524309 | 0.258724 | 5.750 | 7.375 | 1.000 |
| oracle | 8 | 0.957091 | 0.477349 | 0.159200 | 6.000 | 8.625 | 1.000 |
| unit | 6 | 1.307544 | 0.653772 | 0.466544 | 10.125 | 2.125 | 0.190 |
| unit | 8 | 1.056345 | 0.528172 | 0.253051 | 14.125 | 2.000 | 0.190 |
| depth-prior | 6 | 0.472906 | 0.207411 | 58.352015 | 1261.875 | 14.500 | 100.812 |
| depth-prior | 8 | 0.390058 | 0.183098 | 29.090730 | 1344.875 | 20.000 | 100.812 |

This is the expected split.  The depth-prior mass model can improve normalized
shape metrics while catastrophically overcounting unnormalized path mass.  Unit
mass undercounts.  Oracle mass remains the control for shape behavior.

## Interpretation

Support conditioning fixed the immediate benchmark issue: target search can now
splice parametric suffix bins, so mass-model choices become visible at the
target level.

The midpoint support-binomial is not a final model.  It is a useful finite
support baseline.  The prior-derived support-binomial shows why a naive prior
mean is not enough: when the prior mean exceeds the observed support width, the
binomial saturates and places nearly all mass at `L_max`.

Next useful work:

- add a bounded mean rule that blends prior mean with support midpoint instead
  of hard-clipping at `p=1`;
- compare skewed small-`n` binomials against measured boundary histograms; and
- replace oracle support intervals with estimated `(L_min, L_max)` once the
  finite-support family behaves well under measured support.

## Validation

```bash
python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_depth_planning_prior_probe tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_parent_boundary_cache_benchmark.py tests/test_lmdb_parent_boundary_cache_benchmark.py
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-shape-model empirical-prior --parametric-mass-model oracle
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-shape-model support-binomial --parametric-mass-model oracle
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-shape-model support-binomial-midpoint --parametric-mass-model oracle
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-shape-model support-binomial-midpoint --parametric-mass-model unit
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-shape-model support-binomial-midpoint --parametric-mass-model depth-prior
```
