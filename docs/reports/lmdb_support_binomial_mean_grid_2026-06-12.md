# LMDB Support-Binomial Mean Grid

This pass adds a small calibration-grid runner for the finite-support binomial
boundary approximation.  The prior PR showed that a midpoint mean is a strong
simple baseline, while raw prior means often saturate the finite support.  This
grid runner makes that comparison repeatable across boundary depths, target
depths, and blend weights.

## Script

`scripts/lmdb_support_binomial_mean_grid.py` runs
`scripts/lmdb_parent_boundary_cache_benchmark.py` across a Cartesian grid:

```text
--boundary-depth-grid
--target-depth-grid
--mean-models
--blend-values
```

Each case fixes:

```text
--parametric-shape-model support-binomial
```

and varies the support-binomial mean rule:

- `midpoint`;
- `blend alpha=...`; and
- any other mean model accepted by the underlying benchmark.

The output is a compact JSON record set plus a Markdown summary table.

## Smoke Command

```bash
python3 scripts/lmdb_support_binomial_mean_grid.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_support_binomial_mean_grid_smoke \
  --boundary-depth-grid 1,2 \
  --target-depth-grid 3 \
  --mean-models midpoint,blend \
  --blend-values 0.0,0.05,0.10 \
  --children-per-node 48 \
  --frontier-limit 300 \
  --boundaries-per-depth 12 \
  --targets-per-depth 4 \
  --boundary-budget 6 \
  --budgets 6,8 \
  --path-cap 30000 \
  --expansion-cap 60000 \
  --seed enwiki-mtc-support-binomial-grid-v1 \
  --output-dir docs/reports
```

Generated artifacts:

```text
docs/reports/enwiki_mtc_support_binomial_mean_grid_smoke_support_binomial_mean_grid.json
docs/reports/enwiki_mtc_support_binomial_mean_grid_smoke_support_binomial_mean_grid.md
```

## Results

All rows use oracle mass, so this grid isolates support-binomial mean placement.

Budget 6:

| boundary_depth | target_depth | mean_model | alpha | mean_l1 | mean_cdf | mean_path_rel | mean_abs_delta | mean_param_bins_spliced |
|---------------:|-------------:|------------|------:|--------:|---------:|--------------:|---------------:|------------------------:|
| 1 | 3 | midpoint | n/a | 0.716738 | 0.354471 | 0.183723 | 6.250 | 6.500 |
| 1 | 3 | blend | 0.000 | 0.716738 | 0.354471 | 0.183723 | 6.250 | 6.500 |
| 1 | 3 | blend | 0.050 | 0.820242 | 0.405313 | 0.185471 | 7.500 | 4.000 |
| 1 | 3 | blend | 0.100 | 0.830316 | 0.409892 | 0.294737 | 11.000 | 2.250 |
| 2 | 3 | midpoint | n/a | 0.005342 | 0.002671 | 0.033654 | 1.750 | 0.250 |
| 2 | 3 | blend | 0.000 | 0.005342 | 0.002671 | 0.033654 | 1.750 | 0.250 |
| 2 | 3 | blend | 0.050 | 0.007867 | 0.003934 | 0.038462 | 2.000 | 0.000 |
| 2 | 3 | blend | 0.100 | 0.007867 | 0.003934 | 0.038462 | 2.000 | 0.000 |

Budget 8:

| boundary_depth | target_depth | mean_model | alpha | mean_l1 | mean_cdf | mean_path_rel | mean_abs_delta | mean_param_bins_spliced |
|---------------:|-------------:|------------|------:|--------:|---------:|--------------:|---------------:|------------------------:|
| 1 | 3 | midpoint | n/a | 0.437500 | 0.215972 | 0.114583 | 2.250 | 3.000 |
| 1 | 3 | blend | 0.000 | 0.437500 | 0.215972 | 0.114583 | 2.250 | 3.000 |
| 1 | 3 | blend | 0.050 | 0.429167 | 0.207639 | 0.162500 | 3.250 | 2.000 |
| 1 | 3 | blend | 0.100 | 0.415278 | 0.202588 | 0.182639 | 3.750 | 2.000 |
| 2 | 3 | midpoint | n/a | 0.034722 | 0.013889 | 0.027778 | 0.500 | 0.500 |
| 2 | 3 | blend | 0.000 | 0.034722 | 0.013889 | 0.027778 | 0.500 | 0.500 |
| 2 | 3 | blend | 0.050 | 0.015873 | 0.007937 | 0.055556 | 1.000 | 0.000 |
| 2 | 3 | blend | 0.100 | 0.015873 | 0.007937 | 0.055556 | 1.000 | 0.000 |

## Interpretation

Midpoint remains the conservative default candidate.  For boundary depth `1`,
adding prior weight increases the chosen binomial probability away from `0.5`
and reduces useful splicing.  On budget `8`, small alpha values improve
normalized L1/CDF slightly, but they worsen path-count error.

For boundary depth `2`, the approximations are already close because fewer
parametric bins are needed in the target search.  Small alpha values again
improve normalized shape on budget `8` while worsening path-count error and
splicing no parametric bins in the summary.

The grid supports the previous conclusion: raw prior mean should not be blended
globally.  A future rule should normalize prior mean by support width or learn a
bucketed alpha from boundary histograms.

## Validation

```bash
python3 -m unittest tests.test_lmdb_support_binomial_mean_grid tests.test_lmdb_parent_boundary_cache_benchmark
python3 scripts/lmdb_support_binomial_mean_grid.py ... --graph-name enwiki_mtc_support_binomial_mean_grid_smoke
```

The explicit `py_compile` command was blocked by the local command launcher in
this session, but the unittest import path exercised the new script and tests.
