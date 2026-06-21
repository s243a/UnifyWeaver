# LMDB Parametric Boundary Mass Models

This pass separates parametric boundary-state shape from path-count mass.  The
previous benchmark always used oracle mass: it aligned the empirical
depth-conditioned prior shape to a boundary node and scaled it by that node's
measured path count.  That is useful for isolating shape/storage behavior, but a
production cache needs an estimated mass model.

## Script Changes

`scripts/lmdb_parent_boundary_cache_benchmark.py` now accepts:

```text
--parametric-mass-model oracle|unit|depth-prior
--parametric-mass-cap
```

The modes are:

- `oracle`: current behavior, using measured boundary `path_count`;
- `unit`: normalized lower baseline with total mass `1`; and
- `depth-prior`: non-oracle branching-pressure estimate
  `(1 + base_mean_excess) ^ L_max`, capped by `--parametric-mass-cap`.

Boundary rows now record:

- `parametric_oracle_path_count`;
- `parametric_path_count`;
- `parametric_mass_delta`;
- `parametric_mass_ratio`; and
- whether the mass estimate was capped.

Comparison rows now also report unnormalized path-count error:

- `abs_path_count_delta`;
- `path_count_relative_error`; and
- histogram versus parametric bins actually spliced.

## Smoke Commands

All d1 comparison runs used:

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
  --output-dir /mnt/c/Users/johnc/Scratch/parametric-boundary-mass-models
```

The three runs changed only:

```text
--graph-name enwiki_mtc_parametric_mass_oracle_d1_smoke --parametric-mass-model oracle
--graph-name enwiki_mtc_parametric_mass_unit_d1_smoke --parametric-mass-model unit
--graph-name enwiki_mtc_parametric_mass_depth_prior_d1_smoke --parametric-mass-model depth-prior
```

## Results

All three runs selected the same nodes:

| boundary_nodes | histogram_cached | parametric_cached | targets | boundary_budget |
|---------------:|-----------------:|------------------:|--------:|----------------:|
| 24 | 6 | 18 | 8 | 6 |

Boundary-state mass differed sharply:

| mass_model | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins |
|------------|----------------------:|----------------------------:|---------------------:|
| oracle | 8.000 | 1.000 | 5.722 |
| unit | 1.000 | 0.190 | 1.000 |
| depth-prior | 609.833 | 100.812 | 13.167 |

Target-search comparison was identical across these three mass models:

| budget | mean_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced |
|-------:|--------:|---------:|-------------------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|
| 6 | 1.139211 | 0.694605 | 0.562386 | 12.250 | 4.000 | 17.625 | 1.500 | 0.000 |
| 8 | 1.070372 | 0.535117 | 0.293339 | 16.125 | 9.500 | 35.125 | 3.000 | 0.000 |

The equal target-search rows are not evidence that mass does not matter.  They
show that the current empirical depth-prior shape has a support-placement
problem for boundary splicing: target search reaches parametric boundary nodes,
but no parametric bins are actually spliced because the approximate suffix bins
are outside the remaining path budget.  In the d1 smoke, `mean_param_hits` is
large while `mean_param_bins_spliced` is exactly zero.

## Interpretation

The benchmark now separates boundary-state mass from shape and exposes mass
diagnostics in the JSONL and markdown summaries.  On this enwiki MTC smoke:

- `oracle` remains the shape-isolation control;
- `unit` gives a normalized lower mass baseline;
- `depth-prior` strongly overestimates boundary mass from branching pressure;
  and
- downstream target error is currently dominated by support placement, not mass.

The next fix should address the path-length support model before treating
target-level mass-model comparisons as meaningful.  Two likely options are:

- condition the empirical prior on the measured or estimated support interval
  `(L_min, L_max)` before scaling; or
- add a path-length family such as binomial over the known support, then compare
  `oracle`, `unit`, and `depth-prior` mass again.

## Validation

```bash
python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_depth_planning_prior_probe tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_parent_boundary_cache_benchmark.py tests/test_lmdb_parent_boundary_cache_benchmark.py
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mass-model oracle
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mass-model unit
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --parametric-mass-model depth-prior
```
