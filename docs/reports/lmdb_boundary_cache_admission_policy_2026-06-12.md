# LMDB Boundary Cache Admission Policy

This pass wires the depth-prior cache admission policy into the boundary-cache
benchmark.  It does not add production cache behavior.  The benchmark still
precomputes boundary histograms for measurement, then the policy decides whether
each measured boundary distribution would enter the cache.

Parametric decisions are recorded, but they are not treated as cache hits.  If a
boundary is classified as `use_parametric_prior` or `skip_cache`, the later
target search falls back to the normal uncached traversal when it reaches that
node.

## Script Changes

`scripts/lmdb_parent_boundary_cache_benchmark.py` now accepts:

```text
--admission-policy baseline|depth-prior
--safety-factor
--max-histogram-bytes
--parametric-bytes
--tail-epsilon
--max-parent-depth
```

For `depth-prior`, boundary candidates are bucketed by measured histogram
`L_max`, a depth-conditioned prior is estimated from root-reaching parent degree
signals, and `cache_admission_policy` decides one of:

- `materialize_exact`
- `materialize_capped`
- `use_parametric_prior`
- `skip_cache`

Only `materialize_exact` and `materialize_capped` insert a histogram into the
boundary cache.

## Smoke Commands

Baseline:

```bash
python3 scripts/lmdb_parent_boundary_cache_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_boundary_cache_policy_baseline_smoke \
  --boundary-depths 2 \
  --target-depths 3 \
  --children-per-node 64 \
  --frontier-limit 600 \
  --boundaries-per-depth 24 \
  --targets-per-depth 8 \
  --boundary-budget 6 \
  --budgets 6,8 \
  --path-cap 50000 \
  --expansion-cap 100000 \
  --seed enwiki-mtc-boundary-policy-v1 \
  --admission-policy baseline \
  --output-dir /mnt/c/Users/johnc/Scratch/boundary-cache-policy
```

Depth-prior, permissive byte cap:

```bash
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... \
  --graph-name enwiki_mtc_boundary_cache_policy_depth_prior_smoke \
  --admission-policy depth-prior \
  --safety-factor 1.25 \
  --max-histogram-bytes 1024 \
  --parametric-bytes 64
```

Depth-prior, strict byte cap:

```bash
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... \
  --graph-name enwiki_mtc_boundary_cache_policy_depth_prior_64b_smoke \
  --admission-policy depth-prior \
  --safety-factor 1.25 \
  --max-histogram-bytes 64 \
  --parametric-bytes 64
```

## Results

All three runs used the same sampled boundary and target sets:

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 35 |
| boundary | 2 | 600 |
| target | 0 | 1 |
| target | 1 | 35 |
| target | 2 | 600 |
| target | 3 | 600 |

Admission outcomes:

| run | boundary_nodes | cached_boundary_nodes | actions |
|-----|---------------:|----------------------:|---------|
| baseline | 24 | 24 | `materialize_exact`: 24 |
| depth-prior, 1024 bytes | 24 | 24 | `materialize_capped`: 24 |
| depth-prior, 64 bytes | 24 | 0 | `use_parametric_prior`: 24 |

Search comparison:

| run | budget | rows | mean_l1 | mean_node_ratio | mean_cache_hits | full_capped | cached_capped |
|-----|-------:|-----:|--------:|----------------:|----------------:|------------:|--------------:|
| baseline | 6 | 8 | 0.000000 | 0.999 | 4.875 | 0 | 0 |
| baseline | 8 | 8 | 0.000845 | 1.000 | 3.625 | 8 | 8 |
| depth-prior, 1024 bytes | 6 | 8 | 0.000000 | 0.999 | 4.875 | 0 | 0 |
| depth-prior, 1024 bytes | 8 | 8 | 0.000845 | 1.000 | 3.625 | 8 | 8 |
| depth-prior, 64 bytes | 6 | 8 | 0.000000 | 1.000 | 0.000 | 0 | 0 |
| depth-prior, 64 bytes | 8 | 8 | 0.000000 | 1.000 | 0.000 | 8 | 8 |

## Interpretation

With a permissive `1024` byte histogram budget, the depth-prior policy admits the
same 24 boundary histograms as the baseline.  They are labeled
`materialize_capped` rather than `materialize_exact` because the policy treats
cycle-skipped boundary histograms as not guaranteed exact boundary conditions.
Benchmark behavior is therefore identical to baseline.

With a strict `64` byte histogram budget, the policy records all 24 boundaries
as `use_parametric_prior` and inserts none into the histogram cache.  Because
the benchmark does not yet simulate parametric boundary hits, target search falls
back to uncached traversal and reports zero cache hits.  This is intentional: the
first integration measures admission decisions without pretending a parametric
state is an exact histogram.

The next benchmark step is to add a separate approximate-boundary path for
parametric priors, then compare accuracy and runtime against exact/capped
histogram materialization.

## Validation

```bash
python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_depth_planning_prior_probe tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_parent_boundary_cache_benchmark.py tests/test_lmdb_parent_boundary_cache_benchmark.py
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy baseline
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy depth-prior --max-histogram-bytes 1024
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy depth-prior --max-histogram-bytes 64
```
