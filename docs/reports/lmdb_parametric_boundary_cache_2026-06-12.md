# LMDB Parametric Boundary Cache Benchmark

This pass adds an approximate boundary-cache path for `use_parametric_prior`
admission decisions.  Histogram-backed actions still splice histograms exactly as
before.  Parametric actions now store a compact approximate suffix distribution
and splice it through a separate parametric cache.

This is still a benchmark feature, not production runtime behavior.

## Approximation Used

The first parametric boundary state uses the empirical depth-conditioned
path-length prior already used by the admission policy.  This keeps three
related but different distributions separate:

- the **parent-count distribution** measures how many root-reaching parents a
  node has;
- the **path-length distribution** measures how many root-reaching parent paths
  land in each finite length bin; and
- the **planning prior** forecasts a path-length distribution from parent-count
  statistics so the cache can decide whether exact histogram materialization is
  worth attempting.

For each `L_max` bucket:

1. estimate the depth-prior distribution from root-reaching parent-degree
   signals;
2. align it to the measured boundary histogram's `L_min`;
3. scale it to the measured boundary path count; and
4. store the resulting compact approximate suffix histogram as the parametric
   boundary state.

The scaling step uses measured boundary mass because this benchmark is isolating
shape/storage effects first.  A later production-oriented pass should replace
that with an estimated mass model.

Gamma-like fits are plausible for the parent-count or branching variation
itself, especially in enwiki where `E[P^2]/E[P]` can be several times larger
than one.  They should not be treated as the default path-length histogram
family.  The path-length histogram is produced by repeated shifted sums of
parent states; when enough finite-variance layers mix, binomial or normal-like
approximations become the natural first candidates.  A binomial approximation
can still retain visible skew for small trial counts such as `n=10`, so it is
not only a symmetric large-`n` approximation.

## Smoke Commands

All runs used:

```text
--boundary-depths 2
--target-depths 3
--children-per-node 64
--frontier-limit 600
--boundaries-per-depth 24
--targets-per-depth 8
--boundary-budget 6
--budgets 6,8
--path-cap 50000
--expansion-cap 100000
--seed enwiki-mtc-parametric-boundary-v1
```

The compared modes were:

```text
--admission-policy baseline

--admission-policy depth-prior
--max-histogram-bytes 1024
--parametric-bytes 64

--admission-policy depth-prior
--max-histogram-bytes 64
--parametric-bytes 64
```

Outputs were written under:

```text
/mnt/c/Users/johnc/Scratch/parametric-boundary-cache
```

## Results

All runs sampled the same shape:

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

| run | histogram_cached | parametric_cached | action |
|-----|-----------------:|------------------:|--------|
| baseline | 24 | 0 | `materialize_exact`: 24 |
| depth-prior, 1024 bytes | 24 | 0 | `materialize_capped`: 24 |
| depth-prior, 64 bytes | 0 | 24 | `use_parametric_prior`: 24 |

Boundary build:

| run | mean_hist_paths | mean_hist_bins | mean_parametric_bins | mean_nodes_expanded |
|-----|----------------:|---------------:|---------------------:|--------------------:|
| baseline | 17.000 | 4.167 | 0.000 | 15227.5 |
| depth-prior, 1024 bytes | 17.000 | 4.167 | 0.000 | 15227.5 |
| depth-prior, 64 bytes | 0.000 | 0.000 | 8.792 | 15227.5 |

Search comparison:

| run | budget | rows | mean_l1 | max_l1 | mean_cdf | mean_node_ratio | mean_hist_hits | mean_param_hits |
|-----|-------:|-----:|--------:|-------:|---------:|----------------:|---------------:|----------------:|
| baseline | 6 | 8 | 0.000000 | 0.000000 | 0.000000 | 0.997 | 6.500 | 0.000 |
| baseline | 8 | 8 | 0.000000 | 0.000000 | 0.000000 | 1.000 | 8.500 | 0.000 |
| depth-prior, 1024 bytes | 6 | 8 | 0.000000 | 0.000000 | 0.000000 | 0.997 | 6.500 | 0.000 |
| depth-prior, 1024 bytes | 8 | 8 | 0.000000 | 0.000000 | 0.000000 | 1.000 | 8.500 | 0.000 |
| depth-prior, 64 bytes | 6 | 8 | 0.027796 | 0.139037 | 0.009887 | 0.997 | 0.000 | 6.500 |
| depth-prior, 64 bytes | 8 | 8 | 0.033983 | 0.259740 | 0.016613 | 1.000 | 0.000 | 8.500 |

## Interpretation

The permissive `1024` byte policy matches baseline behavior: all 24 boundaries
enter as histogram states, and target search uses histogram hits only.

The strict `64` byte policy now gets approximate boundary hits instead of falling
back to uncached traversal.  It records 24 parametric boundary states, with mean
parametric support around `8.8` bins.  The target searches see the same hit
rates as the histogram-backed runs, but the hits are approximate and introduce
small distribution error on this sample.

The node expansion ratio remains near the histogram-backed runs because the
parametric states are installed at the same boundary nodes.  The useful
difference is now measurable accuracy versus representation choice:

- exact/capped histograms: no observed error in this smoke;
- parametric prior states: mean L1 around `0.028` to `0.034`, max L1 up to
  `0.260`.

This creates the benchmark surface needed for the next step: compare multiple
parametric families and mass-estimation rules.

## Validation

```bash
python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_depth_planning_prior_probe tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_parent_boundary_cache_benchmark.py tests/test_lmdb_parent_boundary_cache_benchmark.py
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy baseline
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy depth-prior --max-histogram-bytes 1024
python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --admission-policy depth-prior --max-histogram-bytes 64
```
