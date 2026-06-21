# LMDB Recurrence Boundary Builder Smoke

Date: 2026-06-12

This smoke compares boundary-cache precomputation with the existing bounded
parent-path search builder and the shifted parent-histogram recurrence builder.
The recurrence builder keeps histograms unnormalized: every bin stores path
count mass.  If a future storage layer uses normalized distributions, each
shifted parent distribution must be multiplied by its parent path mass `N_p`
before summing, and the resulting `N_v` must be stored separately.

Common setup:

```text
--lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident
--root 7345184
--boundary-depths 1
--target-depths 2
--children-per-node 32
--frontier-limit 256
--boundaries-per-depth 12
--targets-per-depth 6
--boundary-budget 6
--budgets 6
--path-cap 20000
--expansion-cap 50000
--seed enwiki-mtc-recurrence-boundary-v1
--admission-policy baseline
```

| boundary_builder | boundary_nodes | cached_boundary_nodes | mean_hist_paths | mean_hist_bins | mean_nodes_or_states | mean_edges_examined | cycle_approximation | mean_l1 | mean_cdf | mean_path_relative_error |
|------------------|---------------:|----------------------:|----------------:|---------------:|---------------------:|--------------------:|--------------------:|--------:|---------:|-------------------------:|
| search | 12 | 12 | 7.750 | 3.667 | 14502.250 | 18762.250 | 0 | 0.000000 | 0.000000 | 0.000000 |
| recurrence | 12 | 12 | 7.750 | 3.667 | 171.500 | 792.917 | 11 | 0.000000 | 0.000000 | 0.000000 |

On this sample the recurrence-built boundary cache produced the same downstream
target-search histograms as the search-built boundary cache while reducing the
boundary precompute work by roughly two orders of magnitude.  The recurrence
builder reported cycle approximation on 11 of 12 boundary rows; this is expected
for enwiki category cones and should remain visible in reports because a
node-only recurrence cannot encode the full simple-path visited set.
