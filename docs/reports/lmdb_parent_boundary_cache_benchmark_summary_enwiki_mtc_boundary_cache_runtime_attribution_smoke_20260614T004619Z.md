# LMDB Boundary Cache Benchmark

Graph: `enwiki_mtc_boundary_cache_runtime_attribution_smoke`

Root: `7345184`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 35 |
| boundary | 2 | 1000 |
| boundary | 3 | 1000 |
| target | 0 | 1 |
| target | 1 | 35 |
| target | 2 | 1000 |
| target | 3 | 1000 |
| target | 4 | 1000 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 158 | 80 | 78 | 158 | 0 | 8 | 8 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_capped | 158 | 158 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_recurrence_cycle_approximation | 158 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 158 | 564.911 | 2464.646 | 158 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 158 | 158 | 0 | 99.222 | 5.867 | 0.000 | 0.000 | 0.000 | 564.9 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 158 | 98.405 | 112 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | full_capped | cached_capped |
|--------|------|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|-------------|---------------|
| 8 | 8 | 0.233354 | 1.284672 | 1.284672 | 0.108989 | 0.295567 | 11.875 | 0.900 | 1.664 | 58170564.5 | 96875805.8 | 67.250 | 0.000 | 22.750 | 0.000 | 6491.000 | 345789.6 | 8 | 7 |

## Cached Runtime Attribution

These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-cap check, and parent lookup timing buckets.

| budget | rows | mean_cached_time_ns | mean_decode_ns | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |
|--------|-----:|--------------------:|---------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|
| 8 | 8 | 96875805.8 | 345789.6 | 56676.6 | 7469802.5 | 15685032.9 | 45101.1 | 23602402.8 | 73273403.0 | 0.004 | 0.001 | 0.077 |
