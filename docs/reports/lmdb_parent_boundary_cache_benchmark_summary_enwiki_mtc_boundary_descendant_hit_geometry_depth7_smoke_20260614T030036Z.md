# LMDB Boundary Cache Benchmark

Graph: `enwiki_mtc_boundary_descendant_hit_geometry_depth7_smoke`

Root: `7345184`

Target selection: `boundary-descendants`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 35 |
| boundary | 2 | 700 |
| boundary | 3 | 700 |
| target | 7 | 16665 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 48 | 48 | 0 | 48 | 0 | 6 | 8 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_capped | 48 | 48 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_recurrence_cycle_approximation | 48 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 48 | 746.708 | 3269.167 | 48 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 48 | 48 | 0 | 612.833 | 6.229 | 0.000 | 0.000 | 0.000 | 746.7 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 48 | 102.750 | 112 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | full_capped | cached_capped |
|--------|------|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------|---------------|
| 8 | 6 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 1.000 | 1.224 | 56780309.7 | 69235530.5 | 35.667 | 0.000 | 3.833 | 0.000 | 170.667 | 21200.0 | 34.000 | 6 | 6 |

## Cache Hit Geometry

| budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | hit_depth_histogram |
|--------|-----:|----------------:|---------------:|----------------------:|-----------------------:|---------------------|
| 8 | 6 | 35.667 | 7.383 | 0.617 | 0.140 | `{"4": 1, "5": 9, "6": 14, "7": 73, "8": 117}` |
