# LMDB Boundary Cache Benchmark

Graph: `enwiki_mtc_boundary_cache_estimator_validation_b4_recurrence`

Root: `7345184`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 10 |
| boundary | 2 | 100 |
| boundary | 3 | 500 |
| boundary | 4 | 500 |
| target | 0 | 1 |
| target | 1 | 10 |
| target | 2 | 97 |
| target | 3 | 500 |
| target | 4 | 500 |
| target | 5 | 500 |
| target | 6 | 500 |
| target | 7 | 500 |
| target | 8 | 500 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 81 | 1 | 80 | 81 | 0 | 2 | 8 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_capped | 81 | 81 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_recurrence_cycle_approximation | 81 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 81 | 701.395 | 3041.864 | 81 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 81 | 81 | 0 | 182.321 | 4.963 | 0.000 | 0.000 | 0.000 | 701.4 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 81 | 87.556 | 88 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | full_capped | cached_capped |
|--------|------|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|-------------|---------------|
| 8 | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 1.000 | 1.139 | 10699788.5 | 12179937.5 | 13.000 | 0.000 | 0.500 | 0.000 | 1144.000 | 47699.5 | 2 | 2 |
