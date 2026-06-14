# LMDB Boundary Cache Benchmark

Graph: `enwiki_mtc_path_count_cap_terminology_smoke`

Root: `7345184`

Target selection: `boundary-descendants`

Path length budgets: `4`

Path count cap: `10`

Expansion cap: `5000`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 8 |
| boundary | 2 | 50 |
| target | 4 | 150 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 4 | 4 | 0 | 4 | 0 | 1 | 4 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_capped | 4 | 4 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_recurrence_cycle_approximation | 4 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 4 | 56.500 | 299.250 | 4 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 4 | 4 | 0 | 2.750 | 1.750 | 0.000 | 0.000 | 0.000 | 56.5 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 4 | 49.000 | 64 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

Here `path_length_budget` is the maximum parent hops in a path. `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row.

| path_length_budget | rows | path_count_cap | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | full_path_count_cap_hits | full_expansion_cap_hits | cached_path_count_cap_hits | cached_expansion_cap_hits |
|-------------------:|-----:|---------------:|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|
| 4 | 1 | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.896 | 0.205 | 1149535.0 | 236191.0 | 1.000 | 0.000 | 1.000 | 0.000 | 52.000 | 8116.0 | 0.000 | 0 | 0 | 0 | 0 |

## Search Termination Diagnostics

`path_count_cap` is a root-reaching path-count cap. It is distinct from `path_length_budget`, which is the maximum number of parent hops allowed in a path.

If a path-count or expansion cap fires, timing and cache-hit statistics describe only the enumerated prefix. Treat full-run cache benefit as an extrapolation unless the unvisited path mass is estimated and assumed to have comparable boundary-hit statistics.

| path_length_budget | rows | path_count_cap | full_stop_reasons | cached_stop_reasons | full_length_budget_cutoff_rows | cached_length_budget_cutoff_rows | full_cycle_skips | cached_cycle_skips |
|-------------------:|-----:|---------------:|-------------------|---------------------|-------------------------------:|---------------------------------:|-----------------:|-------------------:|
| 4 | 1 | 10 | `{"path_length_budget": 1}` | `{"path_length_budget": 1}` | 1 | 1 | 10 | 10 |

## Cache Hit Geometry

| path_length_budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | hit_depth_histogram | remaining_budget_histogram |
|-------------------:|-----:|----------------:|---------------:|----------------------:|-----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|---------------------|----------------------------|
| 4 | 1 | 1.000 | 2.000 | 2.000 | 1.000 | 2.000 | 2.000 | 1 | 0 | 0 | `{"2": 1}` | `{"2": 1}` |
