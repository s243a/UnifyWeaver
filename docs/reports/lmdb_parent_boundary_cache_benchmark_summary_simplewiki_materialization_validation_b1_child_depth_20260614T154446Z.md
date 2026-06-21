# LMDB Boundary Cache Benchmark

Graph: `simplewiki_materialization_validation_b1_child_depth`

Root: `2`

Target selection: `child-depth`

Path length budgets: `4,6,8`

Path count cap: `100000`

Expansion cap: `250000`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 128 |
| target | 0 | 1 |
| target | 1 | 128 |
| target | 2 | 15 |
| target | 3 | 0 |
| target | 4 | 0 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 33 | 32 | 1 | 33 | 0 | 3 | 8 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_exact | 33 | 33 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_uncapped_histogram | 33 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 33 | 4.303 | 4.333 | 0 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 33 | 33 | 0 | 1.030 | 1.030 | 0.000 | 0.000 | 0.000 | 4.3 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 33 | 40.364 | 52 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

Here `path_length_budget` is the maximum parent hops in a path. `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row.

| path_length_budget | rows | path_count_cap | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | full_path_count_cap_hits | full_expansion_cap_hits | cached_path_count_cap_hits | cached_expansion_cap_hits |
|-------------------:|-----:|---------------:|---------|--------|--------|----------|-------------------------------:|--------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|
| 4 | 3 | 100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.676 | 2.204 | 9966.3 | 22633.3 | 1.000 | 0.000 | 1.000 | 0.000 | 13.333 | 1300.0 | 0.667 | 0 | 0 | 0 | 0 |
| 6 | 3 | 100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.676 | 1.946 | 7466.7 | 14500.0 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 0 | 0 | 0 |
| 8 | 3 | 100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.676 | 1.809 | 7566.7 | 13433.3 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 | 1.000 | 0 | 0 | 0 | 0 |

## Search Termination Diagnostics

`path_count_cap` is a root-reaching path-count cap. It is distinct from `path_length_budget`, which is the maximum number of parent hops allowed in a path.

If a path-count or expansion cap fires, timing and cache-hit statistics describe only the enumerated prefix. Treat full-run cache benefit as an extrapolation unless the unvisited path mass is estimated and assumed to have comparable boundary-hit statistics.

| path_length_budget | rows | path_count_cap | full_stop_reasons | cached_stop_reasons | full_length_budget_cutoff_rows | cached_length_budget_cutoff_rows | full_cycle_skips | cached_cycle_skips |
|-------------------:|-----:|---------------:|-------------------|---------------------|-------------------------------:|---------------------------------:|-----------------:|-------------------:|
| 4 | 3 | 100000 | `{"complete": 3}` | `{"complete": 3}` | 0 | 0 | 0 | 0 |
| 6 | 3 | 100000 | `{"complete": 3}` | `{"complete": 3}` | 0 | 0 | 0 | 0 |
| 8 | 3 | 100000 | `{"complete": 3}` | `{"complete": 3}` | 0 | 0 | 0 | 0 |

## Cache Hit Geometry

| path_length_budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | hit_depth_histogram | remaining_budget_histogram |
|-------------------:|-----:|----------------:|---------------:|----------------------:|-----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|---------------------|----------------------------|
| 4 | 3 | 1.000 | 1.000 | 3.000 | 1.000 | 3.000 | 3.000 | 3 | 0 | 0 | `{"1": 3}` | `{"3": 3}` |
| 6 | 3 | 1.000 | 1.000 | 5.000 | 1.000 | 5.000 | 5.000 | 3 | 3 | 0 | `{"1": 3}` | `{"5": 3}` |
| 8 | 3 | 1.000 | 1.000 | 7.000 | 1.000 | 7.000 | 7.000 | 3 | 3 | 3 | `{"1": 3}` | `{"7": 3}` |

## Cached Runtime Attribution

These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-count-cap check, and parent lookup timing buckets.

| path_length_budget | rows | mean_cached_time_ns | mean_decode_ns | mean_decode_memo_hits | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_count_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |
|-------------------:|-----:|--------------------:|---------------:|----------------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|
| 4 | 3 | 22633.3 | 1300.0 | 0.667 | 1333.3 | 1366.7 | 866.7 | 166.7 | 5033.3 | 17600.0 | 0.057 | 0.059 | 0.060 |
| 6 | 3 | 14500.0 | 0.0 | 1.000 | 1166.7 | 1233.3 | 766.7 | 200.0 | 3366.7 | 11133.3 | 0.000 | 0.080 | 0.085 |
| 8 | 3 | 13433.3 | 0.0 | 1.000 | 1100.0 | 1266.7 | 700.0 | 200.0 | 3266.7 | 10166.7 | 0.000 | 0.082 | 0.094 |
