# LMDB Boundary Cache Benchmark

Graph: `enwiki_mtc_boundary_aggregate_root_cone_b2_t4_bp_decay_smoke`

Root: `7345184`

Target selection: `child-depth`

Parent filter: `root-cone`

Path length budgets: `6,8`

Path count cap: `50000`

Expansion cap: `100000`

Aggregate kernel: `bp-decay`

Aggregate branching factor: `2.164`

Aggregate power: `1.0`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 35 |
| boundary | 2 | 1000 |
| target | 0 | 1 |
| target | 1 | 35 |
| target | 2 | 1000 |
| target | 3 | 1000 |
| target | 4 | 1000 |

| root_cone_depth | root_cone_nodes | root_cone_children_per_node | root_cone_frontier_limit |
|----------------:|----------------:|----------------------------:|-------------------------:|
| 10 | 8897 | 64 | 1000 |

| root_cone_child_depth | nodes |
|----------------------:|------:|
| 0 | 1 |
| 1 | 35 |
| 2 | 984 |
| 3 | 968 |
| 4 | 989 |
| 5 | 989 |
| 6 | 991 |
| 7 | 990 |
| 8 | 987 |
| 9 | 974 |
| 10 | 989 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 22 | 20 | 2 | 22 | 0 | 4 | 10 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_exact | 22 | 22 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_uncapped_histogram | 22 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 22 | 6.636 | 10.227 | 0 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 22 | 22 | 0 | 4.682 | 2.545 | 0.000 | 0.000 | 0.000 | 6.6 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 22 | 58.545 | 124 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

Here `path_length_budget` is the maximum parent hops in a path. `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row.

| path_length_budget | rows | path_count_cap | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_aggregate_relative_error | mean_abs_aggregate_delta | mean_abs_mean_length_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | mean_full_filtered_parent_skips | mean_cached_filtered_parent_skips | full_path_count_cap_hits | full_expansion_cap_hits | cached_path_count_cap_hits | cached_expansion_cap_hits |
|-------------------:|-----:|---------------:|---------|--------|--------|----------|-------------------------------:|--------------------:|------------------------------:|-------------------------:|---------------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------------------------:|---------------------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|
| 6 | 4 | 50000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.821 | 1.315 | 11599.2 | 15524.5 | 0.250 | 0.000 | 0.500 | 0.000 | 13.000 | 1574.8 | 0.000 | 4.500 | 2.500 | 0 | 0 | 0 | 0 |
| 8 | 4 | 50000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.821 | 1.351 | 7274.8 | 8399.8 | 0.250 | 0.000 | 0.500 | 0.000 | 0.000 | 0.0 | 0.250 | 4.500 | 2.500 | 0 | 0 | 0 | 0 |

## Search Termination Diagnostics

`path_count_cap` is a root-reaching path-count cap. It is distinct from `path_length_budget`, which is the maximum number of parent hops allowed in a path.

If a path-count or expansion cap fires, timing and cache-hit statistics describe only the enumerated prefix. Treat full-run cache benefit as an extrapolation unless the unvisited path mass is estimated and assumed to have comparable boundary-hit statistics.

| path_length_budget | rows | path_count_cap | full_stop_reasons | cached_stop_reasons | full_length_budget_cutoff_rows | cached_length_budget_cutoff_rows | full_cycle_skips | cached_cycle_skips |
|-------------------:|-----:|---------------:|-------------------|---------------------|-------------------------------:|---------------------------------:|-----------------:|-------------------:|
| 6 | 4 | 50000 | `{"complete": 4}` | `{"complete": 4}` | 0 | 0 | 0 | 0 |
| 8 | 4 | 50000 | `{"complete": 4}` | `{"complete": 4}` | 0 | 0 | 0 | 0 |

## Cache Hit Geometry

| path_length_budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | hit_depth_histogram | remaining_budget_histogram |
|-------------------:|-----:|----------------:|---------------:|----------------------:|-----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|---------------------|----------------------------|
| 6 | 4 | 0.250 | 1.000 | 5.000 | 2.000 | 5.000 | 5.000 | 1 | 1 | 0 | `{"1": 1}` | `{"5": 1}` |
| 8 | 4 | 0.250 | 1.000 | 7.000 | 2.000 | 7.000 | 7.000 | 1 | 1 | 1 | `{"1": 1}` | `{"7": 1}` |

## Cached Runtime Attribution

These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-count-cap check, and parent lookup timing buckets.

| path_length_budget | rows | mean_cached_time_ns | mean_decode_ns | mean_decode_memo_hits | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_count_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |
|-------------------:|-----:|--------------------:|---------------:|----------------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|
| 6 | 4 | 15524.5 | 1574.8 | 0.000 | 550.0 | 600.0 | 275.0 | 50.0 | 3049.8 | 12474.8 | 0.101 | 0.035 | 0.039 |
| 8 | 4 | 8399.8 | 0.0 | 0.250 | 425.0 | 425.0 | 199.8 | 50.0 | 1099.8 | 7300.0 | 0.000 | 0.051 | 0.051 |
