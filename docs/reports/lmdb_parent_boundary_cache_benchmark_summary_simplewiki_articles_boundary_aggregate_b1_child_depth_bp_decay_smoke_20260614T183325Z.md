# LMDB Boundary Cache Benchmark

Graph: `simplewiki_articles_boundary_aggregate_b1_child_depth_bp_decay_smoke`

Root: `2`

Target selection: `child-depth`

Path length budgets: `6,8`

Path count cap: `100000`

Expansion cap: `250000`

Aggregate kernel: `bp-decay`

Aggregate branching factor: `1.029`

Aggregate power: `1.0`

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 128 |
| target | 0 | 1 |
| target | 1 | 128 |
| target | 2 | 20 |
| target | 3 | 0 |

| boundary_nodes | selected_boundary_nodes | target_ancestor_boundary_nodes_added | cached_boundary_nodes | parametric_boundary_nodes | targets | boundary_budget | boundary_builder |
|----------------|------------------------:|-------------------------------------:|----------------------:|--------------------------:|--------:|----------------:|-----------------|
| 54 | 50 | 4 | 54 | 0 | 12 | 8 | recurrence |

## Admission Policy

| policy | safety_factor | max_histogram_bytes | parametric_bytes | parametric_shape_model | parametric_mean_model | parametric_mean_blend | parametric_support_source | parametric_mass_model | parametric_mass_cap |
|--------|--------------:|--------------------:|-----------------:|------------------------|-----------------------|----------------------:|---------------------------|-----------------------|--------------------:|
| baseline | 1.25 | 1024 | 64 | empirical-prior | prior-clipped | 0.5 | measured | oracle | 1000000 |

## Boundary Admission Outcomes

| action | rows | histogram_cached | parametric_cached |
|--------|-----:|-----------------:|------------------:|
| materialize_exact | 54 | 54 | 0 |

## Boundary Admission Reasons

| reason | rows |
|--------|-----:|
| baseline_uncapped_histogram | 54 |

## Boundary Builders

| builder | rows | mean_nodes_or_states | mean_edges_examined | cycle_approximation | capped |
|---------|-----:|---------------------:|--------------------:|--------------------:|-------:|
| recurrence | 54 | 4.593 | 4.593 | 0 | 0 |

## Boundary Cache Build

| entries | histogram_cached | parametric_cached | mean_hist_paths | mean_hist_bins | mean_parametric_paths | mean_parametric_mass_ratio | mean_parametric_bins | mean_nodes_expanded | capped_entries |
|---------|-----------------:|------------------:|----------------:|---------------:|----------------------:|----------------------------:|---------------------:|--------------------:|---------------:|
| 54 | 54 | 0 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.6 | 0 |

## Boundary Cache Payloads

| role | entries | mean_payload_bytes | max_payload_bytes | mean_decoded_cdf |
|------|--------:|-------------------:|------------------:|-----------------:|
| histogram | 54 | 40.000 | 40 | 0.000000 |
| parametric | 0 | 0.000 | 0 | 0.000000 |

## Full Search Versus Boundary Cache

Here `path_length_budget` is the maximum parent hops in a path. `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row.

| path_length_budget | rows | path_count_cap | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_path_count_relative_error | mean_abs_path_delta | mean_aggregate_relative_error | mean_abs_aggregate_delta | mean_abs_mean_length_delta | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_param_hits | mean_hist_bins_spliced | mean_param_bins_spliced | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits | full_path_count_cap_hits | full_expansion_cap_hits | cached_path_count_cap_hits | cached_expansion_cap_hits |
|-------------------:|-----:|---------------:|---------|--------|--------|----------|-------------------------------:|--------------------:|------------------------------:|-------------------------:|---------------------------:|-----------------|----------------:|------------------:|--------------------:|---------------:|----------------:|-----------------------:|------------------------:|------------------------:|---------------:|----------------------:|-------------------------:|------------------------:|---------------------------:|--------------------------:|
| 6 | 12 | 100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.585 | 1.993 | 12917.4 | 25751.7 | 1.083 | 0.000 | 1.083 | 0.000 | 13.333 | 1466.9 | 0.750 | 0 | 0 | 0 | 0 |
| 8 | 12 | 100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.585 | 5.055 | 12484.2 | 64079.2 | 1.083 | 0.000 | 1.083 | 0.000 | 0.000 | 0.0 | 1.083 | 0 | 0 | 0 | 0 |

## Search Termination Diagnostics

`path_count_cap` is a root-reaching path-count cap. It is distinct from `path_length_budget`, which is the maximum number of parent hops allowed in a path.

If a path-count or expansion cap fires, timing and cache-hit statistics describe only the enumerated prefix. Treat full-run cache benefit as an extrapolation unless the unvisited path mass is estimated and assumed to have comparable boundary-hit statistics.

| path_length_budget | rows | path_count_cap | full_stop_reasons | cached_stop_reasons | full_length_budget_cutoff_rows | cached_length_budget_cutoff_rows | full_cycle_skips | cached_cycle_skips |
|-------------------:|-----:|---------------:|-------------------|---------------------|-------------------------------:|---------------------------------:|-----------------:|-------------------:|
| 6 | 12 | 100000 | `{"complete": 12}` | `{"complete": 12}` | 0 | 0 | 0 | 0 |
| 8 | 12 | 100000 | `{"complete": 12}` | `{"complete": 12}` | 0 | 0 | 0 | 0 |

## Cache Hit Geometry

| path_length_budget | rows | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | hit_depth_histogram | remaining_budget_histogram |
|-------------------:|-----:|----------------:|---------------:|----------------------:|-----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|---------------------|----------------------------|
| 6 | 12 | 1.083 | 1.000 | 5.000 | 1.000 | 5.000 | 5.000 | 13 | 13 | 0 | `{"1": 13}` | `{"5": 13}` |
| 8 | 12 | 1.083 | 1.000 | 7.000 | 1.000 | 7.000 | 7.000 | 13 | 13 | 13 | `{"1": 13}` | `{"7": 13}` |

## Cached Runtime Attribution

These columns attribute only the cached search path. `unattributed` is the remaining cached wall time after decode, splice, cache-probe, path-count-cap check, and parent lookup timing buckets.

| path_length_budget | rows | mean_cached_time_ns | mean_decode_ns | mean_decode_memo_hits | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_count_cap_check_ns | mean_attributed_ns | mean_unattributed_ns | decode_share | splice_share | parent_lookup_share |
|-------------------:|-----:|--------------------:|---------------:|----------------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|-------------:|-------------:|--------------------:|
| 6 | 12 | 25751.7 | 1466.9 | 0.750 | 2083.5 | 2458.4 | 1341.8 | 233.3 | 7583.9 | 18167.8 | 0.057 | 0.081 | 0.095 |
| 8 | 12 | 64079.2 | 0.0 | 1.083 | 1341.8 | 2433.4 | 1358.6 | 191.7 | 5325.4 | 58753.8 | 0.000 | 0.021 | 0.038 |
