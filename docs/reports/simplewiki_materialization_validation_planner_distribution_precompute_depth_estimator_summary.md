# Distribution Precompute Depth Estimator

Graph: `simplewiki_materialization_validation_planner`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `1.075000`

Target depth: `2`

Cap mode: `validation`

Recommendation score: `value-weighted`

## Cost Model

| uncached_cost_per_state | cached_eval_base_cost | cached_eval_cost_per_point | splice_cost_per_point | decode_cost_per_byte | storage_cost_per_byte | build_cost_per_state |
|------------------------:|----------------------:|---------------------------:|----------------------:|---------------------:|----------------------:|---------------------:|
| 1.000000 | 1.000000 | 0.020000 | 0.000000 | 0.020000 | 0.010000 | 1.000000 |

## Validation Measurements

| boundary_depth | rows | positive_hit_rows | zero_hit_rows | mean_time_ratio | mean_cache_hits | measured_saved_per_hit_ns | clipped_saved_per_hit_ns | usable_for_cap | measured_pays |
|---------------:|-----:|------------------:|--------------:|----------------:|----------------:|--------------------------:|-------------------------:|----------------|---------------|
| 1 | 9 | 9 | 0 | 1.986 | 1.000 | -8522.333 | 0.000 | yes | no |

## Path-Value Sweep Measurements

Planning hit scale field: `mean_estimated_root_value_boundary_hit_fraction`

| variant | mode | budget | boundary_depth | target_depth | rows | kernel | b_p | power | boundary_hit_fraction | root_boundary_hit_fraction | root_value_boundary_hit_fraction |
|---------|------|-------:|---------------:|-------------:|-----:|--------|----:|------:|----------------------:|---------------------------:|---------------------------------:|
| bp_decay_auto | root-sample | 4 | 1 | 2 | 2 | bp-decay | 1.075000 | n/a | 0.544130 | 0.730769 | 0.721811 |
| bp_decay_explicit_2p0 | root-sample | 4 | 1 | 2 | 2 | bp-decay | 2.000000 | n/a | 0.544130 | 0.730769 | 0.650000 |
| count | root-sample | 4 | 1 | 2 | 2 | count | n/a | n/a | 0.544130 | 0.730769 | 0.730769 |
| weighted_power_1p0 | root-sample | 4 | 1 | 2 | 2 | weighted-power | n/a | 1.000 | 0.544130 | 0.730769 | 0.681818 |
| weighted_power_2p0 | root-sample | 4 | 1 | 2 | 2 | weighted-power | n/a | 2.000 | 0.544130 | 0.730769 | 0.637931 |

## Depth Recommendation

| variant | boundary_depth | suffix_hops | expected_hits | hit_scale | planning_hits | suffix_states | cap_limited_suffix_states | build_states | best_representation | validation_time_ratio | hits_to_break_even | economic_net | value_weighted_net | recommendation_net | pays |
|---------|---------------:|------------:|--------------:|----------:|--------------:|--------------:|--------------------------:|-------------:|--------------------|----------------------:|-------------------:|-------------:|-------------------:|-------------------:|------|
| bp_decay_auto | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | n/a | -1.480 | -1.480 | -1.480 | no |
| bp_decay_auto | 1 | 1 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | 1.986 | n/a | -2.035 | -2.035 | -2.035 | no |
| bp_decay_explicit_2p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | n/a | -1.480 | -1.480 | -1.480 | no |
| bp_decay_explicit_2p0 | 1 | 1 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | 1.986 | n/a | -2.035 | -2.035 | -2.035 | no |
| count | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | n/a | -1.480 | -1.480 | -1.480 | no |
| count | 1 | 1 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | 1.986 | n/a | -2.035 | -2.035 | -2.035 | no |
| weighted_power_1p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | n/a | -1.480 | -1.480 | -1.480 | no |
| weighted_power_1p0 | 1 | 1 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | 1.986 | n/a | -2.035 | -2.035 | -2.035 | no |
| weighted_power_2p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | n/a | -1.480 | -1.480 | -1.480 | no |
| weighted_power_2p0 | 1 | 1 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | 1.986 | n/a | -2.035 | -2.035 | -2.035 | no |

## Validation Agreement

| variant | boundary_depth | predicted_pays | measured_pays | matches | validation_time_ratio | measured_saved_per_hit_ns | recommendation_net |
|---------|---------------:|----------------|---------------|---------|----------------------:|--------------------------:|-------------------:|
| bp_decay_auto | 1 | no | no | yes | 1.986 | -8522.333 | -2.035 |
| bp_decay_explicit_2p0 | 1 | no | no | yes | 1.986 | -8522.333 | -2.035 |
| count | 1 | no | no | yes | 1.986 | -8522.333 | -2.035 |
| weighted_power_1p0 | 1 | no | no | yes | 1.986 | -8522.333 | -2.035 |
| weighted_power_2p0 | 1 | no | no | yes | 1.986 | -8522.333 | -2.035 |

- validation agreement: `5/5` best rows matched measured pay/no-pay.

## Representation Detail

| variant | boundary_depth | suffix_hops | representation | points | bytes | expected_hits | hit_scale | planning_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | validation_time_ratio | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | economic_net | value_weighted_net | recommendation_net | pays |
|---------|---------------:|------------:|----------------|-------:|------:|--------------:|----------:|--------------:|--------------:|--------------------------:|--------------:|----------------------:|---------------:|------------:|--------------:|-------------------:|-------------:|-------------------:|-------------------:|------|
| bp_decay_auto | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.320 | 0.000 | 1.480 | n/a | -1.480 | -1.480 | -1.480 | no |
| bp_decay_auto | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.160 | 0.000 | 2.240 | n/a | -2.240 | -2.240 | -2.240 | no |
| bp_decay_auto | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 1.280 | 0.000 | 34.920 | n/a | -34.920 | -34.920 | -34.920 | no |
| bp_decay_auto | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 0.000 | 1.986 | 0.640 | 0.000 | 2.035 | n/a | -2.035 | -2.035 | -2.035 | no |
| bp_decay_auto | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 0.000 | 1.986 | 0.320 | 0.000 | 3.555 | n/a | -3.555 | -3.555 | -3.555 | no |
| bp_decay_auto | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 0.000 | 1.986 | 1.280 | 0.000 | 34.995 | n/a | -34.995 | -34.995 | -34.995 | no |
| bp_decay_explicit_2p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.320 | 0.000 | 1.480 | n/a | -1.480 | -1.480 | -1.480 | no |
| bp_decay_explicit_2p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.160 | 0.000 | 2.240 | n/a | -2.240 | -2.240 | -2.240 | no |
| bp_decay_explicit_2p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 1.280 | 0.000 | 34.920 | n/a | -34.920 | -34.920 | -34.920 | no |
| bp_decay_explicit_2p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 0.000 | 1.986 | 0.640 | 0.000 | 2.035 | n/a | -2.035 | -2.035 | -2.035 | no |
| bp_decay_explicit_2p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 0.000 | 1.986 | 0.320 | 0.000 | 3.555 | n/a | -3.555 | -3.555 | -3.555 | no |
| bp_decay_explicit_2p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 0.000 | 1.986 | 1.280 | 0.000 | 34.995 | n/a | -34.995 | -34.995 | -34.995 | no |
| count | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.320 | 0.000 | 1.480 | n/a | -1.480 | -1.480 | -1.480 | no |
| count | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.160 | 0.000 | 2.240 | n/a | -2.240 | -2.240 | -2.240 | no |
| count | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 1.280 | 0.000 | 34.920 | n/a | -34.920 | -34.920 | -34.920 | no |
| count | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 0.000 | 1.986 | 0.640 | 0.000 | 2.035 | n/a | -2.035 | -2.035 | -2.035 | no |
| count | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 0.000 | 1.986 | 0.320 | 0.000 | 3.555 | n/a | -3.555 | -3.555 | -3.555 | no |
| count | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 0.000 | 1.986 | 1.280 | 0.000 | 34.995 | n/a | -34.995 | -34.995 | -34.995 | no |
| weighted_power_1p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.320 | 0.000 | 1.480 | n/a | -1.480 | -1.480 | -1.480 | no |
| weighted_power_1p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.160 | 0.000 | 2.240 | n/a | -2.240 | -2.240 | -2.240 | no |
| weighted_power_1p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 1.280 | 0.000 | 34.920 | n/a | -34.920 | -34.920 | -34.920 | no |
| weighted_power_1p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 0.000 | 1.986 | 0.640 | 0.000 | 2.035 | n/a | -2.035 | -2.035 | -2.035 | no |
| weighted_power_1p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 0.000 | 1.986 | 0.320 | 0.000 | 3.555 | n/a | -3.555 | -3.555 | -3.555 | no |
| weighted_power_1p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 0.000 | 1.986 | 1.280 | 0.000 | 34.995 | n/a | -34.995 | -34.995 | -34.995 | no |
| weighted_power_2p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.320 | 0.000 | 1.480 | n/a | -1.480 | -1.480 | -1.480 | no |
| weighted_power_2p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 0.160 | 0.000 | 2.240 | n/a | -2.240 | -2.240 | -2.240 | no |
| weighted_power_2p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 0.000 | n/a | 1.280 | 0.000 | 34.920 | n/a | -34.920 | -34.920 | -34.920 | no |
| weighted_power_2p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 0.000 | 1.986 | 0.640 | 0.000 | 2.035 | n/a | -2.035 | -2.035 | -2.035 | no |
| weighted_power_2p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 0.000 | 1.986 | 0.320 | 0.000 | 3.555 | n/a | -3.555 | -3.555 | -3.555 | no |
| weighted_power_2p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 0.000 | 1.986 | 1.280 | 0.000 | 34.995 | n/a | -34.995 | -34.995 | -34.995 | no |

## Summary

- deepest recommended boundary depth: `none`
- mean best recommendation net value: `-1.757`
- note: point count is a representation cost input, not the break-even hit count.
