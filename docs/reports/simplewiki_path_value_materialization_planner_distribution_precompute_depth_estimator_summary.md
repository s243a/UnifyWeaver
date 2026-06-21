# Distribution Precompute Depth Estimator

Graph: `simplewiki_path_value_materialization_planner`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `1.075000`

Target depth: `2`

Cap mode: `uncapped`

Recommendation score: `value-weighted`

## Cost Model

| uncached_cost_per_state | cached_eval_base_cost | cached_eval_cost_per_point | splice_cost_per_point | decode_cost_per_byte | storage_cost_per_byte | build_cost_per_state |
|------------------------:|----------------------:|---------------------------:|----------------------:|---------------------:|----------------------:|---------------------:|
| 5.000000 | 0.000000 | 0.010000 | 0.000000 | 0.000000 | 0.010000 | 1.000000 |

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
| bp_decay_auto | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| bp_decay_auto | 1 | 1 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | n/a | 0.261 | 4980.000 | 3594.233 | 3594.233 | yes |
| bp_decay_explicit_2p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| bp_decay_explicit_2p0 | 1 | 1 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | n/a | 0.261 | 4980.000 | 3236.512 | 3236.512 | yes |
| count | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| count | 1 | 1 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | n/a | 0.261 | 4980.000 | 3638.855 | 3638.855 | yes |
| weighted_power_1p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| weighted_power_1p0 | 1 | 1 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | n/a | 0.261 | 4980.000 | 3395.011 | 3395.011 | yes |
| weighted_power_2p0 | 0 | 2 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 1.000 | exact_sparse_histogram | n/a | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| weighted_power_2p0 | 1 | 1 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 1.075 | exact_sparse_histogram | n/a | 0.261 | 4980.000 | 3176.392 | 3176.392 | yes |

## Representation Detail

| variant | boundary_depth | suffix_hops | representation | points | bytes | expected_hits | hit_scale | planning_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | validation_time_ratio | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | economic_net | value_weighted_net | recommendation_net | pays |
|---------|---------------:|------------:|----------------|-------:|------:|--------------:|----------:|--------------:|--------------:|--------------------------:|--------------:|----------------------:|---------------:|------------:|--------------:|-------------------:|-------------:|-------------------:|-------------------:|------|
| bp_decay_auto | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 1.160 | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| bp_decay_auto | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 2.080 | 0.361 | 5766.045 | 5766.045 | 5766.045 | yes |
| bp_decay_auto | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.738 | n/a | 0.000 | 0.000 | 33.640 | 5.863 | 5704.485 | 5704.485 | 5704.485 | yes |
| bp_decay_auto | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 1.395 | 0.261 | 4980.000 | 3594.233 | 3594.233 | yes |
| bp_decay_auto | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 3.235 | 0.604 | 4978.160 | 3592.393 | 3592.393 | yes |
| bp_decay_auto | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.721811 | 671.453 | 1.075 | 1.075 | 5.335 | n/a | 0.000 | 0.000 | 33.715 | 6.320 | 4929.076 | 3548.484 | 3548.484 | yes |
| bp_decay_explicit_2p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 1.160 | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| bp_decay_explicit_2p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 2.080 | 0.361 | 5766.045 | 5766.045 | 5766.045 | yes |
| bp_decay_explicit_2p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.738 | n/a | 0.000 | 0.000 | 33.640 | 5.863 | 5704.485 | 5704.485 | 5704.485 | yes |
| bp_decay_explicit_2p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 1.395 | 0.261 | 4980.000 | 3236.512 | 3236.512 | yes |
| bp_decay_explicit_2p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 3.235 | 0.604 | 4978.160 | 3234.672 | 3234.672 | yes |
| bp_decay_explicit_2p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.650000 | 604.651 | 1.075 | 1.075 | 5.335 | n/a | 0.000 | 0.000 | 33.715 | 6.320 | 4929.076 | 3192.099 | 3192.099 | yes |
| count | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 1.160 | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| count | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 2.080 | 0.361 | 5766.045 | 5766.045 | 5766.045 | yes |
| count | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.738 | n/a | 0.000 | 0.000 | 33.640 | 5.863 | 5704.485 | 5704.485 | 5704.485 | yes |
| count | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 1.395 | 0.261 | 4980.000 | 3638.855 | 3638.855 | yes |
| count | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 3.235 | 0.604 | 4978.160 | 3637.015 | 3637.015 | yes |
| count | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.730769 | 679.785 | 1.075 | 1.075 | 5.335 | n/a | 0.000 | 0.000 | 33.715 | 6.320 | 4929.076 | 3592.940 | 3592.940 | yes |
| weighted_power_1p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 1.160 | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| weighted_power_1p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 2.080 | 0.361 | 5766.045 | 5766.045 | 5766.045 | yes |
| weighted_power_1p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.738 | n/a | 0.000 | 0.000 | 33.640 | 5.863 | 5704.485 | 5704.485 | 5704.485 | yes |
| weighted_power_1p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 1.395 | 0.261 | 4980.000 | 3395.011 | 3395.011 | yes |
| weighted_power_1p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 3.235 | 0.604 | 4978.160 | 3393.171 | 3393.171 | yes |
| weighted_power_1p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.681818 | 634.249 | 1.075 | 1.075 | 5.335 | n/a | 0.000 | 0.000 | 33.715 | 6.320 | 4929.076 | 3350.006 | 3350.006 | yes |
| weighted_power_2p0 | 0 | 2 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 1.160 | 0.201 | 5766.965 | 5766.965 | 5766.965 | yes |
| weighted_power_2p0 | 0 | 2 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.768 | n/a | 0.000 | 0.000 | 2.080 | 0.361 | 5766.045 | 5766.045 | 5766.045 | yes |
| weighted_power_2p0 | 0 | 2 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1.000000 | 1000.000 | 1.156 | 1.156 | 5.738 | n/a | 0.000 | 0.000 | 33.640 | 5.863 | 5704.485 | 5704.485 | 5704.485 | yes |
| weighted_power_2p0 | 1 | 1 | exact_sparse_histogram | 2 | 32.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 1.395 | 0.261 | 4980.000 | 3176.392 | 3176.392 | yes |
| weighted_power_2p0 | 1 | 1 | sampled_up_to_50_point_distribution | 2 | 16.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 5.355 | n/a | 0.000 | 0.000 | 3.235 | 0.604 | 4978.160 | 3174.552 | 3174.552 | yes |
| weighted_power_2p0 | 1 | 1 | parametric_closed_form | 4 | 64.0 | 930.233 | 0.637931 | 593.424 | 1.075 | 1.075 | 5.335 | n/a | 0.000 | 0.000 | 33.715 | 6.320 | 4929.076 | 3132.203 | 3132.203 | yes |

## Summary

- deepest recommended boundary depth: `1`
- mean best recommendation net value: `4587.583`
- note: point count is a representation cost input, not the break-even hit count.
