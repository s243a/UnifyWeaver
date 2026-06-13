# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator_validation_calibrated`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

Target depth: `8`

Cap mode: `validation`

## Calibration

| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |
|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|
| 2 | 8 | 66999.284 | 111665.496 | 28270.750 | 68.088 | 1.080 |

## Validation Measurements

| boundary_depth | rows | positive_hit_rows | zero_hit_rows | mean_time_ratio | mean_cache_hits | measured_saved_per_hit_ns | clipped_saved_per_hit_ns | usable_for_cap | measured_pays |
|---------------:|-----:|------------------:|--------------:|----------------:|----------------:|--------------------------:|-------------------------:|----------------|---------------|
| 2 | 2 | 1 | 1 | 1.228 | 60.000 | -37783.300 | 0.000 | yes | no |
| 3 | 2 | 1 | 1 | 1.116 | 27.000 | -45514.741 | 0.000 | yes | no |
| 4 | 2 | 1 | 1 | 1.161 | 26.000 | -63099.962 | 0.000 | yes | no |
| 5 | 2 | 2 | 0 | 1.089 | 16.000 | -57381.188 | 0.000 | yes | no |
| 6 | 2 | 1 | 1 | 1.064 | 2.000 | -340450.000 | 0.000 | yes | no |
| 7 | 2 | 0 | 2 | n/a | 0.000 | n/a | 0.000 | no | no |

## Depth Recommendation

| boundary_depth | suffix_hops | expected_hits | suffix_states | cap_limited_suffix_states | build_states | best_representation | validation_time_ratio | hits_to_break_even | net_value | pays |
|---------------:|------------:|--------------:|--------------:|--------------------------:|-------------:|--------------------|----------------------:|-------------------:|----------:|------|
| 0 | 8 | 1000.000 | 1952.426 | 1952.426 | 1.000 | sampled_up_to_50_point_distribution | n/a | 0.001 | 130782184210.125 | yes |
| 1 | 7 | 1000.000 | 1952.426 | 1952.426 | 1.000 | sampled_up_to_50_point_distribution | n/a | 0.001 | 130753368207.088 | yes |
| 2 | 6 | 600.000 | 1171.455 | 1171.455 | 1.667 | sampled_up_to_50_point_distribution | 1.228 | n/a | -187746.560 | no |
| 3 | 5 | 360.000 | 702.873 | 702.873 | 2.778 | sampled_up_to_50_point_distribution | 1.116 | n/a | -312365.207 | no |
| 4 | 4 | 168.000 | 328.007 | 328.007 | 5.952 | sampled_up_to_50_point_distribution | 1.161 | n/a | -667404.731 | no |
| 5 | 3 | 39.476 | 77.075 | 77.075 | 25.332 | sampled_up_to_50_point_distribution | 1.089 | n/a | -2831934.840 | no |
| 6 | 2 | 9.276 | 18.111 | 18.111 | 107.803 | sampled_up_to_50_point_distribution | 1.064 | n/a | -12041746.540 | no |
| 7 | 1 | 2.180 | 4.256 | 4.256 | 458.779 | parametric_closed_form | n/a | 305.534 | -50868672.675 | no |
| 8 | 0 | 0.512 | 1.000 | 1.000 | 1952.426 | parametric_closed_form | n/a | n/a | -218022955.366 | no |

## Representation Detail

| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | validation_time_ratio | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | net_value | pays |
|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------------------:|--------------:|----------------------:|---------------:|------------:|--------------:|-------------------:|----------:|------|
| 0 | 8 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1952.426 | 1952.426 | 130781751.714 | n/a | 1089.414 | 0.000 | 112755.071 | 0.001 | 130781638959.088 | yes |
| 0 | 8 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1952.426 | 1952.426 | 130782296.421 | n/a | 544.707 | 0.000 | 112211.284 | 0.001 | 130782184210.125 | yes |
| 0 | 8 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1952.426 | 130693671.221 | n/a | 4357.658 | 0.000 | 116055.794 | 0.001 | 130693555164.871 | yes |
| 1 | 7 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 1952.426 | 1952.426 | 130752391.550 | n/a | 2178.829 | 0.000 | 113844.645 | 0.001 | 130752277705.016 | yes |
| 1 | 7 | sampled_up_to_50_point_distribution | 2 | 16.0 | 1000.000 | 1952.426 | 1952.426 | 130753480.964 | n/a | 1089.414 | 0.000 | 112757.071 | 0.001 | 130753368207.088 | yes |
| 1 | 7 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1952.426 | 130693671.221 | n/a | 4357.658 | 0.000 | 116055.794 | 0.001 | 130693555164.871 | yes |
| 2 | 6 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 1171.455 | 1171.455 | 0.000 | 1.228 | 3268.243 | 0.000 | 189377.921 | n/a | -189377.921 | no |
| 2 | 6 | sampled_up_to_50_point_distribution | 3 | 24.0 | 600.000 | 1171.455 | 1171.455 | 0.000 | 1.228 | 1634.122 | 0.000 | 187746.560 | n/a | -187746.560 | no |
| 2 | 6 | parametric_closed_form | 4 | 64.0 | 600.000 | 1171.455 | 1171.455 | 0.000 | 1.228 | 4357.658 | 0.000 | 190499.496 | n/a | -190499.496 | no |
| 3 | 5 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 702.873 | 702.873 | 0.000 | 1.116 | 4357.658 | 0.000 | 314540.356 | n/a | -314540.356 | no |
| 3 | 5 | sampled_up_to_50_point_distribution | 4 | 32.0 | 360.000 | 702.873 | 702.873 | 0.000 | 1.116 | 2178.829 | 0.000 | 312365.207 | n/a | -312365.207 | no |
| 3 | 5 | parametric_closed_form | 4 | 64.0 | 360.000 | 702.873 | 702.873 | 0.000 | 1.116 | 4357.658 | 0.000 | 314572.356 | n/a | -314572.356 | no |
| 4 | 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 328.007 | 328.007 | 0.000 | 1.161 | 5447.072 | 0.000 | 670123.668 | n/a | -670123.668 | no |
| 4 | 4 | sampled_up_to_50_point_distribution | 5 | 40.0 | 168.000 | 328.007 | 328.007 | 0.000 | 1.161 | 2723.536 | 0.000 | 667404.731 | n/a | -667404.731 | no |
| 4 | 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 328.007 | 328.007 | 0.000 | 1.161 | 4357.658 | 0.000 | 669066.093 | n/a | -669066.093 | no |
| 5 | 3 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 77.075 | 77.075 | 0.000 | 1.089 | 6536.487 | 0.000 | 2835197.563 | n/a | -2835197.563 | no |
| 5 | 3 | sampled_up_to_50_point_distribution | 6 | 48.0 | 39.476 | 77.075 | 77.075 | 0.000 | 1.089 | 3268.243 | 0.000 | 2831934.840 | n/a | -2831934.840 | no |
| 5 | 3 | parametric_closed_form | 4 | 64.0 | 39.476 | 77.075 | 77.075 | 0.000 | 1.089 | 4357.658 | 0.000 | 2833050.414 | n/a | -2833050.414 | no |
| 6 | 2 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 18.111 | 18.111 | 0.000 | 1.064 | 7625.901 | 0.000 | 12045553.051 | n/a | -12045553.051 | no |
| 6 | 2 | sampled_up_to_50_point_distribution | 7 | 56.0 | 9.276 | 18.111 | 18.111 | 0.000 | 1.064 | 3812.951 | 0.000 | 12041746.540 | n/a | -12041746.540 | no |
| 6 | 2 | parametric_closed_form | 4 | 64.0 | 9.276 | 18.111 | 18.111 | 0.000 | 1.064 | 4357.658 | 0.000 | 12042316.327 | n/a | -12042316.327 | no |
| 7 | 1 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 4.256 | 4.256 | 50246.472 | n/a | 8715.316 | 0.000 | 51238506.360 | 1019.743 | -51128984.200 | no |
| 7 | 1 | sampled_up_to_50_point_distribution | 8 | 64.0 | 2.180 | 4.256 | 4.256 | 54604.130 | n/a | 4357.658 | 0.000 | 51234156.062 | 938.284 | -51115135.522 | no |
| 7 | 1 | parametric_closed_form | 4 | 64.0 | 2.180 | 4.256 | 4.256 | 167687.130 | n/a | 4357.658 | 0.000 | 51234180.062 | 305.534 | -50868672.675 | no |
| 8 | 0 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 1.000 | 1.000 | 0.000 | n/a | 9804.730 | 0.000 | 218028371.239 | n/a | -218028371.239 | no |
| 8 | 0 | sampled_up_to_50_point_distribution | 9 | 72.0 | 0.512 | 1.000 | 1.000 | 0.000 | n/a | 4902.365 | 0.000 | 218023477.153 | n/a | -218023477.153 | no |
| 8 | 0 | parametric_closed_form | 4 | 64.0 | 0.512 | 1.000 | 1.000 | 0.000 | n/a | 4357.658 | 0.000 | 218022955.366 | n/a | -218022955.366 | no |

## Summary

- deepest recommended boundary depth: `1`
- mean best net value: `29027846621.255`
- note: point count is a representation cost input, not the break-even hit count.
