# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator_cap_aware_smoke_matched`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

Target depth: `8`

Cap mode: `measured`

## Calibration

| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |
|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|
| 2 | 8 | 66999.284 | 111665.496 | 28270.750 | 68.088 | 1.080 |

## Depth Recommendation

| boundary_depth | suffix_hops | expected_hits | suffix_states | cap_limited_suffix_states | build_states | best_representation | hits_to_break_even | net_value | pays |
|---------------:|------------:|--------------:|--------------:|--------------------------:|-------------:|--------------------|-------------------:|----------:|------|
| 0 | 8 | 1000.000 | 1952.426 | 1.000 | 1.000 | sampled_up_to_50_point_distribution | 11.321 | 9799865.879 | yes |
| 1 | 7 | 1000.000 | 1952.426 | 1.000 | 1.000 | sampled_up_to_50_point_distribution | n/a | -112757.071 | no |
| 2 | 6 | 600.000 | 1171.455 | 1.000 | 1.667 | sampled_up_to_50_point_distribution | n/a | -187746.560 | no |
| 3 | 5 | 360.000 | 702.873 | 1.000 | 2.778 | sampled_up_to_50_point_distribution | n/a | -312365.207 | no |
| 4 | 4 | 168.000 | 328.007 | 1.000 | 5.952 | sampled_up_to_50_point_distribution | n/a | -667404.731 | no |
| 5 | 3 | 39.476 | 77.075 | 1.000 | 25.332 | sampled_up_to_50_point_distribution | n/a | -2831934.840 | no |
| 6 | 2 | 9.276 | 18.111 | 1.000 | 107.803 | sampled_up_to_50_point_distribution | n/a | -12041746.540 | no |
| 7 | 1 | 2.180 | 4.256 | 1.000 | 458.779 | sampled_up_to_50_point_distribution | n/a | -51234156.062 | no |
| 8 | 0 | 0.512 | 1.000 | 1.000 | 1952.426 | parametric_closed_form | n/a | -218022955.366 | no |

## Representation Detail

| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | net_value | pays |
|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------------------:|--------------:|---------------:|------------:|--------------:|-------------------:|----------:|------|
| 0 | 8 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1952.426 | 1.000 | 9367.370 | 1089.414 | 28270.750 | 112755.071 | 12.037 | 9254614.843 | yes |
| 0 | 8 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1952.426 | 1.000 | 9912.077 | 544.707 | 28270.750 | 112211.284 | 11.321 | 9799865.879 | yes |
| 0 | 8 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1.000 | 0.000 | 4357.658 | 113083.000 | 116055.794 | n/a | -116055.794 | no |
| 1 | 7 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 1952.426 | 1.000 | 0.000 | 2178.829 | 56541.500 | 113844.645 | n/a | -113844.645 | no |
| 1 | 7 | sampled_up_to_50_point_distribution | 2 | 16.0 | 1000.000 | 1952.426 | 1.000 | 0.000 | 1089.414 | 56541.500 | 112757.071 | n/a | -112757.071 | no |
| 1 | 7 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1.000 | 0.000 | 4357.658 | 113083.000 | 116055.794 | n/a | -116055.794 | no |
| 2 | 6 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 1171.455 | 1.000 | 0.000 | 3268.243 | 84812.250 | 189377.921 | n/a | -189377.921 | no |
| 2 | 6 | sampled_up_to_50_point_distribution | 3 | 24.0 | 600.000 | 1171.455 | 1.000 | 0.000 | 1634.122 | 84812.250 | 187746.560 | n/a | -187746.560 | no |
| 2 | 6 | parametric_closed_form | 4 | 64.0 | 600.000 | 1171.455 | 1.000 | 0.000 | 4357.658 | 113083.000 | 190499.496 | n/a | -190499.496 | no |
| 3 | 5 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 702.873 | 1.000 | 0.000 | 4357.658 | 113083.000 | 314540.356 | n/a | -314540.356 | no |
| 3 | 5 | sampled_up_to_50_point_distribution | 4 | 32.0 | 360.000 | 702.873 | 1.000 | 0.000 | 2178.829 | 113083.000 | 312365.207 | n/a | -312365.207 | no |
| 3 | 5 | parametric_closed_form | 4 | 64.0 | 360.000 | 702.873 | 1.000 | 0.000 | 4357.658 | 113083.000 | 314572.356 | n/a | -314572.356 | no |
| 4 | 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 328.007 | 1.000 | 0.000 | 5447.072 | 141353.750 | 670123.668 | n/a | -670123.668 | no |
| 4 | 4 | sampled_up_to_50_point_distribution | 5 | 40.0 | 168.000 | 328.007 | 1.000 | 0.000 | 2723.536 | 141353.750 | 667404.731 | n/a | -667404.731 | no |
| 4 | 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 328.007 | 1.000 | 0.000 | 4357.658 | 113083.000 | 669066.093 | n/a | -669066.093 | no |
| 5 | 3 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 77.075 | 1.000 | 0.000 | 6536.487 | 169624.500 | 2835197.563 | n/a | -2835197.563 | no |
| 5 | 3 | sampled_up_to_50_point_distribution | 6 | 48.0 | 39.476 | 77.075 | 1.000 | 0.000 | 3268.243 | 169624.500 | 2831934.840 | n/a | -2831934.840 | no |
| 5 | 3 | parametric_closed_form | 4 | 64.0 | 39.476 | 77.075 | 1.000 | 0.000 | 4357.658 | 113083.000 | 2833050.414 | n/a | -2833050.414 | no |
| 6 | 2 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 18.111 | 1.000 | 0.000 | 7625.901 | 197895.250 | 12045553.051 | n/a | -12045553.051 | no |
| 6 | 2 | sampled_up_to_50_point_distribution | 7 | 56.0 | 9.276 | 18.111 | 1.000 | 0.000 | 3812.951 | 197895.250 | 12041746.540 | n/a | -12041746.540 | no |
| 6 | 2 | parametric_closed_form | 4 | 64.0 | 9.276 | 18.111 | 1.000 | 0.000 | 4357.658 | 113083.000 | 12042316.327 | n/a | -12042316.327 | no |
| 7 | 1 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 4.256 | 1.000 | 0.000 | 8715.316 | 226166.000 | 51238506.360 | n/a | -51238506.360 | no |
| 7 | 1 | sampled_up_to_50_point_distribution | 8 | 64.0 | 2.180 | 4.256 | 1.000 | 0.000 | 4357.658 | 226166.000 | 51234156.062 | n/a | -51234156.062 | no |
| 7 | 1 | parametric_closed_form | 4 | 64.0 | 2.180 | 4.256 | 1.000 | 0.000 | 4357.658 | 113083.000 | 51234180.062 | n/a | -51234180.062 | no |
| 8 | 0 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 1.000 | 1.000 | 0.000 | 9804.730 | 254436.750 | 218028371.239 | n/a | -218028371.239 | no |
| 8 | 0 | sampled_up_to_50_point_distribution | 9 | 72.0 | 0.512 | 1.000 | 1.000 | 0.000 | 4902.365 | 254436.750 | 218023477.153 | n/a | -218023477.153 | no |
| 8 | 0 | parametric_closed_form | 4 | 64.0 | 0.512 | 1.000 | 1.000 | 0.000 | 4357.658 | 113083.000 | 218022955.366 | n/a | -218022955.366 | no |

## Summary

- deepest recommended boundary depth: `0`
- mean best net value: `-30623466.722`
- note: point count is a representation cost input, not the break-even hit count.
