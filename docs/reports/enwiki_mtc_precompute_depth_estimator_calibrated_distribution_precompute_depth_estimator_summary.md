# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator_calibrated`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

Target depth: `8`

## Calibration

| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |
|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|
| 2 | 8 | 66999.284 | 111665.496 | 28270.750 | 68.088 | 1.080 |

## Depth Recommendation

| boundary_depth | suffix_hops | expected_hits | suffix_states | build_states | best_representation | hits_to_break_even | net_value | pays |
|---------------:|------------:|--------------:|--------------:|-------------:|--------------------|-------------------:|----------:|------|
| 0 | 8 | 1000.000 | 1952.426 | 1.000 | sampled_up_to_50_point_distribution | 0.001 | 130782728917.374 | yes |
| 1 | 7 | 1000.000 | 1952.426 | 1.000 | sampled_up_to_50_point_distribution | 0.001 | 130754457621.586 | yes |
| 2 | 6 | 600.000 | 1171.455 | 1.667 | sampled_up_to_50_point_distribution | 0.002 | 47040906113.140 | yes |
| 3 | 5 | 360.000 | 702.873 | 2.778 | sampled_up_to_50_point_distribution | 0.007 | 16912084077.661 | yes |
| 4 | 4 | 168.000 | 328.007 | 5.952 | parametric_closed_form | 0.031 | 3672343216.786 | yes |
| 5 | 3 | 39.476 | 77.075 | 25.332 | parametric_closed_form | 0.561 | 196557671.118 | yes |
| 6 | 2 | 9.276 | 18.111 | 107.803 | parametric_closed_form | 10.944 | -1835423.400 | no |
| 7 | 1 | 2.180 | 4.256 | 458.779 | parametric_closed_form | 297.796 | -50859174.295 | no |
| 8 | 0 | 0.512 | 1.000 | 1952.426 | parametric_closed_form | n/a | -218022955.366 | no |

## Representation Detail

| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | saved_per_hit | one_time_cost | hits_to_break_even | net_value | pays |
|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------:|--------------:|-------------------:|----------:|------|
| 0 | 8 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1952.426 | 130782841.129 | 112755.071 | 0.001 | 130782728373.586 | yes |
| 0 | 8 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1952.426 | 130782841.129 | 112211.284 | 0.001 | 130782728917.374 | yes |
| 0 | 8 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 130698028.879 | 116055.794 | 0.001 | 130697912822.863 | yes |
| 1 | 7 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 1952.426 | 130754570.379 | 113844.645 | 0.001 | 130754456534.012 | yes |
| 1 | 7 | sampled_up_to_50_point_distribution | 2 | 16.0 | 1000.000 | 1952.426 | 130754570.379 | 112757.071 | 0.001 | 130754457621.586 | yes |
| 1 | 7 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 130698028.879 | 116055.794 | 0.001 | 130697912822.863 | yes |
| 2 | 6 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 1171.455 | 78401838.780 | 189377.921 | 0.002 | 47040904481.778 | yes |
| 2 | 6 | sampled_up_to_50_point_distribution | 3 | 24.0 | 600.000 | 1171.455 | 78401838.780 | 187746.560 | 0.002 | 47040906113.140 | yes |
| 2 | 6 | parametric_closed_form | 4 | 64.0 | 600.000 | 1171.455 | 78373568.030 | 190499.496 | 0.002 | 47023940913.596 | yes |
| 3 | 5 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 702.873 | 46978897.800 | 314540.356 | 0.007 | 16912081902.512 | yes |
| 3 | 5 | sampled_up_to_50_point_distribution | 4 | 32.0 | 360.000 | 702.873 | 46978897.800 | 312365.207 | 0.007 | 16912084077.661 | yes |
| 3 | 5 | parametric_closed_form | 4 | 64.0 | 360.000 | 702.873 | 46978897.800 | 314572.356 | 0.007 | 16912081870.512 | yes |
| 4 | 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 328.007 | 21834904.888 | 670123.668 | 0.031 | 3667592674.795 | yes |
| 4 | 4 | sampled_up_to_50_point_distribution | 5 | 40.0 | 168.000 | 328.007 | 21834904.888 | 667404.731 | 0.031 | 3667595393.731 | yes |
| 4 | 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 328.007 | 21863175.638 | 669066.093 | 0.031 | 3672343216.786 | yes |
| 5 | 3 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 77.075 | 4994334.554 | 2835197.563 | 0.568 | 194323465.538 | yes |
| 5 | 3 | sampled_up_to_50_point_distribution | 6 | 48.0 | 39.476 | 77.075 | 4994334.554 | 2831934.840 | 0.567 | 194326728.262 | yes |
| 5 | 3 | parametric_closed_form | 4 | 64.0 | 39.476 | 77.075 | 5050876.054 | 2833050.414 | 0.561 | 196557671.118 | yes |
| 6 | 2 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 18.111 | 1015526.047 | 12045553.051 | 11.861 | -2625390.493 | no |
| 6 | 2 | sampled_up_to_50_point_distribution | 7 | 56.0 | 9.276 | 18.111 | 1015526.047 | 12041746.540 | 11.858 | -2621583.982 | no |
| 6 | 2 | parametric_closed_form | 4 | 64.0 | 9.276 | 18.111 | 1100338.297 | 12042316.327 | 10.944 | -1835423.400 | no |
| 7 | 1 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 4.256 | 58961.788 | 51238506.360 | 869.012 | -51109987.439 | no |
| 7 | 1 | sampled_up_to_50_point_distribution | 8 | 64.0 | 2.180 | 4.256 | 58961.788 | 51234156.062 | 868.938 | -51105637.141 | no |
| 7 | 1 | parametric_closed_form | 4 | 64.0 | 2.180 | 4.256 | 172044.788 | 51234180.062 | 297.796 | -50859174.295 | no |
| 8 | 0 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 1.000 | 0.000 | 218028371.239 | n/a | -218028371.239 | no |
| 8 | 0 | sampled_up_to_50_point_distribution | 9 | 72.0 | 0.512 | 1.000 | 0.000 | 218023477.153 | n/a | -218023477.153 | no |
| 8 | 0 | parametric_closed_form | 4 | 64.0 | 0.512 | 1.000 | 0.000 | 218022955.366 | n/a | -218022955.366 | no |

## Summary

- deepest recommended boundary depth: `5`
- mean best net value: `36565373340.512`
- note: point count is a representation cost input, not the break-even hit count.
