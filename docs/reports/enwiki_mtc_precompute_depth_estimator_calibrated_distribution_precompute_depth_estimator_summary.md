# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator_calibrated`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

## Calibration

| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |
|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|
| 2 | 8 | 66999.284 | 111665.496 | 28270.750 | 68.088 | 1.080 |

## Depth Recommendation

| depth | expected_hits | path_states | best_representation | hits_to_break_even | net_value | pays |
|------:|--------------:|------------:|--------------------|-------------------:|----------:|------|
| 0 | 1000.000 | 1.000 | exact_sparse_histogram | 2.911 | 38614779.341 | yes |
| 1 | 1000.000 | 1.000 | exact_sparse_histogram | 10.887 | 10342939.766 | yes |
| 2 | 600.000 | 1.667 | exact_sparse_histogram | 7.053 | 15921966.668 | yes |
| 3 | 360.000 | 2.778 | exact_sparse_histogram | 4.307 | 25974520.340 | yes |
| 4 | 168.000 | 5.952 | parametric_closed_form | 2.342 | 47332112.651 | yes |
| 5 | 39.476 | 25.332 | parametric_closed_form | 1.788 | 59702077.658 | yes |
| 6 | 9.276 | 107.803 | parametric_closed_form | 1.694 | 53907984.983 | yes |
| 7 | 2.180 | 458.779 | parametric_closed_form | 1.673 | 15518615.324 | yes |
| 8 | 0.512 | 1952.426 | parametric_closed_form | 1.668 | -151081590.705 | no |

## Representation Detail

| depth | representation | points | bytes | expected_hits | saved_per_hit | one_time_cost | hits_to_break_even | net_value | pays |
|------:|----------------|-------:|------:|--------------:|--------------:|--------------:|-------------------:|----------:|------|
| 0 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 38727.534 | 112755.071 | 2.911 | 38614779.341 | yes |
| 0 | sampled_50_point_distribution | 50 | 400.0 | 1000.000 | 0.000 | 138954.859 | n/a | -138954.859 | no |
| 0 | parametric_closed_form | 4 | 64.0 | 1000.000 | 0.000 | 116055.794 | n/a | -116055.794 | no |
| 1 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 10456.784 | 113844.645 | 10.887 | 10342939.766 | yes |
| 1 | sampled_50_point_distribution | 50 | 400.0 | 1000.000 | 0.000 | 138954.859 | n/a | -138954.859 | no |
| 1 | parametric_closed_form | 4 | 64.0 | 1000.000 | 0.000 | 116055.794 | n/a | -116055.794 | no |
| 2 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 26852.246 | 189377.921 | 7.053 | 15921966.668 | yes |
| 2 | sampled_50_point_distribution | 50 | 400.0 | 600.000 | 0.000 | 213398.560 | n/a | -213398.560 | no |
| 2 | parametric_closed_form | 4 | 64.0 | 600.000 | 0.000 | 190499.496 | n/a | -190499.496 | no |
| 3 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 73025.198 | 314540.356 | 4.307 | 25974520.340 | yes |
| 3 | sampled_50_point_distribution | 50 | 400.0 | 360.000 | 0.000 | 337471.421 | n/a | -337471.421 | no |
| 3 | parametric_closed_form | 4 | 64.0 | 360.000 | 73025.198 | 314572.356 | 4.308 | 25974488.340 | yes |
| 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 257450.647 | 670123.668 | 2.603 | 42581570.660 | yes |
| 4 | sampled_50_point_distribution | 50 | 400.0 | 168.000 | 0.000 | 691965.158 | n/a | -691965.158 | no |
| 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 285721.397 | 669066.093 | 2.342 | 47332112.651 | yes |
| 5 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 1527570.230 | 2835197.563 | 1.856 | 57467872.078 | yes |
| 5 | sampled_50_point_distribution | 50 | 400.0 | 39.476 | 283657.230 | 2855949.479 | 10.068 | 8341834.676 | yes |
| 5 | parametric_closed_form | 4 | 64.0 | 39.476 | 1584111.730 | 2833050.414 | 1.788 | 59702077.658 | yes |
| 6 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 7024857.923 | 12045553.051 | 1.715 | 53118017.890 | yes |
| 6 | sampled_50_point_distribution | 50 | 400.0 | 9.276 | 5809215.673 | 12065215.391 | 2.077 | 41821886.919 | yes |
| 6 | parametric_closed_form | 4 | 64.0 | 9.276 | 7109670.173 | 12042316.327 | 1.694 | 53907984.983 | yes |
| 7 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 30511700.711 | 51238506.360 | 1.679 | 15267802.179 | yes |
| 7 | sampled_50_point_distribution | 50 | 400.0 | 2.180 | 29324329.211 | 51257079.126 | 1.748 | 12661117.525 | yes |
| 7 | parametric_closed_form | 4 | 64.0 | 2.180 | 30624783.711 | 51234180.062 | 1.673 | 15518615.324 | yes |
| 8 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 130556675.129 | 218028371.239 | 1.670 | -151159405.626 | no |
| 8 | sampled_50_point_distribution | 50 | 400.0 | 0.512 | 129397574.379 | 218045854.431 | 1.685 | -151770561.015 | no |
| 8 | parametric_closed_form | 4 | 64.0 | 0.512 | 130698028.879 | 218022955.366 | 1.668 | -151081590.705 | no |

## Summary

- deepest recommended depth: `7`
- mean best net value: `12914822.892`
- note: point count is a representation cost input, not the break-even hit count.
