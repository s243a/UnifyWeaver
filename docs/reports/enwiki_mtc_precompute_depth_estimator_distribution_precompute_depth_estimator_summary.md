# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

## Depth Recommendation

| depth | expected_hits | path_states | best_representation | hits_to_break_even | net_value | pays |
|------:|--------------:|------------:|--------------------|-------------------:|----------:|------|
| 0 | 1000.000 | 1.000 | exact_sparse_histogram | n/a | -1.480 | no |
| 1 | 1000.000 | 1.000 | exact_sparse_histogram | n/a | -1.960 | no |
| 2 | 600.000 | 1.667 | exact_sparse_histogram | 5.121 | 360.893 | yes |
| 3 | 360.000 | 2.778 | exact_sparse_histogram | 2.767 | 606.502 | yes |
| 4 | 168.000 | 5.952 | exact_sparse_histogram | 1.721 | 806.848 | yes |
| 5 | 39.476 | 25.332 | exact_sparse_histogram | 1.165 | 927.575 | yes |
| 6 | 9.276 | 107.803 | exact_sparse_histogram | 1.042 | 878.262 | yes |
| 7 | 2.180 | 458.779 | exact_sparse_histogram | 1.011 | 534.853 | yes |
| 8 | 0.512 | 1952.426 | exact_sparse_histogram | 1.003 | -957.350 | no |

## Representation Detail

| depth | representation | points | bytes | expected_hits | saved_per_hit | one_time_cost | hits_to_break_even | net_value | pays |
|------:|----------------|-------:|------:|--------------:|--------------:|--------------:|-------------------:|----------:|------|
| 0 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 0.000 | 1.480 | n/a | -1.480 | no |
| 0 | sampled_50_point_distribution | 50 | 400.0 | 1000.000 | 0.000 | 63.000 | n/a | -63.000 | no |
| 0 | parametric_closed_form | 4 | 64.0 | 1000.000 | 0.000 | 34.920 | n/a | -34.920 | no |
| 1 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 0.000 | 1.960 | n/a | -1.960 | no |
| 1 | sampled_50_point_distribution | 50 | 400.0 | 1000.000 | 0.000 | 63.000 | n/a | -63.000 | no |
| 1 | parametric_closed_form | 4 | 64.0 | 1000.000 | 0.000 | 34.920 | n/a | -34.920 | no |
| 2 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 0.607 | 3.107 | 5.121 | 360.893 | yes |
| 2 | sampled_50_point_distribution | 50 | 400.0 | 600.000 | 0.000 | 63.667 | n/a | -63.667 | no |
| 2 | parametric_closed_form | 4 | 64.0 | 600.000 | 0.587 | 35.587 | 60.659 | 316.413 | yes |
| 3 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 1.698 | 4.698 | 2.767 | 606.502 | yes |
| 3 | sampled_50_point_distribution | 50 | 400.0 | 360.000 | 0.778 | 64.778 | 83.286 | 215.223 | yes |
| 3 | parametric_closed_form | 4 | 64.0 | 360.000 | 1.698 | 36.698 | 21.615 | 574.502 | yes |
| 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 4.852 | 8.352 | 1.721 | 806.848 | yes |
| 4 | sampled_50_point_distribution | 50 | 400.0 | 168.000 | 3.952 | 67.952 | 17.193 | 596.048 | yes |
| 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 4.872 | 39.872 | 8.183 | 778.688 | yes |
| 5 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 24.212 | 28.212 | 1.165 | 927.575 | yes |
| 5 | sampled_50_point_distribution | 50 | 400.0 | 39.476 | 23.332 | 87.332 | 3.743 | 833.716 | yes |
| 5 | parametric_closed_form | 4 | 64.0 | 39.476 | 24.252 | 59.252 | 2.443 | 898.114 | yes |
| 6 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 106.663 | 111.163 | 1.042 | 878.262 | yes |
| 6 | sampled_50_point_distribution | 50 | 400.0 | 9.276 | 105.803 | 169.803 | 1.605 | 811.644 | yes |
| 6 | parametric_closed_form | 4 | 64.0 | 9.276 | 106.723 | 141.723 | 1.328 | 848.258 | yes |
| 7 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 457.619 | 462.619 | 1.011 | 534.853 | yes |
| 7 | sampled_50_point_distribution | 50 | 400.0 | 2.180 | 456.779 | 520.779 | 1.140 | 474.862 | yes |
| 7 | parametric_closed_form | 4 | 64.0 | 2.180 | 457.699 | 492.699 | 1.076 | 504.947 | yes |
| 8 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 1951.246 | 1956.746 | 1.003 | -957.350 | no |
| 8 | sampled_50_point_distribution | 50 | 400.0 | 0.512 | 1950.426 | 2014.426 | 1.033 | -1015.450 | no |
| 8 | parametric_closed_form | 4 | 64.0 | 0.512 | 1951.346 | 1986.346 | 1.018 | -986.899 | no |

## Summary

- deepest recommended depth: `7`
- mean best net value: `350.460`
- note: point count is a representation cost input, not the break-even hit count.
