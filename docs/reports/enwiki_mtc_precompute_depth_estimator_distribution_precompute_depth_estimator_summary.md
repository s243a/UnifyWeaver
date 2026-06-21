# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `4.255699`

Target depth: `8`

## Depth Recommendation

| boundary_depth | suffix_hops | expected_hits | suffix_states | build_states | best_representation | hits_to_break_even | net_value | pays |
|---------------:|------------:|--------------:|--------------:|-------------:|--------------------|-------------------:|----------:|------|
| 0 | 8 | 1000.000 | 1952.426 | 1.000 | exact_sparse_histogram | 0.001 | 1951404.043 | yes |
| 1 | 7 | 1000.000 | 1952.426 | 1.000 | exact_sparse_histogram | 0.001 | 1951383.563 | yes |
| 2 | 6 | 600.000 | 1171.455 | 1.667 | exact_sparse_histogram | 0.003 | 702233.800 | yes |
| 3 | 5 | 360.000 | 702.873 | 2.778 | exact_sparse_histogram | 0.007 | 252640.648 | yes |
| 4 | 4 | 168.000 | 328.007 | 5.952 | exact_sparse_histogram | 0.026 | 54912.069 | yes |
| 5 | 3 | 39.476 | 77.075 | 25.332 | exact_sparse_histogram | 0.371 | 2970.217 | yes |
| 6 | 2 | 9.276 | 18.111 | 107.803 | exact_sparse_histogram | 6.550 | 46.262 | yes |
| 7 | 1 | 2.180 | 4.256 | 458.779 | exact_sparse_histogram | 149.439 | -455.871 | no |
| 8 | 0 | 0.512 | 1.000 | 1952.426 | exact_sparse_histogram | n/a | -1956.746 | no |

## Representation Detail

| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | saved_per_hit | one_time_cost | hits_to_break_even | net_value | pays |
|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------:|--------------:|-------------------:|----------:|------|
| 0 | 8 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 1952.426 | 1951.406 | 1.480 | 0.001 | 1951404.043 | yes |
| 0 | 8 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 1952.426 | 1951.406 | 2.240 | 0.001 | 1951403.283 | yes |
| 0 | 8 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1951.346 | 34.920 | 0.018 | 1951310.603 | yes |
| 1 | 7 | exact_sparse_histogram | 2 | 32.0 | 1000.000 | 1952.426 | 1951.386 | 1.960 | 0.001 | 1951383.563 | yes |
| 1 | 7 | sampled_up_to_50_point_distribution | 2 | 16.0 | 1000.000 | 1952.426 | 1951.386 | 3.480 | 0.002 | 1951382.043 | yes |
| 1 | 7 | parametric_closed_form | 4 | 64.0 | 1000.000 | 1952.426 | 1951.346 | 34.920 | 0.018 | 1951310.603 | yes |
| 2 | 6 | exact_sparse_histogram | 3 | 48.0 | 600.000 | 1171.455 | 1170.395 | 3.107 | 0.003 | 702233.800 | yes |
| 2 | 6 | sampled_up_to_50_point_distribution | 3 | 24.0 | 600.000 | 1171.455 | 1170.395 | 5.387 | 0.005 | 702231.520 | yes |
| 2 | 6 | parametric_closed_form | 4 | 64.0 | 600.000 | 1171.455 | 1170.375 | 35.587 | 0.030 | 702189.320 | yes |
| 3 | 5 | exact_sparse_histogram | 4 | 64.0 | 360.000 | 702.873 | 701.793 | 4.698 | 0.007 | 252640.648 | yes |
| 3 | 5 | sampled_up_to_50_point_distribution | 4 | 32.0 | 360.000 | 702.873 | 701.793 | 7.738 | 0.011 | 252637.608 | yes |
| 3 | 5 | parametric_closed_form | 4 | 64.0 | 360.000 | 702.873 | 701.793 | 36.698 | 0.052 | 252608.648 | yes |
| 4 | 4 | exact_sparse_histogram | 5 | 80.0 | 168.000 | 328.007 | 326.907 | 8.352 | 0.026 | 54912.069 | yes |
| 4 | 4 | sampled_up_to_50_point_distribution | 5 | 40.0 | 168.000 | 328.007 | 326.907 | 12.152 | 0.037 | 54908.269 | yes |
| 4 | 4 | parametric_closed_form | 4 | 64.0 | 168.000 | 328.007 | 326.927 | 39.872 | 0.122 | 54883.909 | yes |
| 5 | 3 | exact_sparse_histogram | 6 | 96.0 | 39.476 | 77.075 | 75.955 | 28.212 | 0.371 | 2970.217 | yes |
| 5 | 3 | sampled_up_to_50_point_distribution | 6 | 48.0 | 39.476 | 77.075 | 75.955 | 32.772 | 0.431 | 2965.657 | yes |
| 5 | 3 | parametric_closed_form | 4 | 64.0 | 39.476 | 77.075 | 75.995 | 59.252 | 0.780 | 2940.756 | yes |
| 6 | 2 | exact_sparse_histogram | 7 | 112.0 | 9.276 | 18.111 | 16.971 | 111.163 | 6.550 | 46.262 | yes |
| 6 | 2 | sampled_up_to_50_point_distribution | 7 | 56.0 | 9.276 | 18.111 | 16.971 | 116.483 | 6.864 | 40.942 | yes |
| 6 | 2 | parametric_closed_form | 4 | 64.0 | 9.276 | 18.111 | 17.031 | 141.723 | 8.322 | 16.258 | yes |
| 7 | 1 | exact_sparse_histogram | 8 | 128.0 | 2.180 | 4.256 | 3.096 | 462.619 | 149.439 | -455.871 | no |
| 7 | 1 | sampled_up_to_50_point_distribution | 8 | 64.0 | 2.180 | 4.256 | 3.096 | 468.699 | 151.403 | -461.951 | no |
| 7 | 1 | parametric_closed_form | 4 | 64.0 | 2.180 | 4.256 | 3.176 | 492.699 | 155.147 | -485.777 | no |
| 8 | 0 | exact_sparse_histogram | 9 | 144.0 | 0.512 | 1.000 | 0.000 | 1956.746 | n/a | -1956.746 | no |
| 8 | 0 | sampled_up_to_50_point_distribution | 9 | 72.0 | 0.512 | 1.000 | 0.000 | 1963.586 | n/a | -1963.586 | no |
| 8 | 0 | parametric_closed_form | 4 | 64.0 | 0.512 | 1.000 | 0.000 | 1986.346 | n/a | -1986.346 | no |

## Summary

- deepest recommended boundary depth: `6`
- mean best net value: `545908.665`
- note: point count is a representation cost input, not the break-even hit count.
