# Distribution Precompute Depth Estimator

Graph: `enwiki_mtc_precompute_depth_estimator_root_conditioned_smoke`

Expected queries: `1000.000`

Default parent branching prior `b = E[p^2] / E[p]`: `2.164035`

Target depth: `8`

Cap mode: `validation`

## Branching Profile

Source paths: `1`

Degree scope: `root_conditioned_parent_degree`

Overall root-conditioned `b`: `2.164035`

Profile nodes: `4999`

Manual branching override: `no`

Manual depth override count: `0`

Profile sample cap: `5000`

Profile truncated: `no`

| child_depth | nodes | mean_parent_degree | b | mean_excess | max_parent_degree |
|------------:|------:|-------------------:|--:|------------:|------------------:|
| 0 | 0 | 0.000000 | n/a | n/a | 0 |
| 1 | 35 | 2.114286 | 2.621622 | 1.621622 | 5 |
| 2 | 1007 | 1.859980 | 2.349706 | 1.349706 | 9 |
| 3 | 3957 | 1.652515 | 2.105674 | 1.105674 | 8 |

## Calibration

| source_summaries | budget_rows | uncached_cost_per_state | build_cost_per_state | cached_eval_cost_per_point | decode_cost_per_byte | mean_parent_reference_reuse |
|-----------------:|------------:|------------------------:|---------------------:|---------------------------:|---------------------:|----------------------------:|
| 2 | 8 | 16099.987 | 35951.933 | 28270.750 | 68.088 | 1.080 |

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
| 0 | 8 | 1000.000 | 615.596 | 615.596 | 1.000 | sampled_up_to_50_point_distribution | n/a | 0.004 | 9882238133.987 | yes |
| 1 | 7 | 381.443 | 234.815 | 234.815 | 2.622 | sampled_up_to_50_point_distribution | n/a | 0.026 | 1419975094.967 | yes |
| 2 | 6 | 162.337 | 99.934 | 99.934 | 6.160 | sampled_up_to_50_point_distribution | 1.228 | n/a | -223102.740 | no |
| 3 | 5 | 77.095 | 47.459 | 47.459 | 12.971 | sampled_up_to_50_point_distribution | 1.116 | n/a | -468516.960 | no |
| 4 | 4 | 35.626 | 21.931 | 21.931 | 28.070 | sampled_up_to_50_point_distribution | 1.161 | n/a | -1011891.570 | no |
| 5 | 3 | 16.463 | 10.134 | 10.134 | 60.744 | sampled_up_to_50_point_distribution | 1.089 | n/a | -2187137.863 | no |
| 6 | 2 | 7.607 | 4.683 | 4.683 | 131.452 | sampled_up_to_50_point_distribution | 1.064 | n/a | -4729776.520 | no |
| 7 | 1 | 3.515 | 2.164 | 2.164 | 284.467 | sampled_up_to_50_point_distribution | n/a | n/a | -10231499.947 | no |
| 8 | 0 | 1.624 | 1.000 | 1.000 | 615.596 | parametric_closed_form | n/a | n/a | -22136264.247 | no |

## Representation Detail

| boundary_depth | suffix_hops | representation | points | bytes | expected_hits | suffix_states | cap_limited_suffix_states | saved_per_hit | validation_time_ratio | per_hit_decode | splice_cost | one_time_cost | hits_to_break_even | net_value | pays |
|---------------:|------------:|----------------|-------:|------:|--------------:|--------------:|--------------------------:|--------------:|----------------------:|---------------:|------------:|--------------:|-------------------:|----------:|------|
| 0 | 8 | exact_sparse_histogram | 1 | 16.0 | 1000.000 | 615.596 | 615.596 | 9881729.924 | n/a | 1089.414 | 0.000 | 37041.507 | 0.004 | 9881692882.951 | yes |
| 0 | 8 | sampled_up_to_50_point_distribution | 1 | 8.0 | 1000.000 | 615.596 | 615.596 | 9882274.632 | n/a | 544.707 | 0.000 | 36497.720 | 0.004 | 9882238133.987 | yes |
| 0 | 8 | parametric_closed_form | 4 | 64.0 | 1000.000 | 615.596 | 615.596 | 9793649.431 | n/a | 4357.658 | 0.000 | 40342.231 | 0.004 | 9793609088.733 | yes |
| 1 | 7 | exact_sparse_histogram | 2 | 32.0 | 381.443 | 234.815 | 234.815 | 3721797.952 | n/a | 2178.829 | 0.000 | 96431.513 | 0.026 | 1419558457.533 | yes |
| 1 | 7 | sampled_up_to_50_point_distribution | 2 | 16.0 | 381.443 | 234.815 | 234.815 | 3722887.367 | n/a | 1089.414 | 0.000 | 95343.938 | 0.026 | 1419975094.967 | yes |
| 1 | 7 | parametric_closed_form | 4 | 64.0 | 381.443 | 234.815 | 234.815 | 3663077.623 | n/a | 4357.658 | 0.000 | 98642.662 | 0.027 | 1397157770.375 | yes |
| 2 | 6 | exact_sparse_histogram | 3 | 48.0 | 162.337 | 99.934 | 99.934 | 0.000 | 1.228 | 3268.243 | 0.000 | 224734.102 | n/a | -224734.102 | no |
| 2 | 6 | sampled_up_to_50_point_distribution | 3 | 24.0 | 162.337 | 99.934 | 99.934 | 0.000 | 1.228 | 1634.122 | 0.000 | 223102.740 | n/a | -223102.740 | no |
| 2 | 6 | parametric_closed_form | 4 | 64.0 | 162.337 | 99.934 | 99.934 | 0.000 | 1.228 | 4357.658 | 0.000 | 225855.676 | n/a | -225855.676 | no |
| 3 | 5 | exact_sparse_histogram | 4 | 64.0 | 77.095 | 47.459 | 47.459 | 0.000 | 1.116 | 4357.658 | 0.000 | 470692.109 | n/a | -470692.109 | no |
| 3 | 5 | sampled_up_to_50_point_distribution | 4 | 32.0 | 77.095 | 47.459 | 47.459 | 0.000 | 1.116 | 2178.829 | 0.000 | 468516.960 | n/a | -468516.960 | no |
| 3 | 5 | parametric_closed_form | 4 | 64.0 | 77.095 | 47.459 | 47.459 | 0.000 | 1.116 | 4357.658 | 0.000 | 470724.109 | n/a | -470724.109 | no |
| 4 | 4 | exact_sparse_histogram | 5 | 80.0 | 35.626 | 21.931 | 21.931 | 0.000 | 1.161 | 5447.072 | 0.000 | 1014610.506 | n/a | -1014610.506 | no |
| 4 | 4 | sampled_up_to_50_point_distribution | 5 | 40.0 | 35.626 | 21.931 | 21.931 | 0.000 | 1.161 | 2723.536 | 0.000 | 1011891.570 | n/a | -1011891.570 | no |
| 4 | 4 | parametric_closed_form | 4 | 64.0 | 35.626 | 21.931 | 21.931 | 0.000 | 1.161 | 4357.658 | 0.000 | 1013552.931 | n/a | -1013552.931 | no |
| 5 | 3 | exact_sparse_histogram | 6 | 96.0 | 16.463 | 10.134 | 10.134 | 0.000 | 1.089 | 6536.487 | 0.000 | 2190400.586 | n/a | -2190400.586 | no |
| 5 | 3 | sampled_up_to_50_point_distribution | 6 | 48.0 | 16.463 | 10.134 | 10.134 | 0.000 | 1.089 | 3268.243 | 0.000 | 2187137.863 | n/a | -2187137.863 | no |
| 5 | 3 | parametric_closed_form | 4 | 64.0 | 16.463 | 10.134 | 10.134 | 0.000 | 1.089 | 4357.658 | 0.000 | 2188253.437 | n/a | -2188253.437 | no |
| 6 | 2 | exact_sparse_histogram | 7 | 112.0 | 7.607 | 4.683 | 4.683 | 0.000 | 1.064 | 7625.901 | 0.000 | 4733583.030 | n/a | -4733583.030 | no |
| 6 | 2 | sampled_up_to_50_point_distribution | 7 | 56.0 | 7.607 | 4.683 | 4.683 | 0.000 | 1.064 | 3812.951 | 0.000 | 4729776.520 | n/a | -4729776.520 | no |
| 6 | 2 | parametric_closed_form | 4 | 64.0 | 7.607 | 4.683 | 4.683 | 0.000 | 1.064 | 4357.658 | 0.000 | 4730346.307 | n/a | -4730346.307 | no |
| 7 | 1 | exact_sparse_histogram | 8 | 128.0 | 3.515 | 2.164 | 2.164 | 0.000 | n/a | 8715.316 | 0.000 | 10235850.245 | n/a | -10235850.245 | no |
| 7 | 1 | sampled_up_to_50_point_distribution | 8 | 64.0 | 3.515 | 2.164 | 2.164 | 0.000 | n/a | 4357.658 | 0.000 | 10231499.947 | n/a | -10231499.947 | no |
| 7 | 1 | parametric_closed_form | 4 | 64.0 | 3.515 | 2.164 | 2.164 | 0.000 | n/a | 4357.658 | 0.000 | 10231523.947 | n/a | -10231523.947 | no |
| 8 | 0 | exact_sparse_histogram | 9 | 144.0 | 1.624 | 1.000 | 1.000 | 0.000 | n/a | 9804.730 | 0.000 | 22141680.120 | n/a | -22141680.120 | no |
| 8 | 0 | sampled_up_to_50_point_distribution | 9 | 72.0 | 1.624 | 1.000 | 1.000 | 0.000 | n/a | 4902.365 | 0.000 | 22136786.035 | n/a | -22136786.035 | no |
| 8 | 0 | parametric_closed_form | 4 | 64.0 | 1.624 | 1.000 | 1.000 | 0.000 | n/a | 4357.658 | 0.000 | 22136264.247 | n/a | -22136264.247 | no |

## Summary

- deepest recommended boundary depth: `1`
- mean best net value: `1251247226.567`
- note: point count is a representation cost input, not the break-even hit count.
