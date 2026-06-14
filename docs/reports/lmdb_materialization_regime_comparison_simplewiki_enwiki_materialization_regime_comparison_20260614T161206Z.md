# LMDB Materialization Regime Comparison

Graph: `simplewiki_enwiki_materialization_regime_comparison`

Point cap: `50`

## Branching Profiles

| dataset | graph | root | retained_nodes | max_observed_depth | truncated | root_conditioned_b | raw_b | mean_root_p | mean_raw_p | max_root_p | max_raw_p |
|---------|-------|-----:|---------------:|-------------------:|-----------|-------------------:|------:|------------:|-----------:|-----------:|----------:|
| enwiki_mtc | `enwiki_mtc_root_conditioned_branching_smoke` | 7345184 | 5000 | 3 | yes | 2.164 | 7.049 | 1.698 | 4.152 | 9 | 202 |
| simplewiki | `simplewiki_articles_root_conditioned_regime_depth4` | 2 | 14680 | 3 | no | 1.029 | 4.966 | 1.014 | 4.155 | 3 | 28 |

## Depth Buckets

| dataset | child_depth | nodes | root_conditioned_b | raw_b | mean_root_p | mean_raw_p | mean_outside_parent_fraction |
|---------|------------:|------:|-------------------:|------:|------------:|-----------:|-----------------------------:|
| enwiki_mtc | 0 | 1 | n/a | 3.000 | 0.000 | 3.000 | 1.000 |
| enwiki_mtc | 1 | 35 | 2.622 | 4.676 | 2.114 | 4.229 | 0.489 |
| enwiki_mtc | 2 | 1007 | 2.350 | 4.819 | 1.860 | 3.993 | 0.495 |
| enwiki_mtc | 3 | 3957 | 2.106 | 7.611 | 1.653 | 4.193 | 0.546 |
| simplewiki | 0 | 1 | n/a | n/a | 0.000 | 0.000 | 0.000 |
| simplewiki | 1 | 13720 | 1.029 | 4.821 | 1.014 | 4.055 | 0.704 |
| simplewiki | 2 | 923 | 1.024 | 6.531 | 1.012 | 5.670 | 0.769 |
| simplewiki | 3 | 36 | 1.000 | 4.190 | 1.000 | 3.500 | 0.632 |

## Boundary Histograms

| dataset | graph | entries | exact | parametric | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_paths | max_paths | mean_states | max_states | mean_payload_bytes | regime |
|---------|-------|--------:|------:|-----------:|--------------:|-------------:|--------------------:|-----------:|----------:|------------:|-----------:|-------------------:|--------|
| enwiki_mtc | `enwiki_mtc_boundary_cache_estimator_validation_b3_recurrence` | 81 | 81 | 0 | 5.963 | 6 | 100.000 | 77.938 | 381 | 506.296 | 1316 | 96.593 | `exact_sparse_high_mass` |
| simplewiki | `simplewiki_materialization_validation_b1_child_depth` | 33 | 33 | 0 | 1.030 | 2 | 100.000 | 1.030 | 2 | 4.303 | 14 | 40.364 | `exact_sparse_low_mass` |

## Cache Search Shape

| dataset | targets | budgets | comparison_rows | mean_cache_hits | positive_hit_rows | mean_full_paths | mean_time_ratio | measured_cache_faster | mean_l1 | mean_cdf |
|---------|--------:|---------|----------------:|----------------:|------------------:|----------------:|----------------:|-----------------------|--------:|---------:|
| enwiki_mtc | 2 | `8` | 2 | 13.500 | 1 | 1.000 | 1.117 | no | 0.000000 | 0.000000 |
| simplewiki | 3 | `4,6,8` | 9 | 1.000 | 9 | 1.000 | 1.986 | no | 0.000000 | 0.000000 |

## Interpretation

- The point cap is a representation upper bound.  A dataset whose measured effective support is far below the cap can stay in exact histogram form without paying for all points.
- `enwiki_mtc` has compact support but larger path mass: mean effective bins 5.963, max effective bins 6, mean path mass 77.938.  This is the regime where recurrence materialization can replace many enumerated paths with a small stored state.
- `simplewiki` is exact-sparse and low-mass in this probe: mean effective bins 1.030, max effective bins 2, mean path mass 1.030.
- `enwiki_mtc` is a constrained profile, so use it as smoke evidence rather than a full-root-cone characterization.
- `enwiki_mtc` has lower root-conditioned branching than raw parent branching (2.164 vs 7.049).  Ancestor-cone planning should prefer the root-conditioned prior when estimating materialization depth.
- `simplewiki` has lower root-conditioned branching than raw parent branching (1.029 vs 4.966).  Ancestor-cone planning should prefer the root-conditioned prior when estimating materialization depth.

## Source Files

- profile `enwiki_mtc`: `docs/reports/lmdb_root_conditioned_branching_profile_enwiki_mtc_root_conditioned_branching_smoke_20260614T001041Z.jsonl`
- profile `simplewiki`: `docs/reports/lmdb_root_conditioned_branching_profile_simplewiki_articles_root_conditioned_regime_depth4_20260614T161057Z.jsonl`
- cache `enwiki_mtc`: `docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b3_recurrence_20260613T164209Z.jsonl`
- cache `simplewiki`: `docs/reports/lmdb_parent_boundary_cache_benchmark_simplewiki_materialization_validation_b1_child_depth_20260614T154446Z.jsonl`
