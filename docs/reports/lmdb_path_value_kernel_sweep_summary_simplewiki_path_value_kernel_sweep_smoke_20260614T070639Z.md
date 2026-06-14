# LMDB Path-Value Kernel Sweep

Graph: `simplewiki_path_value_kernel_sweep_smoke`

Root: `2`

LMDB: `/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_cats/lmdb_resident`

Mode: `all`

Parent filter: `all`

Selection source: `root-cone`

Budgets: `4`

Samples per target: `50`

This report reruns one fixed target-selection recipe for several path-value kernels. Path-count estimates remain the branch-product estimates from the boundary probe; value-sum estimates apply the selected kernel to root-reaching paths or boundary suffix histograms.

## Variants

| variant | kernel | b_p | b_p_source | power | targets | boundary_nodes | elapsed_ms |
|---------|--------|----:|------------|------:|--------:|---------------:|-----------:|
| count | count | n/a | n/a | n/a | 2 | 64 | 35.346 |
| bp_decay_auto | bp-decay | 1.075000 | root_cone_eligible_parent_e_p2_over_e_p | n/a | 2 | 64 | 27.576 |
| bp_decay_explicit_2p0 | bp-decay | 2.000000 | user | n/a | 2 | 64 | 22.866 |
| weighted_power_1p0 | weighted-power | n/a | n/a | 1.000 | 2 | 64 | 23.676 |
| weighted_power_2p0 | weighted-power | n/a | n/a | 2.000 | 2 | 64 | 24.646 |

## Estimate Comparison

| variant | mode | budget | targets | observed_root_paths | observed_boundary_hits | mean_est_root_paths | mean_est_root_value_sum | mean_kernel_mean_length | mean_spliced_paths | mean_spliced_value_sum | mean_boundary_hit_fraction | elapsed_ms |
|---------|------|-------:|--------:|--------------------:|-----------------------:|--------------------:|------------------------:|------------------------:|-------------------:|-----------------------:|---------------------------:|-----------:|
| bp_decay_auto | exact | 4 | 2 | 1 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.399 |
| bp_decay_auto | root-sample | 4 | 2 | 21 | 23 | 1.680 | 1.490103 | 1.722 | n/a | n/a | 0.544130 | 8.637 |
| bp_decay_auto | sample | 4 | 2 | 11 | 21 | 0.440 | 0.409302 | n/a | 1.280 | 1.136182 | 0.210000 | 8.500 |
| bp_decay_explicit_2p0 | exact | 4 | 2 | 1 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.220 |
| bp_decay_explicit_2p0 | root-sample | 4 | 2 | 21 | 23 | 1.680 | 0.560000 | 1.650 | n/a | n/a | 0.544130 | 8.765 |
| bp_decay_explicit_2p0 | sample | 4 | 2 | 11 | 21 | 0.440 | 0.220000 | n/a | 1.280 | 0.430000 | 0.210000 | 8.620 |
| count | exact | 4 | 2 | 1 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | 1.104 |
| count | root-sample | 4 | 2 | 21 | 23 | 1.680 | 1.680000 | 1.731 | n/a | n/a | 0.544130 | 8.551 |
| count | sample | 4 | 2 | 11 | 21 | 0.440 | 0.440000 | n/a | 1.280 | 1.280000 | 0.210000 | 8.553 |
| weighted_power_1p0 | exact | 4 | 2 | 1 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.231 |
| weighted_power_1p0 | root-sample | 4 | 2 | 21 | 23 | 1.680 | 0.653333 | 1.682 | n/a | n/a | 0.544130 | 9.185 |
| weighted_power_1p0 | sample | 4 | 2 | 11 | 21 | 0.440 | 0.220000 | n/a | 1.280 | 0.500000 | 0.210000 | 8.689 |
| weighted_power_2p0 | exact | 4 | 2 | 1 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.239 |
| weighted_power_2p0 | root-sample | 4 | 2 | 21 | 23 | 1.680 | 0.264444 | 1.638 | n/a | n/a | 0.544130 | 9.614 |
| weighted_power_2p0 | sample | 4 | 2 | 11 | 21 | 0.440 | 0.110000 | n/a | 1.280 | 0.203333 | 0.210000 | 8.940 |

## Notes

- `count` is the control: its value sum equals the path-count quantity.
- `bp-decay:auto` estimates `b_p = E[p^2] / E[p]` from the selected root-cone scope when available.
- `bp-decay:explicit` uses the value passed by `--explicit-branching-factor`.
- `weighted-power:n` computes `(L + 1)^(-n)` and therefore needs prefix-length-aware suffix evaluation.
