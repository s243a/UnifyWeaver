# SimpleWiki Tail-Pruning Calibration

Date: 2026-06-11

Branch: `codex/simplewiki-tail-pruning-report`

## Scope

This report runs the tail-pruning version of `scripts/distribution_fit_comparison.py` on the local SimpleWiki Articles numeric artifact:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv
```

The artifact is numeric-id based. Prior SimpleWiki reports identify root id `2` as the Articles root, so all commands below use `--root 2`.

Generated sampled TSVs, target lists, JSONL, and generated markdown summaries were written under Scratch and are not committed:

```text
/mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration
```

## Sampling

Command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 3 --output /mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/simplewiki_articles_root2_depth3.tsv --targets-output /mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/simplewiki_articles_root2_depth3_targets.txt
```

Result:

```text
root=2
selected_nodes=14680
sampled_edges=14887
max_depth=3
```

This matches the earlier SimpleWiki prior-calibration sample size, so the new results are comparable to `distribution_fit_simplewiki_prior_calibration_2026-06-11.md`.

## Fit And Tail-Pruning Comparison

Command:

```text
python3 scripts/distribution_fit_comparison.py --edge-file /mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/simplewiki_articles_root2_depth3.tsv --graph-name simplewiki_articles_root2_depth3_tail_pruning --root 2 --targets-file /mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/simplewiki_articles_root2_depth3_targets.txt --depths 1,2,3,4,6,8,10,12 --tail-epsilon 0.001 --continuous-sample-points 100 --prune-thresholds 0.01,0.001,0.0001 --output-dir /mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration
```

Generated outputs:

```text
/mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/distribution_fit_comparison_simplewiki_articles_root2_depth3_tail_pruning_20260611T171819Z.jsonl
/mnt/c/Users/johnc/Scratch/simplewiki-tail-pruning-calibration/distribution_fit_comparison_summary_simplewiki_articles_root2_depth3_tail_pruning_20260611T171819Z.md
```

### Realized Histogram Fits

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|
| binomial_fit | 14680 | 0.000590 | 0.000000 | 1.000000 | 0.000148 | 1800.0 | 6098.2 | 3.609 |
| shifted_gamma_fit | 14680 | 0.010107 | 0.000000 | 1.073430 | 0.005054 | 2493.4 | 6098.2 | 3.609 |

### Realized Support By Root Distance

| L_min | targets | mean_bins | p95_bins | max_bins | mean_effective_bins | mean_path_count | mean_parent_degree |
|-------|---------|-----------|----------|----------|---------------------|-----------------|--------------------|
| 0 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 0.000000 |
| 1 | 13720 | 1.015 | 1.000 | 3 | 1.015 | 1.015 | 1.014359 |
| 2 | 923 | 1.001 | 1.000 | 2 | 1.001 | 1.013 | 1.011918 |
| 3 | 36 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000000 |

### Realized Tail-Pruned Support

| tail_threshold | distributions | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | mean_weighted_power_error |
|----------------|---------------|--------------------|----------------|-------------------|-------------------|---------------------------|
| 0.0001 | 14680 | 1.014 | 1.014 | 0.000 | 0.000000 | 0.000000 |
| 0.001 | 14680 | 1.014 | 1.014 | 0.000 | 0.000000 | 0.000000 |
| 0.01 | 14680 | 1.014 | 1.014 | 0.000 | 0.000000 | 0.000000 |

### Depth-Conditioned Prior Distributions

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|
| binomial_prior | 8 | 0.020596 | 0.035789 | 0.035789 | 0.005698 | 6062.5 | 28300.0 | 1.250 |
| shifted_gamma_prior | 8 | 0.143635 | 0.227540 | 0.227540 | 0.066402 | 267062.9 | 28300.0 | 1.250 |

### Prior Support By Depth

| depth | rows | mean_bins | mean_effective_bins | max_effective_bins | mean_excess |
|-------|------|-----------|---------------------|--------------------|-------------|
| 1 | 2 | 3.000 | 3.000 | 3 | 0.028750 |
| 2 | 2 | 5.000 | 3.000 | 3 | 0.057500 |
| 3 | 2 | 7.000 | 3.000 | 3 | 0.086250 |
| 4 | 2 | 9.000 | 3.000 | 3 | 0.115000 |
| 6 | 2 | 13.000 | 4.000 | 4 | 0.172499 |
| 8 | 2 | 17.000 | 4.000 | 4 | 0.229999 |
| 10 | 2 | 21.000 | 4.000 | 4 | 0.287499 |
| 12 | 2 | 25.000 | 4.000 | 4 | 0.344999 |

### Prior Tail-Pruned Support

| tail_threshold | distributions | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | mean_weighted_power_error |
|----------------|---------------|--------------------|----------------|-------------------|-------------------|---------------------------|
| 0.0001 | 8 | 12.500 | 4.125 | 8.375 | 0.000035 | 0.000002 |
| 0.001 | 8 | 12.500 | 3.500 | 9.000 | 0.000273 | 0.000015 |
| 0.01 | 8 | 12.500 | 2.500 | 10.000 | 0.004282 | 0.000419 |

## Interpretation

The realized SimpleWiki Articles histograms are still effectively exact-sparse. The mean realized support is `1.014` bins, the 95th percentile is one bin, and the maximum support is three bins. Tail pruning therefore removes no realized bins at thresholds `1e-2`, `1e-3`, or `1e-4`: there is almost no tail to remove.

The depth-conditioned prior tells a different but compatible story. Algebraic support grows with horizon, reaching a mean of `12.5` bins across the tested prior rows. Under the same prior, suffix pruning is very effective: with threshold `1e-3`, the mean kept support is `3.5` bins, mean dropped support is `9.0` bins, mean dropped mass is `0.000273`, and mean weighted-power error is `0.000015`.

This means tail pruning is more useful as a prior/planning compression mechanism than as a realized SimpleWiki histogram optimization. For the measured parent-only Articles subtree, exact sparse histograms remain the right realized representation because most nodes already have one bin. For deeper forecast distributions, pruning the light suffix can keep the prior representation small while preserving the mass and weighted-power functional to tight thresholds.

The comparison also preserves the previous family conclusion: binomial remains a better SimpleWiki prior than shifted Gamma in this near-chain regime. The Gamma fit is still useful as a candidate for wider enwiki-style branching, but it is not the default for this SimpleWiki subtree.

## Policy Implication

For SimpleWiki Articles parent-only metrics, the first policy should be:

```text
if realized_support_bins <= 3:
    store exact sparse histogram
elif prior_tail_pruned_bins(depth, threshold=1e-3) <= 4:
    keep exact prefix plus tail-pruning metadata for planning
else:
    compare exact histogram, parametric fit, and sampled-continuous approximation by functional error and storage cost
```

The important decision is not to prune by position, such as blindly dropping half the bins. The admissible rule is to drop the largest suffix whose mass and selected-functional contribution are below threshold. In this run, the realized histograms are too narrow for pruning to matter, but the depth prior shows that light-tail pruning can aggressively reduce forecast support.

## Next Work

The next useful calibration is the same run on either a deeper or broader SimpleWiki root and then an enwiki sample. The key question is where realized histograms stop being one-bin dominated. That is the point where tail pruning can become a real storage optimization rather than only a prior compression tool.
