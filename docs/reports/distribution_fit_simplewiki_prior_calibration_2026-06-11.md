# SimpleWiki Distribution Fit Prior Calibration

Date: 2026-06-11

Branch: `codex/simplewiki-prior-calibration`

## Scope

This report calibrates the parent-path distribution approximation harness on the
local SimpleWiki Articles numeric artifact:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv
```

The artifact uses numeric category ids. The prior SimpleWiki reports identify
category id `2` as the Articles root, so all commands below use `--root 2`.

Generated TSVs, target lists, JSONL, and generated markdown summaries were
written under Scratch and are not committed:

```text
/mnt/c/Users/johnc/Scratch/distribution-prior-calibration
```

## Sampling

Depth 3 command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 3 --output /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth3.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth3_targets.txt
```

Depth 3 result:

```text
selected_nodes=14680
sampled_edges=14887
```

Depth 4 command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 4 --output /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth4.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth4_targets.txt
```

Depth 4 result:

```text
selected_nodes=14680
sampled_edges=14887
```

The depth-4 sample is identical to depth 3 for this filtered Articles artifact,
so the calibration run uses the depth-3 graph.

## Fit/Prior Comparison

Command:

```text
python3 scripts/distribution_fit_comparison.py --edge-file /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth3.tsv --graph-name simplewiki_articles_root2_depth3_prior_calibration --root 2 --targets-file /mnt/c/Users/johnc/Scratch/distribution-prior-calibration/simplewiki_articles_root2_depth3_targets.txt --depths 1,2,3,4,6,8,10,12 --tail-epsilon 0.001 --output-dir /mnt/c/Users/johnc/Scratch/distribution-prior-calibration
```

Generated outputs:

```text
/mnt/c/Users/johnc/Scratch/distribution-prior-calibration/distribution_fit_comparison_simplewiki_articles_root2_depth3_prior_calibration_20260611T041620Z.jsonl
/mnt/c/Users/johnc/Scratch/distribution-prior-calibration/distribution_fit_comparison_summary_simplewiki_articles_root2_depth3_prior_calibration_20260611T041620Z.md
```

### Realized Histogram Fits

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|
| binomial_fit | 14680 | 0.000590 | 0.000000 | 1.000000 | 0.000148 | 1890.9 | 4835.7 |
| shifted_gamma_fit | 14680 | 0.010107 | 0.000000 | 1.073430 | 0.005054 | 2288.8 | 4835.7 |

### Realized Support By Root Distance

| L_min | targets | mean_bins | p95_bins | max_bins | mean_effective_bins | mean_path_count | mean_parent_degree |
|-------|---------|-----------|----------|----------|---------------------|-----------------|--------------------|
| 0 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 0.000000 |
| 1 | 13720 | 1.015 | 1.000 | 3 | 1.015 | 1.015 | 1.014359 |
| 2 | 923 | 1.001 | 1.000 | 2 | 1.001 | 1.013 | 1.011918 |
| 3 | 36 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000000 |

### Depth-Conditioned Prior Distributions

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|
| binomial_prior | 8 | 0.020596 | 0.035789 | 0.035789 | 0.005698 | 5844.6 | 24371.5 |
| shifted_gamma_prior | 8 | 0.143635 | 0.227540 | 0.227540 | 0.066402 | 69489.2 | 24371.5 |

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

## Interpretation

The realized histograms are extremely narrow. Across `14,680` targets, the mean
support is nearly one bin, the 95th percentile is one bin, and the maximum is
only three bins. This supports carrying exact sparse histograms for this
SimpleWiki parent-only Articles subtree: the representation cost is dominated by
the number of cached nodes rather than by per-node histogram width.

The depth-conditioned prior tells the same story. With `epsilon_tail=0.001`,
the effective prior support stays at three bins through depth `4` and only rises
to four bins through depth `12`. The full algebraic support width grows with the
horizon, but most prior mass stays in the first few excess-parent bins.

For this near-chain SimpleWiki regime, binomial is the better approximation
family. It has much lower realized-fit and prior-fit error than the
shifted-Gamma approximation, and it is cheaper to build in the prior comparison.
Gamma remains useful as a candidate for wider enwiki-style branching, not as
the default for this SimpleWiki sample.

The report therefore suggests a cheap first prior: use scalar support bounds to
set the candidate binomial support, and use the measured size-biased parent
branching signal as the Bernoulli excess-parent proxy. In this run, that means
`n ~= L_max - L_min` and `p ~= E[P^2]/E[P] - 1 = 0.028750` globally, with the
understanding that the root-distance table already shows why a future
layer-conditioned `p_i` may be better.

The binomial result should be read as a sparse-event model, not as a symmetric
Gaussian claim. In this sample, excess-parent probability is small, so the
binomial prior is right-skewed and compact. A normal approximation would be a
later large-depth approximation after checking tail error and finite support.

The root-distance table shows the depth-varying signal that motivates a future
non-stationary prior: mean parent degree declines from `1.014359` at `L_min=1`
to `1.011918` at `L_min=2` and `1.000000` at `L_min=3`. The current stationary
prior is conservative enough here, but a future planner could use
layer-conditioned priors `P(Y_i | L_min=i)` once deeper samples show the
difference matters.

## Policy Implication

The `Binomial(10, 0.1)` example is a useful warning case: it is clearly skewed
even though the tail falls quickly. For this reason, symmetric approximations
should wait for a skewness/tail-error gate. Binomial fits can still be useful
before then as compression: they may reduce storage from many bins to `(n, p,
error_metadata)`, but that is different from avoiding the initial exact or
sampled computation needed to validate the fit.

A reasonable first SimpleWiki policy is:

```text
if realized_support_width <= 3:
    store exact sparse histogram
elif prior_effective_support_bins(depth, epsilon_tail=0.001) <= 4:
    prefer exact sparse histogram when reuse is non-trivial
else:
    compare exact histogram, scalar min/max bounds, and fitted state by cost
```

This is intentionally SimpleWiki-specific. Enwiki should be recalibrated rather
than inheriting these thresholds.
