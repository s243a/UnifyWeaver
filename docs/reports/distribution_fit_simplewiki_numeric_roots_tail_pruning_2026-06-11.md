# SimpleWiki Numeric Root Tail-Pruning Calibration

Date: 2026-06-11

Branch: `codex/simplewiki-container-tail-pruning`

## Scope

This report follows up on the Articles-root tail-pruning calibration by testing three large non-Articles numeric roots from the broader `simplewiki_cats` LMDB artifact.

Important caveat: this artifact was produced by `examples/benchmark/simplewiki_post_ingest.py`. Its topology is valid, but its nodes are numeric page/linktarget ids rather than resolved category titles. The local resolved SimpleWiki dumps or `data/simplewiki/simplewiki_categories.db` are not present in this checkout, so these roots cannot be named as `Category:...` container categories without rebuilding the resolved export. This is therefore a topology-level proxy for individual large category roots, not a title-level container-category report.

The broader artifact is:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_cats/lmdb_resident
```

Scratch outputs were written under:

```text
/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration
```

## Export And Root Selection

A Scratch-only helper exported the numeric `category_parent` LMDB relation and ranked true roots by bounded descendant-subtree size:

```text
/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/export_lmdb_edges_and_roots.py
```

Export result:

```text
edges=297283
unique_children=91514
unique_parents=43974
true_roots=43178
wrote_edges=/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/simplewiki_cats_category_parent.tsv
```

Top roots by depth-10 subtree size:

| rank | root | subtree_size_depth10 | out_degree |
|------|------|----------------------|------------|
| 1 | 2 | 14680 | 13720 |
| 2 | 672 | 14661 | 14283 |
| 3 | 374 | 12959 | 12550 |
| 4 | 110 | 6404 | 6106 |
| 5 | 673 | 5286 | 5125 |
| 6 | 677 | 5127 | 4964 |
| 7 | 2922 | 4851 | 4642 |
| 8 | 127 | 4180 | 3915 |

Root `2` is the previously studied Articles-root fixture, so this report uses roots `672`, `374`, and `110` as the largest non-`2` numeric-root samples.

## Sampling

Each root was sampled with `--max-depth 4` from the exported numeric edge TSV.

| root | selected_nodes | sampled_edges |
|------|----------------|---------------|
| 672 | 14661 | 14974 |
| 374 | 12959 | 13242 |
| 110 | 6404 | 6405 |

Example command shape:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/simplewiki_cats_category_parent.tsv --root 672 --max-depth 4 --no-default-excludes --output /mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/simplewiki_root672_depth4.tsv --targets-output /mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/simplewiki_root672_depth4_targets.txt
```

## Tail-Pruning Runs

Each sampled root was run with:

```text
python3 scripts/distribution_fit_comparison.py --edge-file <sample.tsv> --graph-name <graph_name> --root <root> --targets-file <targets.txt> --depths 1,2,3,4,6,8,10,12 --tail-epsilon 0.001 --continuous-sample-points 100 --prune-thresholds 0.01,0.001,0.0001 --output-dir /mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration
```

Generated summary files:

```text
/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/distribution_fit_comparison_summary_simplewiki_root672_depth4_tail_pruning_20260611T221928Z.md
/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/distribution_fit_comparison_summary_simplewiki_root374_depth4_tail_pruning_20260611T221928Z.md
/mnt/c/Users/johnc/Scratch/simplewiki-container-tail-pruning-calibration/distribution_fit_comparison_summary_simplewiki_root110_depth4_tail_pruning_20260611T221928Z.md
```

All three runs completed with `errors = 0`.

## Realized Histogram Width

| root | targets | mean_bins | p95_bins | max_bins | mean_path_count |
|------|---------|-----------|----------|----------|-----------------|
| 672 | 14661 | 1.022 | 1.000 | 3 | about 1.022 at L_min=1 |
| 374 | 12959 | 1.022 | 1.000 | 3 | about 1.023 at L_min=1 |
| 110 | 6404 | 1.000 | 1.000 | 2 | about 1.000 at L_min=1 |

The broader numeric roots remain one-bin dominated. Roots `672` and `374` have slightly wider realized support than Articles, but the maximum support is still only three bins. Root `110` is even closer to a pure chain.

## Realized Tail-Pruned Support

| root | threshold | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | weighted_power_error |
|------|-----------|--------------------|----------------|-------------------|-------------------|----------------------|
| 672 | 0.001 | 1.022 | 1.022 | 0.000 | 0.000000 | 0.000000 |
| 374 | 0.001 | 1.022 | 1.022 | 0.000 | 0.000000 | 0.000000 |
| 110 | 0.001 | 1.000 | 1.000 | 0.000 | 0.000000 | 0.000000 |

At realized-state level, tail pruning still removes nothing. The distributions are already too narrow for suffix pruning to matter.

## Depth-Conditioned Prior Support

| root | mean_excess at depth 12 | mean_prior_bins | effective_bins at depth 12 | binomial_prior mean_l1 | gamma_prior mean_l1 |
|------|-------------------------|-----------------|----------------------------|------------------------|---------------------|
| 672 | 0.512889 | 12.500 | 5 | 0.017599 | 0.176300 |
| 374 | 0.521976 | 12.500 | 5 | 0.013130 | 0.180229 |
| 110 | 0.007494 | 6.750 | 2 | 0.000000 | 0.259432 |

Roots `672` and `374` have a larger excess-parent prior than Articles, but the effective support remains small. Root `110` is almost a chain by this statistic.

## Prior Tail-Pruned Support

At pruning threshold `1e-3`:

| root | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | weighted_power_error |
|------|--------------------|----------------|-------------------|-------------------|----------------------|
| 672 | 12.500 | 3.750 | 8.750 | 0.000280 | 0.000017 |
| 374 | 12.500 | 3.625 | 8.875 | 0.000357 | 0.000027 |
| 110 | 6.750 | 1.875 | 4.875 | 0.000086 | 0.000020 |

Tail pruning is again useful for prior/planning distributions. It removes most algebraic suffix bins with very small dropped mass and weighted-power error.

## Interpretation

The main finding from Articles generalizes to the largest non-`2` numeric SimpleWiki roots in the available LMDB artifact: realized parent-only histograms stay extremely narrow. Even roots with thousands of immediate children do not create wide realized hop-count histograms, because most nodes have effectively one admissible parent-path length to the chosen root.

This suggests that, for SimpleWiki parent-only category metrics, exact sparse histograms can be carried aggressively across large rooted subtrees. Tail pruning is not yet a realized-state optimization because there is little realized tail to prune.

The depth-conditioned prior is still valuable. It shows how algebraic support grows with horizon and how light-tail pruning can keep forecast distributions compact. That matters for planning, admission decisions, and future enwiki runs where realized histograms may become wider.

The binomial prior remains a better fit than shifted Gamma for these SimpleWiki roots. Gamma is still a candidate for broader enwiki-style branching, not the default for SimpleWiki near-chain regimes.

## Policy Implication

For SimpleWiki numeric-root parent-only metrics:

```text
if realized_support_bins <= 3:
    store exact sparse histogram
elif prior_tail_pruned_bins(depth, threshold=1e-3) <= 4 or 5:
    keep exact prefix plus tail-pruning metadata for planning
else:
    compare exact histogram, fitted binomial, shifted Gamma, and sampled-continuous approximation by functional error and storage cost
```

The current SimpleWiki evidence has not yet found a rooted subtree where realized tail pruning pays off. The next meaningful stress test is therefore enwiki, especially a filtered `Category:Main_topic_classifications` sample or another named root from a resolved enwiki export.

## Tooling Note

The earlier attempt to use `scripts/export_distribution_cache_edges.py` was not the right path for this artifact. That script expects a resolved SQLite DB with category titles. The available broad SimpleWiki artifact is LMDB numeric-keyed and was produced by `examples/benchmark/simplewiki_post_ingest.py`, so the correct path here was to export the LMDB `category_parent` relation directly and treat roots as numeric topology ids.
