# SimpleWiki Parent Support Bounds Benchmark

Date: 2026-06-10

## Purpose

This run validates the scalar support-bounds idea from
`docs/design/DISTRIBUTIONAL_FIT_POLICY.md` and
`docs/design/DISTRIBUTION_CACHE_BENCHMARK_PLAN.md` against the same shallow
SimpleWiki Articles sample used by the distribution-cache depth grid.

The question is whether parent-only path support bounds are exact enough to use
as a cheap planner state before materialising full path-length distributions:

```text
L_min(v) = shortest parent-only path length from v to root
L_max(v) = longest parent-only path length from v to root
support_width(v) = L_max(v) - L_min(v)
```

For min-only and max-only functionals, those scalar recurrences are the result.
For bounded aggregate functionals, they are pruning and planning signals:

```text
if L_min(v) > remaining_budget: suffix contributes zero
if L_max(v) <= remaining_budget: suffix is fully inside the budget
otherwise: exact histogram, cached boundary, or fitted state is still needed
```

## Input

The run used the prior sampled depth-3 SimpleWiki Articles artifact:

```text
/mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3.tsv
```

The root was numeric category id `2`, matching the previous SimpleWiki Articles
artifact sidecar. The sampled graph contains `14680` reachable target nodes and
`14887` parent edges.

## Command

```sh
python3 scripts/distribution_cache_support_bounds.py \
  --edge-file /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3.tsv \
  --graph-name simplewiki_articles_root2_depth3 \
  --root 2 \
  --budgets 2,4,6 \
  --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-support-bounds \
  --fail-on-error
```

Raw generated outputs were written to:

```text
/mnt/c/Users/johnc/Scratch/distribution-cache-support-bounds/distribution_support_bounds_simplewiki_articles_root2_depth3_20260610T231212Z.jsonl
/mnt/c/Users/johnc/Scratch/distribution-cache-support-bounds/distribution_support_bounds_summary_simplewiki_articles_root2_depth3_20260610T231212Z.md
```

## Results

### Target Bounds

| targets | validated | unreachable | exact_errors | bounds_failures | mean_width | p95_width | max_width | mean_path_count |
|---------|-----------|-------------|--------------|-----------------|------------|-----------|-----------|-----------------|
| 14680 | 14680 | 0 | 0 | 0 | 0.014 | 0.000 | 2 | 1.014 |

### Budget Signals

| B_search | targets | zero_by_min | fully_covered_by_max | partial_by_bounds | narrow_support | wide_support |
|----------|---------|-------------|----------------------|-------------------|----------------|--------------|
| 2 | 14680 | 36 | 14633 | 11 | 14680 | 0 |
| 4 | 14680 | 0 | 14680 | 0 | 14680 | 0 |
| 6 | 14680 | 0 | 14680 | 0 | 14680 | 0 |

### Root-Distance Buckets

| L_min | targets | mean_width | max_width | mean_path_count |
|-------|---------|------------|-----------|-----------------|
| 0 | 1 | 0.000 | 0 | 1.000 |
| 1 | 13720 | 0.015 | 2 | 1.015 |
| 2 | 923 | 0.001 | 1 | 1.013 |
| 3 | 36 | 0.000 | 0 | 1.000 |

## Interpretation

For this shallow SimpleWiki Articles sample, scalar bounds are exact against the
full parent-path histograms and are strong enough to classify almost every
budgeted query without looking at the full distribution. At `B_search=2`, bounds
prove `36` targets contribute zero and `14633` targets are fully covered, leaving
only `11` partial cases. At `B_search=4` and `B_search=6`, every sampled target is
fully covered by `L_max`.

This confirms that min/max support bounds are a good early planner layer for the
SimpleWiki parent-only subtree. The next useful stress case is a deeper target
sample or an enwiki sample where support widths and shortcut paths should be
larger. Full histograms and fitted distribution states should be reserved for
the partial/wide-support region rather than materialised uniformly.
