# Enwiki Main Topic Parent Branching Diagnostic

Date: 2026-06-12

Branch: codex/full-parent-branching-diagnostic

## Scope

This diagnostic follows up on the numeric-root enwiki tail-pruning pilot. The previous pilot measured parent degree from a sampled TSV, so parent degree meant parents retained inside the sampled subtree. That undercounts Wikipedia category branching when a target has parents outside the capped subtree.

The important correction is that the earlier numeric root 97688913 was not the named Category:Main_topic_classifications root. The title-resolved enwiki_cats_correct artifact identifies the relevant root as:

    artifact=/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct
    root_title=Main_topic_classifications
    root_id=7345184
    root_child_count=35
    edge_count=9927768

Definitions:

    full_parent_degree          = all LMDB category_parent entries for the target
    root_reaching_parent_degree = direct parents that can reach the selected root
                                  by parent-only traversal within the search cap

The first statistic describes raw enwiki category branching. The second is the admissible branching for a root-anchored parent-path histogram.

## Main Topic Classifications Sample

Command:

    python3 scripts/lmdb_parent_branching_diagnostic.py --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident --root 7345184 --graph-name enwiki_mtc_full_parent_branching --child-depths 1,2,3,4 --children-per-node 256 --frontier-limit 5000 --targets-per-depth 200 --max-parent-depth 48 --seed enwiki-mtc-parent-branching-v1 --output-dir /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic

Generated outputs:

    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_branching_diagnostic_enwiki_mtc_full_parent_branching_20260612T003149Z.jsonl
    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_branching_diagnostic_summary_enwiki_mtc_full_parent_branching_20260612T003149Z.md

Selection:

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 35 |
| 2 | 1023 |
| 3 | 5000 |
| 4 | 5000 |

Target summary:

| targets | root_reachable | truncated | cycle_skipped |
|---------|----------------|-----------|---------------|
| 633 | 633 | 632 | 632 |

Most sampled targets hit the max_parent_depth=48 cap or a skipped cycle while searching parent paths. Buckets at L_max = 48 are therefore censored lower bounds, not exact maximum root distances.

## MTC Parent Degree By L_max

| L_max | targets | mean_full_p | p95_full_p | max_full_p | b_full | mean_excess_full | mean_root_p | p95_root_p | max_root_p | b_root | mean_excess_root |
|-------|---------|-------------|------------|------------|--------|------------------|-------------|------------|------------|--------|------------------|
| 1 | 5 | 2.400 | 4.000 | 4 | 2.833333 | 1.833333 | 1.000 | 1.000 | 1 | 1.000000 | 0.000000 |
| 2 | 6 | 2.833 | 4.000 | 4 | 3.000000 | 2.000000 | 1.500 | 2.000 | 2 | 1.666667 | 0.666667 |
| 3 | 4 | 3.250 | 5.000 | 5 | 3.769231 | 2.769231 | 1.500 | 2.000 | 2 | 1.666667 | 0.666667 |
| 4 | 4 | 3.500 | 4.000 | 4 | 3.714286 | 2.714286 | 1.750 | 3.000 | 3 | 2.142857 | 1.142857 |
| 6 | 1 | 5.000 | 5.000 | 5 | 5.000000 | 4.000000 | 3.000 | 3.000 | 3 | 3.000000 | 2.000000 |
| 7 | 2 | 4.000 | 4.000 | 4 | 4.000000 | 3.000000 | 1.500 | 2.000 | 2 | 1.666667 | 0.666667 |
| 8 | 1 | 5.000 | 5.000 | 5 | 5.000000 | 4.000000 | 4.000 | 4.000 | 4 | 4.000000 | 3.000000 |
| 9 | 5 | 4.400 | 5.000 | 5 | 4.454545 | 3.454545 | 2.000 | 3.000 | 3 | 2.200000 | 1.200000 |
| 19 | 5 | 4.600 | 6.000 | 6 | 4.826087 | 3.826087 | 2.200 | 3.000 | 3 | 2.272727 | 1.272727 |
| 20 | 10 | 4.100 | 6.000 | 6 | 4.414634 | 3.414634 | 1.800 | 3.000 | 3 | 2.000000 | 1.000000 |
| 21 | 1 | 7.000 | 7.000 | 7 | 7.000000 | 6.000000 | 4.000 | 4.000 | 4 | 4.000000 | 3.000000 |
| 22 | 20 | 3.850 | 7.000 | 7 | 4.272727 | 3.272727 | 2.150 | 3.000 | 3 | 2.348837 | 1.348837 |
| 30 | 7 | 2.857 | 4.000 | 4 | 3.000000 | 2.000000 | 2.143 | 3.000 | 3 | 2.200000 | 1.200000 |
| 34 | 50 | 4.240 | 7.000 | 8 | 4.764151 | 3.764151 | 2.560 | 5.000 | 6 | 2.968750 | 1.968750 |
| 41 | 19 | 3.316 | 6.000 | 9 | 4.333333 | 3.333333 | 2.421 | 4.000 | 5 | 2.695652 | 1.695652 |
| 42 | 1 | 2.000 | 2.000 | 2 | 2.000000 | 1.000000 | 2.000 | 2.000 | 2 | 2.000000 | 1.000000 |
| 48 | 492 | 4.142 | 8.000 | 26 | 5.290481 | 4.290481 | 3.299 | 6.000 | 24 | 4.255699 | 3.255699 |

This is the more relevant enwiki result. The MTC sample reaches substantial frontiers at depths 2-4, and both full and root-reaching parent branching are well above the earlier TSV-local estimate. In the censored L_max = 48 bucket, mean full parent degree is 4.142, p95 full degree is 8, and the largest sampled full parent degree is 26. The root-reaching signal remains large: mean root-reaching parent degree is 3.299, p95 is 6, and the largest sampled root-reaching parent degree is 24.

## Numeric-Root Contrast

The older numeric root 97688913 came from the separate enwiki_cats/lmdb_resident artifact metadata, not from the MTC title-resolved artifact. It still usefully demonstrated the difference between full parent degree and sampled-TSV degree, but it should not be treated as representative of the MTC subtree.

Numeric-root parent degree by L_max:

| L_max | targets | mean_full_p | p95_full_p | max_full_p | b_full | mean_excess_full | mean_root_p | p95_root_p | max_root_p | b_root | mean_excess_root |
|-------|---------|-------------|------------|------------|--------|------------------|-------------|------------|------------|--------|------------------|
| 1 | 145 | 5.469 | 9.000 | 10 | 6.157629 | 5.157629 | 1.000 | 1.000 | 1 | 1.000000 | 0.000000 |
| 2 | 18 | 6.556 | 10.000 | 12 | 7.254237 | 6.254237 | 1.556 | 2.000 | 2 | 1.714286 | 0.714286 |

This confirms that enwiki categories in the numeric-root sample have much more raw parent branching than the previous TSV-local estimate showed. The earlier mean near 1.01 was not a credible estimate of full category parent degree; it was mostly a measure of how many sampled-subtree parent edges survived the capped export.

## Interpretation

Both parent-degree signals are needed:

    full_parent_degree          -> graph complexity / raw Wikipedia branching
    root_reaching_parent_degree -> admissible branching for this root and filter

The distribution prior for root-anchored parent paths should use root_reaching_parent_degree, bucketed by L_max, because parents that cannot reach the selected root do not contribute paths to the root histogram. The full degree should still be reported because it tells us when the category graph has substantial side branching that a different root, broader filter, or child-inclusive search may admit.

This also explains why user-provided examples such as categories with three or five parent categories do not contradict a low root-reaching estimate under a specific root. They do contradict the previous TSV-local degree estimate, and they show why the full LMDB parent-degree column must be carried separately.

## Next Sampling Step

The MTC run shows enough branching that the next enwiki run should move from degree diagnostics to bounded exact histograms over selected ancestor cones. Because many targets hit the depth cap or cycle policy, the next run should select deeper MTC targets, export their bounded parent ancestor cones, and then compare:

    1. exact root-reaching histograms;
    2. L_min/L_max scalar bounds;
    3. priors bucketed by capped or exact L_max;
    4. full versus root-reaching parent-degree moments.

That deeper-target approach is the right place to recompute exact histograms, because it will let the prior use L_max and root_reaching_parent_degree for the actual nodes whose distributions we care about.

## Validation

- python3 -m unittest tests.test_lmdb_parent_branching_diagnostic
- python3 scripts/lmdb_parent_branching_diagnostic.py ... --graph-name enwiki_mtc_full_parent_branching
- python3 scripts/lmdb_parent_branching_diagnostic.py ... --graph-name enwiki_root97688913_full_parent_branching
- python3 scripts/lmdb_parent_branching_diagnostic.py ... --graph-name enwiki_root97688913_full_parent_branching_wide
