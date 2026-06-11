# Enwiki Numeric Root Tail-Pruning Pilot

Date: 2026-06-11

Branch: `codex/enwiki-tail-pruning-pilot`

## Scope

This report is the first enwiki-side tail-pruning pilot using the local numeric-keyed LMDB artifact:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_resident
```

The artifact metadata identifies best numeric root `97688913`:

```text
subcat_edges=9932244
unique_ids=3783423
unique_children=2626951
unique_parents=1182212
true_roots=1156472
best_root_id=97688913
best_root_subtree_size=796695
best_root_out_degree=765354
```

Important caveat: this is a numeric-root LMDB pilot. The local resolved enwiki category-title export is not present in this checkout, so this is not yet a named `Category:Main_topic_classifications` run. The sample is also capped to stay small and to avoid exporting all `9.9M` category-parent edges.

## Disk Check

Before running the pilot, disk space was checked with `df -h`.

Relevant free space:

```text
/dev/sdd  251G  171G   68G  72% /
C:\       932G  667G  266G  72% /mnt/c
```

The `snapfuse` mounts report `100%` usage, but those are read-only snap images and are not the project disk filling up.

## Sampling

A Scratch-only helper sampled a bounded subtree directly from LMDB, using `category_child` for traversal and `category_parent` to emit the sampled parent-edge TSV:

```text
/mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/sample_enwiki_lmdb_subtree.py
```

Final sample command:

```text
python3 /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/sample_enwiki_lmdb_subtree.py --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_resident --root 97688913 --max-depth 3 --node-limit 30000 --max-children-per-node 10000 --output /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/enwiki_root97688913_depth3_cap30k.tsv --targets-output /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/enwiki_root97688913_depth3_cap30k_targets.txt --stats-output /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/enwiki_root97688913_depth3_cap30k_stats.txt
```

Result:

```text
root=97688913
selected_nodes=10403
sampled_edges=10505
depth_counts=0:1,1:10000,2:380,3:22
```

This confirms the sample is dominated by immediate children of the high-degree root. It is useful as a smoke/pilot, but it is not a deep balanced enwiki subtree sample.

## Fit And Tail-Pruning Comparison

Command:

```text
python3 scripts/distribution_fit_comparison.py --edge-file /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/enwiki_root97688913_depth3_cap30k.tsv --graph-name enwiki_root97688913_depth3_cap30k_tail_pruning --root 97688913 --targets-file /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/enwiki_root97688913_depth3_cap30k_targets.txt --depths 1,2,3,4,6,8,10,12 --tail-epsilon 0.001 --continuous-sample-points 100 --prune-thresholds 0.01,0.001,0.0001 --output-dir /mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot
```

Generated outputs:

```text
/mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/distribution_fit_comparison_enwiki_root97688913_depth3_cap30k_tail_pruning_20260611T234527Z.jsonl
/mnt/c/Users/johnc/Scratch/enwiki-tail-pruning-pilot/distribution_fit_comparison_summary_enwiki_root97688913_depth3_cap30k_tail_pruning_20260611T234527Z.md
```

### Realized Histogram Fits

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|
| binomial_fit | 10403 | 0.000385 | 0.000000 | 1.000000 | 0.000096 | 1770.4 | 6349.8 | 3.607 |
| shifted_gamma_fit | 10403 | 0.007563 | 0.000000 | 0.819939 | 0.003781 | 2603.3 | 6349.8 | 3.607 |

### Realized Support By Root Distance

| L_min | targets | mean_bins | p95_bins | max_bins | mean_effective_bins | mean_path_count | mean_parent_degree |
|-------|---------|-----------|----------|----------|---------------------|-----------------|--------------------|
| 0 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 0.000000 |
| 1 | 10102 | 1.011 | 1.000 | 3 | 1.011 | 1.010 | 1.010196 |
| 2 | 282 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000000 |
| 3 | 18 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000000 |

### Realized Tail-Pruned Support

| tail_threshold | distributions | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | mean_weighted_power_error |
|----------------|---------------|--------------------|----------------|-------------------|-------------------|---------------------------|
| 0.0001 | 10403 | 1.010 | 1.010 | 0.000 | 0.000000 | 0.000000 |
| 0.001 | 10403 | 1.010 | 1.010 | 0.000 | 0.000000 | 0.000000 |
| 0.01 | 10403 | 1.010 | 1.010 | 0.000 | 0.000000 | 0.000000 |

### Depth-Conditioned Prior Distributions

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |
|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|
| binomial_prior | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5384.6 | 16397.9 | 0.675 |
| shifted_gamma_prior | 8 | 0.126675 | 0.188226 | 0.188226 | 0.058938 | 47715.8 | 16397.9 | 0.675 |

### Prior Support By Depth

| depth | rows | mean_bins | mean_effective_bins | max_effective_bins | mean_excess |
|-------|------|-----------|---------------------|--------------------|-------------|
| 1 | 2 | 2.000 | 2.000 | 2 | 0.019610 |
| 2 | 2 | 3.000 | 2.000 | 2 | 0.039219 |
| 3 | 2 | 4.000 | 3.000 | 3 | 0.058829 |
| 4 | 2 | 5.000 | 3.000 | 3 | 0.078439 |
| 6 | 2 | 7.000 | 3.000 | 3 | 0.117658 |
| 8 | 2 | 9.000 | 3.000 | 3 | 0.156878 |
| 10 | 2 | 11.000 | 3.000 | 3 | 0.196097 |
| 12 | 2 | 13.000 | 4.000 | 4 | 0.235317 |

### Prior Tail-Pruned Support

| tail_threshold | distributions | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | mean_weighted_power_error |
|----------------|---------------|--------------------|----------------|-------------------|-------------------|---------------------------|
| 0.0001 | 8 | 6.750 | 3.375 | 3.375 | 0.000018 | 0.000001 |
| 0.001 | 8 | 6.750 | 2.875 | 3.875 | 0.000230 | 0.000018 |
| 0.01 | 8 | 6.750 | 2.250 | 4.500 | 0.002683 | 0.000310 |

## Interpretation

This capped numeric-root enwiki pilot does not yet show the wider realized histograms we expected. The realized sample remains one-bin dominated: mean realized support is about `1.010` bins, p95 is one bin, and max support is three bins. Tail pruning removes no realized bins.

The likely reason is sample shape. The root has very high out-degree, and even with a per-node fanout cap the sample is dominated by immediate children: `10,102` of `10,403` targets have `L_min = 1`. This is a valid smoke test for the LMDB path and comparison harness, but it is not a deep enwiki stress sample.

The prior still shows light-tail behavior. At threshold `1e-3`, prior support drops from mean `6.75` bins to `2.875` kept bins with mean dropped mass `0.000230` and weighted-power error `0.000018`.

Binomial again dominates shifted Gamma for this near-chain sample. That should not be generalized to all enwiki; it says the current capped numeric-root sample is still structurally close to the SimpleWiki parent-only cases.

## Policy Implication

This pilot does not justify changing the SimpleWiki-derived policy. For parent-only rooted samples that remain one-bin dominated, store exact sparse histograms. Use tail-pruned priors for admission/planning, not realized-state compression.

The next enwiki run should avoid the high-degree-root immediate-child bias. Better options are:

```text
1. resolve and sample Category:Main_topic_classifications by title;
2. choose target nodes at deeper root distance and compute their ancestor cones back to the root;
3. stratify samples by L_min bucket rather than BFS order from a giant root;
4. run child-inclusive paths, where branching pressure should grow much faster.
```

## Validation

- `df -h` showed `/` has about `68G` free and `/mnt/c` has about `266G` free.
- Scratch-only LMDB sampler produced the capped `10,403` node sample.
- `distribution_fit_comparison.py` completed with `errors = 0`.
