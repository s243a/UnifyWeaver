# Enwiki MTC Root-Cone Ancestor Boundary Validation

Date: 2026-06-14

This follow-up validates the boundary-splice identity at budgets `10` and `20`
after making target-ancestor boundary collection respect the active root-cone
scope during traversal.

The previous shallow EnWiki validation proved the splice identity at budget
`10`, but an early budget-`20` trial showed that target-ancestor boundary
collection could dominate before target rows were evaluated.  The collector now
accepts an optional parent predicate and an optional root-cone depth map, so
root-cone experiments prune off-cone parents while collecting target ancestors
instead of walking the full parent graph and filtering afterward.

## Run

```bash
python3 scripts/lmdb_boundary_coverage_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_shallow_splice_validation_b10_20 \
  --mode exact \
  --parent-filter root-cone \
  --selection-source root-cone \
  --root-cone-depth 3 \
  --root-cone-children-per-node 0 \
  --root-cone-frontier-limit 3000 \
  --boundary-depths 1,2 \
  --target-depths 2,3 \
  --boundaries-per-depth 12 \
  --targets-per-depth 2 \
  --include-target-ancestor-boundaries \
  --target-ancestor-boundary-limit 20 \
  --max-parent-depth 4 \
  --budgets 10,20 \
  --path-count-cap 100000 \
  --expansion-cap 200000 \
  --measure-filtered-boundary-suffix-mass \
  --validate-full-exact \
  --seed enwiki-mtc-splice-validation-v1 \
  --output-dir docs/reports
```

## Results

| budget | targets | boundary_nodes | terminal_prefixes | direct_root_paths | boundary_hit_prefixes | spliced_root_paths | comparable_validation_rows | exact_match_rows | max_abs_root_path_delta | max_abs_value_sum_delta | max_abs_mean_path_length_delta |
|-------:|--------:|---------------:|------------------:|------------------:|----------------------:|-------------------:|---------------------------:|----------------:|------------------------:|------------------------:|-------------------------------:|
| 10 | 4 | 40 | 11 | 1 | 10 | 63 | 4 | 4 | 0.000 | 0.000000 | 0.000000 |
| 20 | 4 | 40 | 11 | 1 | 10 | 63 | 4 | 4 | 0.000 | 0.000000 | 0.000000 |

All eight validation rows were comparable.  Neither boundary-stopped search nor
full filtered DFS hit a path-count or expansion cap.  Boundary-spliced mass,
value sum, and mean path length matched full filtered DFS exactly for both path
budgets.

The generated report records `Target-ancestor boundary collection scope` as
`root-cone`, confirming that the boundary candidates were collected under the
same root-cone policy used for target search and suffix measurement.

## Interpretation

For shallow/general EnWiki MTC targets, root-cone-scoped target-ancestor
collection makes the validation pass cheap enough to include budget `20`.
Because the root cone in this run is only depth `3`, the result should be read
as a correctness check for the splice identity and collector scope, not as a
claim about deeper target performance.

The next larger experiment can now move target depths from `2,3` to `3,4` while
keeping root-cone-scoped ancestor collection enabled.  If exact validation stays
comparable at budget `20`, that is the point where cache-boundary cost and hit
geometry become the main planning question again.

## Artifacts

- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_shallow_splice_validation_b10_20_20260614T235232Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_shallow_splice_validation_b10_20_20260614T235232Z.jsonl`
