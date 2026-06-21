# Enwiki MTC Shallow Splice Validation

Date: 2026-06-14

This run validates the boundary-splice identity on the title-resolved enwiki
`Category:Main_topic_classifications` artifact:

```text
lmdb_dir=/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident
root=7345184
root_title=Category:Main_topic_classifications
```

The target selection intentionally stays near the top of the MTC child cone.
Targets are sampled at child depths `2,3`, and boundaries are sampled at child
depths `1,2`.  This follows the current policy preference to validate on more
general child categories before moving to leafier categories where full exact
enumeration and target-ancestor collection become more expensive.

## Run

```bash
python3 scripts/lmdb_boundary_coverage_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_shallow_splice_validation \
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
  --budgets 10 \
  --path-count-cap 100000 \
  --expansion-cap 200000 \
  --measure-filtered-boundary-suffix-mass \
  --validate-full-exact \
  --seed enwiki-mtc-splice-validation-v1 \
  --output-dir docs/reports
```

## Results

| graph | targets | budget | boundary_nodes | terminal_prefixes | direct_root_paths | boundary_hit_prefixes | spliced_root_paths | comparable_validation_rows | exact_match_rows | max_abs_root_path_delta | max_abs_value_sum_delta | max_abs_mean_path_length_delta |
|-------|--------:|-------:|---------------:|------------------:|------------------:|----------------------:|-------------------:|---------------------------:|----------------:|------------------------:|------------------------:|-------------------------------:|
| enwiki_mtc_shallow_splice_validation | 4 | 10 | 43 | 44 | 27 | 17 | 63 | 4 | 4 | 0.000 | 0.000000 | 0.000000 |

All four validation rows were comparable: neither the boundary-stopped run nor
the full filtered DFS hit a path-count or expansion cap.  The boundary-spliced
mass, value sum, and mean path length exactly matched the full filtered DFS for
each sampled target.

One target row had `root_paths=0` before suffix splicing but a positive boundary
hit count.  In this validation mode that is expected: the boundary-stopped pass
hands off to a suffix histogram before it reaches root, and the full exact
validation confirms that the spliced suffix accounts for the missing root paths.

## Interpretation

This is a correctness smoke for boundary suffixes on enwiki MTC, not yet a
performance decision.  It shows that for shallow/general categories and budget
`10`, stopping at selected boundaries and adding their filtered suffix
histograms reproduces full filtered DFS exactly under the same simple-path and
root-cone policies.

Budget `20` and deeper target depths should be a separate follow-up.  A first
trial showed that target-ancestor collection can dominate before the validation
rows even run unless `--max-parent-depth` is kept small.  The next larger run
should either keep ancestor collection bounded, select known general child
targets explicitly, or add a cheaper root-cone-aware ancestor-boundary selector.

## Artifacts

- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_shallow_splice_validation_20260614T234122Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_shallow_splice_validation_20260614T234122Z.jsonl`
