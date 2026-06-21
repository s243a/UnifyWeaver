# LMDB Ancestor Cache Policy Resource Controls - 2026-06-12

This update makes the ancestor-cache policy benchmark cheaper by default and adds
a summary-only sweep driver.

## Changes

- The single-config benchmark now writes only `*_summary.json` by default.
- Full per-query/per-entry JSONL traces require `--write-jsonl`.
- The default sample and traversal caps are smaller:
  - `children_per_node`: 64
  - `frontier_limit`: 800
  - `targets_per_depth`: 12
  - `max_ancestor_nodes`: 1500
  - `max_ancestor_edges`: 8000
  - `root_distance_cap`: 32
  - `cache_slots`: 256
- `scripts/lmdb_ancestor_cache_policy_sweep.py` samples targets once, collects
  ancestor cones once, computes root-distance summaries once, then replays cache
  policy for each cache/admission setting.

## Validation

Unit tests:

```bash
python3 -m unittest tests.test_lmdb_ancestor_cache_policy_benchmark
```

Result: 8 tests passed.

Tiny sweep smoke:

```bash
python3 scripts/lmdb_ancestor_cache_policy_sweep.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_cache_policy_tiny_sweep \
  --target-depths 3 \
  --children-per-node 32 \
  --frontier-limit 200 \
  --targets-per-depth 4 \
  --max-ancestor-nodes 500 \
  --max-ancestor-edges 2000 \
  --root-distance-cap 24 \
  --cache-slots 64 \
  --admit-l-min 6 \
  --admit-l-max 12 \
  --sweep-cache-slots 64,128 \
  --sweep-admit-l-max 8,12 \
  --seed enwiki-mtc-cache-policy-tiny-sweep \
  --output-dir /mnt/c/Users/johnc/Scratch/ancestor-cache-controls
```

Result: 4 summary rows, no JSONL trace. The deliberately low caps produced a
`capped_query_rate` of 0.75, which is useful for resource-control validation but
not a policy recommendation.

Single-config summary-only smoke:

```bash
python3 scripts/lmdb_ancestor_cache_policy_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_cache_policy_single_summary_smoke \
  --target-depths 3 \
  --children-per-node 16 \
  --frontier-limit 80 \
  --targets-per-depth 2 \
  --max-ancestor-nodes 250 \
  --max-ancestor-edges 1000 \
  --root-distance-cap 20 \
  --cache-slots 64 \
  --admit-l-min 6 \
  --admit-l-max 10 \
  --seed enwiki-mtc-cache-policy-single-summary \
  --output-dir /mnt/c/Users/johnc/Scratch/ancestor-cache-controls
```

Result: one summary JSON and no JSONL trace. The smoke reported
`capped_query_rate = 0.5`, `mean_ancestor_nodes = 243.0`, and final occupancy
`40 / 64` cache slots.

## Next Use

Use the sweep script for first-pass policy scans. If a setting has an acceptable
capped rate and stable reuse metrics, rerun the single-config benchmark with
larger caps and `--write-jsonl` only when per-query diagnostics are needed.
