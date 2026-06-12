# LMDB Ancestor Cache Policy MTC Smoke - 2026-06-12

This smoke run exercises fixed-size cache residency for root-anchored ancestor
histograms on the enwiki `Category:Main_topic_classifications` subtree. The
cache policy is intentionally separate from histogram path-length budgets:
targets define an ancestor search space, near-root nodes inside that space are
eligible for histogram caching, and collisions are resolved using current
ancestor-cone membership plus root-distance priority.

## Command

```bash
python3 scripts/lmdb_ancestor_cache_policy_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_ancestor_cache_policy_smoke \
  --target-depths 3,4 \
  --children-per-node 64 \
  --frontier-limit 800 \
  --targets-per-depth 12 \
  --max-ancestor-nodes 10000 \
  --max-ancestor-edges 80000 \
  --root-distance-cap 32 \
  --cache-slots 256 \
  --admit-l-min 6 \
  --admit-l-max 12 \
  --seed enwiki-mtc-cache-policy-smoke \
  --output-dir docs/reports
```

## Results

- Targets: 24 sampled category nodes at child depths 3 and 4.
- Ancestor collection caps: 1 capped query after raising the cap to 10,000
  ancestor nodes and 80,000 ancestor edges.
- Mean ancestor search-space size: 3,626.92 nodes.
- P95 ancestor search-space size: 6,918 nodes.
- Mean admission candidates per query: 147.75 near-root/root-reaching nodes.
- P95 admission candidates per query: 185 nodes.
- Fixed cache size: 256 slots.
- Final occupied cache entries: 138.
- Mean cache hits before admission updates: 102.08 entries already resident in
  the current ancestor cone.

Cache actions:

| Action | Count |
| --- | ---: |
| `refresh` | 2,399 |
| `keep_existing` | 911 |
| `insert` | 138 |
| `overwrite_lower_priority` | 51 |
| `overwrite_outside_cone` | 47 |

## Interpretation

The high `refresh` count means near-root ancestor candidates are repeatedly
encountered across nearby target cones, so fixed-size residency has useful
reuse even without tying cache admission to a search-length budget.

The collision policy behaved as intended in this run: when both entries were in
the current cone, most collisions kept the existing nearer-root or otherwise
higher-priority entry. The `overwrite_outside_cone` count shows the separate
case where an old colliding entry was not useful for the current target cone and
could be replaced without sacrificing local reuse.

The remaining capped query says the diagnostic collection bound is still visible
for a small part of the sampled workload. The next benchmark should sweep cache
slots and root-distance admission thresholds before treating these numbers as a
policy recommendation.
