# F# V1/V3 depth-likeness probe

Production-quality counterpart to the Python prototypes in `../scripts/`.
Uses the F# WAM bidirectional kernel template directly, talks to LMDB via
`LightningDB` cursors, scales to enwiki (2.26M nodes from
`Category:Main_topic_classifications`).

## Files

- `Kernel.fs` — bundles the bidirectional kernel from
  `templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`
  (mustache var `{{edge_pred}}` resolved to `category_parent`). Adds a
  `_withMinDist` variant that accepts a precomputed BFS distance map so
  calibration runs once across many seed queries. Without this, each
  seed call re-runs full graph BFS — ~31s/seed on enwiki vs ~0.05s with
  caching.
- `Program.fs` — per-seed harness:
  - V1 mode (default): `B = depth(seed)` — only child-via shortcuts admissible
  - V3 mode (`--variant v3`): `B = max acyclic parent distance(seed)` — admits
    parent-direction shortcuts via alternate ancestor chains. Includes
    children-BFS-depth-order DP to compute `max_dist`.
- `uw_v1_probe.fsproj` — F# project file. Single dependency: LightningDB 0.21.0.

## Build + run

```bash
# Requires .NET 9 SDK
cd fsharp_v1_v3_probe
dotnet build -c Release    # ~5 seconds

# Generate seeds (Python helper)
# ... see scripts/generate_seeds.py for the pattern, or any seeds TSV
#     with columns seed_id\tdepth

# V1: B = depth
dotnet run -c Release --no-build -- \
    /path/to/topical_lmdb 7345184 \
    seeds.tsv results_v1.tsv

# V3: B = max parent distance
dotnet run -c Release --no-build -- \
    /path/to/topical_lmdb 7345184 \
    seeds.tsv results_v3.tsv \
    --variant v3
```

## Key parameter: `Kernel.maxPaths`

The kernel has a configurable cap on enumerated paths (default 200K, the
probe sets it to 100K). Hit by V3 on enwiki at depths 5-10 where
budgets are large and path counts blow up. Truncation order isn't
random — DFS hits paths in a specific order — so capped seeds have a
small bias toward shorter paths. The mean `d_wPow` values are slightly
biased toward shorter values when the cap is hit, but the qualitative
findings (region of seed pairs that are MIN-LIKE/AVG-LIKE/SHORTCUT/etc.)
are robust.

## Cross-target parity (simplewiki)

The F# probe was validated against the Python V1 prototype on
simplewiki Articles topical subgraph:

```
            calibration                shortcut rate
Python V1:  D=4.914, b_eff=14.828      0% at depths 1-9
F# V1:      D=4.914, b_eff=14.828      0% at depths 1-9
```

Per-seed `d_wPow` values match to floating-point precision. The F# is
~10× faster end-to-end on simplewiki (calibration in 0.3s instead of
the Python's 3s for an equivalent LMDB read).

## Results

- `../results/enwiki_v1_fsharp_B_equals_min_depth.tsv` — V1 on enwiki MTC, 240 seeds
- `../results/enwiki_v3_fsharp_B_equals_max_parent_dist.tsv` — V3 on enwiki MTC, 240 seeds
- `../results/simplewiki_v3_fsharp_B_equals_max_parent_dist.tsv` — V3 simplewiki for comparison

See `docs/reports/depth_likeness_budget_variants.md` (the "enwiki
extension" section) for full analysis.

## Building the post-fix enwiki MTC LMDB

The probe needs an enwiki LMDB with the closed-subtree schema
(`s2i`/`i2s`/`category_parent`/`category_child`). The build steps:

```bash
# Step 1: full-graph LMDB via mysql_stream_lmdb (correct mode)
BIN=src/unifyweaver/runtime/rust/mysql_stream/target/release/mysql_stream_lmdb
DUMPS=path/to/enwiki/dumps
$BIN $DUMPS/enwiki-latest-categorylinks.sql.gz /tmp/enwiki_post_fix_lmdb \
    --mode correct \
    --linktarget-dump $DUMPS/enwiki-latest-linktarget.sql.gz \
    --page-dump $DUMPS/enwiki-latest-page.sql.gz \
    --cl-type subcat --refresh

# Step 2: extract MTC subgraph (Python helper, see report appendix)
python3 build_enwiki_mtc_subgraph.py /tmp/enwiki_post_fix_lmdb /tmp/enwiki_mtc_lmdb 7345184
```

The full-graph ingest takes ~3.5 minutes (9.9M edges from 2.6M
categories). MTC subgraph extraction takes ~1 minute (2.26M reachable
nodes, 6.7M edges).
