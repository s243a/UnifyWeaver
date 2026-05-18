# R5 simplewiki handoff: ingest column-mapping blocker (RESOLVED)

**Status**: R5 complete. The ingest column-mapping fix landed (`UW_VAL_COL = 2 → 6`), simplewiki LMDB re-ingested with real `cl_target_id` edges, post-ingest selected a 14,680-node root subtree, and the Rust LMDB-mode matrix bench produced deterministic results across 5 trials.

## TL;DR — the fix that was applied

In `examples/streaming/simplewiki_category_ingest_text.pl`, changed:

```prolog
'UW_VAL_COL'        = 2,    % WRONG — col 2 is cl_timestamp
```

to:

```prolog
'UW_VAL_COL'        = 6,    % cl_target_id (newer MediaWiki schema replaces cl_to varchar)
```

Confirmed against the CREATE TABLE in the dump: col 0 = `cl_from`, col 4 = `cl_type`, col 6 = `cl_target_id` (bigint, the linktarget reference). Older docs/comments saying "cl_to (column 2)" reflected a pre-linktarget MediaWiki schema.

## R5 results (5 trials, median, simplewiki, Rust LMDB cursor, lmdb_zero, -N1)

| Metric | Median |
| --- | ---: |
| `load_ms` (open LMDB + i2s load + reach-root cursor BFS) | 43 |
| `query_ms` (134,692 seed iterations through the kernel) | 370 |
| `aggregation_ms` | 29 |
| `total_ms` | 462 |
| `seed_count` | 134,692 |
| `demand_set_size` | 14,680 (root_id=2 subtree) |
| `tuple_count` | 126 (paths within max_depth=10) |
| peak RSS | ~80 MB |
| wall | ~0.48 s |

`tuple_count` is deterministic across all 5 trials — correctness signal.

### Comparison to Haskell Phase L#7

Haskell `resident_cursor` at simplewiki: 226 ms -N1 / 207 ms -N2 / 234 ms -N4 (root id 265340 with 14,661 descendants, **5,000 seeds**).

Rust ran the same per-seed workload but across **134,692 seeds** (the full demand-graph seed set) — 27× more seeds. At ~3.4 µs/seed, the Rust cursor path is competitive on a per-seed basis; total time is higher because the seed denominator differs. For a true head-to-head comparison, both runs would need to use the same seed cardinality.

## What's staged but uncommitted on this branch

`feat/wam-rust-bench-simplewiki` is **zero commits ahead of main**. These two files are staged:

```
A  docs/handoff/wam_rust_simplewiki_blocker.md         (this file)
A  examples/benchmark/simplewiki_post_ingest.py
```

Plus the untracked LMDB fixture at `data/benchmark/simplewiki_cats/` (gitignored, but the LMDB content is currently the BROKEN version with timestamps as edges — should be re-ingested after applying the fix).

## How I confirmed the column mapping

Inspecting the actual mysql_stream output for subcat rows with the same `cl_from=4985`:

```
col0: 4985                                                          ← cl_from
col1: .FBHRP2L\x04N.:2D.2\x01\x14\x01\xdc\x13                       ← cl_sortkey (binary)
col2: 2025-02-17 19:20:03                                           ← cl_timestamp
col3:                                                               ← cl_sortkey_prefix
col4: subcat                                                        ← cl_type
col5: 1                                                             ← cl_collation
col6: 664772 / 1886545 / 2377498  (varies per row)                  ← cl_to (linktarget_id)
```

Three subcat rows with `cl_from=4985` had three different col-6 values — that's the shape of an outgoing edge list, which is what `cl_to` should look like. Column 6 it is.

## What the post-ingest script (`examples/benchmark/simplewiki_post_ingest.py`) does

Already saved and ready to use. After running the (fixed) ingest pipeline:

1. Walks `category_parent` (DUPSORT), writes reversed edges into a new `category_child` DUPSORT sub-db on the same LMDB env (idempotent — safe to rerun)
2. Scans all true-roots (nodes with no parents) for the largest reachable subtree at `max_depth=10`
3. Emits `seed_ids.txt`, `root_ids.txt`, `root_categories.tsv`, `article_category.tsv` (using `i2s`-decoded names from the same env to avoid the readonly-cursor quirk), `metadata.json`

The article_category fix: reads i2s from the same RW env in one session — Python lmdb's `open_db` on a readonly env over a never-opened-named-subdb fails with EINVAL.

## Recovery procedure for next session

```bash
# 0. confirm we're on the right branch (or recreate it)
git checkout feat/wam-rust-bench-simplewiki 2>/dev/null || git checkout -b feat/wam-rust-bench-simplewiki

# 1. apply the column fix to the ingest declaration
#    (edit examples/streaming/simplewiki_category_ingest_text.pl
#     change UW_VAL_COL = 2 to UW_VAL_COL = 6)

# 2. wipe and re-ingest the simplewiki LMDB
rm -rf data/benchmark/simplewiki_cats/lmdb_resident
mkdir -p data/benchmark/simplewiki_cats/lmdb_resident
swipl -q -g "process_dump('context/gemini/UnifyWeaver/data/simplewiki/simplewiki-latest-categorylinks.sql.gz', 'data/benchmark/simplewiki_cats/lmdb_resident')." \
    -t halt examples/streaming/simplewiki_category_ingest_text.pl

# 3. run post-ingest (adds category_child + picks root + emits fixture files)
python3 examples/benchmark/simplewiki_post_ingest.py

# 4. sanity check: best_root_subtree_size should be much larger than 1222
#    (with correct edges we expect a deep hierarchy — Phase L#7 Haskell
#    had a root with 14,661 descendants)
cat data/benchmark/simplewiki_cats/metadata.json

# 5. regen the Rust LMDB-mode bench (R4's wiring is already on main)
rm -rf /tmp/uw_swiki_bench
swipl -q -s examples/benchmark/generate_wam_rust_matrix_benchmark.pl -- \
    data/benchmark/1k/facts.pl /tmp/uw_swiki_bench accumulated interpreter kernels_on cursor lmdb_zero
(cd /tmp/uw_swiki_bench && cargo build --release)

# 6. run the 5-trial sweep
BIN=/tmp/uw_swiki_bench/target/release/wam_rust_matrix_bench
FIXTURE=$PWD/data/benchmark/simplewiki_cats
for T in 1 2 3 4 5; do
    /usr/bin/time -f "wall=%es rss=%MkB" "$BIN" "$FIXTURE" 2>&1 \
        | grep -E "(load_ms|query_ms|total_ms|seed_count|demand_set_size|tuple_count|wall|rss)" \
        | tr '\n' ' '; echo
done

# 7. compare to Haskell Phase L#7 numbers in WAM_PERF_OPTIMIZATION_LOG.md:
#    resident_cursor at simplewiki: 226 ms -N1 / 207 ms -N2 / 234 ms -N4
#    (Haskell used root id 265340 with 14,661 descendants, 5000 seeds)
#    With corrected ingest, our Rust numbers should be in the same regime.

# 8. commit + open PR with R5 results
```

## Caveat that survives the fix

`cl_to` in the newer schema is a `linktarget_id`, not a category name. The graph topology will be correct (real subcat edges connecting categories), but the node STRINGS in s2i/i2s will be `cl_from` page_id-strings (children) and `cl_to` linktarget_id-strings (parents) — different namespaces.

For PERF measurement this is fine (wall-clock, demand_set_size, kernel throughput are all valid). For human-readable output (knowing the root is "Physics"), we'd need a second-pass ingest of `simplewiki-latest-linktarget.sql.gz` to map linktarget_id → category title. That's a separate enhancement — file as R5.5 if/when needed.

## Negative-baseline bench numbers (with broken topology)

For the record, the BROKEN simplewiki LMDB (page_id→timestamp edges) gave us these numbers — useful as a "kernel-fail throughput" reference:

| Metric | Median (5 trials) |
| --- | ---: |
| `load_ms` | 20 |
| `query_ms` | 322 |
| `aggregation_ms` | 24 |
| `total_ms` | 380 |
| `seed_count` | 118,697 |
| `demand_set_size` | 1,222 |
| `tuple_count` | 0 (kernel finds no paths) |
| peak RSS | 64 MB |

Single-threaded Rust, lmdb-zero crate, cursor BFS. The 322 ms query_ms is dominated by iterating 118k seeds where every kernel call fails fast (no edges → empty result). With the column fix, query_ms should *decrease* (fewer dead-end iterations) AND `tuple_count` should go up (real paths exist).

## What's on main (unaffected by this blocker)

R1–R4 of the Rust-LMDB arc are merged. Specifically:

| PR | Commit | What |
| --- | --- | --- |
| #2258 | `21b65f31` + `181b72ab` | `LmdbFactSource` + configurable lmdb_crate (lmdb_zero default, heed opt-in) |
| #2260 | `005f5dfc` | Cursor `reachable_to_root` BFS + LMDB codegen tests |
| #2278 | `aa126442` | Matrix bench main.rs wired to LmdbFactSource (R4) |

These run end-to-end at 1k_cats with `total_ms=12`, `tuple_count=48` — proving the infrastructure works. R5's job was to validate at 297k-edge scale, which is what this blocker is about.

## Reference points

- Haskell Phase L#7 (simplewiki resident_cursor): `docs/design/WAM_PERF_OPTIMIZATION_LOG.md`
- Haskell Phase L#8/9 (enwiki — also needs the same ingest fix when we get to R6): same doc
- ingest pipeline source: `examples/streaming/simplewiki_category_ingest_text.pl`
- mysql_stream binary: `src/unifyweaver/runtime/rust/mysql_stream/target/release/mysql_stream`
- SQL dumps: `context/gemini/UnifyWeaver/data/simplewiki/` (gitignored)
