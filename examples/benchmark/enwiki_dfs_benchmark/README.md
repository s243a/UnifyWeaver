# enwiki-dfs-benchmark

Standalone Haskell benchmark that compares **IntMap** vs **LMDB raw** as
backing store for the same DFS workload, running against the full
enwiki subcat graph (~10M edges) ingested via the streaming pipeline
at `examples/streaming/enwiki_category_ingest.pl`.

Purpose: measure whether the IntMap/LMDB crossover projected in
`docs/design/WAM_HASKELL_SCALING_INSIGHTS.md` actually happens at
this scale.

## Build

```sh
cd examples/benchmark/enwiki_dfs_benchmark
cabal v2-build
```

## Run

```sh
# LMDB backend, 10,000 random seeds
./dist-newstyle/.../enwiki-dfs PATH/TO/LMDB lmdb 10000

# IntMap backend (loads all edges into memory at startup)
./dist-newstyle/.../enwiki-dfs PATH/TO/LMDB intmap 10000
```

Samples N random seeds using reservoir sampling with a fixed RNG
seed (reproducible). Runs a depth-12 DFS from each seed, reports
mean-per-seed wall time, seeds/sec throughput, and total nodes
visited.

## Why two backends on the same LMDB

The **LMDB backend** keeps a long-lived read transaction and does
dupsort cursor iteration per lookup — reads directly from mmap'd
pages, no materialization.

The **IntMap backend** iterates the LMDB once at startup, builds an
`IntMap [Int]` adjacency list in the GHC heap, then closes the
LMDB. All queries hit memory.

Both sides use the same DFS algorithm and same seed samples. The
only variable is the backing storage.

## Results at 10M edges

| Scale (seeds) | IntMap per-seed | LMDB per-seed | Ratio |
|---------------|----------------:|--------------:|------:|
| 10,000 | 0.166 ms | 0.159 ms | 0.96× (LMDB faster) |

At this scale, LMDB's on-demand mmap lookups are slightly faster
than the in-memory IntMap — the projected crossover has happened.
IntMap also pays ~50s of startup cost to load 9.93M edges; LMDB
has negligible startup.
