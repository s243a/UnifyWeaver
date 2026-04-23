# WAM Haskell Benchmark Strategy — Notes

Working notes for the IntMap-vs-LMDB scaling comparison. These are
decisions-in-progress rather than a finished spec. Captured here so we can
revisit them when the data lands.

## Current benchmark registry

| Directory | Shape | Scale | Generator |
|-----------|-------|-------|-----------|
| `data/benchmark/dev/` | Dev (~20 articles from physics JSONL) | tiny | `generate_facts_from_db.py` |
| `data/benchmark/300/` | Physics subset, 300 articles | 300 | `generate_facts.py` (API) |
| `data/benchmark/1k/` | Physics subset, 1k articles | 1k | `generate_facts_from_db.py` |
| `data/benchmark/5k/` | Physics subset, 5k articles | 5k | `generate_facts_from_db.py` |
| `data/benchmark/10k/` | Science subset, 10k articles | 10k (25k edges) | `generate_facts_from_db.py --root Science --max-articles 10000` |
| `data/benchmark/100k_cats/` | Full simplewiki category hierarchy, categories as seeds | ~84k cats / ~197k edges | `generate_category_only_benchmark.py` |
| `data/benchmark/10x/` | Legacy scale variant | varies | ad hoc |

**What each scale is good for:**

- **dev/300/1k** — fast correctness checks; WAM-interpreter timing fits in
  seconds.
- **5k/10k** — backend throughput comparisons (IntMap vs LMDB raw). 10k
  is the documented baseline in scaling-insights.
- **100k_cats** — next step up in graph size; ~8× the hierarchy edges of
  10k. Still fits in RAM comfortably so IntMap should still win, but
  starts to exercise GC. First chance to see the IntMap/LMDB ratio
  shift under pressure.
- **Future 1M / full-383k-articles** — crossover territory. Requires
  rerunning `generate_facts_from_db.py` with `--max-articles 383000`
  against the local DB.

## What we're trying to measure

Two separable questions hide inside "how does fact access scale":

1. **Backend throughput** — for a fixed graph, how does `EdgeLookup`
   perform when implemented as IntMap vs LMDB raw? Dominated by per-lookup
   cost.
2. **Scaling behaviour** — as the graph grows, at what size does
   in-memory IntMap lose (RAM, GC pressure) and LMDB's mmap story win?
   Dominated by working-set vs RAM and GC pause time.

At 10k (25k edges) we answered (1): LMDB raw is 1.29x IntMap warm. We
haven't stressed (2) yet — IntMap is still trivially resident.

100k is the next step up. Simplewiki has ~300k category edges total, so
100k articles with the full category hierarchy gives us roughly 12x the
hierarchy size of the 10k run — enough to start exercising GC but still
well inside RAM.

## Axes of variation

| Axis | Choices | What it tests |
|------|---------|---------------|
| Article count | 1k, 5k, 10k, 100k, 1M | Seed-level work; parallelism scale |
| Category set | Full simplewiki (~97k cats / ~297k edges), descendant-of-root, descendant-of-Science | Graph connectivity vs sparsity; orphan rate |
| Article filter | All, top-k by category count, top-k semantic (embeddings), random-seeded | Information density per seed; reproducibility |
| Backend | IntMap (FFI), LMDB raw (FFI) | Backend throughput |
| Workload | effective-distance, closure, shortest-path | Access pattern (DFS, BFS, Dijkstra) |

## Recommended strategy (for 100k)

**Phase 1 — confirm the crossover direction is what we projected:**
- Full category set (~300k hierarchy edges).
- Articles: top-100k by category count (deterministic, reproducible,
  no embedding dependency).
- Effective-distance workload, Science root, max_depth=20.
- Both backends, FFI kernels, same seeds for both runs.

If IntMap still wins comfortably, crossover is past 100k and we should
jump to ~1M or the full 383k-article set to see it.

If LMDB closes the gap (< 1.29x), we've started hitting GC territory in
IntMap and can measure GC pause time directly.

**Phase 2 — variant axis:**
- Descendant-of-root vs full category set, same article filter. Tests
  how orphan categories affect DFS — the descendant-filtered run should
  be faster for both backends because the graph is smaller, but the
  ratio between backends is what's interesting.

**Phase 3 — only if phases 1–2 motivate it:**
- Top-k semantic filter (embeddings) to compare "rich" seeds against
  "shallow" seeds. Probably doesn't change the scaling story, but
  affects per-seed workload variance — relevant for parallelism.

## Embedding-based filtering (for 1M+ scale)

Not needed at 100k — deferred notes:

- Cohere's Wikipedia embeddings on HuggingFace
  (`Supabase/wikipedia-en-embeddings`) use MiniLM embeddings. Locally
  cached folder exists but parquet data is not populated.
- The typical pipeline for "rich" embeddings over a subset: **filter
  top-k with MiniLM first, then recompute with a heavier model
  (BGE/E5/nomic) on the smaller set**. We have BGE, E5, ModernBERT,
  nomic-embed-text cached locally (`~/.cache/huggingface/hub`).
- For benchmark reproducibility, prefer deterministic ranking (e.g.,
  MiniLM cosine to a fixed centroid) over randomness.

## Category-only workloads (small scales)

At ≤100k we don't need articles in the graph at all — simplewiki's
full category hierarchy (~97k categories / ~297k child→parent edges)
is already near the 100k target. Shape for a category-only run:

- `category_parent.tsv`: full hierarchy unchanged.
- `article_category.tsv`: synthesized as `(C, C)` for each seed
  category so the existing `effective_distance` workload still has
  "article" seeds to iterate — each seed enters `category_ancestor`
  at exactly its own category.
- `root_categories.tsv`: one or more well-connected top-level
  categories (e.g., `Main_topic_classifications`) that reach most of
  the hierarchy.

This keeps the `effective_distance.pl` workload shape unchanged —
only the data it runs on differs.

## Data generation considerations

- **Reproducibility**: prefer deterministic article selection (top-k by
  some fixed metric, or fixed-seed random sample) over API-driven
  crawls. The simplewiki SQLite DB at
  `context/gemini/UnifyWeaver/data/simplewiki/simplewiki_categories.db`
  is already a frozen snapshot — use it.
- **Correctness check**: both backends must produce identical query
  results. Already enforced at 10k — repeat at every scale.
- **Idempotent ingestion**: LMDB directory should be reused across runs
  once populated. First run pays ingestion cost; subsequent runs don't.
- **Cold vs warm**: `echo 3 > /proc/sys/vm/drop_caches` (requires sudo)
  to force cold pages. Without that, "warm" is the default after first
  run.

## Profiling considerations

- GHC `-p`: watch for `peekElemOff`, `mdb_get'`, `mapM` showing up in
  the LMDB run. In the IntMap run, watch for `IntMap.lookup` and GC
  time.
- RTS `-s`: compare total GC time between backends at each scale. This
  is the signal for LMDB's eventual win.
- `+RTS -hT`: heap profile by closure type. IntMap heap residency
  should grow roughly linearly with graph size; LMDB stays near-flat.

## Optimization hypotheses (to evaluate, not pre-commit)

Neither should be implemented until the profile justifies it.

- **LRU hot-key cache over LMDB**: effective-distance DFS revisits
  high-degree nodes often. If `mdb_get'` dominates the LMDB profile,
  a small cache (1k-10k entries) should help. Risk: adds complexity
  for narrow gain if profile shows the bottleneck is elsewhere.
- **Demand-set pre-materialization**: walk the demand set once via
  LMDB, build a transient IntMap of just the touched keys, query
  against that. Half-defeats the point of LMDB (working set must fit
  in RAM) but useful when demand sets are small relative to the full
  graph.

## What we're NOT testing (yet)

- SQLite backend (Termux target — deferred to B2)
- Raw mmap `.uwbr` artifact (deferred to B3)
- Multi-predicate workloads (category_ancestor is the only FFI-backed
  predicate at scale right now)
- Write throughput (everything here is read-only after ingestion)

## Open questions

- Does the GC cost of a 300k-edge IntMap actually show up in total
  query time, or does generational GC promote it to old-gen once and
  forget about it?
- Is the long-lived read transaction's page-cache pinning helping or
  hurting at larger scales? At 1M+ it could pin more than we want.
- For the "orphan" variant — do disconnected categories just sit
  unqueried (zero cost) or do they somehow slow down lookups in both
  backends?
