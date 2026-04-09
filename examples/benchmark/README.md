# Cross-Target Effective Distance Benchmark

## Overview

Computes effective distance `d_eff = (Σ dᵢ^(-n))^(-1/n)` from Wikipedia
articles to root categories via the category hierarchy, compiled to
multiple targets. Demonstrates UnifyWeaver's core value: write Prolog once,
compile to the best target for your scale.

See `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_*.md` for theory
and specification.

## Benchmark Results

### Accumulated Prolog and Query Engine vs DFS Pipelines

The effective-distance benchmark now has a dedicated C# query-runtime
weight-sum path. Instead of materializing all `(seed, ancestor, hops)`
rows and regrouping them in the benchmark harness, the runtime now:

- streams `category_parent` and `article_category` into the engine
- reuses path-aware edge state inside `QueryRuntime`
- computes per-article/per-root weight sums directly
- emits compact `(article, root, weight_sum)` rows for the final
  `d_eff` calculation

That moves the retained state into the engine and flips the old headline:
the query engine is now faster than accumulated Prolog and all DFS
pipelines across the full benchmark range while preserving the same
all-path semantics. It also now supports cost-guided selection between
compact grouped weight sums and the legacy seeded-row regrouping path,
using the same grouped-summary policy layer that now also drives the
shortest-path minima family. The path-aware family now coordinates that
grouped-summary selector with the earlier edge-retention selector through the
shared internal materialization planner layer that also now covers the DAG
family's relation-retention planning and the generic scan family used by
`RelationScanNode`, `PatternScanNode`, and scan-heavy join/negation/aggregate
consumers. That policy now records measured cost buckets like `load_roots`,
`load_seeds`, `strategy_select`, `build_*`, and `group_reduce`, while the
earlier path-aware edge-retention boundary now also records buckets like
`edge_strategy_select`, `edge_probe_*`, `edge_materialize_replayable`, and
`edge_build_*`. Generic scan planning now adds corresponding scan buckets such
as `scan_strategy_select`, `scan_probe_*`, `scan_materialize_*`, and
`scan_build_fact_set`.

| Target | 300 art | 1K art | 5K art | 10K art |
|--------|---------|--------|--------|---------|
| **C# Query Engine** | **0.279s** | **0.229s** | **0.453s** | **0.902s** |
| Prolog accumulated | 0.488s | 0.333s | 1.040s | 2.283s |
| C# DFS pipeline | 0.509s | 1.465s | 6.494s | 11.863s |
| Rust DFS pipeline | 0.413s | 1.718s | 9.761s | 16.344s |
| Go DFS pipeline | 0.580s | 2.496s | 15.875s | 24.587s |

**Speedups of the current C# query engine:**

| Scale | vs Prolog accumulated | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------------------:|----------:|------------:|----------:|
| 300 art | 1.75x | 1.83x | 1.48x | 2.08x |
| 1K art | 1.45x | 6.40x | 7.50x | 10.90x |
| 5K art | 2.30x | 14.34x | 21.55x | 35.04x |
| 10K art | 2.53x | 13.15x | 18.12x | 27.26x |

Semantic note:

- the current C# query-engine path still preserves per-path cycle
  checks without collapsing distinct simple paths that happen to reach
  the same ancestor with the same hop count
- `benchmark_effective_distance.py` compares the query-engine output
  against both the C# DFS reference and accumulated Prolog, reporting
  `query_vs_csharp_dfs` and `query_vs_prolog_accumulated`
- current runs report `query_vs_csharp_dfs = match` at `300`, `1k`,
  `5k`, and `10k`
- current runs also report `query_vs_prolog_accumulated = match` at
  `300`, `1k`, `5k`, and `10k`

### Why Seeded Closures Win

The main advantage comes from **seed deduplication**: many articles
share the same categories, so the benchmark computes ancestors once per
unique category seed, not once per article.

| Scale | Articles | Unique category seeds | Redundancy factor |
|-------|----------|-----------------------|-------------------|
| 300 | 289 | 386 | ~0.7x (more seeds than articles) |
| 1K | 1,000 | 89 | ~11x |
| 5K | 5,000 | 284 | ~18x |
| 10K | 10,000 | 888 | ~11x |

At 1K scale, 1000 articles map to only 89 unique category seeds — the
seeded benchmark does ~11x less seed expansion work. Even with full
per-path semantics restored, the precomputed ancestor index still makes
per-article aggregation much cheaper than rerunning DFS from scratch for
each article. The current Prolog effective-distance path now uses that
same strategy directly.

### DFS Pipeline Execute Time Scaling

For historical context, the original DFS-only pipeline comparison:

| Target | 300 art | 1K art | 5K art | 10K art | Trend |
|--------|---------|--------|--------|---------|-------|
| **C#** | 0.43s | **1.13s** | **4.74s** | **9.48s** | Scales best among DFS pipelines |
| **Rust** | **0.33s** | 1.33s | 6.86s | 12.44s | Fastest at small scale, overtaken by C# |
| **Go** | 0.43s | 1.96s | 11.36s | 18.71s | Falls behind at scale |
| **Codon** | 0.67s | 2.55s | 10.98s | 22.14s | Compiled Python, approaching Go |
| **CPython** | 0.73s | 2.93s | 15.71s | — | Dropped at 10K |
| **Prolog** | 1.26s | — | — | — | Dropped: needs demand analysis |
| **AWK** | 2.46s | — | — | — | Dropped: interpreter overhead |

### Compile Time (file-based loading)

| Target | 300 art | 1K art | 5K art | 10K art |
|--------|---------|--------|--------|---------|
| Go | 0.50s | 0.35s | 0.13s | 0.17s |
| Rust | 0.65s | 0.72s | 0.31s | 0.36s |
| C# | 3.00s | 2.94s | 1.21s | 2.76s |
| Codon | 4.16s | 4.75s | 4.00s | 5.53s |

### Key Findings

1. **The C# query engine is still the fastest effective-distance path on
   this benchmark surface** — it beats accumulated Prolog by `1.75x` at
   `300`, `1.45x` at `1k`, `2.30x` at `5k`, and `2.53x` at `10k`.

2. **The C# query engine still beats the DFS baselines by a wide margin** —
   at `10k` it is `13.15x` faster than C# DFS, `18.12x` faster than Rust
   DFS, and `27.26x` faster than Go DFS.

3. **Retained-state strategy matters more than branch pruning on this
   workload** — seeded reuse is the first big win, and moving per-article
   root-weight aggregation into the query runtime removes the huge path-row
   materialization cost that used to dominate the C# query path. The new
   `Auto` selector also chooses the compact grouped weight-sum path on the
   tested scales; forcing legacy seeded-row regrouping is dramatically
   worse (`300`: `0.726s` vs `0.275s`, `10k`: `3.106s` vs `0.828s`).

4. **Accumulated Prolog is still a useful retained-state reference** — it
   keeps only per-seed weight sums (`48` tuples at `1k`, `151` at `5k`,
   `462` at `10k`), but the query engine still reaches a better end-to-end
   result by combining streamed ingestion with operator-owned retained
   state.

5. **The new direct article/root Prolog helper is currently experimental**
   — it emits compact `(article, root, weight_sum)` rows directly, but it
   is slower than the seed-accumulated path on this workload, so it
   remains an optional benchmark variant rather than the default.

6. **The current C# query-engine path now matches the intended ownership
   boundary much more closely** — the parser streams tuples, and the engine
   decides what retained form the effective-distance operator actually needs.

### Target Recommendations by Audience

| Audience | Recommended Target | Why |
|----------|-------------------|-----|
| SWI / Prolog-first | Accumulated Prolog | Strong retained-state reference and compact effective-distance helper |
| Enterprise / .NET | C# Query Engine | Fastest current effective-distance path, with engine-owned retained state |
| Cloud / DevOps | Go | Fast compile, good ecosystem |
| Systems / Embedded | Rust | No runtime overhead, strong DFS baseline |

## Data Provenance

### Source: Simple English Wikipedia

All data comes from **Simple English Wikipedia** database dumps, downloaded
from `https://dumps.wikimedia.org/simplewiki/latest/`:

| File | Size | Contents |
|------|------|----------|
| `simplewiki-latest-page.sql.gz` | ~32 MB | Page table (page_id, title, namespace) |
| `simplewiki-latest-categorylinks.sql.gz` | ~27 MB | Category links (page_id → target_id, type) |
| `simplewiki-latest-linktarget.sql.gz` | ~36 MB | Link target resolution (target_id → title) |

These are stored in `data/simplewiki/` (not committed — too large).

### Why Simple Wikipedia?

The Cohere Wikipedia embeddings dataset (`Supabase/wikipedia-en-embeddings`)
used elsewhere in this project is sourced from Simple English Wikipedia.
Using simplewiki category data ensures article titles match.

### Schema Note (MediaWiki post-2024)

The categorylinks table uses a new schema where `cl_to` (category name string)
was replaced by `cl_target_id` (a reference to the `linktarget` table). The
parser resolves this JOIN:

    categorylinks.cl_target_id → linktarget.lt_id → linktarget.lt_title

### SQLite Database

Parsed dumps are stored in `data/simplewiki/simplewiki_categories.db`:

```
Tables:
  page             - 483K rows (articles ns=0: 392K, categories ns=14: 92K)
  linktarget       - 336K rows (category namespace targets only)
  categorylinks_raw - 2.2M rows (raw cl_from → cl_target_id)
  categorylinks    - 2.2M rows (resolved: cl_from → category_name)
    page:   1.9M (article → category)
    subcat: 297K (category → parent category)
    file:   250
```

## Scripts

| Script | Purpose |
|--------|---------|
| `parse_simplewiki_dump.py` | Parse Wikipedia SQL dumps → SQLite |
| `generate_facts_from_db.py` | Extract facts from SQLite → Prolog/TSV (no crawling) |
| `generate_facts.py` | Alternative: fetch from Wikipedia API (for small datasets) |
| `generate_pipeline.py` | Generate self-contained pipeline per target |
| `benchmark_common.py` | Shared build/run utilities for cross-target benchmark runners |
| `compute_effective_distance.py` | Post-processing aggregation (validation tool) |
| `benchmark_effective_distance.py` | Rebuild and time effective distance across the C# query engine, accumulated Prolog, optional direct article/root and root-bound Prolog variants, and C#/Rust/Go DFS binaries |
| `benchmark_shortest_path_cross_target.py` | Compare shortest-path-to-root across C# query, seeded Prolog `min`, C# DFS, Rust DFS, and Go DFS |
| `benchmark_dependency_depth_cross_target.py` | Compare synthetic dependency reach-count across C# query, C# DFS, Rust DFS, and Go DFS |
| `benchmark_dependency_longest_depth_cross_target.py` | Compare true DAG longest dependency-chain depth across C# query, C# DFS, Rust DFS, and Go DFS |
| `benchmark_path_aware_accumulation.py` | Measure counted-closure vs generalized accumulation overhead |
| `benchmark_weighted_shortest_path.py` | Measure `PathAwareAccumulationNode` `All` vs `Min` pruning on positive weighted paths |
| `benchmark_weighted_shortest_path_cross_target.py` | Compare positive weighted shortest path across C# query, seeded Prolog `min`, C# DFS, Rust DFS, and Go DFS |
| `benchmark_category_influence_cross_target.py` | Compare category influence propagation across the C# query engine, Rust DFS, Go DFS, and an optional Prolog accumulated path |
| `generate_prolog_shortest_path_benchmark.pl` | Generate standalone SWI-Prolog shortest-path benchmark scripts with `branch_pruning(auto|false)` |
| `benchmark_prolog_branch_pruning.py` | Compare handwritten Prolog shortest-path source against generated pruned and unpruned Prolog scripts |
| `generate_prolog_effective_distance_benchmark.pl` | Generate standalone SWI-Prolog effective-distance scripts for seeded closure reuse, generated accumulation helpers, optional direct article/root helpers, and optional branch pruning |
| `benchmark_prolog_effective_distance.py` | Compare seeded, pruned, accumulated, and optional direct article/root and root-bound Prolog effective-distance scripts and report phase/work metrics |
| `generate_prolog_category_influence_benchmark.pl` | Generate standalone SWI-Prolog category-influence scripts using the PPV `category_ancestor/4` closure with optional seeded accumulation helpers |
| `benchmark_prolog_category_influence.py` | Compare seeded and accumulated Prolog category-influence scripts and report phase/work metrics |
| `generate_prolog_shortest_path_seeded_benchmark.pl` | Generate standalone SWI-Prolog shortest-path scripts for seeded `all` vs mode-directed `min` closure, loading `facts.pl` at runtime |
| `benchmark_prolog_seeded_min_closure.py` | Compare seeded Prolog `all` vs `min` closure and report `load_ms`, `query_ms`, `aggregation_ms`, and work metrics |
| `generate_prolog_weighted_shortest_path_benchmark.pl` | Generate standalone SWI-Prolog weighted-shortest-path scripts for seeded `all` vs mode-directed `min` closure |
| `benchmark_prolog_weighted_min_closure.py` | Compare seeded Prolog weighted `all` vs `min` closure and report `load_ms`, `query_ms`, `aggregation_ms`, and work metrics |
| `generate_dependency_benchmark_data.py` | Generate deterministic synthetic package-DAG benchmark data |
| `effective_distance.pl` | Benchmark Prolog program |
| `run_benchmark.sh` | Compile all targets + generate reference output |

Packaging note:

- the shared `generate_pipeline.py` generator now owns packaged C#
  benchmark output for the cross-target runners, including:
  - `Program.cs`
  - `QueryRuntime.cs` where needed for `csharp_query`
  - `benchmark.csproj`
- the benchmark runners are correspondingly narrower now: they mainly
  generate, build, run, and compare, rather than assembling one-off C#
  projects in-script
- shared Rust/Go/C# runner utilities now live in `benchmark_common.py`
  so new workload runners do not need to reimplement the same temp/build
  scaffolding
- `benchmark_common.py` now also carries shared output-digest and summary
  helpers, so only workload-specific normalization and extra reporting
  stay in the individual runners
- safe shared normalization helpers also live there now for:
  - exact sorted-line outputs
  - rounded float row outputs in common 2-column and 3-column forms
- workloads can still keep custom normalization when their semantics or
  ordering rules are genuinely different
- the seeded Prolog effective-distance, shortest-path, and
  weighted-shortest-path benchmarks now load `facts.pl` at runtime from
  a small generated driver instead of inlining facts into the generated
  script, so `load_ms` is reported separately from `query_ms`

## Usage

```bash
# One-time: parse dumps into SQLite
python examples/benchmark/parse_simplewiki_dump.py

# Generate dataset (e.g., 5000 articles, Physics root)
python examples/benchmark/generate_facts_from_db.py \
    --output data/benchmark/5k/ \
    --max-articles 5000 \
    --root Physics

# Generate pipeline for a target
python examples/benchmark/generate_pipeline.py \
    --facts data/benchmark/5k/facts.pl --root Physics \
    --target go --output pipelines/effective_distance.go \
    --max-depth 10

# Compile and run (file-based loading)
go build -o bench pipelines/effective_distance.go
./bench data/benchmark/5k/category_parent.tsv data/benchmark/5k/article_category.tsv

# Re-run the current non-regression benchmark
python examples/benchmark/benchmark_effective_distance.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-accumulated

# Compare seeded, pruned, and accumulated effective-distance Prolog variants
python examples/benchmark/benchmark_prolog_effective_distance.py \
    --scales 300,1k,5k,10k \
    --targets prolog-seeded,prolog-pruned,prolog-accumulated

# Compare minimum-hop shortest path across query engine and DFS targets
python examples/benchmark/benchmark_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min

# Compare dependency reach count across query engine and DFS targets
python examples/benchmark/benchmark_dependency_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs

# Compare true DAG longest dependency depth across query engine and DFS targets
python examples/benchmark/benchmark_dependency_longest_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs

# Measure overhead of the generalized path-aware accumulation runtime
python examples/benchmark/benchmark_path_aware_accumulation.py \
    --scales 300,1k,5k,10k

# Measure directed-table pruning on weighted path-aware accumulation
python examples/benchmark/benchmark_weighted_shortest_path.py \
    --scales 300,1k,5k,10k

# Compare positive weighted minimum path distance across query engine and DFS targets
python examples/benchmark/benchmark_weighted_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min

# Compare category influence propagation across the C# query engine, Rust, and Go
python examples/benchmark/benchmark_category_influence_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,rust-dfs,go-dfs

# Compare seeded vs accumulated Prolog category influence
python examples/benchmark/benchmark_prolog_category_influence.py \
    --scales 300,1k \
    --targets prolog-seeded,prolog-accumulated

# Optional: add the accumulated Prolog path to the category-influence cross-target run
python examples/benchmark/benchmark_category_influence_cross_target.py \
    --scales 300,1k \
    --targets csharp-query,rust-dfs,go-dfs,prolog-accumulated

# Compare handwritten Prolog vs generated pruned and unpruned Prolog
python examples/benchmark/benchmark_prolog_branch_pruning.py \
    --scales 300,1k,5k,10k \
    --targets prolog-source,prolog-pruned,prolog-unpruned

# Compare seeded Prolog all-path closure vs mode-directed min closure
python examples/benchmark/benchmark_prolog_seeded_min_closure.py \
    --scales 300,1k,5k,10k \
    --targets prolog-all,prolog-min

# Compare seeded Prolog weighted all-path closure vs mode-directed min closure
python examples/benchmark/benchmark_prolog_weighted_min_closure.py \
    --scales 300,1k,5k,10k \
    --targets prolog-all,prolog-min
```

The current comparison surface across query and non-query targets is:

- C# query:
  - all-path effective distance
  - minimum hop distance
  - dependency reach count
  - positive weighted minimum path distance
  - category influence propagation

- Rust DFS:
  - all-path effective distance
  - minimum hop distance
  - dependency reach count
  - dependency longest depth
  - positive weighted minimum path distance
  - category influence propagation
- Go DFS:
  - all-path effective distance
  - minimum hop distance
  - dependency reach count
  - dependency longest depth
  - positive weighted minimum path distance
  - category influence propagation
- Prolog seeded closures and accumulations:
  - all-path effective distance
  - minimum hop distance
  - positive weighted minimum path distance

The benchmark split is now:

- cross-target:
  - `benchmark_effective_distance.py`
  - `benchmark_shortest_path_cross_target.py`
  - `benchmark_dependency_depth_cross_target.py`
  - `benchmark_dependency_longest_depth_cross_target.py`
  - `benchmark_weighted_shortest_path_cross_target.py`
  - `benchmark_category_influence_cross_target.py`
- Prolog target:
  - `benchmark_prolog_effective_distance.py`
  - `benchmark_prolog_branch_pruning.py`
  - `benchmark_prolog_seeded_min_closure.py`
  - `benchmark_prolog_weighted_min_closure.py`
- C# query-engine internal mode comparison:
  - `benchmark_shortest_path_to_root.py`
  - `benchmark_weighted_shortest_path.py`

### Prolog Effective-Distance Retained-State Results

Command:

```bash
python examples/benchmark/benchmark_prolog_effective_distance.py \
    --scales 300,1k,5k,10k --repetitions 1
```

Latest local results:

| Scale | Seeded | Pruned | Accumulated | Output Match | Note |
|-------|--------|--------|-------------|--------------|------|
| 300 | 0.380s | 0.416s | 0.369s | match | accumulated edges out seeded once load is amortized |
| 1k | 0.293s | 0.364s | 0.257s | match | accumulated remains the best Prolog path |
| 5k | 1.176s | 1.096s | 0.850s | match | retained-state reduction dominates |
| 10k | 2.627s | 2.577s | 2.249s | match | accumulated stays best at the largest tested scale |

Current interpretation:

- seeded closure reuse is the real win on effective distance
- branch pruning does not currently reduce tuple count or inferences on this workload
- the accumulated variant now uses the generated selected helper surface,
  which routes bound root queries to a dedicated bound-key fast path
  instead of paying the generic helper's key-enumeration overhead
- the experimental `article_accumulated` variant now uses the generated
  `category_ancestor$effective_distance_article_sum` helper to emit
  compact `(article, root, weight_sum)` rows directly, but it is slower
  than `accumulated` on this workload:
  - `300`: `0.457s` vs `0.393s`
  - `1k`: `0.493s` vs `0.274s`
  - `5k`: `6.275s` vs `0.973s`
- the new `root_accumulated` profiling path uses the generated
  `category_ancestor$effective_distance_article_sum_by_root/3` and
  `category_ancestor$effective_distance_article_sum_pairs_by_root/2`
  helpers to benchmark the bound-root path directly
- that root-bound path is competitive at smaller scales but not the best
  default retained-state plan overall:
  - `300`: `0.360s` vs accumulated `0.354s`
  - `1k`: `0.236s` vs accumulated `0.288s`
  - `5k`: `0.934s` vs accumulated `0.797s`
  - `10k`: `2.629s` vs accumulated `1.943s`
- pre-aggregating per-seed weight sums materially reduces retained state:
  - `1k`: tuple count `10976 -> 48`
  - `5k`: tuple count `41132 -> 151`
  - `10k`: tuple count `105287 -> 462`
- the bound-key fast path also reduces query work for accumulated Prolog:
  - `1k`: inferences `4078570 -> 3686379`
  - `5k`: inferences `16176114 -> 14105020`
  - `10k`: inferences `40693380 -> 37778802`
- the seeded, pruned, and accumulated variants are semantically identical at all tested scales

### Prolog Branch-Pruning Results

Command:

```bash
python examples/benchmark/benchmark_prolog_branch_pruning.py \
    --scales 300,1k,5k,10k --repetitions 1
```

Latest local results:

| Scale | Source | Pruned | Unpruned | Output Match | Note |
|-------|--------|--------|----------|--------------|------|
| 300 | 2.718s | 2.748s | 2.804s | match | pruning overhead slightly visible |
| 1k | 1.694s | 1.786s | 1.776s | match | still slightly slower than source |
| 5k | 8.916s | 8.968s | 8.915s | match | essentially neutral |
| 10k | 17.530s | 17.490s | 17.475s | match | effectively tied |

This is a useful validation result even though it is not yet a strong
speedup result: the generated pruned and unpruned Prolog variants match
the handwritten source output across all tested dataset scales, and the
current shortest-path workload gives a stable baseline for future
benchmarks that should stress pruning more directly.

### Prolog Seeded Min-Closure Results

Command:

```bash
python examples/benchmark/benchmark_prolog_seeded_min_closure.py \
    --scales 300,1k,5k,10k --repetitions 1
```

Latest local results:

| Scale | Prolog All | Prolog Min | Output Match |
|-------|-----------:|-----------:|--------------|
| 300 | 0.405s | 0.109s | match |
| 1k | 0.283s | 0.115s | match |
| 5k | 0.964s | 0.246s | match |
| 10k | 2.294s | 0.491s | match |

Speedups of seeded Prolog `min` over seeded `all`:

| Scale | Speedup |
|-------|--------:|
| 300 | 3.71x |
| 1k | 2.45x |
| 5k | 3.92x |
| 10k | 4.67x |

The retained work also drops sharply:

| Scale | `all` tuple count | `min` tuple count |
|-------|------------------:|------------------:|
| 300 | 11172 | 213 |
| 1k | 10976 | 48 |
| 5k | 41132 | 151 |
| 10k | 105287 | 462 |

### Cross-Target Shortest-Path Results

Command:

```bash
python examples/benchmark/benchmark_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min \
    --repetitions 1
```

Latest local results:

| Scale | C# Query | Prolog Min | C# DFS | Rust DFS | Go DFS | Outputs |
|-------|---------:|-----------:|--------:|---------:|-------:|---------|
| 300 | 0.117s | 0.136s | 0.537s | 0.418s | 0.600s | match |
| 1k | 0.107s | 0.112s | 1.465s | 1.718s | 2.531s | match |
| 5k | 0.148s | 0.278s | 6.434s | 8.819s | 15.760s | match |
| 10k | 0.212s | 0.510s | 12.139s | 16.450s | 25.110s | match |

Direct C# query vs Prolog seeded `min`:

| Scale | Faster target | Speedup |
|-------|---------------|--------:|
| 300 | C# Query | 1.15x |
| 1k | C# Query | 1.05x |
| 5k | C# Query | 1.87x |
| 10k | C# Query | 2.40x |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 4.57x | 3.56x | 5.11x |
| 1k | 13.73x | 16.10x | 23.72x |
| 5k | 43.34x | 59.41x | 106.16x |
| 10k | 57.21x | 77.53x | 118.35x |

Comparison note:

- the query runtime now emits compact `(article, root, min_depth)` rows
  directly instead of materializing broad seeded ancestor rows and then
  regrouping them in the benchmark harness
- on the current one-root benchmark shape, the retained row count now
  collapses to the final article result count, which is why the C# query
  path improves so sharply at every tested scale
- the new `Auto` selector still chooses the compact grouped minima
  path at every tested scale; this now flows through the same grouped-summary
  policy layer used by grouped weight sums, while still preserving the
  per-family override knob
- the path-aware edge relation now also goes through measured retention
  selection, and on the current benchmark surface `Auto` prefers
  `ReplayableBuffer`; that selection now runs through the same internal
  relation-retention policy layer that also drives the DAG family, while the
  combined path-aware materialization decision now flows through the shared
  internal planner layer and the benchmark-visible override knobs stay
  separate
- trace output now exposes edge buckets such as
  `edge_strategy_select`, `edge_probe_streaming_direct`,
  `edge_probe_replayable_buffer`, `edge_materialize_replayable`, and
  `edge_build_replayable_buffer`
- forcing legacy seeded-row regrouping is
  materially worse (`300`: `0.160s` vs `0.083s`, `10k`: `0.419s` vs
  `0.161s`)
- seeded Prolog `min` remains competitive, but the C# query engine is
  now faster at every tested scale, including `300`
- output digests match across all five targets at every tested scale

### Cross-Target Weighted Shortest-Path Results

Command:

```bash
python examples/benchmark/benchmark_weighted_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min \
    --repetitions 1
```

Latest local results:

| Scale | C# Query | Prolog Min | C# DFS | Rust DFS | Go DFS | Outputs |
|-------|---------:|-----------:|--------:|---------:|-------:|---------|
| 300 | 0.190s | 0.134s | 0.528s | 0.386s | 0.628s | match |
| 1k | 0.175s | 0.155s | 1.450s | 1.730s | 2.622s | match |
| 5k | 0.246s | 0.305s | 6.742s | 9.137s | 15.661s | match |
| 10k | 0.414s | 0.636s | 12.798s | 17.239s | 24.495s | match |

Direct C# query vs Prolog seeded `min`:

| Scale | Faster target | Speedup |
|-------|---------------|--------:|
| 300 | Prolog Min | 1.42x |
| 1k | Prolog Min | 1.13x |
| 5k | C# Query | 1.24x |
| 10k | C# Query | 1.54x |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 2.78x | 2.03x | 3.31x |
| 1k | 8.30x | 9.90x | 15.01x |
| 5k | 27.39x | 37.12x | 63.62x |
| 10k | 30.93x | 41.66x | 59.20x |

Comparison note:

- the query runtime now emits compact `(article, root, min_weight)` rows
  directly instead of materializing broad seeded weighted-path rows and
  regrouping them afterward in the benchmark harness
- on the current one-root benchmark shape, the retained row count now
  also collapses to the final article result count
- the new `Auto` selector also chooses the compact grouped minima path
  here; this now uses the same grouped-summary policy layer as shortest
  path while keeping a separate benchmark override
- the path-aware edge relation also now goes through measured retention
  selection, and the same edge buckets are available under trace for this
  family too
- forcing legacy seeded-row regrouping regresses both ends of the
  benchmark (`300`: `0.202s` vs `0.151s`, `10k`: `0.430s` vs `0.320s`)
- seeded Prolog `min` still wins narrowly at `300` and `1k`, but the C#
  query engine is now clearly faster from `5k` onward
- the cross-target weighted benchmark normalizes floating-point outputs
  with a tolerance appropriate for cross-language evaluation order
  differences

### Prolog Weighted Min-Closure Results

Command:

```bash
python examples/benchmark/benchmark_prolog_weighted_min_closure.py \
    --scales 300,1k,5k,10k --repetitions 1
```

Latest local results:

| Scale | Prolog All | Prolog Min | Output Match |
|-------|-----------:|-----------:|--------------|
| 300 | 0.441s | 0.117s | match |
| 1k | 0.303s | 0.129s | match |
| 5k | 1.189s | 0.292s | match |
| 10k | 2.762s | 0.552s | match |

Speedups of seeded Prolog weighted `min` over seeded `all`:

| Scale | Speedup |
|-------|--------:|
| 300 | 3.78x |
| 1k | 2.35x |
| 5k | 4.08x |
| 10k | 5.01x |

The retained work also drops sharply:

| Scale | `all` tuple count | `min` tuple count |
|-------|------------------:|------------------:|
| 300 | 11172 | 213 |
| 1k | 10976 | 48 |
| 5k | 41132 | 151 |
| 10k | 105287 | 462 |

### Cross-Target Dependency Reach Results

Command:

```bash
python examples/benchmark/benchmark_dependency_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs \
    --repetitions 1
```

This workload uses a deterministic synthetic package DAG and reports, for
each project, the size of its unique transitive dependency closure. It is
named `dependency_depth` in the generator and runner for continuity with
the original benchmark idea, but the current metric is more precisely a
dependency **reach count** than a maximum-chain depth.

Latest local results:

| Scale | C# Query | C# DFS | Rust DFS | Go DFS | Query vs C# DFS |
|-------|---------:|--------:|---------:|-------:|-----------------|
| 300 | 0.081s | 0.051s | 0.002s | 0.003s | match |
| 1k | 0.102s | 0.056s | 0.008s | 0.008s | match |
| 5k | 0.090s | 0.183s | 0.157s | 0.118s | match |
| 10k | 0.098s | 0.583s | 0.683s | 0.501s | match |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 0.63x | 0.03x | 0.03x |
| 1k | 0.55x | 0.08x | 0.08x |
| 5k | 2.03x | 1.74x | 1.31x |
| 10k | 5.95x | 6.96x | 5.11x |

Comparison note:

- the current C# query path uses a project-grouped reachability node
  with direct count aggregation inside the query runtime
- relation ingestion now also goes through a measured DAG retention
  selector, and the generated benchmarks expose
  `UNIFYWEAVER_DAG_RETENTION_STRATEGY=auto|streaming|replayable|external`
- the DAG selector now shares the same internal relation-retention policy
  layer as path-aware edge retention, and now also flows through the same
  shared internal materialization planner framework while preserving a
  DAG-specific public override surface
- on the current synthetic benchmark surface, `Auto` picks
  `StreamingDirect` for the reach-count DAG path; at smaller scales it can
  use bounded probes (`dag_probe_streaming_direct`,
  `dag_probe_replayable_buffer`), and at larger scales it short-circuits
  structurally
- trace output now exposes DAG retention buckets such as
  `dag_strategy_select`, `dag_probe_*`, `dag_materialize_replayable_*`,
  and `dag_build_*`
- for the one-shot generated benchmark programs, disabling query-runtime
  cache reuse still helps slightly by removing bookkeeping that is not
  reused within a single run
- this keeps the benchmark closer to the intended ownership boundary:
  the parser streams tuples, and the engine decides what state to retain
  for the operator
- the query engine is still behind at `300` and `1k`, but is clearly
  ahead of all DFS targets from `5k` onward
- outputs still match across all four targets

### Cross-Target Dependency Longest-Depth Results

Command:

```bash
python examples/benchmark/benchmark_dependency_longest_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs \
    --repetitions 1
```

This workload is the true DAG longest-path version of the dependency
benchmark family. For each project, it reports the maximum dependency
chain length over the synthetic package DAG.

Latest local results:

| Scale | C# Query | C# DFS | Rust DFS | Go DFS | Query vs C# DFS |
|-------|---------:|--------:|---------:|-------:|-----------------|
| 300 | 0.070s | 0.044s | 0.002s | 0.002s | match |
| 1k | 0.069s | 0.044s | 0.003s | 0.004s | match |
| 5k | 0.086s | 0.052s | 0.008s | 0.006s | match |
| 10k | 0.085s | 0.064s | 0.014s | 0.012s | match |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 0.63x | 0.03x | 0.03x |
| 1k | 0.63x | 0.04x | 0.05x |
| 5k | 0.61x | 0.09x | 0.07x |
| 10k | 0.76x | 0.16x | 0.14x |

Comparison note:

- this benchmark includes a real `csharp-query` DAG longest-depth path
  built on a dedicated query-runtime node
- DAG relation ingestion now also goes through a measured retention
  selector, and the generated benchmarks expose
  `UNIFYWEAVER_DAG_RETENTION_STRATEGY=auto|streaming|replayable|external`
- on the current benchmark surface, `Auto` also picks
  `StreamingDirect` for longest depth; at smaller scales it can use
  bounded probes, and at larger scales it short-circuits structurally
- set `UNIFYWEAVER_BENCH_TRACE=1` when you want the generated
  `csharp-query` longest-depth benchmark to print both per-phase DAG
  timings and DAG retention strategy buckets to `stderr`
- cache reuse remains disabled for these one-shot generated benchmark
  programs, and trace creation is now opt-in rather than always-on
- the hand-written C# DFS baseline is still cheaper after its lighter
  custom TSV loader work, but the query engine now sits closer to that
  baseline while keeping the intended retention boundary in the engine
  rather than the loader
- outputs still match across all four targets, and longest depth remains
  a separate optimization track from reach-count

### Cross-Target Category Influence Results

Command:

```bash
python examples/benchmark/benchmark_category_influence_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,rust-dfs,go-dfs \
    --repetitions 1
```

Latest local results:

| Scale | C# Query | Rust DFS | Go DFS | Outputs |
|-------|---------:|---------:|-------:|---------|
| 300 | 0.320s | 0.469s | 1.516s | match |
| 1k | 0.340s | 2.861s | 2.636s | match |
| 5k | 0.455s | 9.366s | 14.989s | match |
| 10k | 0.882s | 16.925s | 25.592s | match |

Speedups of C# query engine:

| Scale | vs Rust DFS | vs Go DFS |
|-------|------------:|----------:|
| 300 | 1.47x | 4.74x |
| 1k | 8.42x | 7.75x |
| 5k | 20.57x | 32.92x |
| 10k | 19.19x | 29.01x |

Comparison note:

- the benchmark now uses the same retained-state idea as effective distance:
  `QueryRuntime` emits compact `(article, root, weight_sum)` rows directly
  instead of materializing all ancestor/hop rows and then summing them later
- the remaining benchmark-side work is only the final per-root sum over those
  compact article/root summaries
- the new `Auto` selector also chooses the compact grouped weight-sum path
  here
- the path-aware edge relation now also goes through measured retention
  selection, and trace output exposes both grouped-summary buckets such as
  `load_roots`, `load_seeds`, `strategy_select`, `build_compact_grouped`, and
  `group_reduce`, plus edge buckets such as `edge_strategy_select`,
  `edge_probe_streaming_direct`, `edge_probe_replayable_buffer`,
  `edge_materialize_replayable`, and `edge_build_replayable_buffer`
- forcing legacy seeded-row regrouping regresses badly (`300`: `0.711s`
  vs `0.237s`, `10k`: `2.998s` vs `0.829s`)
- Rust and Go remain generated DFS/pipeline binaries
- so this runner is still intentionally a mixed execution-model comparison:
  query engine versus non-query target pipelines
- the C# path is now faster at every tested scale, and the gap widens quickly
  once the path-row materialization cost is removed

Optional grouped-Prolog spot check:

```bash
python examples/benchmark/benchmark_category_influence_cross_target.py \
    --scales 300,1k \
    --targets csharp-query,rust-dfs,go-dfs,prolog-accumulated \
    --repetitions 1
```

| Scale | C# Query | Rust DFS | Go DFS | Prolog Accumulated | Outputs |
|-------|---------:|---------:|-------:|-------------------:|---------|
| 300 | 0.222s | 0.381s | 0.487s | 1.748s | match |
| 1k | 0.149s | 1.506s | 2.133s | 1.050s | match |

This grouped Prolog path now calls the generated
`category_ancestor$power_sum_selected` helper. When the root is
unbound, that selector routes to the grouped helper rather than the
generic per-root fallback. That makes the accumulated Prolog path viable
again, but C# query is still clearly faster on this workload.

### Prolog Category-Influence Accumulation Results

The category-influence benchmark now uses the generated
`category_ancestor$power_sum_selected` helper for the cycle-safe PPV
`category_ancestor/4` closure from the effective-distance workload.
Because the root argument is left unbound in this workload, the selected
helper routes to the grouped fast path automatically. The generic seeded
accumulation helper remains available as the fallback path.

Command:

```bash
python examples/benchmark/benchmark_prolog_category_influence.py \
    --scales 300,1k --repetitions 1
```

Latest local results:

| Scale | Seeded | Accumulated | Output Match | Note |
|-------|-------:|------------:|--------------|------|
| 300 | 1.770s | 1.868s | match | selected helper keeps grouped path close |
| 1k | 1.175s | 1.231s | match | selected helper stays close |

Current interpretation:

- the specialized grouped helper is semantically correct on category
  influence and dramatically better than the old generic accumulated
  path
- it still trades a higher query cost for much smaller retained state:
  - `300`: tuple count `602808 -> 30968`
  - `1k`: tuple count `352522 -> 10328`
- that retained-state drop is enough to make accumulated Prolog roughly
  competitive with the seeded baseline instead of catastrophically
  slower
- the generic seeded accumulation helper remains the right fallback for
  workloads like effective distance where the grouped helper is not the
  better consumer fit

### Weighted `Min` Results

The current positive-additive weighted `Min` fast path in the C# query
engine is now materially faster than `All` while preserving exact output
agreement on the benchmarked workload.

Command:

```bash
python examples/benchmark/benchmark_weighted_shortest_path.py \
    --scales 300,1k,5k,10k --repetitions 1
```

Latest local results:

| Scale | All | Min | Speedup | Output Match |
|-------|-----|-----|---------|--------------|
| 300 | 0.627s | 0.184s | 3.41x | match |
| 1k | 0.440s | 0.139s | 3.17x | match |
| 5k | 1.188s | 0.236s | 5.03x | match |
| 10k | 2.930s | 0.419s | 6.98x | match |

The same run also reports SCC metrics for the category graph. At `10k`
the graph has:

- `8247` nodes
- `25227` edges
- `8204` SCCs
- `17` cyclic SCCs
- largest cyclic SCC size `35`

That matters because it suggests the remaining hard cyclic structure is
small and localized, which is favorable for future SCC-condensed
strategies on broader weighted `Min` workloads.

### Available Targets

| Target | Flag | Run Command |
|--------|------|-------------|
| Go | `--target go` | `go build -o bench file.go && ./bench edges.tsv articles.tsv` |
| Rust | `--target rust` | `rustc -O -o bench file.rs && ./bench edges.tsv articles.tsv` |
| C# | `--target csharp` | `dotnet build -c Release && ./bin/Release/prog edges.tsv articles.tsv` |
| Python | `--target python` | `python3 file.py edges.tsv articles.tsv` |
| Codon | `--target codon` | `codon build -release -o bench file.py && ./bench edges.tsv articles.tsv` |
| AWK | `--target awk` | `awk -f file.awk /dev/null` (embedded data, small scale only) |

## Self-Contained Pipelines

### Current Limitation: Generator Script vs. Full Compilation

The self-contained pipelines are produced by `generate_pipeline.py`,
which wraps the DFS transitive closure with aggregation logic per target.
This is a **validation and benchmarking tool**, not the final architecture.

The long-term approach is for UnifyWeaver to compile the **full**
`effective_distance.pl` directly to each target language. This requires:

- **aggregate_all/3 compilation** across all targets
- **Full predicate composition** (predicates calling predicates)
- **Per-path visited pattern** compilation (design docs in
  `docs/design/PER_PATH_VISITED_*.md`)

## Dataset Scales

| Scale | Articles | Edges | Used For |
|-------|----------|-------|----------|
| dev | 19 | 198 | Correctness validation |
| 10x | 195 | 3932 | Initial profiling |
| 300 | 289 | 6008 | All-target comparison |
| 1K | 1000 | 5933 | Scaling analysis |
| 5K | 5000 | 12981 | Performance divergence |
| 10K | 10000 | 25227 | C# lead confirmed, Codon vs Go |

## Design Documents

| Document | Purpose |
|----------|---------|
| `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_THEORY.md` | Philosophy, spectral dimensionality, future work |
| `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_SPEC.md` | Prolog predicates, dataset spec, correctness criteria |
| `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_PLAN.md` | Implementation phases, gap analysis |
| `docs/design/PER_PATH_VISITED_RECURSION.md` | New recursion pattern: all-simple-paths with per-path visited |
| `docs/design/PER_PATH_VISITED_IMPLEMENTATION_PLAN.md` | Cross-target implementation guide |
| `docs/design/PROLOG_TARGET_OPTIMIZATION.md` | Prolog optimization strategies |
| `docs/proposals/PROLOG_TARGET_DEMAND_ANALYSIS.md` | Demand-driven optimization (for Codex/GPT) |
| `docs/proposals/RUST_AWK_PROPOSAL.md` | Rust-based AWK with JIT (back-burner) |
