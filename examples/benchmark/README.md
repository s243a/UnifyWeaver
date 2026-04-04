# Cross-Target Effective Distance Benchmark

## Overview

Computes effective distance `d_eff = (Σ dᵢ^(-n))^(-1/n)` from Wikipedia
articles to root categories via the category hierarchy, compiled to
multiple targets. Demonstrates UnifyWeaver's core value: write Prolog once,
compile to the best target for your scale.

See `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_*.md` for theory
and specification.

## Benchmark Results

### C# Query Engine vs DFS Pipelines

The C# parameterized query engine (`PathAwareTransitiveClosureNode`) is
dramatically faster than all DFS (depth-first search) pipeline
implementations, including compiled Rust:

| Target | 300 art | 1K art | 5K art | 10K art |
|--------|---------|--------|--------|---------|
| **C# Query Engine** | 0.60s | **0.41s** | **1.25s** | **3.13s** |
| C# DFS pipeline | **0.44s** | 1.25s | 5.36s | 9.96s |
| Rust DFS pipeline | — | 1.36s | 7.33s | 13.68s |
| Go DFS pipeline | 0.43s | 1.96s | 11.36s | 18.71s |
| Codon DFS pipeline | 0.67s | 2.55s | 10.98s | 22.14s |
| CPython DFS pipeline | 0.73s | 2.93s | 15.71s | — |
| Prolog | 1.26s | — | — | — |
| AWK | 2.46s | — | — | — |

**Speedup of C# Query Engine over DFS pipelines:**

| vs Target | 300 art | 1K art | 5K art | 10K art |
|-----------|---------|--------|--------|---------|
| vs C# DFS | 0.7x | 3.1x | 4.3x | 3.2x |
| vs Rust | — | 3.3x | 5.9x | 4.4x |
| vs Go | 1.1x | 8.9x | 17.2x | 12.4x |

Semantic note:

- The current C# query-engine benchmark path preserves per-path cycle
  checks without collapsing distinct simple paths that happen to reach
  the same ancestor with the same hop count.
- `benchmark_effective_distance.py` compares the query-engine output
  against the C# DFS reference and reports the result via
  `query_vs_csharp_dfs`.
- Current post-fix runs report `query_vs_csharp_dfs = match` at
  `300`, `1k`, `5k`, and `10k`.

### Why the Query Engine Wins

The query engine's advantage comes from **seed deduplication**: many
articles share the same categories, so the engine computes ancestors
once per unique category seed, not once per article.

| Scale | Articles | Unique category seeds | Redundancy factor |
|-------|----------|-----------------------|-------------------|
| 300 | 289 | 386 | ~0.7x (more seeds than articles) |
| 1K | 1,000 | 89 | ~11x |
| 5K | 5,000 | 284 | ~18x |
| 10K | 10,000 | 888 | ~11x |

At 1K scale, 1000 articles map to only 89 unique category seeds — the
engine does ~11x less seed expansion work. Even with full per-path
semantics restored, the precomputed ancestor index still makes
per-article aggregation much cheaper than rerunning DFS from scratch for
each article.

### DFS Pipeline Execute Time Scaling

For completeness, the original DFS pipeline comparison:

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

1. **The C# Query Engine remains the fastest target in the intended
   large-scale regime** — after restoring full per-path semantics, it is
   still 3.1x faster than C# DFS at `1k`, 4.3x faster at `5k`, and 3.2x
   faster at `10k`.

2. **At small scale, the semantic fix makes the query engine slightly
   slower than C# DFS** — at `300`, the query engine is `0.60s` vs
   `0.44s` for C# DFS because it now preserves full path multiplicity
   instead of collapsing equal `(ancestor, hops)` tuples.

3. **Among DFS pipelines, C# still leads at scale** — at `10k` it is
   faster than Rust DFS (`9.96s` vs `13.68s`) and much faster than Go.

4. **The query engine still beats Rust DFS comfortably at scale** —
   `3.3x` faster at `1k`, `5.9x` at `5k`, and `4.4x` at `10k`.

5. **Go falls behind at scale** — `map[string]bool` copy-on-branch
   semantics are heavier than C#'s `HashSet`. Go is 2x slower than C#
   at 10K.

6. **Codon approaches Go** — at 5K they were nearly tied (11.0s vs 11.4s).
   At 10K Go pulled ahead again (18.7s vs 22.1s), but the gap is narrowing.

7. **Prolog needs demand analysis** to compete — naive all-simple-paths
   exploration is combinatorially explosive. Design docs for this optimization
   are in `docs/proposals/PROLOG_TARGET_DEMAND_ANALYSIS_*.md`.

### Target Recommendations by Audience

| Audience | Recommended Target | Why |
|----------|-------------------|-----|
| Enterprise / .NET | C# Query Engine | Fastest at all scales, purpose-built |
| ML / Data Science | Codon | Stay in Python, competitive performance |
| Cloud / DevOps | Go | Fast compile, good ecosystem |
| Systems / Embedded | Rust | No runtime overhead, fastest DFS at small scale |

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
| `benchmark_effective_distance.py` | Rebuild and time the C# query engine vs C#/Rust/Go DFS binaries |
| `benchmark_shortest_path_cross_target.py` | Compare shortest-path-to-root across C# query, seeded Prolog `min`, C# DFS, Rust DFS, and Go DFS |
| `benchmark_dependency_depth_cross_target.py` | Compare synthetic dependency reach-count across C# query, C# DFS, Rust DFS, and Go DFS |
| `benchmark_dependency_longest_depth_cross_target.py` | Compare true DAG longest dependency-chain depth across C# query, C# DFS, Rust DFS, and Go DFS |
| `benchmark_path_aware_accumulation.py` | Measure counted-closure vs generalized accumulation overhead |
| `benchmark_weighted_shortest_path.py` | Measure `PathAwareAccumulationNode` `All` vs `Min` pruning on positive weighted paths |
| `benchmark_weighted_shortest_path_cross_target.py` | Compare positive weighted shortest path across C# query, C# DFS, Rust DFS, and Go DFS |
| `benchmark_category_influence_cross_target.py` | Compare category influence propagation across the C# query engine, Rust DFS, and Go DFS |
| `generate_prolog_shortest_path_benchmark.pl` | Generate standalone SWI-Prolog shortest-path benchmark scripts with `branch_pruning(auto|false)` |
| `benchmark_prolog_branch_pruning.py` | Compare handwritten Prolog shortest-path source against generated pruned and unpruned Prolog scripts |
| `generate_prolog_shortest_path_seeded_benchmark.pl` | Generate standalone SWI-Prolog shortest-path scripts for seeded `all` vs mode-directed `min` closure, loading `facts.pl` at runtime |
| `benchmark_prolog_seeded_min_closure.py` | Compare seeded Prolog `all` vs `min` closure and report `load_ms`, `query_ms`, `aggregation_ms`, and work metrics |
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
- the seeded Prolog shortest-path benchmark now loads `facts.pl` at
  runtime from a small generated driver instead of inlining facts into
  the generated script, so `load_ms` is reported separately from
  `query_ms`

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
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs

# Compare minimum-hop shortest path across query engine and DFS targets
python examples/benchmark/benchmark_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-min

# Compare dependency reach count across query engine and DFS targets
python examples/benchmark/benchmark_dependency_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs

# Compare true DAG longest dependency depth across DFS targets
python examples/benchmark/benchmark_dependency_longest_depth_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-dfs,rust-dfs,go-dfs

# Measure overhead of the generalized path-aware accumulation runtime
python examples/benchmark/benchmark_path_aware_accumulation.py \
    --scales 300,1k,5k,10k

# Measure directed-table pruning on weighted path-aware accumulation
python examples/benchmark/benchmark_weighted_shortest_path.py \
    --scales 300,1k,5k,10k

# Compare positive weighted minimum path distance across query engine and DFS targets
python examples/benchmark/benchmark_weighted_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs

# Compare category influence propagation across the C# query engine, Rust, and Go
python examples/benchmark/benchmark_category_influence_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,rust-dfs,go-dfs

# Compare handwritten Prolog vs generated pruned and unpruned Prolog
python examples/benchmark/benchmark_prolog_branch_pruning.py \
    --scales 300,1k,5k,10k \
    --targets prolog-source,prolog-pruned,prolog-unpruned

# Compare seeded Prolog all-path closure vs mode-directed min closure
python examples/benchmark/benchmark_prolog_seeded_min_closure.py \
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
- Prolog seeded `min`:
  - minimum hop distance

The benchmark split is now:

- cross-target:
  - `benchmark_effective_distance.py`
  - `benchmark_shortest_path_cross_target.py`
  - `benchmark_dependency_depth_cross_target.py`
  - `benchmark_dependency_longest_depth_cross_target.py`
  - `benchmark_weighted_shortest_path_cross_target.py`
  - `benchmark_category_influence_cross_target.py`
- Prolog target:
  - `benchmark_prolog_branch_pruning.py`
  - `benchmark_prolog_seeded_min_closure.py`
- C# query-engine internal mode comparison:
  - `benchmark_shortest_path_to_root.py`
  - `benchmark_weighted_shortest_path.py`

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
| 300 | 0.170s | 0.117s | 0.489s | 0.388s | 0.501s | match |
| 1k | 0.105s | 0.121s | 1.243s | 1.469s | 2.109s | match |
| 5k | 0.187s | 0.277s | 5.777s | 7.648s | 12.378s | match |
| 10k | 0.429s | 0.535s | 10.408s | 14.443s | 20.399s | match |

Direct C# query vs Prolog seeded `min`:

| Scale | Faster target | Speedup |
|-------|---------------|--------:|
| 300 | Prolog Min | 1.45x |
| 1k | C# Query | 1.16x |
| 5k | C# Query | 1.48x |
| 10k | C# Query | 1.25x |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 2.87x | 2.28x | 2.95x |
| 1k | 11.87x | 14.02x | 20.13x |
| 5k | 30.86x | 40.85x | 66.12x |
| 10k | 24.24x | 33.63x | 47.50x |

Comparison note:

- seeded Prolog `min` is now competitive with the C# query engine on
  this shortest-path workload: it wins at `300`, trails only slightly at
  `1k`, and remains within about `1.25x` to `1.48x` of the C# query
  engine at `5k` and `10k`
- output digests match across all five targets at every tested scale

### Cross-Target Weighted Shortest-Path Results

Command:

```bash
python examples/benchmark/benchmark_weighted_shortest_path_cross_target.py \
    --scales 300,1k,5k,10k \
    --targets csharp-query,csharp-dfs,rust-dfs,go-dfs \
    --repetitions 1
```

Latest local results:

| Scale | C# Query | C# DFS | Rust DFS | Go DFS | Query vs C# DFS |
|-------|---------:|--------:|---------:|-------:|-----------------|
| 300 | 0.181s | 0.466s | 0.353s | 0.487s | match |
| 1k | 0.142s | 1.409s | 1.447s | 2.206s | match |
| 5k | 0.250s | 5.957s | 7.994s | 12.404s | match |
| 10k | 0.381s | 10.593s | 14.234s | 19.460s | match |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 2.57x | 1.95x | 2.69x |
| 1k | 9.91x | 10.18x | 15.51x |
| 5k | 23.81x | 31.94x | 49.57x |
| 10k | 23.57x | 33.02x | 44.65x |

Comparison note:

- the earlier apparent weighted `10k` mismatch turned out to be
  benchmark normalization sensitivity at roughly `1e-12`, not a semantic
  query-engine bug
- the cross-target weighted benchmark now normalizes floating-point
  outputs with a tolerance appropriate for cross-language evaluation
  order differences

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
| 300 | 0.067s | 0.046s | 0.003s | 0.007s | match |
| 1k | 0.086s | 0.051s | 0.008s | 0.007s | match |
| 5k | 0.088s | 0.180s | 0.143s | 0.113s | match |
| 10k | 0.103s | 0.516s | 0.546s | 0.469s | match |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 0.69x | 0.04x | 0.10x |
| 1k | 0.59x | 0.09x | 0.09x |
| 5k | 2.04x | 1.61x | 1.27x |
| 10k | 4.99x | 5.28x | 4.54x |

Comparison note:

- the current C# query path now uses a project-grouped reachability node
  rather than raw per-seed closure followed by regrouping
- the latest runtime-overhead pass pushes the final count aggregation
  into the query runtime too, so the benchmark no longer materializes the
  full `(project, reachable_dependency)` relation just to count it
- for the one-shot generated benchmark programs, disabling query-runtime
  cache reuse also helped slightly by removing cache bookkeeping that is
  not reused within a single run
- this changed the cost profile materially: the query engine is still
  behind at `300` and `1k`, but is now clearly ahead of all DFS targets
  from `5k` onward
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
| 300 | 0.057s | 0.043s | 0.002s | 0.002s | match |
| 1k | 0.056s | 0.045s | 0.002s | 0.003s | match |
| 5k | 0.087s | 0.056s | 0.006s | 0.006s | match |
| 10k | 0.091s | 0.055s | 0.013s | 0.012s | match |

Speedups of C# query engine:

| Scale | vs C# DFS | vs Rust DFS | vs Go DFS |
|-------|----------:|------------:|----------:|
| 300 | 0.75x | 0.03x | 0.04x |
| 1k | 0.81x | 0.04x | 0.05x |
| 5k | 0.64x | 0.07x | 0.07x |
| 10k | 0.60x | 0.14x | 0.14x |

Comparison note:

- this benchmark now includes a real `csharp-query` DAG longest-depth
  path built on a dedicated query-runtime node
- the latest runtime-overhead passes remove some setup overhead inside
  the longest-depth node, disable cache reuse for the one-shot generated
  benchmark program, and make phase tracing opt-in instead of always-on
- set `UNIFYWEAVER_BENCH_TRACE=1` when you want the generated
  `csharp-query` longest-depth benchmark to print per-phase timings to
  `stderr`
- the hand-written C# DFS baseline is now cheaper too, after moving it to a
  lighter-weight custom TSV loader with manual tab scanning and better
  pre-sizing
- the query engine still matches all DFS outputs and remains somewhat
  behind the hand-written C# DFS baseline, so longest depth is still a
  separate optimization track from reach-count
- that makes it a good DAG benchmark baseline for further query-runtime
  optimization rather than a finished performance story

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
| 300 | 0.715s | 0.371s | 0.487s | match |
| 1k | 0.554s | 1.484s | 2.141s | match |
| 5k | 1.673s | 7.905s | 12.509s | match |
| 10k | 4.080s | 14.932s | 20.369s | match |

Speedups of C# query engine:

| Scale | vs Rust DFS | vs Go DFS |
|-------|------------:|----------:|
| 300 | 0.52x | 0.68x |
| 1k | 2.68x | 3.87x |
| 5k | 4.72x | 7.47x |
| 10k | 3.66x | 4.99x |

Comparison note:

- the benchmark now includes a dedicated C# query-engine path
- that path now uses `QueryRuntime` for both:
  - the recursive `category_ancestor` expansion
  - the outer grouped influence sum via `AggregateNode`
- the C# benchmark package is now generated through
  `generate_pipeline.py` as `target=csharp_query`, including:
  - `Program.cs`
  - `QueryRuntime.cs`
  - `benchmark_qe.csproj`
- Rust and Go remain generated DFS/pipeline binaries
- so this runner is intentionally a mixed execution-model comparison:
  query engine versus non-query target pipelines
- the remaining benchmark-only orchestration is now just build/run
  invocation in the runner, not embedded C# workload or package wiring
- the C# path is still weaker only at `300` and clearly faster from `1k`
  onward on the current workload

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
