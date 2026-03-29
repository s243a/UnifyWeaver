# Cross-Target Effective Distance Benchmark: Implementation Plan

## Related Documents

- **Theory**: `CROSS_TARGET_EFFECTIVE_DISTANCE_THEORY.md` — Philosophy,
  theoretical foundations, decomposition principle, future work
- **Specification**: `CROSS_TARGET_EFFECTIVE_DISTANCE_SPEC.md` — Prolog
  program, dataset definition, correctness criteria, output format

## Prior Project Work Referenced

| Resource | Location | Relevance |
|----------|----------|-----------|
| Hand-crafted Python implementation | `src/unifyweaver/data/wikipedia_categories.py` | Reference implementation; ground truth for correctness |
| Wikipedia category fetcher | `scripts/fetch_wikipedia_categories.py` | Parses enwiki categorylinks SQL → SQLite |
| Cohere Wikipedia dataset fetcher | `scripts/fetch_wikipedia_physics.py` | Fetches from `Supabase/wikipedia-en-embeddings` |
| Physics articles (300) | `reports/wikipedia_physics_articles.jsonl` | Dev-phase dataset |
| Wikipedia hierarchy bridge proposal | `docs/proposals/wikipedia_hierarchy_bridge.md` | Architecture, SQLite schema, incremental training |
| AWK target (with transitive closure) | `src/unifyweaver/targets/awk_target.pl` | Recently added TC support |
| AWK aggregation | `docs/AWK_TARGET_EXAMPLES.md`, `docs/AWK_TARGET_STATUS.md` | sum/count/avg/min/max supported |
| Go target aggregation | `src/unifyweaver/targets/go_target.pl` | Richest aggregation: stddev, median, percentile, collect |
| C# query engine | `src/unifyweaver/targets/csharp_target.pl` | FixpointNode, TransitiveClosureNode, AggregateNode |
| C# query runtime | `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` | 13K-line runtime library |
| Python target | `src/unifyweaver/targets/python_target.pl` | Tail recursion, memoized recursion, mutual recursion |
| Effective distance formula | `src/unifyweaver/data/wikipedia_categories.py:89-117` | `d_eff = (Σ dᵢ^(-n))^(-1/n)` with n=5 |

## Phase 0: Dataset Preparation

### 0.1 Dev Dataset (300 articles)

Reuse existing `reports/wikipedia_physics_articles.jsonl`. Extract category
relationships:

```bash
# Use existing tooling to get categories for the 300 physics articles
python scripts/fetch_wikipedia_categories.py \
    --articles reports/wikipedia_physics_articles.jsonl \
    --output data/benchmark/dev/
```

Output files:
- `article_category.tsv` — article_id → category_name
- `category_parent.tsv` — child_category → parent_category
- `root_categories.tsv` — root category names (e.g., "Science")

We use **Approach A** (Wikipedia-only with assumed root) as defined in the
spec. The semantic filter selects science articles, so "Science" is the
natural root. See spec doc for alternatives (discovered root, Pearltrees
bridging).

### 0.2 Prolog Fact File

Generate `benchmark_facts.pl` from TSV:

```prolog
% Auto-generated from Wikipedia category data
article_category('Bose-Einstein statistics', 'Statistical mechanics').
article_category('Classical mechanics', 'Mechanics').
category_parent('Statistical mechanics', 'Physics').
category_parent('Mechanics', 'Physics').
category_parent('Physics', 'Science').
root_category('Science').
```

### 0.3 Scale-Up Dataset (50K articles)

Broaden the semantic filter in `fetch_wikipedia_physics.py` or take the first
50K articles without filtering:

```bash
python scripts/fetch_wikipedia_physics.py \
    --no-filter --top-k 50000 \
    --output data/benchmark/bench/
```

This will require building category hierarchy for all 50K articles, which
exercises the SQLite-backed category lookup.

**Prerequisite**: The enwiki categorylinks SQLite database must be built.
See `docs/proposals/wikipedia_hierarchy_bridge.md` Phase 1.

## Phase 1: Prolog Program & Native Execution

### 1.1 Write Benchmark Prolog Program

File: `examples/benchmark/effective_distance.pl`

Core predicates as specified in the spec doc. Verify correctness by running
in SWI-Prolog directly:

```bash
swipl -g "effective_distance('Bose-Einstein statistics', Folder, D), write(D), nl, fail ; true" \
    -t halt examples/benchmark/effective_distance.pl
```

### 1.2 Validate Against Reference Python

Compare SWI-Prolog output against `wikipedia_categories.py` for the 300-article
dev set. All (article, folder, distance) triples must match within 1e-6.

## Phase 2: Target Compilation — Gap Analysis & Implementation

For each target, attempt compilation and identify gaps.

### 2.1 C# Query Engine

**Expected to work**: The C# query engine has the richest support for this
workload pattern.

- `category_ancestor/3` → `FixpointNode` or `TransitiveClosureNode`
- `aggregate_all(sum(W), ...)` → `AggregateNode`
- `is/2` arithmetic → typed delegates

**Potential gaps**:
- Arithmetic expressions inside aggregate goals (`W is Hops ** (-5)`)
  — verify the C# target handles `is/2` within aggregate subplans
- Power operator (`**`) — verify it maps to `Math.Pow()` in generated code

**Files to modify**: `src/unifyweaver/targets/csharp_target.pl`

### 2.2 Go

**Expected to mostly work**: Go has strong aggregation and recursion support.

- `category_ancestor/3` → compiled fixpoint loop
- `aggregate_all(sum(W), ...)` → sum with grouping
- `is/2` → inline Go arithmetic

**Potential gaps**:
- Power operator (`**`) → needs `math.Pow()` mapping
- Aggregation over recursive results (sum over transitive closure output)
  — verify this composition works

**Files to modify**: `src/unifyweaver/targets/go_target.pl`

### 2.3 AWK

**Most likely to need new features**: AWK's streaming model is fundamentally
different from the relational aggregation pattern.

- Transitive closure — recently added, verify it works for this use case
- `sum` aggregation — supported, but **inside a transitive closure** is new
- Power operator — AWK has `^` natively, should work
- Multi-pass — AWK may need to: (1) compute transitive closure → temp file,
  (2) aggregate from temp file, (3) apply final arithmetic

**Potential gaps**:
- **Aggregation over recursive results**: AWK processes line-by-line;
  collecting all paths per (article, folder) pair before summing requires
  associative array accumulation. May need a new compilation pattern.
- **Multi-relation input**: Three input files (articles, categories, folders)
  need multi-file AWK processing or pre-joined input.

**Files to modify**: `src/unifyweaver/targets/awk_target.pl`

### 2.4 Python

**Two deliverables**: transpiled version AND comparison to hand-crafted.

- Recursion → memoized or tail-recursive function
- `aggregate_all/3` → **GAP**: needs list comprehension + `sum()` generation
- Arithmetic → straightforward

**Potential gaps**:
- `aggregate_all/3` transpilation: The Python target does not currently
  generate aggregation code. Need to add a compilation pattern that maps
  `aggregate_all(sum(Expr), Goal, Result)` to:
  ```python
  result = sum(expr for ... in goal_solutions(...))
  ```
- Collecting all solutions from a recursive predicate — needs generator or
  list accumulation pattern.

**Files to modify**: `src/unifyweaver/targets/python_target.pl`

**Reference comparison**: Run both the transpiled Python and the hand-crafted
`wikipedia_categories.py` on the same input, compare output and performance.

## Phase 3: Small-Scale Validation (300 articles)

### 3.1 Compile & Run All Targets

For each target:
1. Compile `effective_distance.pl` → target source
2. Build/run with dev dataset (300 articles)
3. Capture TSV output
4. Diff against reference (SWI-Prolog output)

### 3.2 Fix Bugs

Iterate until all four targets produce identical output on the dev dataset.

## Phase 4: Scale-Up Benchmark (50K articles)

### 4.1 Generate Large Dataset

Build the 50K-article fact base with full category hierarchy.

### 4.2 Run All Targets

Measure wall-clock time, peak memory, and throughput for each target.

### 4.3 Report

Create a benchmark report with:
- Performance comparison table
- Analysis of where each target excels/struggles
- Feature gap summary (what was added to each target)
- Recommendations for further target improvements

## Implementation Order

| Step | Task | Depends On | Estimated Effort |
|------|------|------------|-----------------|
| 0.1 | Dev dataset extraction | Existing tooling | Low |
| 0.2 | Prolog fact file generator | 0.1 | Low |
| 1.1 | Benchmark Prolog program | 0.2 | Medium |
| 1.2 | Validate against Python reference | 1.1 | Low |
| 2.1 | C# Query compilation + gaps | 1.1 | Medium |
| 2.2 | Go compilation + gaps | 1.1 | Medium |
| 2.3 | AWK compilation + gaps | 1.1 | High (most gaps) |
| 2.4 | Python compilation + gaps | 1.1 | Medium-High |
| 3.1 | Small-scale validation | 2.1-2.4 | Low |
| 3.2 | Bug fixes | 3.1 | Variable |
| 4.1 | 50K dataset generation | 3.2 | Medium |
| 4.2 | Benchmark runs | 4.1 | Low |
| 4.3 | Report | 4.2 | Low |

Steps 2.1 through 2.4 can be parallelized across developers/agents.

## Directory Structure

```
examples/benchmark/
    effective_distance.pl          # Source Prolog program
    generate_facts.py              # Dataset → Prolog facts generator
    run_benchmark.sh               # Run all targets, collect results
    compare_outputs.py             # Diff target outputs against reference

data/benchmark/
    dev/                           # 300-article dataset
        article_category.tsv
        category_parent.tsv
        root_categories.tsv
        facts.pl                   # Materialized Prolog facts
    bench/                         # 50K-article dataset
        ...

docs/proposals/
    CROSS_TARGET_EFFECTIVE_DISTANCE_THEORY.md   # This doc set
    CROSS_TARGET_EFFECTIVE_DISTANCE_SPEC.md
    CROSS_TARGET_EFFECTIVE_DISTANCE_PLAN.md

reports/
    benchmark_results.md           # Final comparison report
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Category graph cycles cause non-termination | High | All targets must implement visited-set / fixpoint convergence |
| AWK cannot handle multi-pass recursive + aggregation | Medium | May need AWK-specific compilation pattern with temp files |
| Python aggregate_all transpilation harder than expected | Medium | Fall back to generating explicit loop + accumulator |
| 50K dataset overwhelming for in-memory targets | Low | C# and Go handle this scale easily; AWK streams; Python may need generators |
| Power operator (`**`) not mapped in some targets | Low | Simple fix: map to target's math library |
