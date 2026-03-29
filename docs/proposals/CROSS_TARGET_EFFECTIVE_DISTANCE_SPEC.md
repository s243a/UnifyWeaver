# Cross-Target Effective Distance Benchmark: Specification

## Overview

A single Prolog program that computes effective distance from Wikipedia articles
to root categories via the category hierarchy, compiled to four targets:
**C# Query Engine**, **Go**, **AWK**, and **Python**. All targets consume the
same materialized fact base and must produce identical results.

### Three Approaches to Root Selection

There are three ways to define the root node(s) for effective distance
computation. They differ in data requirements and complexity:

**Approach A: Wikipedia-Only with Assumed Root (chosen for benchmark)**

Apply a semantic filter (e.g., science articles from the Cohere Wikipedia
dataset), then assume the filter's domain category as root (e.g., "Science").
The category hierarchy is traversed purely within Wikipedia's categorylinks.

- Simplest pipeline: filter → materialize facts → run Prolog
- No external data dependency beyond Wikipedia
- Root is a reasonable assumption when the semantic filter is domain-specific

**Approach B: Wikipedia-Only with Discovered Root**

Instead of assuming a root, navigate up the categorylinks from each article
until paths converge on a common ancestor. The deepest common category that
all (or most) articles reach becomes the root.

- More robust than Approach A for broad/mixed datasets
- Still Wikipedia-only; no Pearltrees dependency
- Requires an additional "root discovery" pass over the category graph

**Approach C: Wikipedia + Pearltrees Bridging**

Use Pearltrees folders as target nodes instead of Wikipedia root categories.
The Wikipedia category hierarchy serves as a bridge connecting articles to
the organizational structure defined in Pearltrees.

- Requires Pearltrees RDF data + Wikipedia categorylinks
- Existing implementation: `src/unifyweaver/data/wikipedia_categories.py`
  (`find_all_folder_connections()`, `walk_hierarchy_all_paths()`)
- Architecture documented in `docs/proposals/wikipedia_hierarchy_bridge.md`
- Category matching: `scripts/fetch_wikipedia_categories.py`
  (`find_connection_point()`, Pearltrees folder matching)
- Multi-account Pearltrees parsing: `scripts/pearltrees_multi_account_generator.py`

**For this benchmark we use Approach A.** It has the fewest moving parts,
making it ideal for validating cross-target transpilation correctness and
performance. Approaches B and C are natural extensions once the core
pipeline is working.

### Semantic Filtering as Pre-Processing

Semantic filtering is a **pre-processing step** external to the Prolog
transpilation path. The filter runs in Python (using existing tooling in
`scripts/fetch_wikipedia_physics.py` with the Cohere Wikipedia dataset),
produces materialized facts (TSV/Prolog), and the Prolog program consumes
those facts. Integrating semantic filtering into the Prolog transpilation
itself is a long-term goal.

## Dataset

### Source

- **Cohere Wikipedia embeddings** via HuggingFace
  (`Supabase/wikipedia-en-embeddings`, ~224K Simple English Wikipedia articles)
- **Wikipedia categorylinks** (`enwiki-latest-categorylinks.sql.gz`)
- Existing tooling: `scripts/fetch_wikipedia_physics.py`,
  `scripts/fetch_wikipedia_categories.py`

### Scale

| Phase | Articles | Category facts (est.) | Purpose |
|-------|----------|----------------------|---------|
| Dev   | ~300     | ~3K                  | Correctness, debugging |
| Test  | ~5K      | ~30K                 | Integration, basic perf |
| Bench | ~50K     | ~300K+               | Performance comparison |

The dev dataset can reuse the existing 300 physics articles in
`reports/wikipedia_physics_articles.jsonl`.

### Materialized Fact Format

The Prolog program operates on three relation types, materialized as facts:

```prolog
% Article belongs to a category
article_category(ArticleId, CategoryName).

% Category hierarchy (child → parent)
category_parent(ChildCategory, ParentCategory).

% Target folders we want to match against (from Pearltrees)
% Root categories to compute distance to (e.g., "Science")
root_category(CategoryName).
```

For the benchmark, facts are exported to target-appropriate formats:
- **Prolog**: `.pl` file with fact clauses
- **AWK**: TSV files (one per relation)
- **Go**: Go source with map literals or JSONL input
- **C#**: `InMemoryRelationProvider` with typed arrays
- **Python**: Python source with list/dict literals or JSONL input

## Prolog Program

### Core Predicates

```prolog
%% category_ancestor(+Cat, -Ancestor, -Hops, +Visited)
%  Transitive closure over category_parent/2 with cycle detection.
%
%  Wikipedia's category graph contains cycles (e.g., mutual subcategories,
%  loops through intermediate categories). We carry a Visited set to avoid
%  infinite recursion. Each target must compile this to an equivalent
%  memoization strategy:
%    - C# Query: semi-naive fixpoint with HashSet deduplication
%    - Go: visited map in fixpoint loop
%    - AWK: associative array of seen nodes
%    - Python: @functools.cache or explicit visited set
%
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

%% path_to_root(+Article, -Root, -Hops)
%  Find all paths from Article to any root_category via category hierarchy.
%  Non-deterministic: backtracks over all paths.
%
%  The natural descendant tree is: root → categories → pages.
%  So an article's distance = 1 hop (article→category) + category hops to root.
%
path_to_root(Article, Root, 1) :-
    article_category(Article, Cat),
    root_category(Cat),
    Root = Cat.

path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    category_ancestor(Cat, AncestorCat, CatHops, [Cat]),
    root_category(AncestorCat),
    Root = AncestorCat,
    Hops is CatHops + 1.

%% effective_distance(+Article, -Root, -Deff)
%  Compute d_eff = (Σ d^(-N))^(-1/N) where N=5.
%  Aggregates over all paths from Article to Root.
effective_distance(Article, Root, Deff) :-
    aggregate_all(sum(W),
        (path_to_root(Article, Root, Hops),
         W is Hops ** (-5)),
        WeightSum),
    WeightSum > 0,
    Deff is WeightSum ** (-0.2).   % (-1/5) = -0.2

%% ranked_articles(+Root, -Article, -Distance)
%  All articles ranked by effective distance to Root.
%  Sorted ascending (closest first).
ranked_articles(Root, Article, Distance) :-
    root_category(Root),
    effective_distance(Article, Root, Distance).
```

### Cycle Detection Strategy

Wikipedia's category graph is **not a DAG** — it contains cycles (e.g.,
"Category:Physics" → "Category:Natural sciences" → "Category:Science" →
"Category:Branches of science" → "Category:Physics"). Without cycle
detection, the transitive closure will not terminate.

The Prolog program uses an explicit `Visited` list with `\+ member/2`.
Each transpilation target must implement an equivalent:

| Target | Cycle detection mechanism |
|--------|--------------------------|
| **C# Query** | Semi-naive fixpoint iteration with `HashSet<T>` — naturally converges; new iterations only process delta (newly discovered) tuples |
| **Go** | `map[string]bool` visited set in fixpoint loop |
| **AWK** | Associative array `visited[node] = 1`; check before traversal |
| **Python** | `@functools.cache` (memoization prevents recomputation) or explicit `visited: set` parameter |

The C# query engine has an advantage here: its `FixpointNode` with semi-naive
evaluation handles cycle detection automatically via delta-set convergence,
without requiring the programmer to thread a visited set through the recursion.

### Auxiliary Queries

These exercise additional target capabilities:

```prolog
%% depth_histogram(-Depth, -Count)
%  Distribution of effective distances (rounded to integer).
%  Useful for tree layering: nodes at d_eff ~ k belong in layer k.
depth_histogram(Depth, Count) :-
    aggregate_all(count,
        (effective_distance(_, _, D),
         Depth is round(D)),
        Count).

%% folder_article_count(-Folder, -Count)
%  How many articles connect to each folder.
folder_article_count(Folder, Count) :-
    root_category(Folder),
    aggregate_all(count,
        effective_distance(_, Folder, _),
        Count).
```

### Notes on Decomposition

The effective distance formula decomposes into primitives available across
targets:

| Step | Prolog construct | Required target support |
|------|-----------------|----------------------|
| Path finding | `category_ancestor/4` | Recursion / transitive closure |
| Cycle detection | `Visited` list + `\+ member/2` | Memoization or visited set |
| Per-path arithmetic | `W is Hops ** (-5)` | `is/2` arithmetic |
| Aggregation | `aggregate_all(sum(W), ...)` | sum aggregation |
| Post-aggregation | `Deff is WeightSum ** (-0.2)` | `is/2` arithmetic |

### Connection to Tree/Mindmap Construction

The effective distance to root produces a **continuous depth** for each node.
In a clean tree, depths would be integers. In a DAG with multiple paths, d_eff
yields fractional values reflecting how "between layers" a node sits:

- d_eff ≈ 1.0 → first layer (direct category match)
- d_eff ≈ 2.0 → second layer
- d_eff ≈ 1.4 → between layers (multiple paths of different lengths)

The dimensionality parameter n controls layer sharpness: high n yields crisper
integer-like layers (shortest path dominates), low n yields fuzzier layering
where alternative paths blur boundaries. This is directly applicable to mindmap
generation where subsequent layers should have roughly integer distance to root.

**Prior work:**
- `docs/proposals/mst_circle_folder_grouping.md` — MST-based hierarchy
  construction where effective distance determines layer assignment

## Correctness Criteria

1. **Exact match on dev dataset**: All targets must produce identical
   (Article, Root, Deff) triples (within floating-point tolerance 1e-6).

2. **Reference comparison**: Results must match the hand-crafted Python
   implementation in `src/unifyweaver/data/wikipedia_categories.py` when
   given the same input facts.

3. **Cycle safety**: Category graphs may contain cycles. All targets must
   terminate (via visited-set or fixpoint convergence).

4. **Empty path handling**: Articles with no path to any root category must
   not appear in output (WeightSum > 0 guard). Note: not all semantically
   similar articles will necessarily be descendants of the root category
   (e.g., "Science"). Coverage depends on what link types are included in
   the category hierarchy — Wikipedia's categorylinks may not capture all
   topical relationships. Articles without a category path to root are
   excluded from results, not treated as errors.

## Performance Metrics

Measured at the 50K-article scale:

| Metric | Description |
|--------|-------------|
| **Wall-clock time** | End-to-end: load facts → compute all effective distances → output |
| **Peak memory** | Maximum RSS during execution |
| **Startup overhead** | Time before first result (relevant for .NET, Go compilation) |
| **Throughput** | Articles processed per second |

## Target-Specific Output Format

All targets emit TSV to stdout for easy comparison:

```
article_id\troot_category\teffective_distance
```

Sorted by (root_category ASC, effective_distance ASC, article_id ASC).

## Target Compilation Requirements

### C# Query Engine
- `FixpointNode` or `TransitiveClosureNode` for `category_ancestor/3`
- `AggregateNode` with sum for the d_eff aggregation
- `SelectionNode` for WeightSum > 0 guard
- Arithmetic delegates for `Hops ** (-5)` and `WeightSum ** (-0.2)`

### Go
- Fixpoint loop or recursive function for transitive closure
- `sum` aggregation (already supported)
- Map-based grouping for per-(article, folder) aggregation
- Standard math for power operations

### AWK
- Transitive closure support (recently added)
- `sum` aggregation (already supported)
- **Gap**: Arithmetic inside aggregation (`Hops ^ (-5)`) — needs verification
- **Gap**: Multi-pass processing (recursive result → aggregation → arithmetic)
  may require temp files or associative array accumulation

### Python
- Recursion with memoization or tail-recursion optimization
- Native sum/arithmetic (straightforward)
- **Gap**: `aggregate_all/3` transpilation — needs to generate
  list comprehension + `sum()` pattern
- Reference: compare output against hand-crafted `wikipedia_categories.py`
