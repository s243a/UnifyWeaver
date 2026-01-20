# Skill: Query Patterns (Sub-Master)

Query and aggregation patterns across SQL generation, runtime streaming, and fuzzy logic.

## When to Use

- User asks "how do I aggregate data?"
- User needs SQL GROUP BY, HAVING, or window functions
- User wants runtime aggregation in Go, C#, Perl, or Ruby
- User asks about score blending or fuzzy search
- User needs to combine rankings from multiple sources

## Skill Hierarchy

```
skill_data_tools.md (parent)
└── skill_query_patterns.md (this file)
    ├── skill_sql_target.md - SQL generation (SQLite, PostgreSQL, MySQL)
    ├── skill_stream_aggregation.md - Runtime aggregation (Go, C#, Perl, Ruby)
    ├── skill_aggregation_patterns.md - Aggregation overview and database selection
    └── skill_fuzzy_search.md - Fuzzy logic DSL, RRF, score blending
```

## Quick Start

### SQL Generation

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

% Count by category
dept_count(Dept, Count) :-
    group_by(Dept, employee(_, Dept, _), count, Count).

% Compiles to:
% SELECT dept, COUNT(*) AS count FROM employee GROUP BY dept

compile_predicate_to_sql(dept_count/2, [dialect(postgres)], SQL).
```

### Stream Aggregation

```prolog
% Runtime aggregation in generated code
total_sales(Region, Total) :-
    aggregate_all(sum(Amount), sale(Region, _, Amount), Region, Total).

% Generates Go:
% func totalSales() map[string]float64 {
%     result := make(map[string]float64)
%     for _, sale := range sales { result[sale.Region] += sale.Amount }
%     return result
% }
```

### Fuzzy Score Blending

```prolog
:- use_module('src/unifyweaver/fuzzy/fuzzy').

% Blend semantic and keyword scores (70%/30%)
combined_search(Query, Items, Combined) :-
    semantic_search(Query, SemanticScores),
    keyword_search(Query, KeywordScores),
    blend_scores(0.7, SemanticScores, KeywordScores, Combined).
```

## The Unifying Theme

All query patterns share a common goal: **reducing many values to fewer values** while preserving meaningful information.

| Paradigm | Input | Output | Example |
|----------|-------|--------|---------|
| **SQL** | Table rows | Grouped summaries | `GROUP BY category` |
| **Stream** | Data stream | Accumulated value | `aggregate_all(count, ...)` |
| **Fuzzy** | Multiple scores | Combined score | `blend_scores(0.7, S1, S2, Combined)` |

## Aggregation Operators

### Standard Operators (SQL & Stream)

| Operator | SQL | Stream | Description |
|----------|-----|--------|-------------|
| `count` | `COUNT(*)` | Yes | Count matching items |
| `sum(V)` | `SUM(col)` | Yes | Sum of values |
| `avg(V)` | `AVG(col)` | Yes | Average value |
| `min(V)` | `MIN(col)` | Yes | Minimum value |
| `max(V)` | `MAX(col)` | Yes | Maximum value |
| `set(V)` | - | Yes | Unique values |
| `bag(V)` | - | Yes | All values with duplicates |

### SQL-Specific

| Feature | Support | Example |
|---------|---------|---------|
| HAVING | Full | `Count >= 10` after GROUP BY |
| Window Functions | PostgreSQL | `RANK() OVER (PARTITION BY dept)` |
| Recursive CTEs | Full | Transitive closure queries |
| Set Operations | Full | UNION, INTERSECT, EXCEPT |

### Fuzzy Logic Operators

| Operator | Formula | Use Case |
|----------|---------|----------|
| `f_and` | Product of weighted terms | All terms must match |
| `f_or` | Probabilistic sum | Any term can match |
| `blend_scores` | `α*S1 + (1-α)*S2` | Weighted combination |
| `rrf_blend` | `Σ 1/(k + rank)` | Rank fusion (incompatible scales) |

## Target Support Matrix

### SQL Generation

| Dialect | GROUP BY | HAVING | Window Fns | CTEs | Recursive |
|---------|----------|--------|------------|------|-----------|
| SQLite | Yes | Yes | Limited | Yes | Yes |
| PostgreSQL | Yes | Yes | Full | Yes | Yes |
| MySQL | Yes | Yes | 8.0+ | 8.0+ | 8.0+ |

### Stream Aggregation

| Target | `aggregate_all/3` | `aggregate_all/4` | HAVING |
|--------|-------------------|-------------------|--------|
| Go | Full | Full | Yes |
| C# | Full | Full | Partial |
| Perl | Basic | Partial | No |
| Ruby | Basic | Partial | No |

## Choosing the Right Approach

### Use SQL Target When

- Output is a database view or query result
- Need complex joins across many tables
- Target database is external (PostgreSQL, MySQL)
- Want pure SQL output (no runtime code)

```prolog
% Best for database views
compile_predicate_to_sql(report/3, [dialect(postgres), format(view)], SQL).
```

### Use Stream Aggregation When

- Processing data streams in generated code
- Need runtime aggregation with custom logic
- Working with embedded databases (BBolt, Redb, LiteDB)
- Target is Go, C#, Perl, or Ruby

```prolog
% Best for runtime processing
aggregate_all(avg(Score), item(_, Score), Average).
```

### Use Fuzzy Logic When

- Combining multiple ranking signals
- Scores have different scales or meanings
- Need soft, weighted combinations
- Building ensemble search systems

```prolog
% Best for search/ranking
blend_scores(0.6, SemanticScores, KeywordScores, Combined).
```

## Common Patterns

### Multi-Source Ranking

```python
# Combine multiple search models
from scripts.experiment_ensemble_blend import blend_scores, rrf_blend_scores

# Score fusion (when scales are compatible)
blended = blend_scores([
    (bge_scores, 0.5),
    (minilm_scores, 0.3),
    (nomic_scores, 0.2)
])

# Rank fusion (when scales differ)
rrf_combined = rrf_blend_scores([bge_scores, minilm_scores, nomic_scores])
```

### Hierarchical Aggregation

```prolog
% SQL: Nested GROUP BY
regional_summary(Region, Category, Total) :-
    group_by([Region, Category], sale(Region, Category, Amount), sum(Amount), Total).

% Stream: Two-pass aggregation
category_totals(Category, Total) :-
    findall(Amount, sale(_, Category, Amount), Amounts),
    sum_list(Amounts, Total).
```

### Filtered Aggregation

```prolog
% SQL with WHERE + GROUP BY + HAVING
active_large_depts(Dept, Count) :-
    employee(Name, Dept, Status),
    Status = active,
    group_by(Dept, employee(Name, Dept, active), count, Count),
    Count >= 10.

% Stream with pre-filter
active_count(Dept, Count) :-
    aggregate_all(count, (employee(_, Dept, Status), Status = active), Dept, Count).
```

## Child Skills

- `skill_sql_target.md` - SQL generation, dialects, views, CTEs
- `skill_stream_aggregation.md` - Runtime aggregation operators
- `skill_aggregation_patterns.md` - Database selection, paradigm overview
- `skill_fuzzy_search.md` - Fuzzy logic DSL, RRF, score blending

## Related

**Parent Skill:**
- `skill_data_tools.md` - Data tools master

**Sibling Sub-Masters:**
- `skill_ml_tools.md` - Machine learning tools
- `skill_data_sources.md` - Data source handling

**Code:**
- `src/unifyweaver/targets/sql_target.pl` - SQL compilation
- `src/unifyweaver/targets/go_target.pl` - Go aggregation
- `src/unifyweaver/targets/csharp_target.pl` - C# aggregation
- `src/unifyweaver/fuzzy/` - Fuzzy logic DSL
- `scripts/experiment_ensemble_blend.py` - Python ensemble utilities
