# Skill: Aggregation Patterns

Master skill for reducing collections to summaries, combining scores, and querying aggregated data across different paradigms.

## When to Use

- User asks "how do I aggregate data?"
- User wants to combine search scores from multiple sources
- User asks about GROUP BY or COUNT/SUM/AVG
- User needs to reduce a stream to a single value
- User asks about fuzzy logic or score blending
- User wants to query databases from transpiled code

## The Unifying Theme

All aggregation patterns share a common goal: **reducing many values to fewer values** while preserving meaningful information.

| Paradigm | Input | Output | Example |
|----------|-------|--------|---------|
| **SQL** | Table rows | Grouped summaries | `GROUP BY category` |
| **Stream** | Data stream | Accumulated value | `aggregate_all(count, ...)` |
| **Fuzzy** | Multiple scores | Combined score | `blend_scores(0.7, S1, S2, Combined)` |

## Sub-Skills

### 1. SQL Target (`skill_sql_target.md`)

Declarative aggregation via SQL generation. Best when:
- You need complex GROUP BY with HAVING
- Target database is SQLite, PostgreSQL, or MySQL
- Output is a database view or query result

```prolog
% Prolog fact-based aggregation → SQL
count_by_category(Category, Count) :-
    product(_, Category, _),
    aggregate_all(count, product(_, Category, _), Count).

% Compiles to:
% SELECT category, COUNT(*) FROM products GROUP BY category
```

### 2. Stream Aggregation (`skill_stream_aggregation.md`)

Procedural aggregation within generated code. Best when:
- Processing data streams in Go, Python, Rust, or C#
- Need runtime aggregation with custom logic
- Working with embedded databases (BBolt, Redb, LiteDB)

```prolog
% Stream-based aggregation
total_sales(Total) :-
    aggregate_all(sum(Price), order(_, _, Price), Total).

% Compiles to Go:
% var total float64
% for _, order := range orders { total += order.Price }
```

### 3. Fuzzy Search (`skill_fuzzy_search.md`)

Score-based aggregation for search and ranking. Best when:
- Combining semantic and lexical search scores
- Implementing soft matching with weighted terms
- Building ensemble search systems

```prolog
% Fuzzy score combination
combined_search(Query, Item, Score) :-
    semantic_score(Query, Item, S1),
    keyword_score(Query, Item, S2),
    blend_scores(0.7, S1, S2, Score).
```

## Database Support by Target

| Target | Database | Type | Use Case |
|--------|----------|------|----------|
| **SQL** | SQLite, PostgreSQL, MySQL | External SQL | Query generation |
| **Python** | SQLite | Embedded | Semantic search, multi-account |
| **Go** | BBolt | Embedded KV | Fast file-based storage |
| **Rust** | Redb | Embedded KV | Pure Rust, transactions |
| **C#** | LiteDB | Embedded Doc | Type-safe document storage |

### Choosing a Database Strategy

**Use SQL Target when:**
- You want pure SQL output (no runtime code)
- Target database is external (PostgreSQL, MySQL)
- Need complex joins across many tables

**Use Embedded Database when:**
- Application is self-contained
- Need fast local storage
- Working with vectors/embeddings

**Use Fuzzy Logic when:**
- Combining multiple ranking signals
- Scores have different scales or meanings
- Need soft, weighted combinations

## Quick Reference

### SQL Aggregation
```prolog
:- use_module('src/unifyweaver/targets/sql_target').
compile_predicate_to_sql(my_pred/2, [dialect(postgres)], SQL).
```

### Stream Aggregation
```prolog
aggregate_all(Op, Goal, Result).
% Op: count, sum(V), min(V), max(V), avg(V), set(V), bag(V)
```

### Fuzzy Score Combination
```prolog
:- use_module('src/unifyweaver/fuzzy/fuzzy').
blend_scores(Weight, Score1, Score2, Combined).
f_and([S1, S2, S3], Min).
f_or([S1, S2, S3], Max).
```

## Related

**Parent Skill:**
- `skill_query_patterns.md` - Query patterns sub-master

**Sibling Skills:**
- `skill_sql_target.md` - SQL generation and database dialects
- `skill_stream_aggregation.md` - Runtime aggregation operators
- `skill_fuzzy_search.md` - Fuzzy logic and score fusion

**Other Skills:**
- `skill_unifyweaver_compile.md` - Basic compilation
- `skill_transpiler_extension.md` - Adding custom targets

**Documentation:**
- `docs/BINDING_MATRIX.md` - Binding coverage by target
- `education/book-10-sql-target/` - SQL target tutorial

**Code:**
- `src/unifyweaver/targets/sql_target.pl` - SQL compilation
- `src/unifyweaver/fuzzy/` - Fuzzy logic DSL
- `src/unifyweaver/targets/go_target.pl` - Stream aggregation example
