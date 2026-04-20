# Target Comparison Matrix

This document compares all UnifyWeaver compilation targets to help choose the appropriate backend for each deployment scenario.

## Full Comparison Matrix

| Capability | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|------------|------|-----|------|--------|------------|----------|-----|---------|
| **Artefact Form** | Shell script | Go binary | Rust binary | Python script | C# source | Query IR | SQL queries | `.wasm` module |
| **Runtime Deps** | POSIX shell | None | None | Python 3 | .NET | .NET | Database | WASM runtime |
| **Binary Output** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Execution Model

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| Facts | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Single Rules | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multiple Rules (OR) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Recursion Support

| Pattern | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| Tail Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Linear Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Mutual Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Transitive Closure | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ (CTE) | ✅ |

### Aggregations

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| count/sum/avg | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| min/max | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| **stddev** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **median** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **percentile** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **collect_list** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **collect_set** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |

### Window Functions

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| row_number | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| rank/dense_rank | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **LAG** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **LEAD** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **first_value** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **last_value** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |

### Joins

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| Inner Join | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LEFT OUTER** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| **RIGHT OUTER** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| **FULL OUTER** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |

### Observability

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|---------|------|-----|------|--------|------------|----------|-----|---------|
| **Progress Reporting** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| **Error Logging** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| **Error Threshold** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Metrics Export** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ | ❌ |

### Data Sources

| Source | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL | WAM-WAT |
|--------|------|-----|------|--------|------------|----------|-----|---------|
| JSONL Stream | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ | ❌ |
| CSV/Delimited | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ | ❌ |
| XML | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Database | ❌ | ✅ (BoltDB) | ✅ (sled) | ⚠️ (SQLite) | ❌ | ❌ | ✅ | ❌ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Implemented

### WAM-WAT notes

The WAM-WAT column reflects the WAM interpreter's feature set
rather than per-backend code generation:

- **Aggregations** (⚠️): The WAM instruction set includes
  `begin_aggregate` / `end_aggregate` framing, but the richer
  aggregation DSL that the source-code backends expose isn't
  fully wired through this target.
- **Outer joins** (⚠️): Disjunctions like
  `(left(X, Y) ; X = null)` execute correctly via `try_me_else`
  chains, but the backend-specific pattern detection that the
  Bash/Go/Rust targets use to generate specialized join code is
  absent. You get "whatever WAM gives you" rather than
  "optimized for join shape".
- **Observability / data sources** (❌): WAM-WAT is a pure
  predicate executor. It doesn't ingest data streams or emit
  metrics. Predicates either encode facts directly or are
  called with pre-materialized arguments.

See [`wam-wat.md`](wam-wat.md) for the target overview and
[`../design/WAM_WAT_ARCHITECTURE.md`](../design/WAM_WAT_ARCHITECTURE.md)
for the implementation-level reference.

---

## Target Selection Guide

### By Use Case

| Use Case | Recommended Target | Reason |
|----------|-------------------|--------|
| Quick prototyping | Bash | Ubiquitous, fast iteration |
| High-performance ETL | Go or Rust | Native binaries, efficient |
| Container deployment | Go | Single binary, no deps |
| Memory-critical systems | Rust | Zero-cost abstractions |
| Data science workflows | Python | ML integration, libraries |
| Enterprise .NET stack | C# Query | LINQ integration |
| Database analytics | SQL | Native window functions |
| Analytics with LAG/LEAD | Go or SQL | Full window function support |
| Browser / sandboxed execution | WAM-WAT | Portable WASM module |
| Cross-runtime predicate logic | WAM-WAT | Same `.wasm` runs anywhere |

### By Feature Priority

| Priority | Best Targets |
|----------|--------------|
| Statistical aggregations | Go, Rust, Python |
| Window functions | Go, SQL |
| Database integration | Go (BoltDB), SQL |
| Observability | Go, Rust, Python |
| Recursion | Bash, Go, Python, C# Query, WAM-WAT |
| Type safety | Rust, Go, C# |
| Portable deployment | WAM-WAT, Go |

---

## Combined `target(csharp)` Behaviour

To keep user experience simple, the generic `target(csharp)` option will behave as a smart facade:
1. Attempt `csharp_codegen` for predicates and features the codegen backend supports.
2. Fall back to `csharp_query` when recursion or other unsupported patterns appear.
3. Provide diagnostics explaining which backend handled each predicate so operators can tune preferences.

This approach gives immediate value from the existing code generator while allowing incremental rollout of the query runtime without breaking existing scripts.

---

## See Also

- [Target Overview](overview.md)
- [Go Target](go.md)
- [Rust Target](rust.md)
- [SQL Target](sql.md)
- [Bash Target](bash.md)
- [C# Targets](csharp-codegen.md)
- [WAM-WAT Target](wam-wat.md)
