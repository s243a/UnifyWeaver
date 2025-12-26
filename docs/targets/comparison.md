# Target Comparison Matrix

This document compares all UnifyWeaver compilation targets to help choose the appropriate backend for each deployment scenario.

## Full Comparison Matrix

| Capability | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|------------|------|-----|------|--------|------------|----------|-----|
| **Artefact Form** | Shell script | Go binary | Rust binary | Python script | C# source | Query IR | SQL queries |
| **Runtime Deps** | POSIX shell | None | None | Python 3 | .NET | .NET | Database |
| **Binary Output** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### Execution Model

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|---------|------|-----|------|--------|------------|----------|-----|
| Facts | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Single Rules | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multiple Rules (OR) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Recursion Support

| Pattern | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|---------|------|-----|------|--------|------------|----------|-----|
| Tail Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Linear Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Mutual Recursion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Transitive Closure | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ (CTE) |

### Aggregations

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|---------|------|-----|------|--------|------------|----------|-----|
| count/sum/avg | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| min/max | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **stddev** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **median** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **percentile** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **collect_list** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **collect_set** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |

### Window Functions

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|---------|------|-----|------|--------|------------|----------|-----|
| row_number | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| rank/dense_rank | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **LAG** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **LEAD** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **first_value** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **last_value** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |

### Observability

| Feature | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|---------|------|-----|------|--------|------------|----------|-----|
| **Progress Reporting** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| **Error Logging** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| **Error Threshold** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Metrics Export** | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ |

### Data Sources

| Source | Bash | Go | Rust | Python | C# Codegen | C# Query | SQL |
|--------|------|-----|------|--------|------------|----------|-----|
| JSONL Stream | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| CSV/Delimited | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| XML | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Database | ❌ | ✅ (BoltDB) | ❌ | ⚠️ (SQLite) | ❌ | ❌ | ✅ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Implemented

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

### By Feature Priority

| Priority | Best Targets |
|----------|--------------|
| Statistical aggregations | Go, Rust, Python |
| Window functions | Go, SQL |
| Database integration | Go (BoltDB), SQL |
| Observability | Go, Rust, Python |
| Recursion | Bash, Go, Python, C# Query |
| Type safety | Rust, Go, C# |

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
