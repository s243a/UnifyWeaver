# Future Work & Enhancement Ideas

This document captures ideas for future development of UnifyWeaver targets and features.

## Known Issues

(None currently open)

### Resolved Issues

#### Go Target: test_go_group_by.pl Failure (Fixed 2025-12-20)
✅ **Fixed**: The `test_go_group_by.pl` failure was due to incorrect test usage (passing atoms instead of variables to `group_by`). The test has been corrected and now passes.

---

## Go Target Enhancements

### Go Target Enhancements

### Database Integration (Completed)

✅ **Implemented**: `bbolt` support is now available in the Go target.
- Pure Go, ACID transactions.
- Use `db_backend(bbolt)` option.

### Advanced JSON Features (Completed)

✅ **Implemented**: Full support for complex JSON structures and validation.
- **Array Support**: Iterate over JSON arrays using `json_array_member/2`.
- **Advanced Schema Validation**: Min/max, regex format, and type checking.
- **Schema Composition**: Nested `object(Schema)` types with recursive deep validation.
- **Variable Mapping**: robust typed compilation for arbitrary projection.

### Custom Functions and Bindings (Completed)

✅ **Implemented**: Extensibility for Go target.
- **Custom Components**: Inject raw Go code via `custom_go` component type.
- **Binding Directives**: Declare bindings in user code with `:- go_binding(...)`.
- **Component Registry**: Full integration for code generation.

### Database Aggregations (Completed)

✅ **Implemented**: Comprehensive aggregation support (Phase 9).
- **Simple Aggregations**: `count`, `sum`, `avg`, `max`, `min`.
- **Grouped Aggregations**: `group_by` with single or multiple operations.
- **Advanced Features**: `HAVING` clause, nested grouping (Phase 9c).
- **Statistical Functions**: `stddev`, `median`, `percentile` (Phase 9d).
- **Array Aggregations**: `collect_list`, `collect_set` (Phase 9e).
- **Window Functions**: `row_number`, `rank`, `dense_rank` (Phase 9f).

### Stream Processing Enhancements (Completed)

✅ **Implemented**: High-performance concurrency and observability.
- **Parallel Processing** - Goroutine-based concurrent record processing with schema validation (`workers(N)`).
- **Error Aggregation** - Collect validation/parsing errors to a separate JSONL file (`error_file(Path)`).
- **Progress Reporting** - Log processed record counts to stderr at configurable intervals (`progress(interval(N))`).
- **Error Thresholds** - Fail if error count exceeds limit (`error_threshold(count(N))`).
- **Metrics Export** - Export processing statistics to JSON (`metrics_file(Path)`).
- **Buffered Channels** - Configurable channel buffer size for parallel workers (`buffer_size(N)`).

### Rust Target (Completed)

✅ **Implemented**: The Rust target is now available with support for:
- Core compilation (facts, rules, constraints, aggregations)
- Regex matching (`regex` crate)
- JSON I/O (`serde`, `serde_json`)
- Project generation (`Cargo.toml`)

See [docs/RUST_TARGET.md](docs/RUST_TARGET.md) for details.


## Other Target Ideas

### WebAssembly (WASM)

Compile predicates to WASM for browser/edge execution:
- JSON processing in browsers
- Edge computing with Cloudflare Workers
- Serverless functions

### Zig

Modern systems language with manual memory management:
- Simpler than Rust but still safe
- Great C interop
- Compile-time execution

### Julia

Scientific computing and data analysis:
- Built-in DataFrames
- Excellent performance
- Great for numerical processing

---

## Cross-Target Feature Parity

This section tracks features that exist in some targets but should be ported to others.

### Quick Wins (Completed)

#### 1. Statistical Aggregations for Rust
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (has stddev, median, percentile)
**Target:** Rust target

Implemented:
- `stddev` - Standard deviation (sample)
- `median` - Median value
- `percentile(N)` - Nth percentile

#### 2. collect_list/collect_set for Rust
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go, Python, C# Query targets
**Target:** Rust target

Implemented:
- `collect_list` - Aggregate into JSON array (with duplicates)
- `collect_set` - Aggregate into sorted JSON array (unique values)

#### 3. Window Functions for Go
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** SQL target (native), Python target
**Target:** Go target

Implemented:
- `lag/3`, `lag/4`, `lag/5` - Access previous row value with offset and default
- `lead/3`, `lead/4`, `lead/5` - Access next row value with offset and default
- `first_value/3` - First value in window partition
- `last_value/3` - Last value in window partition

#### 4. Observability for Rust
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (comprehensive), Python target
**Target:** Rust target

Implemented:
- `progress(interval(N))` - Progress reporting every N records
- `error_file(Path)` - Error logging to JSON file
- `error_threshold(count(N))` - Exit after N errors
- `metrics_file(Path)` - Performance metrics export

#### 5. Window Functions for Rust
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (full window function support)
**Target:** Rust target

Implemented:
- `lag/3`, `lag/4`, `lag/5` - Access previous row value with offset and default
- `lead/3`, `lead/4`, `lead/5` - Access next row value with offset and default
- `first_value/3` - First value in window partition
- `last_value/3` - Last value in window partition
- `row_number/2`, `rank/2`, `dense_rank/2` - Ranking functions

### Medium Priority

#### 6. XML Processing (Rust)
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (streaming + flattening)
**Target:** Rust target

Implemented:
- `compile_rust_xml_mode/4` - XML input mode compilation
- Streaming XML parsing with `quick-xml` crate
- Attribute extraction (prefixed with `@`)
- Text content extraction
- Tag filtering with `tags([...])` option
- Deduplication with `unique(true)`
- File and stdin input sources

#### 7. Full Outer Joins
**Status:** ✅ PARTIAL (2025-12-25)
**Source:** Go, Python, Rust, SQL targets
**Target:** Bash ✅, C# Codegen ✅, PowerShell 📋

Implemented for Bash and C# Codegen:
- LEFT JOIN: `(LeftGoals, (RightGoal ; X = null))`
- RIGHT JOIN: `((LeftGoal ; X = null), RightGoals)`
- FULL OUTER JOIN: `((L ; L = null), (R ; R = null))`
- Automatic pattern detection and optimized code generation
- Bash: Nested loops with associative arrays for deduplication
- C# LINQ: GroupJoin + SelectMany + DefaultIfEmpty patterns

#### 8. Schema Validation for JSON (Rust)
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (comprehensive)
**Target:** Rust target

Implemented:
- `field(Name, Type, Options)` - Field definitions with options
- `min(N)`, `max(N)` - Numeric and string length constraints
- `format(email)`, `format(date)` - Format validation
- `required`, `optional` - Field presence requirements
- `object(SchemaName)` - Nested schema validation
- `generate_rust_schema_validator/2` - Code generation for validators

### Lower Priority (Specialized)

#### 9. Database Integration
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go target (BoltDB with secondary indexes)
**Target:** Rust target (sled embedded database)

Implemented:
- `db_backend(sled)` - Enable sled persistence for Rust target
- `db_file(Path)` - Database file path configuration
- `db_tree(Name)` - Tree/bucket name (default: predicate name)
- `db_key_field(Field)` - Single field as primary key
- `db_key_strategy(Strategy)` - Complex key generation (field, composite, hash, uuid)
- `db_mode(read|write|analyze)` - Database operation modes
- Secondary index support with automatic index updates
- Predicate pushdown optimization (direct lookup, prefix scan, index scan)
- Statistics collection for cost-based optimization (`db_mode(analyze)`)

#### 10. Cost-Based Optimization
**Status:** 📋 PLANNED
**Source:** Go target (statistics-based)
**Target:** All targets
**Effort:** High - requires statistics collection infrastructure

### Feature Parity Matrix

| Feature | Go | Python | Rust | C# | Bash | SQL |
|---------|-----|--------|------|-----|------|-----|
| Statistical Aggs | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| collect_list/set | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Window Functions | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |
| Observability | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| XML Processing | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Full Outer Join | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Database Integration | ✅ | ⚠️ | ✅ | ❌ | ❌ | ✅ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Missing

---

## Cross-Target Features

### Query Optimization (Completed - The "Codd" Phase)

✅ **Implemented**: Heuristic-based join reordering.
- **Join Optimization** - Automatically reorder rule bodies to prioritize ground or highly selective goals. Minimize intermediate result set sizes. Respects variable dependencies.
- **Constraint Integration** - Integrated into Stream, Recursive, Go, and Rust targets.

✅ **Implemented**: Secondary Indexes and Predicate Pushdown (Phase 8b).
- **Index Hints** - Manual index selection via `:- index(predicate/arity, field).` directive.
- **Predicate Pushdown** - Pushes equality filters into Bbolt using secondary indexes (`cursor.Seek`) and prefix scans.
- **Key Optimization** - Automatically detects direct lookup and prefix scan opportunities.

✅ **Implemented**: Cost-Based Optimization (Statistics).
- **Cost Model** - Goal reordering based on estimated execution cost (cardinality * selectivity).
- **Statistics Collection** - `db_mode(analyze)` generates tools to collect Bbolt bucket statistics.

### Query Optimization (Planned)

### Incremental Compilation

- **Partial Recompilation** - Only recompile changed predicates
- **Dependency Tracking** - Smart rebuild based on dependencies
- **Cache Generated Code** - Avoid regenerating identical code

### Testing Infrastructure

- **Property-Based Testing** - Generate random inputs for validation
- **Benchmark Suite** - Performance regression testing
- **Cross-Target Tests** - Ensure consistent behavior across targets

### Documentation

- **Interactive Tutorial** - Step-by-step guide with examples
- **Video Demos** - Screen recordings of real-world usage
- **Cookbook** - Common patterns and solutions
- **Performance Guide** - Optimization tips per target

## Integration Projects

### Data Pipeline Framework

Complete ETL framework using UnifyWeaver:
1. Ingest from various sources (JSON, CSV, APIs)
2. Transform with Prolog logic
3. Validate with schemas
4. Store in embedded databases
5. Export to various formats

### Real-World Applications

**Log Processing:**
- Parse web server logs
- Extract patterns
- Store in database
- Generate reports

**Data Migration:**
- Read from legacy formats
- Transform and validate
- Write to modern databases
- Maintain data integrity

**API Gateway:**
- Validate incoming requests
- Transform data formats
- Route to backends
- Cache results

## Research Areas

### Incremental Maintenance

Explore incremental view maintenance:
- Only recompute changed results
- Maintain materialized views efficiently
- Delta processing for updates

### Distributed Execution (Completed)

✅ **Implemented**: Distributed parallel execution backends are now available.
- **Hadoop Streaming**: MapReduce via stdin/stdout for any language
- **Hadoop Native**: In-process JVM API (Java, Scala, Kotlin, Clojure)
- **Apache Spark**: PySpark + native JVM modes (Java, Scala, Kotlin, Clojure)
- **Dask Distributed**: Python distributed computing with threads/processes/cluster
- **JVM Glue**: Complete cross-language bridges for JVM ecosystem

See `docs/proposals/parallel_architecture.md` and `examples/demo_distributed_backends.pl` for details.

### AI/ML Integration

Explore integration with machine learning:
- Feature extraction from structured data
- Training data preparation
- Model validation pipelines

## Priority Ranking

**Immediate Priorities:**
1.  **Semantic Search Validation** (Held-Out Test Set)
2.  **Optimizations** (Eliminate allocations)
3.  **Semantic Runtime** (Vector embeddings integration)

**Short Term (Next 3-6 Months):**


**Medium Term (6-12 Months):**
6. WebAssembly target
7. Complete ETL framework
8. Performance optimization suite

**Long Term (12+ Months):**
9. Incremental maintenance
10. ~~Distributed execution~~ ✅ COMPLETE (v0.0.5)
11. AI/ML integration

## Contributing

Ideas from the community are welcome! If you want to work on any of these:
1. Open an issue to discuss the approach
2. Check existing PRs to avoid duplication
3. Start with a design document
4. Implement with tests
5. Update documentation

## References

- Go database comparison: https://github.com/avelino/awesome-go#database
- Rust database ecosystem: https://lib.rs/database
- Embedded databases overview: https://dbdb.io/browse?type=embedded

## Semantic Search Enhancements and Validation

### Held-Out Test Set Evaluation (Completed)
✅ **Implemented**: Created dedicated test set and validation script.
- **Test Set**: `playbooks/lda-training-data/raw/qa_pairs_test.json` contains novel queries.
- **Script**: `scripts/validate_multi_head_search.py` evaluates Recall@1 and MRR on the test set.

### Hyperparameter Tuning
**Description**: Experiment with different `temperature` values (τ) for the multi-head model to optimize retrieval performance (Recall@1, MRR) on a held-out validation set.
**Goal**: Fine-tune the multi-head model for optimal balance between sharp and diffuse routing.
**Related**: `scripts/train_multi_head_projection.py` (`--temperature` flag).

### Advanced Multi-Head Architectures
**Description**: Explore "Future Work" ideas from the Multi-Head LDA Projection Theory document, such as "Learnable Temperature" or "Hierarchical Multi-Head" routing for scenarios with very large numbers of clusters.
**Goal**: Enhance model capacity and accuracy for more complex or larger knowledge bases.
**Related**: `docs/proposals/MULTI_HEAD_PROJECTION_THEORY.md`.

### User-Friendly CLI Tool (Completed)
✅ **Implemented**: `./unifyweaver search` command available.
- Wraps `scripts/skills/lookup_example.py` for easy access to semantic search.

### Database Cleanup (Completed)
✅ **Implemented**: `scripts/cleanup_lda_db.py` utility.
- Identifies and purges broken clusters (no embeddings) from `lda.db`.
- Supports dry-run and forced deletion.

### Playbook Clarity Review (Completed)
✅ **Implemented**: Updated all playbooks to use `./unifyweaver search` CLI.
- Standardized "Finding Examples" section across all 35+ playbooks.
- Verified example IDs in sampled playbooks.
