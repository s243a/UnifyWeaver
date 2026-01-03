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
**Status:** ✅ COMPLETE (2025-12-25)
**Source:** Go, Python, Rust, SQL targets
**Target:** Bash ✅, C# Codegen ✅, PowerShell ✅

Implemented for Bash, C# Codegen, and PowerShell:
- LEFT JOIN: `(LeftGoals, (RightGoal ; X = null))`
- RIGHT JOIN: `((LeftGoal ; X = null), RightGoals)`
- FULL OUTER JOIN: `((L ; L = null), (R ; R = null))`
- Automatic pattern detection and optimized code generation
- Bash: Nested loops with associative arrays for deduplication
- C# LINQ: GroupJoin + SelectMany + DefaultIfEmpty patterns
- PowerShell: Hashtable lookups with PSCustomObject output

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

### Incremental Compilation (Complete)

**Status:** ✅ COMPLETE - All Phases Implemented (2025-01-02)
**Proposal:** [`docs/proposals/INCREMENTAL_COMPILATION.md`](docs/proposals/INCREMENTAL_COMPILATION.md)

**Core Features:**
- **Predicate Hashing** - Detect source changes via `term_hash/2` with variable normalization
- **Dependency Tracking** - Reverse graph traversal via `get_transitive_dependents/2`
- **Compilation Cache** - In-memory cache indexed by predicate + target + hash
- **Invalidation Cascade** - Automatic invalidation of dependent predicates
- **Multi-Target Support** - All 20 targets supported with independent caching
- **Disk Persistence** - Save/load cache to `.unifyweaver_cache/` directory
- **Cache Management** - Stats, clear, save/load commands
- **Performance** - ~11x speedup from cache hits (benchmarked)

**Optional by Design:** Incremental compilation can be disabled at multiple levels:
- Per-call: `compile_incremental(foo/2, bash, [incremental(false)], Code)`
- Per-session: `set_prolog_flag(unifyweaver_incremental, false)`
- Environment: `UNIFYWEAVER_CACHE=0`

**Implementation Phases (All Complete):**
1. ✅ Core infrastructure (hasher, cache manager)
2. ✅ Dependency integration (reverse graph traversal)
3. ✅ Compiler wrapper (Bash target proof of concept)
4. ✅ Multi-target support (all 20 targets)
5. ✅ File persistence (survive restarts)
6. ✅ CLI management commands
7. ✅ Documentation & benchmarks

**Files:**
- `src/unifyweaver/incremental/hasher.pl`
- `src/unifyweaver/incremental/cache_manager.pl`
- `src/unifyweaver/incremental/incremental_compiler.pl`
- `src/unifyweaver/incremental/cache_persistence.pl`
- `src/unifyweaver/incremental/test_integration.pl`
- `src/unifyweaver/incremental/benchmark.pl`

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

---

## Python Bridges Enhancement

Cross-runtime Python bridges allow non-Python languages to access Python's ML/AI ecosystem. The bridge infrastructure supports multiple approaches with different trade-offs.

### Tested Bridges (7 Languages)

| Language | Bridge | Technology | Status |
|----------|--------|------------|--------|
| C# | Python.NET | CPython embedding | ✅ Tested |
| C# | CSnakes | Source generators | ✅ Documented |
| Java | JPype | CPython embedding | ✅ Tested |
| Java | jpy | Bi-directional | ✅ Tested |
| Ruby | PyCall.rb | CPython embedding | ✅ Tested |
| Rust | PyO3 | In-process | ✅ Tested |
| Go | Rust FFI | Go → Rust (PyO3) → CPython | ✅ Tested |

### Rust FFI Bridge Architecture

For languages without mature CPython embedding (Go, Node.js, Lua), we use Rust as a universal bridge:

```
┌────────────────────────────────────────────────────────┐
│ Go/Node/Lua Application                                │
│ ┌────────────────────────────────────────────────────┐ │
│ │ C FFI (CGO / node-ffi / LuaJIT FFI)                │ │
│ │ ┌────────────────────────────────────────────────┐ │ │
│ │ │ Rust cdylib (librpyc_bridge.so)                │ │ │
│ │ │ ┌────────────────────────────────────────────┐ │ │ │
│ │ │ │ PyO3 (CPython embedding)                   │ │ │ │
│ │ │ │ ┌────────────────────────────────────────┐ │ │ │ │
│ │ │ │ │ RPyC Client                            │ │ │ │ │
│ │ │ │ └────────────────────────────────────────┘ │ │ │ │
│ │ │ └────────────────────────────────────────────┘ │ │ │
│ │ └────────────────────────────────────────────────┘ │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Future Bridge Improvements

#### 1. Node.js FFI Example
**Status:** ✅ COMPLETE
**Location:** `examples/python-bridges/rust-ffi-node/`
**Features:**
- Express API server with REST endpoints (`src/server.ts`)
- TypeScript FFI wrapper using koffi (`src/rpyc_bridge.ts`)
- Test script (`src/test.ts`)
- React frontend with NumPy/Math UI
- Full documentation

```bash
# Run the example
cd examples/python-bridges/rust-ffi-node
npm install && npm run dev
```

#### 2. Additional FFI Languages
**Status:** 📋 TODO
**Priority:** Medium
**Languages:** Lua (LuaJIT FFI), PHP (FFI extension), Crystal

Each language with C FFI capability can use the same Rust bridge library.

#### 3. Bridge Integration Tests
**Status:** ✅ COMPLETE
**Priority:** High
**Scope:** Automated CI testing for all 7+ bridges

```bash
# CI workflow (implemented)
./test_bridges.sh --all        # Test all bridges
./test_bridges.sh --jvm        # Test Java bridges only
./test_bridges.sh --dotnet     # Test .NET bridges only
./test_bridges.sh --ffi        # Test FFI bridges only
```

Test script: `examples/python-bridges/test_bridges.sh`

#### 4. Cross-Runtime Pipeline Examples
**Status:** 📋 TODO
**Priority:** Medium
**Scope:** Pipeline examples combining Rust FFI bridges with other targets

Example: Go extract → Python (via Rust FFI) transform → Rust persist

```prolog
% cross_runtime_pipeline.pl already supports:
predicate_runtime(rust_ffi:go:transform/2, rust_ffi_go).

% Desired pipeline
?- compile_pipeline([
       stage(extract, go:read_csv/2),
       stage(transform, rust_ffi:go:numpy_stats/2),
       stage(persist, rust:write_db/2)
   ], Options, Code).
```

#### 5. Performance Benchmarks
**Status:** 📋 TODO
**Priority:** Low
**Scope:** Compare bridge overhead across technologies

| Metric | Python.NET | JPype | PyO3 | Rust FFI (Go) |
|--------|------------|-------|------|---------------|
| Call overhead | TBD | TBD | TBD | TBD |
| NumPy array transfer | TBD | TBD | TBD | TBD |
| Connection setup | TBD | TBD | TBD | TBD |

#### 6. Go Native Bridge Monitoring
**Status:** 📋 WATCH
**Priority:** Low (wait for maturity)
**Libraries to Watch:**
- **go-python3** (DataDog): Currently archived (2021)
- **go-embed-python** (kluctl): Uses subprocess (loses live proxy benefit)

When a mature Go CPython embedding library emerges, add direct support.

### Implementation Files

| File | Purpose |
|------|---------|
| `src/unifyweaver/glue/python_bridges_glue.pl` | Bridge detection, code generation |
| `src/unifyweaver/glue/cross_runtime_pipeline.pl` | Pipeline orchestration |
| `examples/python-bridges/rust-ffi-go/` | Working Go + Rust FFI example |
| `docs/research/RPYC_LANGUAGE_COMPATIBILITY.md` | Compatibility matrix |

---

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

### Hyperparameter Tuning (Completed)
✅ **Implemented**: Optimized softmax temperature (τ) for routing.
- **Experiment**: Tested τ=1.0 (7% Recall@1) vs τ=0.1 (87% Recall@1 on training data).
- **Result**: τ=0.1 provides sharp routing and generalizes well to novel queries (68% Recall@1 vs 58% for direct similarity).

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

## C# Target Enhancements

### Procedural Recursion and Native Pipeline (Planned)

We have implemented procedural recursion in the `csharp_native` target. The next steps involve enhancing this target to reach parity with the Bash and Python pipeline models.

#### 1. Implement C# Native Pipeline
**Goal:** Implement `compile_csharp_native_pipeline/3` in `csharp_native_target.pl`.
- Currently a stub.
- Should generate a main entry point that orchestrates multiple stages (predicates) using fixpoint iteration logic similar to the Bash target's `generate_bash_pipeline_connector`.
- Enables full end-to-end pipeline execution within a single compiled C# binary.

#### 2. Refine C# Target Selection
**Goal:** Update `target(csharp)` preference logic to smartly choose between `csharp_native` (procedural) and `csharp_query` (runtime engine).
- **Evaluation Required:** We need to evaluate what the best default is. The `csharp_native` target offers standalone code and potentially better performance for simple recursion, while `csharp_query` handles complex mutual recursion and optimizations.
- **Action:** Benchmarking and analysis to determine the default strategy for `target(csharp)`.

#### 3. Extend Native Built-in Support
**Goal:** Add comprehensive support for arithmetic and string built-ins in `csharp_native_target`.
- Currently limited.
- Needs parity with other targets for standard Prolog built-ins (is/2, string manipulation, etc.) within the procedural generation mode.

#### 4. Integration Testing
**Goal:** Verify generated C# code in a full .NET environment.
- Requires an environment with the .NET SDK.
- End-to-end verification of recursive predicates and pipelines to ensure runtime correctness.