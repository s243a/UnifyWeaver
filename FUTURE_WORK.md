# Future Work & Enhancement Ideas

This document captures ideas for future development of UnifyWeaver targets and features.

## Known Issues

(None currently open)

### Resolved Issues

#### Go Target: test_go_group_by.pl Failure (Fixed 2025-12-20)
✅ **Fixed**: The `test_go_group_by.pl` failure was due to incorrect test usage (passing atoms instead of variables to `group_by`). The test has been corrected and now passes.

---

## Go Target Enhancements

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

### Stream Processing Enhancements (Completed)

✅ **Implemented**: High-performance concurrency and observability.
- **Parallel Processing** - Goroutine-based concurrent record processing with schema validation (`workers(N)`).
- **Error Aggregation** - Collect validation/parsing errors to a separate JSONL file (`error_file(Path)`).
- **Progress Reporting** - Log processed record counts to stderr at configurable intervals (`progress(interval(N))`).
- **Buffered Channels** - Configurable channel buffer size for parallel workers (`buffer_size(N)`).

### Stream Processing Enhancements (Planned)

- **Error Thresholds** - Option to fail if error count exceeds a threshold
- **Metrics Export** - Export processing metrics to Prometheus/JSON

## Rust Target (Completed)

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

## Cross-Target Features

### Query Optimization (The "Codd" Phase)

- **Join Optimization** - Automatically reorder rule bodies to prioritize ground or highly selective goals. Minimize intermediate result set sizes.
- **Index Hints** - Allow manual index selection via `:- index(predicate/arity, field).` directive.
- **Statistics** - Collect and use statistics (cardinality, selectivity) for query planning.
- **Predicate Pushdown** - Push filters as deep as possible into the data source (e.g., SQL `WHERE` or Bbolt range queries).

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

### Distributed Execution

Investigate distributed compilation:
- Split predicates across nodes
- Coordinate via message passing
- Aggregate results

### AI/ML Integration

Explore integration with machine learning:
- Feature extraction from structured data
- Training data preparation
- Model validation pipelines

## Priority Ranking

**Immediate (Next 1-2 Milestones):**
1. ✅ Advanced JSON features (arrays, advanced schemas)
2. ✅ Stream processing enhancements (Parallel Processing)
3. Error Aggregation & Progress Reporting

**Short Term (Next 3-6 Months):**
4. Query optimization basics


**Medium Term (6-12 Months):**
6. WebAssembly target
7. Complete ETL framework
8. Performance optimization suite

**Long Term (12+ Months):**
9. Incremental maintenance
10. Distributed execution
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

### Held-Out Test Set Evaluation
**Description**: Conduct a rigorous evaluation of the multi-head semantic search model's performance on a held-out test set of queries. This involves creating a dedicated test set with novel queries and their expected answers to assess generalization beyond training data.
**Goal**: Obtain realistic Recall@1 and MRR metrics for the unified multi-head model.
**Related**: `scripts/validate_multi_head_search.py` (to be created), `playbooks/lda-training-data/raw/qa_pairs_test.json` (to be created).

### Hyperparameter Tuning
**Description**: Experiment with different `temperature` values (τ) for the multi-head model to optimize retrieval performance (Recall@1, MRR) on a held-out validation set.
**Goal**: Fine-tune the multi-head model for optimal balance between sharp and diffuse routing.
**Related**: `scripts/train_multi_head_projection.py` (`--temperature` flag).

### Advanced Multi-Head Architectures
**Description**: Explore "Future Work" ideas from the Multi-Head LDA Projection Theory document, such as "Learnable Temperature" or "Hierarchical Multi-Head" routing for scenarios with very large numbers of clusters.
**Goal**: Enhance model capacity and accuracy for more complex or larger knowledge bases.
**Related**: `docs/proposals/MULTI_HEAD_PROJECTION_THEORY.md`.

### User-Friendly CLI Tool
**Description**: Create a more streamlined command-line interface or sub-command (e.g., `unifyweaver search`) for accessing the semantic search functionality provided by `scripts/skills/lookup_example.py`.
**Goal**: Improve developer experience and direct usability of the search feature.

### Database Cleanup
**Description**: Implement a utility to identify and purge "broken" clusters (those with imported structure but no associated embeddings) from the `lda.db` database. This occurred due to initial failures in the embedding process.
**Goal**: Maintain a clean and consistent database state.
**Related**: `scripts/migrate_to_lda_db.py`.

### Playbook Clarity Review
**Description**: Review all updated playbooks (`playbooks/*_playbook.md`) for clarity and consistency in their instructions, especially regarding the new "Finding Examples" section. Ensure example IDs and semantic search queries are accurate and helpful.
**Goal**: Optimize agent understanding and execution reliability.
