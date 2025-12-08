# Future Work: Rust Vector Search

## Overview

This document outlines potential improvements and next steps for UnifyWeaver's Rust vector search implementation using Candle ML framework.

---

## Short-Term Improvements (Next Sprint)

### 1. Test Additional Embedding Models

**Priority**: Medium
**Effort**: Low

Test the remaining documented models to complete the evaluation matrix:

- **intfloat/e5-base-v2** (768-dim, 512 ctx)
  - Already documented in MODEL_DOWNLOAD.md
  - Should work with current BERT implementation
  - Provides higher quality than e5-small-v2
  - Useful for comparing 768-dim models with shorter context

**Benefits**:
- Complete model comparison data
- Validate BERT implementation with larger models
- Provide more options for different use cases

---

### 2. Sync Test Output Directory

**Priority**: High
**Effort**: Trivial

Ensure `output/rust_vector_test/src/embedding.rs` stays in sync with source:

```bash
# Automated sync script or build process
cp src/unifyweaver/targets/rust_runtime/embedding.rs \
   output/rust_vector_test/src/embedding.rs
cargo build --release
```

**Benefits**:
- Prevents version drift
- Ensures tests use latest code
- Reduces debugging confusion

**Suggested Implementation**:
- Add Makefile or shell script for sync
- Or use Cargo workspace to share code
- Add pre-build hook to verify sync

---

### 3. Performance Benchmarking

**Priority**: Medium
**Effort**: Medium

Create systematic benchmarks comparing models:

**Metrics to Measure**:
- Embedding generation speed (embeddings/second)
- CPU vs GPU speedup factor
- Memory usage under load
- Database size vs dataset size
- Search latency (text vs semantic)
- Batch processing throughput

**Test Scenarios**:
- Single document embedding
- Batch embedding (10, 100, 1000 docs)
- Various document lengths (100, 500, 1000, 5000 tokens)
- Cold start vs warm cache
- Concurrent embedding requests

**Deliverables**:
- Performance comparison table (add to RUST_VECTOR_MODELS.md)
- Graphs showing scaling characteristics
- Recommendations for production deployment

---

## Medium-Term Features (1-2 Months)

### 4. Real-World Dataset Testing

**Priority**: High
**Effort**: Medium

Test with realistic datasets beyond Pearltrees RDF:

**Suggested Datasets**:
- Wikipedia articles (long documents, leverage 8192 context)
- Code documentation (varied length, technical content)
- Research papers (structured documents)
- User queries + knowledge base (search quality evaluation)

**Evaluation Criteria**:
- Search result relevance (manual evaluation)
- Embedding quality on domain-specific content
- Performance with large datasets (10K, 100K, 1M documents)
- Database size scaling

**Benefits**:
- Validate approach for production use
- Identify edge cases and failure modes
- Build confidence in model selection recommendations

---

### 5. Production Integration

**Priority**: High
**Effort**: High

Integrate embedding provider into UnifyWeaver's main runtime:

**Tasks**:
- Wire up EmbeddingProvider to runtime initialization
- Add configuration system for model selection
- Implement embedding cache for frequently accessed documents
- Add persistence layer for pre-computed embeddings
- Handle model loading errors gracefully
- Add logging and monitoring

**Configuration Design**:
```toml
[embeddings]
model_type = "modernbert"  # or "bert"
model_path = "models/modernbert-base-safetensors"
device = "cuda"  # or "cpu", "metal", "auto"
cache_size = 10000  # number of embeddings to cache
precompute_on_load = true  # compute all embeddings at startup
```

**Benefits**:
- Makes vector search available to UnifyWeaver users
- Enables semantic search in production queries
- Foundation for advanced search features

---

### 6. Batch Processing Optimization

**Priority**: Medium
**Effort**: Medium

Optimize for bulk embedding generation:

**Improvements**:
- Batch tokenization (process multiple docs together)
- Dynamic batching based on document length
- Parallel processing across CPU cores
- GPU memory management for large batches
- Progress reporting for long operations

**Target Performance**:
- 100+ documents/second on GPU (short docs)
- 10+ documents/second on GPU (long docs with ModernBERT)
- Graceful degradation on CPU

**Benefits**:
- Faster initial indexing
- Better GPU utilization
- Supports large-scale deployments

---

## Long-Term Vision (3+ Months)

### 7. Advanced Model Support

**Priority**: Medium
**Effort**: High

Expand model architecture support as Candle evolves:

**Potential Additions**:
- **Multilingual models**: Support non-English embeddings
  - intfloat/multilingual-e5-base
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Specialized models**: Domain-specific embeddings
  - Code embeddings (for source code search)
  - Scientific paper embeddings
- **Newer architectures**: As Candle adds support
  - Check Candle releases for new model types
  - Evaluate emerging embedding models

**Implementation Strategy**:
- Extend ModelType enum for each new architecture
- Add architecture-specific forward pass logic
- Update documentation with new model specs
- Benchmark against existing models

---

### 8. Embedding Caching & Persistence

**Priority**: Medium
**Effort**: Medium

Implement intelligent caching to avoid recomputation:

**Features**:
- **In-memory cache**: LRU cache for hot embeddings
- **Persistent cache**: Store embeddings in database or file
- **Incremental updates**: Only recompute changed documents
- **Cache invalidation**: Handle document updates correctly

**Design Considerations**:
- Hash-based cache keys (document content → embedding)
- Configurable cache size limits
- Option to precompute all embeddings at startup
- Separate cache per model (different models = different embeddings)

**Benefits**:
- Faster startup time (no recomputation)
- Lower GPU usage in production
- Better resource utilization

---

### 9. Search Quality Improvements

**Priority**: Medium
**Effort**: Medium-High

Enhance search relevance and user experience:

**Features**:
- **Hybrid search**: Combine text search + semantic search scores
- **Query expansion**: Use embeddings to suggest related terms
- **Re-ranking**: Use embeddings to re-rank initial results
- **Faceted search**: Filter by metadata + semantic similarity
- **Relevance feedback**: Learn from user interactions

**Research Areas**:
- Optimal weighting of text vs semantic scores
- Query preprocessing (normalization, expansion)
- Context-aware search (use query context for better results)

---

### 10. Multi-Modal Embeddings

**Priority**: Low
**Effort**: Very High

Extend beyond text to other modalities:

**Potential Modalities**:
- **Images**: Embed diagrams, screenshots, visualizations
  - CLIP-based models
  - Image-text retrieval
- **Code**: Specialized code embeddings
  - CodeBERT, GraphCodeBERT
  - Function-level search
- **Structured data**: Embed tables, graphs, JSON
  - Table embeddings
  - Knowledge graph embeddings

**Use Cases**:
- Search across documentation with images
- Find code snippets semantically
- Query knowledge graphs with natural language

**Dependencies**:
- Candle support for vision models
- Multi-modal model availability
- Additional storage for non-text data

---

## Infrastructure & Tooling

### 11. Model Management Tools

**Priority**: Low
**Effort**: Medium

Create utilities to simplify model operations:

**Tools**:
- **Model downloader**: Interactive CLI for downloading models
  - `unifyweaver download-model modernbert`
  - Automatic verification of checksums
  - Resume interrupted downloads
- **Model validator**: Verify model files are correct
  - Check file integrity
  - Validate config.json structure
  - Test model loading
- **Model benchmark**: Quick performance test
  - `unifyweaver benchmark-model modernbert --device cuda`
  - Reports speed, memory usage

**Benefits**:
- Easier onboarding for new users
- Reduces setup errors
- Provides quick performance insights

---

### 12. Documentation Improvements

**Priority**: Medium
**Effort**: Low-Medium

Enhance documentation based on user feedback:

**Additions**:
- **Performance comparison table**: Real benchmark numbers
- **Use case examples**: Code snippets for common scenarios
- **Troubleshooting guide**: Expand with more edge cases
- **Best practices**: Model selection, caching, batching
- **Architecture diagrams**: Visual explanation of system

**Interactive Elements**:
- Model selection wizard (based on requirements)
- Configuration generator (based on system specs)
- Cost calculator (RAM, GPU, storage estimates)

---

### 13. Continuous Integration

**Priority**: Medium
**Effort**: Medium

Add CI/CD for embedding models:

**Tests**:
- **Model loading tests**: Verify all models load correctly
- **Forward pass tests**: Ensure inference works
- **Regression tests**: Catch embedding quality degradation
- **Performance tests**: Track speed over time

**Automation**:
- Run tests on each commit
- Benchmark on main branch merges
- Generate performance reports
- Alert on significant regressions

**Benefits**:
- Catch breaking changes early
- Track performance trends
- Build confidence in releases

---

## Research & Exploration

### 14. Custom Model Fine-Tuning

**Priority**: Low
**Effort**: Very High

Explore fine-tuning models for specific domains:

**Approach**:
- Start with pre-trained model (e.g., ModernBERT-base)
- Fine-tune on domain-specific data
- Evaluate improvement on target task
- Document fine-tuning process

**Challenges**:
- Requires domain-specific training data
- GPU resources for training
- Validation of fine-tuned quality
- Model distribution/versioning

**Potential Domains**:
- UnifyWeaver query language syntax
- Software documentation search
- Domain-specific knowledge bases

---

### 15. Quantization & Optimization

**Priority**: Low
**Effort**: High

Reduce model size and inference time:

**Techniques**:
- **Quantization**: INT8 or FP16 instead of FP32
- **Pruning**: Remove less important weights
- **Distillation**: Train smaller model to mimic larger one
- **ONNX Runtime**: Optimize inference with ONNX

**Benefits**:
- Lower memory usage
- Faster inference (especially on CPU)
- Smaller model files
- Better mobile/edge deployment

**Trade-offs**:
- Potential quality loss
- Compatibility with Candle
- Additional complexity

---

## Summary of Priorities

### High Priority (Do Next)
1. ✅ Sync test output directory
2. ✅ Real-world dataset testing
3. ✅ Production integration

### Medium Priority (Important)
1. Test additional models (e5-base-v2)
2. Performance benchmarking
3. Batch processing optimization
4. Search quality improvements
5. Documentation improvements
6. Continuous integration

### Low Priority (Future Exploration)
1. Advanced model support
2. Multi-modal embeddings
3. Model management tools
4. Custom fine-tuning
5. Quantization & optimization

---

## Contributing

If you'd like to work on any of these items:
1. Check if there's an existing issue/PR
2. Open an issue to discuss approach
3. Reference this document in your PR
4. Update this document as work progresses

---

## Related Documents

- [RUST_VECTOR_MODELS.md](RUST_VECTOR_MODELS.md) - Model specifications and comparisons
- [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) - Model download instructions
- [README.md](../README.md) - Project overview
