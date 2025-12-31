# Pearltrees Rust Bookmark Filing

Complete Rust implementation for semantic bookmark organization using vector embeddings and tree-based context.

## Overview

This implementation demonstrates a production-ready bookmark filing system that:
- Ingests XML data using AWK-based streaming
- Generates semantic embeddings using BERT models via candle-transformers
- Performs vector similarity search for intelligent bookmark placement
- Builds hierarchical tree context for placement suggestions

## Architecture

### Core Components

**Ingestion Pipeline** (`rust_ingest.rs`)
- Reads null-delimited XML fragments from stdin (AWK preprocessor)
- Parses nested XML with CDATA support for extracting titles
- Generates 384-dimensional embeddings using all-MiniLM-L6-v2
- Stores objects and embeddings in embedded redb database

**XML Parser** (`crawler.rs`)
- Namespace-agnostic attribute lookup via `get_local()` helper
- Nested element parsing with CDATA extraction
- Handles `<dcterms:title><![CDATA[...]]></dcterms:title>` patterns
- Supports both Text and CData XML events

**Embedding Provider** (`embedding.rs`)
- BERT model integration via candle-transformers
- Automatic device selection (CUDA > Metal > CPU)
- Mean pooling with L2 normalization
- 384-dimensional sentence embeddings

**Database Layer** (`importer.rs`)
- redb embedded key-value store
- Three tables: objects, embeddings, links
- Atomic upsert operations
- JSON serialization for objects

**Vector Search** (`searcher.rs`)
- Cosine similarity search over embeddings
- Tree context building with ancestors/siblings
- Bookmark filing with placement suggestions
- LLM integration for intelligent recommendations

## Quick Start

### Prerequisites

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Download embedding model (all-MiniLM-L6-v2 in SafeTensors format)
mkdir -p models/all-MiniLM-L6-v2-safetensors
# Download model.safetensors, tokenizer.json, config.json to the above directory
```

### Build

```bash
cd examples/pearltrees
cargo build --release
```

### Ingestion

Process XML data using AWK fragment extraction:

```bash
# Extract XML fragments and pipe to Rust ingester
awk -f ../../scripts/utils/select_xml_elements.awk \
    -v tag="pt:Tree|pt:RefPearl|pt:Pearl" \
    pearltrees_export.rdf | \
    ./target/release/rust_ingest
```

This creates `pt_ingest_test.redb` with all objects and embeddings.

### Search

```bash
# Vector search for quantum physics content
./target/release/test_rust_search "quantum entanglement" 10
```

Example output:
```
Top 10 results:
 1. [0.8672] quantum-mechanics
 2. [0.8638] semi-classical-quantum-physics
 3. [0.8558] kinetic-quantum-mechanics
 ...
```

### Bookmark Filing Demo

```bash
cargo run --bin demo_bookmark_filing
```

Demonstrates intelligent bookmark placement with tree context.

## Performance

### Tested Scale
- **Dataset**: 11,867 XML fragments (full Pearltrees export)
- **Database**: 73MB with objects and embeddings
- **Search**: Sub-second semantic queries
- **Ingestion**: ~10 minutes on CPU (all-MiniLM-L6-v2)

### Optimization Opportunities
- GPU acceleration with CUDA support
- Batch embedding generation
- Vector index structures (HNSW, IVF)
- Streaming ingestion with progress tracking

## XML Parsing Features

### Namespace-Agnostic Lookups

The `get_local()` helper finds attributes regardless of namespace:

```rust
// Matches: "title", "@title", "dcterms:title", "@dcterms:title"
let text = get_local(&obj, "title")
    .or_else(|| get_local(&obj, "about"))
    .or_else(|| obj.get("text"))
    .and_then(|v| v.as_str());
```

### Nested Element Parsing

Properly extracts CDATA content from nested elements:

```xml
<pt:Pearl rdf:about="https://example.com/id123">
  <dcterms:title><![CDATA[Article Title]]></dcterms:title>
  <pt:category>Science</pt:category>
</pt:Pearl>
```

Extracted as:
```json
{
  "@rdf:about": "https://example.com/id123",
  "dcterms:title": "Article Title",
  "pt:category": "Science"
}
```

## Dependencies

```toml
[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = "0.20"
redb = "2.0"
quick-xml = "0.36"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
```

## File Structure

```
examples/pearltrees/
├── Cargo.toml           # Rust project configuration
├── rust_ingest.rs       # Main ingestion binary
├── crawler.rs           # XML parser with CDATA support
├── embedding.rs         # BERT embedding provider
├── importer.rs          # redb database operations
├── searcher.rs          # Vector search and filing
├── llm.rs               # LLM integration
├── demo_bookmark_filing.rs  # Interactive demo
└── README.md            # This file
```

## Integration with UnifyWeaver

This implementation uses the shared Rust runtime components:

- `src/unifyweaver/targets/rust_runtime/crawler.rs` - Core XML parser
- `src/unifyweaver/targets/rust_runtime/importer.rs` - Database layer

The examples demonstrate how to build domain-specific applications on top of the UnifyWeaver runtime infrastructure.

## Future Enhancements

- **GPU Acceleration**: Enable CUDA features for faster embedding generation
- **Incremental Updates**: Support for partial re-indexing
- **Graph Search**: Leverage relationship links for improved context
- **Multi-Model Support**: Experiment with different embedding models
- **REST API**: Expose search and filing via HTTP endpoints

## Testing Results

Full dataset testing confirms:
- ✅ 11,867 fragments processed without errors
- ✅ Semantic search returns relevant results (0.86+ similarity)
- ✅ Namespace-agnostic parsing handles all XML variants
- ✅ CDATA extraction works correctly
- ✅ No UTF-8 encoding issues

## References

- [candle](https://github.com/huggingface/candle) - Rust ML framework
- [redb](https://github.com/cberner/redb) - Embedded key-value database
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence embedding model
- [quick-xml](https://github.com/tafia/quick-xml) - Fast XML parser
