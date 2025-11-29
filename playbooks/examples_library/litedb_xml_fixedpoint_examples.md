<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# LiteDB XML Fixed-Point Crawl (C#)

## `unifyweaver.execution.litedb_xml_fixedpoint`

> [!example-record]
> id: unifyweaver.execution.litedb_xml_fixedpoint
> name: LiteDB XML Fixed-Point Crawl
> platform: csharp

This example shows how to:
- Stream XML fragments (with pearltrees defaults) via `XmlStreamReader`
- Upsert projected records into LiteDB (POCO + Raw `BsonDocument`)
- Iteratively fetch children (parentTree/seeAlso) for N hops (e.g., 5)
- Filter out private entries (`pt:privacy > 1`)

### Files
- `test_data/test_xml_fragments.txt` — scrubbed sample (pearltrees-style, physics-themed)
- `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` — `XmlStreamReader` with namespace/CDATa support
- (This example uses `BsonDocument` for Raw and a POCO for typed fields.)

### Suggested pipeline (outline)
1. Define a dynamic XML source with `pearltrees(true)` and `record_format(xml)`.
2. Project to a row dictionary with keys: `id`, `title`, `privacy`, `parentTree`, `children[]`, plus `Raw` for the whole map.
3. Upsert into LiteDB (`_id = id`).
4. Query LiteDB for next-hop child tree IDs (from `children` or `parentTree`).
5. Repeat for a bounded number of iterations (crawler supports `FixedPoint(seeds, configForId, maxDepth)`; demo uses 2 hops).

### Minimal C# harness (conceptual)
```csharp
var xmlConfig = new XmlSourceConfig {
    InputPath = "test_data/scrubbed_pearltrees.xml",
    RecordSeparator = RecordSeparatorKind.LineFeed,
    NamespacePrefixes = new Dictionary<string,string> {
        { "http://www.pearltrees.com/rdf/0.1/#", "pt" },
        { "http://purl.org/dc/elements/1.1/", "dcterms" }
    },
    TreatPearltreesCDataAsText = true
};

using var crawler = new PtCrawler("pearltrees.db", xmlConfig);
crawler.IngestOnce(emitEmbeddings: true); // single pass + dummy embeddings
// Example fixed-point crawl (demo): seeds = ["physics-001"], config always same XML, maxDepth=2
// crawler.FixedPoint(seeds, id => BuildConfigForId(id), maxDepth: 2);
```

### Testing
A working test harness is available in `tests/core/test_pearltrees_csharp.pl` that:
- Runs a console app (`tmp/pt_ingest_test/`) to test ingestion
- Verifies correct type detection (pt:Tree vs pt:RefPearl)
- Checks that documents are properly inserted into LiteDB collections

Run the test with:
```bash
swipl -q -g "run_tests([pearltrees_csharp])" -t halt tests/core/test_pearltrees_csharp.pl
```

The test successfully ingests the scrubbed sample (`test_data/scrubbed_pearltrees.xml`):
- 2 trees with Type: `pt:Tree`
- 1 pearl with Type: `pt:RefPearl`

### Notes
- `Raw` can capture the full dictionary projection for unmapped fields.
- The XML parser recognizes empty lines as fragment delimiters and automatically sets the Type field from the root element name.

## Vector Embeddings and Semantic Search

### Overview

The system now supports **real vector embeddings** and **semantic search** over Pearltrees data using ONNX Runtime with the all-MiniLM-L6-v2 model.

### Features

**Embedding Generation:**
- 384-dimensional sentence embeddings via all-MiniLM-L6-v2 (ONNX)
- ~100MB RAM footprint, fully offline operation
- Pluggable architecture (`IEmbeddingProvider` interface)
- Embeddings stored in LiteDB alongside entities

**Vector Similarity Search:**
- Cosine similarity search via `PtSearcher`
- Top-k results with configurable score thresholds
- Semantic matching across entire knowledge base

### Setup

1. **Download the ONNX model** (one-time):
```bash
mkdir -p models/all-MiniLM-L6-v2
cd models/all-MiniLM-L6-v2

# Download model and vocabulary
curl -L -o model.onnx "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
curl -L -o vocab.txt "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
```

2. **Create the embedding provider**:
```csharp
var embeddingProvider = new OnnxEmbeddingProvider(
    "models/all-MiniLM-L6-v2/model.onnx",
    "models/all-MiniLM-L6-v2/vocab.txt",
    dimensions: 384,        // Model output size
    maxLength: 512,         // Max token length (~350-400 words)
    modelName: "all-MiniLM-L6-v2"
);
```

### Usage: Ingestion with Embeddings

```csharp
// Configure XML source
var config = new XmlSourceConfig {
    InputPath = "context/PT/pearltrees_export.rdf",
    RecordSeparator = RecordSeparatorKind.LineFeed,
    NamespacePrefixes = new Dictionary<string,string> {
        { "http://www.pearltrees.com/rdf/0.1/#", "pt" },
        { "http://purl.org/dc/elements/1.1/", "dcterms" }
    },
    TreatPearltreesCDataAsText = true
};

// Ingest with embeddings
using var crawler = new PtCrawler("pearltrees.db", config, embeddingProvider);
crawler.IngestOnce(emitEmbeddings: true);
```

### Usage: Semantic Search

```csharp
using var searcher = new PtSearcher("pearltrees.db", embeddingProvider);

// Search for similar documents
var results = searcher.SearchSimilar(
    query: "quantum mechanics and physics",
    topK: 10,              // Return top 10 results
    minScore: 0.1          // Minimum similarity threshold (0-1)
);

// Display results
foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F3}] {result.Title}");
    Console.WriteLine($"  Type: {result.Type}, ID: {result.Id}");
}
```

### Real-World Performance

**Dataset**: 19MB Pearltrees RDF export
- **11,867 documents** (5,002 trees + 6,865 pearls)
- **File size**: Too large for Notepad to open comfortably!

**Processing via Streaming Pipeline:**
```
RDF (19MB) → AWK extraction → XML parsing → Namespace injection
           → Entity mapping → Embedding generation → LiteDB storage
```

**Results:**
- ✅ All 11,867 documents embedded successfully
- ✅ Processing time: Minutes (not hours)
- ✅ Memory usage: ~150MB peak (model + streaming)
- ✅ No API costs, fully offline

**Why Streaming Wins:**

Traditional approach (Notepad, text editors):
```
❌ Load entire 19MB file into memory
❌ Parse full document tree
❌ High memory pressure, slow/freeze
```

Our streaming approach:
```
✅ Process one XML fragment at a time
✅ Generate one embedding per document
✅ Write incrementally to LiteDB
✅ Constant memory usage regardless of file size
✅ Could handle 100GB+ files the same way!
```

### Search Quality Examples

**Query:** "quantum mechanics and physics"
```
1. [0.932] Quantum Mechanics
2. [0.824] Relativistic Quantum Mechanics
3. [0.810] Analysis & Quantum Mechanics
```

**Query:** "artificial intelligence machine learning"
```
1. [0.832] Machine Learning
2. [0.791] Artificial Intelligence
3. [0.607] data mining
```

**Interpretation:**
- Scores 0.8-1.0: Excellent semantic match
- Scores 0.5-0.8: Good related content
- Scores below 0.5: Weak match

### Architecture: Pluggable Embeddings

The `IEmbeddingProvider` interface supports multiple backends:

**Currently Implemented:**
- `OnnxEmbeddingProvider` - Local ONNX models (all-MiniLM-L6-v2)

**Easy to Add:**
- Together AI provider (ModernBERT for 8K token context)
- Ollama provider (nomic-embed-text, mxbai-embed-large)
- OpenAI/Cohere API providers

**Interface:**
```csharp
public interface IEmbeddingProvider
{
    double[] GetEmbedding(string text);
    int Dimensions { get; }
    string ModelName { get; }
}
```

### Performance Comparison

**Local ONNX (our approach):**
- Speed: Milliseconds per embedding
- Cost: FREE (one-time model download)
- Privacy: Fully offline, no data leaves your machine
- Scalability: Process millions of docs locally

**API-based (OpenAI, Cohere):**
- Speed: 1-2 seconds per request (network latency)
- Cost: $$$ per million tokens
- Privacy: Data sent to external service
- Scalability: Rate limits, quotas

**For 11,867 documents:**
- ONNX: ~5-10 minutes, $0
- API: ~6-12 hours, $10-50+ (depending on service)

### Technical Details

**Tokenization:**
- Simple BERT WordPiece tokenizer with vocab.txt
- Lowercase normalization
- Max 512 tokens (~350-400 words)
- Special tokens: [CLS], [SEP], [UNK], [PAD]

**Embedding Process:**
1. Tokenize text → token IDs
2. ONNX inference → hidden states (batch_size × seq_len × 384)
3. Mean pooling over sequence length (weighted by attention mask)
4. L2 normalization → final embedding vector

**Storage:**
- Embeddings stored in LiteDB `embeddings` collection
- Format: `{ _id: "doc-id", vector: [double array] }`
- No special indexing (brute-force cosine similarity search)

**Future Optimizations:**
- FAISS/HNSW indexing for large-scale search
- Batch embedding generation
- GPU acceleration via ONNX GPU providers
