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

## Semantic Seed Selection for Focused Crawling

### Overview

One of the most powerful features is **semantic seed selection**: using vector similarity search to automatically find relevant starting points for graph crawling. Instead of manually specifying crawler seeds (e.g., `["physics-001", "physics-002"]`), you can use natural language queries to discover them.

**Use Case**: "I want to build a focused subset of my knowledge graph containing only physics-related content, but I don't know the IDs of physics documents."

**Solution**: Use semantic search to find the top 100 most physics-related documents, then crawl from those seeds to capture their children and related content.

### API: GetSeedIds

The `PtSearcher.GetSeedIds()` method returns document IDs ranked by semantic relevance:

```csharp
using var searcher = new PtSearcher("full_database.db", embeddingProvider);

// Find top 100 physics-related documents (all types)
var seedIds = searcher.GetSeedIds(
    query: "physics",
    topK: 100,           // Return top 100 matches
    minScore: 0.40       // Recommended threshold for broad topics
);

// Find top 100 physics trees only (more diverse coverage)
var treeSeedIds = searcher.GetSeedIds(
    query: "physics",
    topK: 100,
    minScore: 0.40,
    typeFilter: "pt:Tree"  // Only return trees, exclude pearls
);

Console.WriteLine($"Found {seedIds.Count} seed documents");
// Output: Found 69 seed documents (at threshold 0.40)
```

**Parameters**:
- `query`: Natural language description of desired topic
- `topK`: Maximum number of seeds to return (default 100)
- `minScore`: Minimum similarity score 0-1 (default 0.5 for quality seeds)
- `typeFilter`: Optional type filter (e.g., "pt:Tree" for trees only, null for all types)

**Returns**: List of document IDs sorted by relevance (highest scores first)

**Why Filter by Type?**

Filtering to trees only (`typeFilter: "pt:Tree"`) provides **more diverse coverage** of the knowledge graph:
- ✅ **Without filter**: May return many pearls from the same tree (redundant starting points)
- ✅ **With tree filter**: Each seed is a distinct organizational node, ensuring broad graph coverage

### Threshold Tuning Guide

**IMPORTANT**: These recommendations are based on **cosine similarity** using the **all-MiniLM-L6-v2** embedding model (384 dimensions). Different models may require different thresholds.

**Cosine Similarity Metric:**
- Formula: `dot(a, b) / (||a|| * ||b||)`
- Range: 0.0 (orthogonal/unrelated) to 1.0 (identical semantic direction)
- Normalized vectors ensure scores reflect semantic similarity, not magnitude

**Threshold Recommendations** (tested on 11,867 Pearltrees documents):

| Threshold | Tree Matches | Coverage | Use Case | Interpretation |
|-----------|--------------|----------|----------|----------------|
| **0.60+** | 6 | Minimal | Exact match only | Perfect semantic alignment - same topic, same terminology |
| **0.50** | 21 | Conservative | High precision | Strong semantic match - core topic with standard terminology |
| **0.40** | 69 | Balanced | **Recommended** | Good semantic match - includes subtopics and related concepts |
| **0.35** | 118 | Comprehensive | Broad exploration | Moderate match - captures specialized subtopics and tangential areas |
| **0.30** | 239 | Very broad | Maximum recall | Weak but relevant - includes loosely related domains |

**Example: Query "physics" on 11,867 Pearltrees documents**

Threshold 0.60 captures:
- ✅ "Physics" (1.000)
- ✅ "Introductory Physics" (0.739)
- ✅ "mechanics" (0.702)
- ❌ Misses "Quantum Mechanics" (0.502) - **too strict!**

Threshold 0.40 captures:
- ✅ All core physics topics
- ✅ "Quantum Mechanics" (0.502)
- ✅ "Classical Mechanics" (0.544)
- ✅ "Energy" (0.555)
- ❌ Excludes "Statistics" (0.362), "advanced calculus" (0.354)

Threshold 0.35 captures:
- ✅ All physics and specialized subtopics
- ✅ "advanced calculus" (0.354) - math foundation
- ✅ "Material Engineering" (0.359) - applied physics
- ✅ ~118 trees for "top 100" broad searches

**Recommendations:**

1. **For broad topic discovery** (e.g., "physics", "machine learning"): Use **0.35-0.40**
   - Captures 100-150 seeds
   - Balances coverage vs noise
   - Includes specialized subtopics

2. **For specific concept search** (e.g., "quantum field theory"): Use **0.50-0.60**
   - Focuses on exact matches
   - Reduces false positives
   - Targets precise terminology

3. **For maximum recall** (research, exploration): Use **0.30**
   - Casts wide net
   - Accepts more tangential matches
   - Good for discovering unexpected connections

**Pro Tip**: Start with 0.40, then adjust based on your first search results. If you're missing obvious relevant content, lower it to 0.35. If you're getting too much noise, raise it to 0.50.

### API: RunSemanticCrawl

The `PtHarness.RunSemanticCrawl()` method combines semantic search with fixed-point crawling:

```csharp
PtHarness.RunSemanticCrawl(
    seedQuery: "physics",
    sourceDb: "full_database.db",           // Database with embeddings
    targetDb: "physics_subset.db",          // Output crawl subset
    embeddingProvider: embeddingProvider,
    fetchConfig: id => new XmlSourceConfig { /* config for fetching by ID */ },
    topSeeds: 150,       // Find top 150 physics documents
    minScore: 0.40,      // Recommended threshold for broad topics
    maxDepth: 3,         // Crawl 3 hops from seeds
    typeFilter: "pt:Tree"  // Only use trees as seeds (recommended for diversity)
);
```

**Workflow**:
1. Search `sourceDb` for documents matching `seedQuery`
2. Apply `typeFilter` if specified (e.g., only trees)
3. Select top `topSeeds` documents with `score >= minScore`
4. Start fixed-point crawl from those seeds
5. Traverse up to `maxDepth` hops following children/parent relationships
6. Write focused subset to `targetDb`

**Result**: A new database containing only the semantically-relevant subset of your knowledge graph.

**Recommended**: Use `typeFilter: "pt:Tree"` to ensure diverse starting points and avoid clustering seeds within the same organizational node.

### Example: Building a Physics-Focused Subset

```csharp
// Load embedding provider
var embeddingProvider = new OnnxEmbeddingProvider(
    "models/all-MiniLM-L6-v2/model.onnx",
    "models/all-MiniLM-L6-v2/vocab.txt"
);

// Run semantic crawl
PtHarness.RunSemanticCrawl(
    seedQuery: "quantum mechanics relativity particle physics",
    sourceDb: "pearltrees_full.db",      // 11,867 documents
    targetDb: "pearltrees_physics.db",   // Output subset
    embeddingProvider: embeddingProvider,
    fetchConfig: id => new XmlSourceConfig {
        InputPath: $"xml_fragments/{id}.xml",  // Fetch by ID
        RecordSeparator: RecordSeparatorKind.LineFeed,
        NamespacePrefixes: new Dictionary<string, string> {
            {"http://www.pearltrees.com/rdf/0.1/#", "pt"}
        },
        TreatPearltreesCDataAsText: true
    },
    topSeeds: 120,       // Request ~100-120 seeds
    minScore: 0.35,      // Balanced threshold for comprehensive coverage
    maxDepth: 3,         // Crawl 3 hops deep
    typeFilter: "pt:Tree"  // Only trees as seeds (avoids pearl clustering)
);

// Result: pearltrees_physics.db contains ~500-2000 physics documents
// (depends on graph connectivity from those 100 tree seeds)
// Using tree filter ensures diverse starting points across different topics
```

### Comparison: Manual vs Semantic Seeds

**Manual Seed Selection (old way)**:
```csharp
var seeds = new[] { "physics-001", "physics-002", "quantum-042" };
// ❌ Requires knowing IDs ahead of time
// ❌ May miss relevant content
// ❌ Tedious for large graphs
```

**Semantic Seed Selection (new way)**:
```csharp
var seeds = searcher.GetSeedIds("quantum mechanics physics", topK: 100, minScore: 0.5);
// ✅ Automatic discovery via natural language
// ✅ Comprehensive coverage (top 100 matches)
// ✅ Configurable quality threshold
```

### Use Cases

1. **Domain-Specific Subsets**: Extract ML/AI content from a general knowledge graph
2. **Exploratory Research**: Find documents related to "transformer neural networks" and their connections
3. **Data Filtering**: Build a clean subset before sharing (e.g., exclude sensitive topics)
4. **Incremental Crawling**: Start with high-quality seeds, expand gradually
5. **Multi-Topic Extraction**: Run multiple semantic crawls with different queries

### Performance

For the 11,867 document Pearltrees dataset:
- **Seed discovery**: <1 second (brute-force search over all embeddings)
- **Crawl execution**: Depends on graph size and depth
- **Result**: Focused subset with 10-50% of original data, but 100% topic-relevant

### Test Output

Example from `tmp/pt_ingest_test/Program.cs`:

```
=== Semantic Crawl Demo ===
Finding seeds via semantic search: "quantum mechanics physics" (type: pt:Tree)
Found 2 seed documents (minScore >= 0.3)
Starting fixed-point crawl from 2 seeds (maxDepth=2)...
Semantic crawl complete
Semantic crawl result: 3 documents in physics-focused subset
SEMANTIC_CRAWL_OK
```

**Note**: With `typeFilter: "pt:Tree"`, only 2 trees were selected as seeds (excluding 1 pearl), ensuring diverse organizational coverage. The crawl then expanded from those 2 tree nodes to capture their children, resulting in a focused physics subset.

This demonstrates the full workflow: semantic search → type filtering → seed selection → focused crawl → subset creation.
