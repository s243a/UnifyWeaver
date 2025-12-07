# Chapter 5: Semantic Crawling and Vector Search

In this chapter, we explore one of the most powerful capabilities of the UnifyWeaver C# target: **Semantic Crawling**. Unlike traditional crawlers that follow links blindly or based on regex patterns, a semantic crawler uses vector embeddings to understand the *meaning* of the content it encounters, allowing it to prioritize relevant information and perform "focused crawling."

## 5.1 The Semantic Runtime Architecture

The C# target's semantic capabilities are built on a set of specialized runtime components that work together:

1.  **`PtCrawler`**: The orchestrator. It fetches XML/HTML content, parses it, and directs the flow of data.
2.  **`PtImporter`**: The storage engine. It wraps **LiteDB** (a NoSQL database) to store:
    *   **Objects**: Flattened representations of the crawled data (e.g., Trees, Pearls).
    *   **Embeddings**: Vector representations of text content.
3.  **`OnnxEmbeddingProvider`**: The brain. It runs a local **ONNX** transformer model (like `all-MiniLM-L6-v2`) to convert text into 384-dimensional vectors.
4.  **`PtSearcher`**: The retriever. It performs cosine similarity searches over the stored vectors to find relevant content.

### Data Flow

```mermaid
graph LR
    A[Source (XML/HTTP)] -->|Fetch| B(PtCrawler)
    B -->|Text| C(OnnxEmbeddingProvider)
    C -->|Vector| B
    B -->|Object + Vector| D(PtImporter / LiteDB)
    E[User Query] -->|Text| F(PtSearcher)
    F -->|Vector| D
    D -->|Results| F
```

## 5.2 Configuring the Embedder

To enable semantic features, your C# application needs access to a pre-trained ONNX model.

1.  **Download the Model**: Get `model.onnx` and `vocab.txt` (e.g., from HuggingFace `sentence-transformers/all-MiniLM-L6-v2`).
2.  **Configure the Provider**:

```csharp
var embedder = new OnnxEmbeddingProvider(
    modelPath: "models/model.onnx",
    vocabPath: "models/vocab.txt"
);
```

## 5.3 Building a Semantic Crawler

In UnifyWeaver, you define the crawler logic in Prolog, and the compiler generates the C# wiring.

### Prolog Definition

```prolog
% Define a semantic search predicate
find_relevant_seeds(Query, Seeds) :-
    semantic_search(Query, 10, Results),
    extract_ids(Results, Seeds).

% Define the crawl logic
crawl_topic(Topic) :-
    find_relevant_seeds(Topic, Seeds),
    crawler_run(Seeds, 3). % Depth 3
```

### Generated C# Logic

The compiler generates a `main` function that initializes the runtime:

```csharp
using UnifyWeaver.QueryRuntime;

// ... inside Main ...
var store = new PtImporter("data.db");
var embedder = new OnnxEmbeddingProvider("model.onnx", "vocab.txt");
var crawler = new PtCrawler("data.db", sourceConfig, embedder);

// Run crawl
crawler.FixedPoint(seedIds, id => GetConfig(id), maxDepth: 3);
```

## 5.4 Vector Search

Once data is crawled and embedded, you can perform semantic searches. This finds documents that are *conceptually similar* to your query, not just keyword matches.

```csharp
var searcher = new PtSearcher("data.db", embedder);
var results = searcher.SearchSimilar("quantum physics applications", topK: 5);

foreach (var result in results)
{
    Console.WriteLine($"{result.Score:F4}: {result.Title}");
}
```

## 5.5 Case Study: Physics Knowledge Base

Imagine building a personal knowledge base from a structured site like Pearltrees or a Wiki.

1.  **Seed**: Start with a general "Physics" category.
2.  **Crawl**: Fetch children. Embed their titles.
3.  **Filter**: Only follow links where `CosineSimilarity(child.Vector, physicsVector) > 0.4`.
4.  **Store**: Save relevant pages to LiteDB.

This creates a highly focused dataset without downloading the entire internet.

## 5.6 Summary

The Semantic Runtime transforms UnifyWeaver from a simple logic-to-script compiler into a tool for building intelligent agents. By leveraging local ONNX models and vector storage, your generated C# programs can understand and organize information at a semantic level.

---

## Navigation

**←** [Previous: Chapter 4: Runtime Libraries and Deployment](04_runtime_libraries_deployment) | [📖 Book 3: C# Target](README) | [Next: Chapter 3: PowerShell Semantic Target (XML & Vecto... →](06_powershell_semantic)
