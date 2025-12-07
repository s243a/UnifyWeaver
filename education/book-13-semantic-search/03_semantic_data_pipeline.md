# Chapter 3: The Semantic Data Pipeline

In the previous chapter, we saw how to *search* data. But how does the data get there?

The Python target includes a robust **Semantic Data Pipeline** capable of ingesting, flattening, and embedding complex hierarchical data (like XML/RDF) into a queryable format.

## 3.1 The Pipeline Stages

The pipeline consists of three stages, orchestrated by `PtCrawler`:

1.  **Fetch**: Retrieve raw content (XML files, URLs).
2.  **Flatten**: Convert hierarchical XML into flat Python dictionaries.
3.  **Enrich & Store**:
    *   **Embed**: Generate vector embeddings for text.
    *   **Link**: Extract graph relationships (`pt:parentTree`).
    *   **Persist**: Save to SQLite.

```mermaid
graph LR
    A[XML Source] -->|lxml iterparse| B(Flattened Dict)
    B -->|Text| C[ONNX Embedder]
    B -->|Attributes| D[Link Extractor]
    C -->|Vector| E[(SQLite: Embeddings)]
    D -->|Source/Target| F[(SQLite: Links)]
    B -->|JSON| G[(SQLite: Objects)]
```

## 3.2 XML Flattening Strategy

UnifyWeaver uses a specific "Flattening" strategy optimized for RDF and Property Graphs.

**Input XML (RDF):**
```xml
<pt:PagePearl rdf:about="http://pearltrees.com/id/123">
    <dcterms:title>My Page</dcterms:title>
    <pt:parentTree rdf:resource="http://pearltrees.com/id/456" />
</pt:PagePearl>
```

**Flattened Python Dict:**
```python
{
    "@tag": "PagePearl",
    "@about": "http://pearltrees.com/id/123",  # Attributes get '@' prefix
    "title": "My Page",                        # Child text becomes key
    # pt:parentTree is processed as a link, not just data
}
```

This structure allows the `PtSearcher` to return clean, JSON-serializable data objects without the complexity of the original DOM.

## 3.3 Link Extraction

One of the most powerful features of the `PtCrawler` is automatic link extraction. It scans the flattened data for standard relationship attributes:

*   `rdf:resource`
*   `resource`
*   Attributes ending in `}resource` (Namespaced)

When found, it creates a directed edge in the `links` table:
*   **Source**: The current object's ID (`@about` or `@id`).
*   **Target**: The value of the resource attribute.

This builds the graph that enables **Graph RAG**.

## 3.4 Running a Crawl

To trigger this pipeline, use the `crawler_run/2` predicate in your Prolog script:

```prolog
index_my_data :-
    % Crawl 'data.rdf', expanding links up to depth 2
    crawler_run(['data.rdf'], 2).
```

The compiler generates Python code that initializes the `PtCrawler` and executes the pipeline efficiently using streams.

---

## Navigation

**←** [Previous: Chapter 2: Graph Retrieval-Augmented Generation (G...](02_graph_rag) | [📖 Book 13: Semantic Search](./) | [Next: Chapter 4: Logic and Recursion in Python →](04_logic_and_recursion)
