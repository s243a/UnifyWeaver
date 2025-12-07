# Chapter 5: Semantic Crawling & RDF Processing

The Go target includes a powerful semantic runtime that enables AI-driven crawling and processing of RDF/XML documents. This chapter explores how to use `semantic_search/3` and `crawler_run/2` to build intelligent data acquisition pipelines.

## The Semantic Runtime

Unlike standard compilation which generates standalone Go code, semantic predicates inject a dedicated runtime library (`unifyweaver/targets/go_runtime/...`) that handles:

1.  **Vector Embeddings**: Generating embeddings for text using ONNX models (e.g., `all-MiniLM-L6-v2`).
2.  **Vector Storage**: Storing and retrieving objects and their embeddings in `bbolt`.
3.  **Semantic Search**: Finding relevant content based on meaning rather than keywords.
4.  **Crawling**: Traversing links in RDF/XML documents.

## Semantic Search

The `semantic_search/3` predicate allows you to query your stored data using natural language.

### Syntax
```prolog
semantic_search(Query, TopK, Results)
```

-   `Query`: A string describing what you are looking for.
-   `TopK`: The number of results to return.
-   `Results`: A list of result objects (currently printed to stdout in the generated code).

### Example
```prolog
find_relevant_docs(Query) :-
    semantic_search(Query, 5, _Results).
```

When compiled, this generates a Go program that:
1.  Initializes the embedding model.
2.  Embeds the `Query` string.
3.  Searches the vector store for the nearest neighbors.
4.  Prints the matching IDs and scores.

## Intelligent Crawling

The `crawler_run/2` predicate starts a crawler that fetches RDF/XML documents, flattens them into JSON-like structures, and stores them with embeddings.

### Syntax
```prolog
crawler_run(Seeds, MaxDepth)
```

-   `Seeds`: A list of starting URLs (strings).
-   `MaxDepth`: The maximum depth to crawl.

### Example
```prolog
start_crawl :-
    crawler_run([
        "http://example.org/data/graph1.xml",
        "http://example.org/data/graph2.xml"
    ], 3).
```

### RDF/XML Processing

The crawler automatically handles RDF/XML documents. It uses a heuristic flattening strategy:
1.  **Attributes**: XML attributes become fields (e.g., `rdf:about` -> `@rdf:about`).
2.  **Text Content**: Element text is stored in the `text` field.
3.  **Child Elements**: Nested elements become nested objects or arrays.
4.  **Embeddings**: If a `text` field is present, it is automatically embedded and stored for semantic search.

## Building a Semantic Pipeline

You can combine these features to build a "crawl-then-search" pipeline:

1.  **Ingest**: Use `crawler_run/2` to fetch data from the web or internal RDF sources.
2.  **Index**: The crawler automatically embeds and indexes the content.
3.  **Query**: Use `semantic_search/3` to retrieve insights from the crawled data.

```prolog
% Full pipeline
run_pipeline :-
    % Step 1: Crawl
    crawler_run(["http://example.org/knowledge-base.xml"], 2),
    
    % Step 2: Search
    semantic_search("concepts related to logic programming", 10, _).
```

## Requirements

To run generated semantic programs, you need:
1.  **ONNX Runtime**: The Go runtime uses `github.com/owulveryck/onnx-go`.
2.  **Model File**: An ONNX model file (e.g., `model.onnx`) in the `models/` directory.
3.  **Internet Access**: For the crawler to fetch URLs.

---

## Navigation

**←** [Previous: Chapter 4: JSON Processing](04_json_processing) | [📖 Book 6: Go Target](./) | [Next: Book 7: Cross-Target Glue →](../book-07-cross-target-glue/)
