# feat(go): Semantic Runtime with Hugot & Bbolt

## Summary
This PR introduces the **Go Semantic Runtime Library** and integrates it into the `go_target` compiler. This enables Go-based agents to perform semantic crawling, embedding generation, and vector similarity search using high-performance native tools.

## Runtime Components
- **Embeddings**: [`knights-analytics/hugot`](https://github.com/knights-analytics/hugot) wrapper for ONNX transformer models.
- **Storage**: [`bbolt`](https://github.com/etcd-io/bbolt) wrapper for key-value storage of objects and vectors.
- **Search**: In-memory cosine similarity search over vectors loaded from `bbolt`.
- **Crawler**: Concurrent-ready crawler using `net/http` and `encoding/xml`.

## Compiler Integration
- **`go_target.pl`**: Updated to recognize semantic predicates:
    - `semantic_search(Query, TopK, Results)`
    - `crawler_run(Seeds, MaxDepth)`
- **Code Generation**: Generates Go code that imports the runtime library (`unifyweaver/targets/go_runtime/...`) and orchestrates the semantic pipeline.

## Usage
```prolog
% Search and crawl
search_and_crawl(Topic) :-
    semantic_search(Topic, 10, Seeds),
    crawler_run(Seeds, 2).
```

## Next Steps
- Ensure the generated Go code can resolve the local runtime module (e.g., via `go.mod` replace directive or by publishing the runtime).