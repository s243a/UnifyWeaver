# feat(go): Semantic Runtime with Hugot & Bbolt

## Summary
This PR introduces the **Go Semantic Runtime Library**, providing the foundation for building semantic agents in Go. It mirrors the Python runtime architecture but leverages Go-native (or CGO-wrapped) tools for high performance.

## Components
- **Embeddings**: Uses [`knights-analytics/hugot`](https://github.com/knights-analytics/hugot) to run ONNX transformer models (e.g., `all-MiniLM-L6-v2`) locally for generating vector embeddings.
- **Storage**: Uses [`bbolt`](https://github.com/etcd-io/bbolt) for pure Go key-value storage of objects and vectors.
- **Search**: Implements in-memory cosine similarity search over vectors loaded from `bbolt`.
- **Crawler**: Implements a concurrent-ready crawler using `net/http` and `encoding/xml` with flattening support.

## Architecture
The runtime is structured as a library in `src/unifyweaver/targets/go_runtime/`:
- `embedder/`: Hugot wrapper.
- `storage/`: Bbolt wrapper.
- `search/`: Vector operations.
- `crawler/`: Crawling logic.

## Next Steps
- Integrate this runtime into `go_target.pl` to allow compiling predicates like `semantic_search/3` into Go code that imports these modules.
