# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
We have achieved **Semantic Capability Parity** across the core targets (Python, Go, C#). The system can now:
1.  **Crawl**: Fetch and parse XML/HTML/RDF (`PtCrawler`, `xml_source`).
2.  **Embed**: Generate vector embeddings using local ONNX models (`all-MiniLM-L6-v2`) on CPU/GPU.
3.  **Store**: Index objects and vectors in local databases (`sqlite` for Python, `bbolt` for Go, `litedb` for C#).
4.  **Search**: Perform cosine similarity searches (`semantic_search/3`).
5.  **Generate**: (Python only) Summarize/Answer using LLMs via `gemini` CLI (`llm_ask/3`).

## Key Architectures

### Python Target (`python_target.pl`)
- **Runtime**: `src/unifyweaver/targets/python_runtime/`
    - `crawler.py`: XML stream processor with flattening (namespace stripping logic).
    - `onnx_embedding.py`: Wraps `onnxruntime` (defaults to CPU provider).
    - `chunker.py`: Hierarchical chunking (Macro/Micro) with configurable sizes (defaults to safe 250 tokens).
    - `llm.py`: Wraps `gemini` CLI for `llm_ask`.
- **Compilation**: Inlines runtime code into generated scripts. Supports procedural mode for side-effects.

### Go Target (`go_target.pl`)
- **Runtime**: `src/unifyweaver/targets/go_runtime/` (uses `hugot`, `bbolt`).
- **Compilation**: Generates `main.go` that imports the local runtime module.

### Data Sources
- **SQLite**: Native `source(sqlite)` plugin with parameter binding via embedded Python.
- **XML**: Native streaming support in all targets.

## Recent Changes
- **LLM Integration**: Added `llm_ask` and `chunk_text` predicates to Python target. Defaults to `gemini-2.5-flash`.
- **Playbook**: `examples/semantic_playbook.pl` demonstrates end-to-end Index -> Search -> Summarize flow on `context/PT` data.
- **Docker**: Added `docker/` directory with full environment (though running in PRoot is limited).

## Next Steps
1.  **PowerShell**: Implement Semantic Parity (XmlReader, Vector Search).
2.  **Graph RAG**: Add predicates for traversing the link graph stored in DB.
3.  **Interactive Shell**: Create a swipl shell extension for running agent commands.
4.  **Testing**: Ensure `gemini` CLI is configured in the environment for the RAG tests to fully pass.
