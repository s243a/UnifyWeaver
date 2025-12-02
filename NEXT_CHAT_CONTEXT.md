# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity achieved (Python, Go, C#).
- **Graph RAG**: **Successfully Verified** on Python target (`feat/rag-integration` branch).
    - Implemented `graph_search/3` (Vector Search + Graph Traversal).
    - Validated with `examples/semantic_playbook.pl` on `context/PT` data.
    - Link extraction added to `PtCrawler`.
- **Environment**: `gemini` CLI v0.18.4 installed. `pwsh` NOT available.

## Key Architectures
### Python Semantic Runtime (`src/unifyweaver/targets/python_runtime/`)
- **`crawler.py`**: Now extracts `rdf:resource` links and stores them in the `links` table.
- **`searcher.py`**: Added `graph_search` method for 1-hop neighborhood retrieval (Parents/Children).
- **`python_target.pl`**: Updated to compile `graph_search/4` predicates.

### Validated Flow (RAG)
1.  **Index**: `PtCrawler` -> `PtImporter` (SQLite `objects`, `links`, `embeddings`).
2.  **Search**: `PtSearcher.graph_search` (Vector -> Graph Expansion).
3.  **Answer**: `LLMProvider` (Gemini CLI) summarizes the graph context.

## Active Proposals
### PowerShell Semantic Target (`POWERSHELL_SEMANTIC_PROPOSAL.md`)
- **Goal**: Achieve parity (XML Source, Vector Search) via pure PowerShell code generation.
- **Plan**:
    - XML: `[System.Xml.XmlReader]` for streaming.
    - Vectors: `[Reflection.Assembly]::LoadFile` for ONNX, or pure PS math for small scale.
    - **Next Step**: Implement `src/unifyweaver/targets/powershell_target.pl` to emit this code.

## Next Steps
1.  **Merge `feat/rag-integration`**: The RAG features are tested and working.
2.  **Start PowerShell Implementation**: Create the compiler infrastructure for the proposed PowerShell semantic features.
