# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG (Vector)**: Implemented and verified.
- **CPU-Only RAG (Text + Lazy Embedding)**: **Implemented and Verified**.
    - **Ingestion**: `crawler_run(..., [embedding(false)])` skips embedding generation.
    - **Retrieval**: `graph_search(..., [mode(text)])` uses SQL LIKE to find anchors.
    - **Lazy Embedding**: When a text search finds an object without an embedding, the runtime automatically generates and stores the vector (if the model is available). This progressively builds the vector index as users query the system.
    - Verified with `examples/lazy_embedding_test.pl`.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Validated Flows
### 1. Full Graph RAG (GPU/Vector)
- Index: Crawl with embeddings.
- Search: Vector Search -> Graph Expansion -> LLM.

### 2. CPU Graph RAG (Text)
- Index: Crawl WITHOUT embeddings (`embedding(false)`).
- Search: Text Search (SQL LIKE) -> Graph Expansion -> LLM.
- **Benefit**: Runs fast on any environment (CI, weak VMs, Termux).

### 3. Progressive Vectorization
- Start with CPU-only index.
- Run Text Searches.
- Runtime transparently adds vectors for found items.
- Future searches can use Vector mode for those items.

## Active Proposals
### PowerShell Semantic Target
- Proposal in `POWERSHELL_SEMANTIC_PROPOSAL.md`.
- Still pending implementation.

## Next Steps
1.  **Push Lazy Embedding changes**: PR and merge `feat/cpu-rag-lazy`.
2.  **PowerShell Implementation**: Start the Code Generation work.
