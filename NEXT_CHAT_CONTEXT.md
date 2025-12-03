# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG (Vector)**: Implemented and verified.
- **CPU-Only RAG**: **Implemented and Verified**.
    - `crawler_run/3` now supports `embedding(false)`.
    - `graph_search/5` supports `mode(text)` (SQL LIKE).
    - Verified with `examples/cpu_rag_playbook.pl`.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Validated Flows
### 1. Full Graph RAG (GPU/Vector)
- Index: Crawl with embeddings.
- Search: Vector Search -> Graph Expansion -> LLM.

### 2. CPU Graph RAG (Text)
- Index: Crawl WITHOUT embeddings (`embedding(false)`).
- Search: Text Search (SQL LIKE) -> Graph Expansion -> LLM.
- **Benefit**: Runs fast on any environment (CI, weak VMs, Termux).

## Active Proposals
### PowerShell Semantic Target
- Proposal in `POWERSHELL_SEMANTIC_PROPOSAL.md`.
- Still pending implementation.

## Next Steps
1.  **Push CPU RAG changes**: PR and merge `feat/cpu-rag`.
2.  **PowerShell Implementation**: Start the Code Generation work.