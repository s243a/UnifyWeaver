# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG (Vector)**: Implemented and verified.
- **CPU-Only RAG**: Verified. Supports Text Search and Lazy Embedding.
- **Bookmark Suggestions**: **Implemented and Verified**.
    - Ported C# `FindBookmarkPlacements` logic to Python `PtSearcher`.
    - `suggest_bookmarks/2` predicate added to Python target.
    - Supports "Tree View" output for context.
    - Supports `mode(text)` for CPU-only usage.

## Validated Flows
### 1. Full Graph RAG
- Index -> Vector Search -> Context -> LLM.

### 2. Bookmark Suggestions
- Index -> `suggest_bookmarks('Topic', [mode(text)], Suggestions)`.
- Returns a formatted ASCII tree showing candidate locations for the new topic.

## Active Proposals
### PowerShell Semantic Target
- Proposal in `POWERSHELL_SEMANTIC_PROPOSAL.md`.
- **Next Step**: Implementation.

## Next Steps
1.  **Merge `feat/python-suggestions`**: Contains the new suggestion logic.
2.  **Start PowerShell Implementation**.