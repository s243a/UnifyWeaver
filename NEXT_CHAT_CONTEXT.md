# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified.
- **Key Construction**: Verified.
- **Query Filters**: **Implemented & Verified**.
    - Ported Go target filtering logic to Python procedural mode.
    - Supports `Age >= 18`, `X \= Y`, etc.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Validated Flows
1.  **Full Graph RAG**: Index -> Search -> Context.
2.  **Bookmark Suggestions**: Tree View output.
3.  **Filtering**: Declarative filtering in procedural Python pipelines.

## Active Proposals
### PowerShell Semantic Target
- **Status**: Ready for implementation.

## Next Steps
1.  **Merge `feat/python-filters`**.
2.  **Start PowerShell Implementation**.