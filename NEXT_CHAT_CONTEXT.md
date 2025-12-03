# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified (Vector & CPU/Text).
- **Bookmark Suggestions**: Verified.
- **Key Construction**: **Ported to Python**.
    - Added `generate_key(Strategy, Var)` to Python target.
    - Supports `composite([F1, F2])`, `hash(E)`, `uuid()`.
    - Allows declarative construction of composite keys for SQLite storage.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Validated Flows
1.  **Full Graph RAG**: Index -> Vector Search -> Context -> LLM.
2.  **Bookmark Suggestions**: Index -> `suggest_bookmarks(...)`.
3.  **Key Generation**: `process(Rec) :- generate_key(composite([field(name), literal(_), field(id)]), Key), ...`

## Active Proposals
### PowerShell Semantic Target
- **Goal**: Native PowerShell implementation of XML streaming and Vector Search.
- **Status**: Proposal accepted (`POWERSHELL_SEMANTIC_PROPOSAL.md`).
- **Next Step**: Implement `src/unifyweaver/targets/powershell_target.pl`.

## Next Steps
1.  **Merge `feat/python-keys`**: Contains the new key generation logic.
2.  **Start PowerShell Implementation**.