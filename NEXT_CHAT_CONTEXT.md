# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified.
- **Bookmark Suggestions**: Verified.
- **Key Construction**: Verified.
- **Rust Semantic Target**: **Phase 4 Complete (Vector Search)**.
    - Implemented `embedding.rs` (Candle/BERT).
    - Implemented `searcher.rs` (Vector/Text/Graph).
    - Updated compiler to generate full projects with dependencies.
    - **Status**: Code complete, verification limited by environment Rust version.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing. `rustc` 1.63 (too old for `anstyle`/`candle`).

## Active Proposals
### 1. PowerShell Semantic Target
- **Status**: Accepted. Pending implementation.

## Next Steps
1.  **Merge `feat/rust-vectors`**.
2.  **Start PowerShell Implementation**.
