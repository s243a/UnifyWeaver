# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified (Vector & CPU/Text).
- **Bookmark Suggestions**: Verified.
- **Key Construction**:
    - **Python**: Verified (`generate_key/2`).
    - **Rust**: **Partially Implemented**.
        - `generate_key/2` compiler logic implemented in `rust_target.pl`.
        - Supports `composite`, `hash`, `uuid`.
        - Dependencies (`sha2`, `hex`, `uuid`) added to Cargo generation.
- **Rust Semantic Runtime**: Proposed (`RUST_SEMANTIC_PROPOSAL.md`).
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Active Proposals
### 1. PowerShell Semantic Target
- **Status**: Accepted. Pending implementation.

### 2. Rust Semantic Target
- **Status**: Phase 1 (Key Construction) Complete.
- **Next Steps**:
    - Implement `rust_runtime` library (importer, crawler).
    - Implement `crawler_run` and `graph_search` in Rust target.

## Next Steps
1.  **Merge `feat/rust-keys`**.
2.  **Start PowerShell Implementation** (as originally planned before Rust diversion).
