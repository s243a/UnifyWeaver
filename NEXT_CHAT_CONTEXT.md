# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified.
- **Bookmark Suggestions**: Verified.
- **Key Construction**: Verified (Python & Rust Phase 1).
- **Rust Semantic Runtime**: **Phase 2 Initiated**.
    - `importer.rs` (Redb) and `crawler.rs` (quick-xml) created in `src/unifyweaver/targets/rust_runtime/`.
    - `rust_target.pl` updated to detect `redb`/`quick-xml` dependencies.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Active Proposals
### 1. PowerShell Semantic Target
- **Status**: Accepted. Pending implementation.

### 2. Rust Semantic Target
- **Status**: Phase 2 (Runtime Foundation) In Progress.
- **Next Steps**:
    - Create `searcher.rs` (Vector/Text Search).
    - Implement the compiler logic to inline these runtime files into the generated project.
    - Implement `crawler_run` translation.

## Next Steps
1.  **Merge `feat/rust-runtime-v1`**.
2.  **Finish Rust Runtime** (Searcher, LLM).
3.  **Start PowerShell Implementation**.