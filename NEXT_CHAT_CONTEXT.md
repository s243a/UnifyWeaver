# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified.
- **Bookmark Suggestions**: Verified.
- **Key Construction**: Verified.
- **Rust Semantic Runtime**: **Phase 2 Almost Complete**.
    - `importer.rs` (Redb), `crawler.rs` (quick-xml), `searcher.rs` (Text/Graph), `llm.rs` (Gemini) are implemented.
    - `rust_target.pl` now copies these files to the generated project.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` missing.

## Active Proposals
### 1. PowerShell Semantic Target
- **Status**: Accepted. Pending implementation.

### 2. Rust Semantic Target
- **Status**: Runtime files created.
- **Next Steps**:
    - Implement `compile_predicate_to_rust` logic to *use* these runtime files (generate `main.rs` calls).
    - Implement `translate_goal` for `crawler_run`, `graph_search`.

## Next Steps
1.  **Merge `feat/rust-runtime-v2`**.
2.  **Finish Rust Compiler Integration**.
3.  **Start PowerShell Implementation**.
