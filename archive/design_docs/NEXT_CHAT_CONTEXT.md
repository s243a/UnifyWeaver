# UnifyWeaver Context for Next Session

## Project State (Dec 2025)
- **Core Targets**: Semantic Capability Parity (Python, Go, C#).
- **Graph RAG**: Verified (Python).
- **Bookmark Suggestions**: Verified (Python).
- **Key Construction**: Verified (Python/Rust).
- **Rust Semantic Runtime**: Implemented & Integrated.
- **PowerShell Semantic Target**: **Implemented**.
    - `powershell_target.pl`: Native semantic compiler.
    - **XML Streaming**: Verified `[System.Xml.XmlReader]` generation.
    - **Vector Search**: Verified Pure PowerShell Cosine Similarity.
- **Environment**: `gemini` CLI v0.18.4. `pwsh` available (with GC limit).

## Next Steps
1.  **Merge `feat/powershell-semantic`**.
2.  **Future**: Full Graph RAG implementation in PowerShell (requires choosing a DB strategy, likely SQLite via .NET or JSONL linear scan).