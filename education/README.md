<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# UnifyWeaver Education

**Educational Materials for [UnifyWeaver](https://github.com/s243a/UnifyWeaver)**

This project contains comprehensive educational materials for learning UnifyWeaver - a Prolog-to-Code compiler that transforms declarative logic programs into efficient streaming scripts across multiple target languages.

## Book Series Overview

The books are designed to be read in sequence, with each building on concepts from previous books. However, after completing Books 1-3, you can explore target-specific books based on your needs.

### Core Sequence (Books 1-4)

| Book | Title | Focus |
|------|-------|-------|
| **1** | [Foundations](book-01-foundations/README.md) | Architecture, Prolog basics, preference system |
| **2** | [Bash Target](book-02-bash-target/README.md) | Stream compilation, templates, data sources |
| **3** | [C# Target](book-03-csharp-target/README.md) | .NET compilation, LINQ, **fixed-point approaches** |
| **4** | [Workflows](book-04-workflows/README.md) | AI agent playbooks, strategic planning |

### Portable Targets (Books 5-6)

| Book | Title | Focus |
|------|-------|-------|
| **5** | [Python Target](book-05-python-target/README.md) | Python code generation, widely portable |
| **6** | [Go Target](book-06-go-target/README.md) | Native binaries, cross-platform, no runtime deps |

### Integration & Security (Books 7-8)

| Book | Title | Focus |
|------|-------|-------|
| **7** | [Cross-Target Glue](book-07-cross-target-glue/README.md) | Multi-language pipelines, Phases 1-7, cloud deployment |
| **8** | [Security & Firewall](book-08-security-firewall/README.md) | Cross-target security policies, production hardening |

### Specialized Targets (Books 9-12)

| Book | Title | Focus |
|------|-------|-------|
| **9** | [Rust Target](book-09-rust-target/README.md) | Memory-safe compilation, performance-critical |
| **10** | [SQL Target](book-10-sql-target/README.md) | Database query generation |
| **11** | [Prolog Target](book-11-prolog-target/README.md) | Prolog-to-Prolog, dialect support |
| **12** | [PowerShell Target](book-12-powershell-target/README.md) | Windows automation, .NET scripting |

### Advanced Applications (Book 13)

| Book | Title | Focus |
|------|-------|-------|
| **13** | [Semantic Search](book-13-semantic-search/README.md) | Graph RAG, embeddings, vector databases |

### Supplementary Material

| Resource | Description |
|----------|-------------|
| [AWK Target](book-awk-target/README.md) | Lightweight text processing (supplementary) |
| [Notebooks](notebooks/README.md) | Interactive Jupyter notebooks |

## Learning Paths

### Path 1: Quick Start (Beginners)
1. Book 1: Foundations (Chapters 1-4)
2. Book 2: Bash Target (Chapters 1-5)
3. Try examples and exercises

### Path 2: Full Stack Developer
1. Books 1-4 (Core sequence)
2. Book 5 or 6 (Python or Go)
3. Book 7 (Cross-Target Glue)
4. Book 8 (Security)

### Path 3: .NET Developer
1. Book 1: Foundations
2. Book 3: C# Target
3. Book 12: PowerShell Target
4. Book 7: Cross-Target Glue

### Path 4: Data Engineer
1. Book 1: Foundations
2. Book 2: Bash Target
3. Book 10: SQL Target
4. Book 7: Cross-Target Glue
5. Book 13: Semantic Search

### Path 5: Systems Programmer
1. Book 1: Foundations
2. Book 6: Go Target
3. Book 9: Rust Target
4. Book 7: Cross-Target Glue
5. Book 8: Security

### Path 6: AI/ML Engineer
1. Book 1: Foundations
2. Book 5: Python Target
3. Book 4: Workflows
4. Book 13: Semantic Search

## Prerequisites

- Basic command-line knowledge
- Programming experience (any language)
- SWI-Prolog 8.0+ installed

Additional requirements per book:
- **Book 2**: Bash 4.0+
- **Book 3, 12**: .NET SDK 8.0+
- **Book 5**: Python 3.8+
- **Book 6**: Go 1.21+
- **Book 9**: Rust/Cargo
- **Book 10**: SQLite or PostgreSQL

## Installation

### Core Requirements
```bash
# Install SWI-Prolog
# Ubuntu/Debian
sudo apt-get install swi-prolog

# macOS
brew install swi-prolog

# Windows
# Download from https://www.swi-prolog.org/download/stable
```

### Clone Repository
```bash
git clone https://github.com/s243a/UnifyWeaver.git
cd UnifyWeaver
```

## Book Status

| Book | Status | Directory |
|------|--------|-----------|
| 1 - Foundations | Complete (chapters 1-3) | `book-01-foundations/` |
| 2 - Bash Target | Complete | `book-02-bash-target/` |
| 3 - C# Target | Complete | `book-03-csharp-target/` |
| 4 - Workflows | Complete | `book-04-workflows/` |
| 5 - Python Target | Planned | `book-05-python-target/` |
| 6 - Go Target | Complete | `book-06-go-target/` |
| 7 - Cross-Target Glue | Complete | `book-07-cross-target-glue/` |
| 8 - Security & Firewall | Planned (extract from Prolog book) | `book-08-security-firewall/` |
| 9 - Rust Target | Complete | `book-09-rust-target/` |
| 10 - SQL Target | Complete | `book-10-sql-target/` |
| 11 - Prolog Target | Complete | `book-11-prolog-target/` |
| 12 - PowerShell Target | Planned | `book-12-powershell-target/` |
| 13 - Semantic Search | Complete | `book-13-semantic-search/` |
| AWK Target | Complete (supplementary) | `book-awk-target/` |

## Progress Tracking

Track your progress through the series:

**Core Sequence:**
- [ ] Book 1: Foundations
- [ ] Book 2: Bash Target
- [ ] Book 3: C# Target (fixed-point approaches)
- [ ] Book 4: Workflows

**Portable Targets:**
- [ ] Book 5: Python Target
- [ ] Book 6: Go Target

**Integration:**
- [ ] Book 7: Cross-Target Glue
- [ ] Book 8: Security & Firewall

**Specialized:**
- [ ] Book 9: Rust Target
- [ ] Book 10: SQL Target
- [ ] Book 11: Prolog Target
- [ ] Book 12: PowerShell Target

**Advanced:**
- [ ] Book 13: Semantic Search

## Additional Resources

### Documentation
- Main README: `../README.md`
- API Documentation: `../docs/`
- Change Log: `CHANGELOG.md`

### Examples
- Code examples: `../examples/`
- Test cases: `../tests/`
- Generated output: `./output/` (after running exercises)

### Community
- GitHub Issues: https://github.com/s243a/UnifyWeaver/issues
- Discussions: https://github.com/s243a/UnifyWeaver/discussions
- Contributing: `CONTRIBUTING.md`

## License

### Educational Content (Documentation)

All chapters, guides, tutorials, and documentation files (`.md`) are dual-licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Creative Commons Attribution 4.0 International** ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0))

**SPDX:** `MIT AND CC-BY-4.0`

### Code Examples

All code example files (`.pl`, `.sh`, `.cs`, `.py`, etc.) are licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))

**SPDX:** `MIT`

### Copyright

Copyright (c) 2025 John William Creighton (s243a)

_Note: Educational materials developed with assistance from AI language models (Claude, Gemini)._

## Contributing

Found an error or want to improve the content?
1. Open an issue describing the problem
2. Or submit a pull request with fixes
3. See `CONTRIBUTING.md` for guidelines

### For Content Developers
- **[Content Planning](docs/development/PLANNING.md)** - Roadmap for future content
- **[License Guide](docs/development/LICENSE_GUIDE.md)** - Licensing guide for contributors

## Acknowledgments

UnifyWeaver educational materials created by John William Creighton (@s243a) with contributions from the community.

---

**Ready to begin?** Start with [Book 1: Foundations](book-01-foundations/README.md)
