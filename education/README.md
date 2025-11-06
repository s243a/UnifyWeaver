---
layout: default
title: UnifyWeaver Education
description: Comprehensive educational materials for learning UnifyWeaver
---

<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# UnifyWeaver_Education

**Educational Materials for [UnifyWeaver](https://github.com/s243a/UnifyWeaver)**

This project contains comprehensive educational materials for learning UnifyWeaver - a Prolog-to-Code compiler that transforms declarative logic programs into efficient streaming scripts. The content is organized into two books covering different aspects of the system.

## Book Structure

### 📘 Book 1: Core UnifyWeaver & Bash Target
**[📖 Read Book 1](book-1-core-bash/README)** | **Location:** `book-1-core-bash/`

Covers fundamental concepts and Bash code generation:
- Prolog basics and UnifyWeaver architecture
- Stream compilation and template system
- Advanced recursion patterns
- Data source plugins (CSV, JSON, Python, HTTP, AWK, XML)
- Partitioning and parallel execution
- Complete ETL pipeline examples

**Start here if you're new to UnifyWeaver.**

### 📗 Book 2: C# Target Language
**[📖 Read Book 2](book-2-csharp-target/README)** | **Location:** `book-2-csharp-target/`

Covers multi-target compilation and C# / .NET:
- C# Stream Target (direct source generation)
- C# Query Runtime (IR + runtime library)
- LINQ-based query execution
- Cross-platform .NET deployment
- Performance optimization

**Complete Book 1 before starting Book 2.**

### 📙 Book-Misc: Emerging Features & Advanced Topics
**[📖 Read Book-Misc](book-misc/README)** | **Location:** `book-misc/`

Documentation for newly implemented or experimental features:
- **Perl Service Infrastructure** ✅ - Python-free data processing
- **XML Perl Splitter** ✅ - Enhanced XML source capabilities
- **Prolog as Target** 🚧 - Pure Prolog code generation (limited)
- **Prolog Bash Service** 🚧 - Fallback compilation strategy
- **Fallback Strategies** 📋 - Pattern matching failure handling

**Status:** Work in Progress - Content may be merged into main books or expanded into Book 3.

### 📓 Jupyter Notebooks
**[📖 View Notebooks](notebooks/README)** | **Location:** `notebooks/`

Interactive notebooks for hands-on experimentation with UnifyWeaver concepts.

## Quick Start

### For Beginners
1. Read `book-1-core-bash/README.md`
2. Work through chapters 1-4 (Foundations)
3. Try the exercises in each chapter
4. Explore case studies and appendices

### For Experienced Users
- Jump to specific topics using the chapter guides
- Review appendices for deep dives on specific patterns
- Check case studies for real-world examples

### For .NET Developers
- Complete Book 1 chapters 1-4 (foundations)
- Skip to Book 2 for C# compilation
- Reference Book 1 as needed for Prolog concepts

## Learning Paths

### Path 1: Data Engineer
Focus on data processing and ETL:
- Book 1: Chapters 1-5, 13-14
- Data Sources Pipeline Guide
- Case Study: Production Pipeline

### Path 2: Language Developer
Focus on compilation and code generation:
- Book 1: Chapters 1-12
- Book 2: All chapters
- Appendices A, B, C

### Path 3: Quick Start
Minimum to get started:
- Book 1: Chapters 1-4
- Data Sources Pipeline Guide
- Your specific use case examples

## Prerequisites

- Basic command-line knowledge
- Programming experience (any language)
- SWI-Prolog 8.0+ installed
- Bash 4.0+ (for Book 1)
- .NET SDK 6.0+ (for Book 2, optional)

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

### Optional (for Book 2)
```bash
# Install .NET SDK
# See: https://dotnet.microsoft.com/download
dotnet --version
```

### Clone Repository
```bash
git clone https://github.com/s243a/UnifyWeaver.git
cd UnifyWeaver
```

## Additional Resources

### Documentation
- Main README: `../README.md`
- API Documentation: `../docs/`
- Change Log: `CHANGELOG.md`
- C# Query Target Test Plan (v0.1): https://github.com/s243a/UnifyWeaver/blob/main/docs/development/testing/v0_1_csharp_test_plan.md

### Examples
- Code examples: `../examples/`
- Test cases: `../tests/`
- Generated output: `./output/` (after running exercises)

### Community
- GitHub Issues: https://github.com/s243a/UnifyWeaver/issues
- Discussions: https://github.com/s243a/UnifyWeaver/discussions
- Contributing: `CONTRIBUTING.md`

## Book Contents Overview

### Book 1 Chapters (13+ chapters)
1. Introduction
2. Prolog Fundamentals
3. UnifyWeaver Architecture
4. Your First Program
5. Stream Compilation
6. Advanced Constraints
7. Variable Scope and Process Substitution
8. Template System
9. Advanced Recursion
10. Prolog Introspection
11. Test Runner Inference
12. Recursive Compilation
13. Partitioning and Parallel Execution
14. XML Source Plugin
- Plus: Appendices, Case Studies, Guides

### Book 2 Chapters (4 chapters)
1. Introduction to Multi-Target Compilation
2. C# Stream Target
3. C# Query Runtime
4. Runtime Libraries and Deployment

## Progress Tracking

Track your progress:
- [ ] Completed Book 1 Foundations (Chapters 1-4)
- [ ] Completed Book 1 Core Techniques (Chapters 5-8)
- [ ] Completed Book 1 Advanced Topics (Chapters 9-14)
- [ ] Completed Book 2 C# Basics (Chapters 1-2)
- [ ] Completed Book 2 Advanced (Chapters 3-4)

## Updates and Versions

This educational content tracks UnifyWeaver development:
- **Book 1:** Covers features up to v0.0.2+
- **Book 2:** Covers C# target (experimental/in-development)

Check `CHANGELOG.md` for updates.

## License

This educational project is separate from the main UnifyWeaver codebase and uses its own licensing:

### Educational Content (Documentation)

All chapters, guides, tutorials, and documentation files (`.md`) are dual-licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Creative Commons Attribution 4.0 International** ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0) or https://creativecommons.org/licenses/by/4.0/)

**SPDX:** `MIT AND CC-BY-4.0`

You are free to share and adapt with attribution under either license.

### Code Examples

All code example files (`.pl`, `.sh`, `.cs`, `.py`, etc.) are licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

**SPDX:** `MIT`

Choose the license that works best for your project.

### Copyright

Copyright (c) 2025 John William Creighton (s243a)

_Note: Educational materials developed with assistance from AI language models (Claude, Gemini)._

## Contributing

Found an error or want to improve the content?
1. Open an issue describing the problem
2. Or submit a pull request with fixes
3. See `CONTRIBUTING.md` for guidelines

### For Content Developers
- **[Content Planning](docs/development/PLANNING)** - Roadmap for future educational content additions
- **[License Guide](docs/development/LICENSE_GUIDE)** - Detailed licensing guide for contributors and users

## Acknowledgments

UnifyWeaver educational materials created by John William Creighton (@s243a) with contributions from the community.

## Get Help

- Check the chapter's "Troubleshooting" section
- Review related appendices
- Search GitHub issues
- Ask in Discussions

---

**Ready to begin?** Start with `book-1-core-bash/README.md` →
