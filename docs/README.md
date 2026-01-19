<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# UnifyWeaver Documentation

This directory contains technical documentation, guides, and reference materials for UnifyWeaver.

## Default Licensing

**All documentation files in this directory (not including subdirectories) are licensed under MIT OR Apache-2.0** unless individual files specify otherwise in their headers.

This applies to:
- Architecture documentation
- Development guides
- Reference materials
- Technical specifications

**Note:** Subdirectories may have different licensing (e.g., `papers/` for published research). Check subdirectory README files for specific licensing information.

## Contents

### Getting Started

- **[EXTENDED_README.md](EXTENDED_README.md)** - Comprehensive documentation
  - Detailed tutorials and examples
  - Advanced recursion patterns
  - Data source plugin system
  - Firewall and security
  - PowerShell target
  - Complete ETL pipelines
  - Troubleshooting guide

### Architecture and Design

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview
  - Module organization
  - Compilation pipeline
  - Generated code structure
  - Design principles

- **[ADVANCED_RECURSION.md](ADVANCED_RECURSION.md)** - Advanced recursion patterns
  - Tail recursion optimization
  - Linear recursion with memoization
  - Tree recursion for structures
  - Mutual recursion with SCC detection
  - Pattern detection and classification

### Features

- **[AWK_INGESTION.md](AWK_INGESTION.md)** - High-performance AWK ingestion pipeline
  - Null-delimited streaming protocol
  - Cross-target support (Python, C#, Go, Rust)
  - Performance benchmarks and use cases
  - AWK fragment extraction scripts
  - Pipeline architecture patterns

- **[POWERSHELL_TARGET.md](POWERSHELL_TARGET.md)** - PowerShell compilation target
  - Pure PowerShell mode
  - BaaS (Bash-as-a-Service) mode
  - Auto mode selection
  - Cross-platform considerations

- **[POWERSHELL_PURE_IMPLEMENTATION.md](POWERSHELL_PURE_IMPLEMENTATION.md)** - Pure PowerShell details
  - Implementation approach
  - Cmdlet-based processing
  - No bash dependencies

- **[POWERSHELL_PURE_VS_BAAS.md](POWERSHELL_PURE_VS_BAAS.md)** - Mode comparison
  - Performance trade-offs
  - Use case recommendations
  - Platform compatibility

- **[FIREWALL_GUIDE.md](FIREWALL_GUIDE.md)** - Security and policy system
  - Network access control
  - Service restrictions
  - Multi-level configuration
  - Policy templates
  - Tool detection
  - Examples and best practices

- **[playbooks/xml_data_source_playbook.md](../playbooks/xml_data_source_playbook.md)** - XML Data Source Playbook
  - Processing XML data using Python.

### Semantic Search and LDA Projection

#### Usage Guide

- **[usage/semantic_search.md](usage/semantic_search.md)** - Semantic Search Usage Guide
  - Prolog API: `find_examples/3,4`, `semantic_search/3,4`
  - Go API: `projection.LoadMultiHead`, `search.SearchWithOptions`
  - Temperature tuning and search modes
  - Example code and testing

#### Runtime Components

- **`src/unifyweaver/runtime/semantic_search.pl`** - Prolog Semantic Search API
  - `find_examples/3,4` - Find playbook examples matching a query
  - `semantic_search/3,4` - Low-level search with options
  - Three search modes: direct, multi-head projection, global projection
  - Component registration with `component_registry`

- **`src/unifyweaver/targets/go_runtime/projection/`** - Go Multi-Head Projection
  - Native Go implementation (no Python subprocess overhead)
  - Softmax routing with configurable temperature
  - NPY file loading for centroids and answer embeddings

#### Theory and Design

- **[proposals/SEMANTIC_PROJECTION_LDA.md](proposals/SEMANTIC_PROJECTION_LDA.md)** - LDA Projection Theory
  - Mathematical foundation for query-to-answer projection
  - Weighted centroid computation
  - Training from Q-A clusters

- **[proposals/MULTI_HEAD_PROJECTION_THEORY.md](proposals/MULTI_HEAD_PROJECTION_THEORY.md)** - Multi-Head Projection
  - Per-cluster attention heads with softmax routing
  - Temperature parameter and its effect on routing sharpness
  - Comparison to transformer attention mechanisms

- **[proposals/LDA_DATABASE_SCHEMA.md](proposals/LDA_DATABASE_SCHEMA.md)** - Database Schema
  - Asymmetric embeddings (different models for queries vs answers)
  - Answer relations graph (chunks, summaries, translations)
  - Training batch tracking with file hash detection

- **[proposals/COMPONENT_REGISTRY.md](proposals/COMPONENT_REGISTRY.md)** - Component Registry
  - Unified component registration system
  - Runtime, source, and binding categories
  - Lazy/eager initialization

- **[TODO_LDA_PROJECTION.md](TODO_LDA_PROJECTION.md)** - LDA Feature Progress
  - Implementation status and next steps
  - Quick test commands

### Testing

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing infrastructure
  - Test environment setup
  - Adding new tests
  - Test discovery system
  - Best practices

- **[TESTING.md](TESTING.md)** - Detailed testing documentation
  - Test organization
  - Running tests
  - Generated script verification

### Development Guides

- **[FILE_HEADERS.md](FILE_HEADERS.md)** - Quick reference for SPDX headers
  - When to add headers
  - Examples for each file type
  - Special cases (papers, generated code)

- **[LINE_ENDINGS.md](LINE_ENDINGS.md)** - Line ending conventions
  - Why Unix (LF) line endings
  - Git configuration
  - Platform-specific issues

- **[development/ai-skills/](development/ai-skills/)** - AI assistant skills and workflows
  - Structured prompts for common development tasks
  - Cross-compatible with Claude Code, Copilot, Cursor, and other AI tools
  - Testing patterns and project-specific workflows

- **[../skills/](../skills/)** - Task-specific skill router for agents
  - Decision tree with LOAD/GOTO navigation ([SKILL_ROUTER.md](../skills/SKILL_ROUTER.md))
  - Mindmap tools: linking, MST grouping, cross-links, folder suggestion
  - Bookmark filing and data processing skills
  - See [AGENT_INSTRUCTIONS.md](../AGENT_INSTRUCTIONS.md) for startup instructions

- **[development/STDIN_LOADING.md](development/STDIN_LOADING.md)** - Loading Prolog code from stdin
  - Using `consult(user)` for quick experiments
  - Testing without temporary files
  - Command-line patterns and examples

- **[development/UNICODE_SPECIFICATION.md](development/UNICODE_SPECIFICATION.md)** - Unicode and emoji handling
  - Cross-platform emoji support
  - Terminal detection and compatibility
  - SWI-Prolog version requirements

### Administration

- **[LICENSE_ADMINISTRATION.md](LICENSE_ADMINISTRATION.md)** - Complete licensing guide
  - Dual licensing administration
  - Contribution policies
  - Third-party code integration
  - Common scenarios and FAQs

### Planning and Design Documents

- **[DATA_SOURCES_IMPLEMENTATION_PLAN.md](DATA_SOURCES_IMPLEMENTATION_PLAN.md)** - Data source plugin system
- **[CONTROL_PLANE.md](CONTROL_PLANE.md)** - Firewall and preferences architecture
- **[CONSTRAINT_SYSTEM.md](CONSTRAINT_SYSTEM.md)** - Constraint detection and optimization
- **[RECURSION_PATTERN_THEORY.md](RECURSION_PATTERN_THEORY.md)** - Theoretical foundations

## For Contributors

If you're contributing to UnifyWeaver:

1. Read [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root
2. Check [FILE_HEADERS.md](FILE_HEADERS.md) for header requirements
3. Review [LICENSE_ADMINISTRATION.md](LICENSE_ADMINISTRATION.md) for licensing details

## License

Documentation in this directory is licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
* MIT license ([LICENSE-MIT](../LICENSE-MIT))

at your option.