<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book-Misc: Emerging Features & Advanced Topics

**Status:** Work in Progress
**Last Updated:** 2025-11-05

---

## Purpose

This folder contains documentation for UnifyWeaver features that are:
- **Newly implemented** and not yet integrated into the main books
- **Advanced topics** that may warrant their own dedicated book
- **Experimental features** undergoing active development
- **Fallback mechanisms** and alternative compilation strategies

Content here may eventually be:
1. Merged into Book 1 (Core Bash) or Book 2 (C# Target)
2. Expanded into a new standalone book (e.g., "Book 3: Prolog Integration")
3. Moved to appendices or advanced topics sections

---

## Current Topics

### 1. **[Perl Service Infrastructure](01_perl_service_infrastructure.md)** ✅ Implemented
- **Status:** Merged to main (2025-11-05)
- **Coverage:**
  - `perl_service.pl` module architecture
  - Inline Perl bash call generation
  - DCG-based code generation
  - Heredoc label collision avoidance
  - Shell argument quoting
  - XML splitter implementation

**Why Here:** New infrastructure that enables Python-free XML processing and could be extended to other use cases. May become part of Book 1 or a new "Advanced Data Sources" book.

### 2. **XML Source Enhancement** ✅ Implemented
- **Status:** Partially documented in Book 1 (Chapter 14), needs update
- **Files:** `02_xml_perl_splitter.md` (planned)
- **Coverage:**
  - Perl-backed xmllint splitter
  - Splitter selection mechanism
  - Firewall-aware tool detection
  - Preference system integration
  - Python vs Perl splitter comparison

**Why Here:** Recent enhancement to existing XML chapter. Should be merged back into Book 1, Chapter 14.

### 3. **[Prolog as Target Language](03_prolog_target.md)** 🚧 Limited Implementation
- **Status:** Partially implemented, no templates
- **Coverage:**
  - Prolog query generation
  - Integration with SWI-Prolog
  - Limitations (no templates, no unique/unordered)
  - Use cases as compilation fallback

**Why Here:** Incomplete feature that could become a fallback strategy when patterns don't match or firewall rules forbid other approaches. May evolve into its own book if expanded.

### 4. **[Prolog Service (Bash-as-a-Service alternative)](04_prolog_bash_service.md)** 🚧 Partial Implementation
- **Status:** Infrastructure exists (`prolog_service_target.pl`), limited testing
- **Coverage:**
  - Inline Prolog execution in bash
  - Comparison to BaaS (Bash-as-a-Service)
  - Use cases and performance characteristics
  - Integration with partitioning system
  - Limitations and future directions

**Why Here:** Experimental fallback mechanism. Could become part of a larger "Multi-Language Services" chapter.

### 5. **Fallback Compilation Strategies** 📋 Conceptual
- **Status:** Design phase
- **Files:** `05_fallback_strategies.md` (planned)
- **Coverage:**
  - Pattern matching failure handling
  - Firewall-driven compilation paths
  - Progressive degradation (Bash → BaaS → Prolog)
  - Error handling and user feedback

**Why Here:** Architectural topic that spans multiple books. Needs real-world examples before documentation.

---

## Topics NOT Included Here

The following are already well-documented elsewhere:

- **PowerShell Target** - See Book 1, Chapters 8-11
- **C# Targets** - See Book 2 (complete coverage)
- **Core Recursion** - See Book 1, Chapters 9, 12, Appendices
- **Data Sources (CSV, JSON, HTTP)** - See Book 1, `data_sources_pipeline_guide.md`
- **AWK/Python Pipelines** - See Book 1, data sources guide

---

## Contributing

When adding new topics to book-misc:

1. **Start with an outline** - Create a stub with key sections
2. **Mark implementation status** - Use ✅ Implemented, 🚧 Partial, 📋 Planned
3. **Link to source code** - Reference actual modules and tests
4. **Include examples** - Working code snippets, not just theory
5. **Note limitations** - Be clear about what doesn't work yet

When content is mature:
- Update this README to mark it as "Ready for Integration"
- Open an issue/PR to merge into the appropriate main book
- Update cross-references in other chapters

---

## Roadmap

### Short-term (v0.1.x)
- [ ] Document Perl service infrastructure (Chapter 01)
- [ ] Update XML chapter 14 with Perl splitter info
- [ ] Document current Prolog target limitations

### Medium-term (v0.2.x)
- [ ] Expand Prolog target with template support
- [ ] Add unique/unordered to Prolog compilation
- [ ] Document fallback strategies with real examples
- [ ] Create "Advanced Services" chapter (Perl, Prolog, BaaS comparison)

### Long-term
- [ ] Potentially split into "Book 3: Alternative Targets & Services"
- [ ] Cross-language optimization guide
- [ ] Performance benchmarking across targets

---

## License

All content in this folder is dual-licensed:
- Code: MIT OR Apache-2.0
- Documentation: CC-BY-4.0

See parent `education/` folder for full license texts.

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
