---
layout: default
title: UnifyWeaver - Prolog Transpiler for Multi-Target Data Pipelines
description: A Prolog transpiler that turns logic programs into LINQ-style data pipelines - Bash, C#, Go, SQL, and more
---

# UnifyWeaver

**A Prolog transpiler that turns logic programs into LINQ-style data pipelines.**

One codebase → Bash streams, C# queries, Go binaries, SQL views, and more.

---

## Quick Links

### 📦 Project Documentation [(code)](https://github.com/s243a/UnifyWeaver)
- **[Project README](project/README)** - Overview, installation, and quick start
- **[Extended README](project/docs/EXTENDED_README)** - Comprehensive tutorials and examples
- **[GitHub Repository](https://github.com/s243a/UnifyWeaver)** - Source code and issues

### 📚 Educational Materials (13-Book Series)

#### Core Sequence
- **[Book 1: Foundations](education/book-01-foundations/)** - Architecture, Prolog basics, preference system
- **[Book 2: Bash Target](education/book-02-bash-target/)** - Stream compilation, templates, data sources
- **[Book 3: C# Target](education/book-03-csharp-target/)** - .NET compilation, LINQ, fixed-point approaches
- **[Book 4: Workflows](education/book-04-workflows/)** - AI agent playbooks, strategic planning

#### Portable Targets
- **[Book 5: Python Target](education/book-05-python-target/)** - Generator-based streaming
- **[Book 6: Go Target](education/book-06-go-target/)** - Native binaries, cross-platform

#### Integration & Advanced
- **[Book 7: Cross-Target Glue](education/book-07-cross-target-glue/)** - Multi-language pipelines
- **[Book 9: Rust Target](education/book-09-rust-target/)** - Memory-safe compilation
- **[Book 10: SQL Target](education/book-10-sql-target/)** - Database query generation
- **[Book 13: Semantic Search](education/book-13-semantic-search/)** - Graph RAG, embeddings

**[View All Books →](education/)**

---

## Key Features

✅ **9 Target Languages** - Bash, C#, Go, Rust, Python, SQL, AWK, Prolog, PowerShell
✅ **Multiple Compilation Approaches** - Stream-based, fixed-point, generator, declarative
✅ **Advanced Recursion** - Linear, tail, tree, mutual recursion with automatic optimization
✅ **Data Source Plugins** - CSV, JSON, XML, HTTP, AWK, Python integration
✅ **Cross-Target Glue** - Compose predicates across languages in unified pipelines
✅ **Parallel Execution** - Automatic partitioning and parallel processing
✅ **Security & Firewall** - Production-ready policy enforcement  

---

## Example: CSV to JSON Pipeline

**Input (Prolog):**
```prolog
% Define data source
csv_source(users/4, [
    file('users.csv'),
    has_header(true)
]).

% Define transformation
process_user(Name, Age, City, Output) :-
    Age >= 18,
    format(string(Output),
           '{"name":"~w","age":~w,"city":"~w"}',
           [Name, Age, City]).

% Compile to bash
?- compile_to_bash(process_user/4, BashScript).
```

**Output (Bash):**
```bash
#!/bin/bash
# Streaming CSV processor with age filter

tail -n +2 users.csv | while IFS=',' read name age city _; do
    if (( age >= 18 )); then
        printf '{"name":"%s","age":%d,"city":"%s"}\n' \
               "$name" "$age" "$city"
    fi
done
```

---

## Use Cases

- **ETL Pipelines** - Transform data between formats efficiently
- **System Administration** - Generate complex bash scripts declaratively
- **Data Analysis** - Process large datasets with streaming algorithms
- **Cross-Platform Tools** - Write once, compile to multiple platforms
- **Educational** - Learn declarative programming and code generation

---

## Getting Started

1. **Installation**
   ```bash
   git clone https://github.com/s243a/UnifyWeaver.git
   cd UnifyWeaver
   swipl examples/load_demo.pl
   ```

2. **Learn the Basics**
   - Start with [Book 1: Foundations](education/book-01-foundations/)
   - Try the [Quick Start Guide](project/docs/EXTENDED_README)

3. **Explore Examples**
   - See [examples/](https://github.com/s243a/UnifyWeaver/tree/main/examples) in the repository
   - Run demos: `swipl examples/data_sources_demo.pl`

---

## Recent Updates

**December 2025**
- ✅ Reorganized 13-book educational series with learning paths
- ✅ Cross-target glue Phase 7 (cloud & enterprise deployment)
- ✅ New README with target comparison tables

**November 2025**
- ✅ C# query runtime with semi-naive fixpoint evaluation
- ✅ SQL target with recursive CTEs and window functions
- ✅ Go/Rust native binary targets

---

## Documentation

### For Users
- **[README](project/README.md)** - Project overview
- **[Testing Guide](project/docs/TESTING_GUIDE.md)** - How to run tests

### For Learners
- **[Book 1: Foundations](education/book-01-foundations/)** - Start here
- **[Book 2: Bash Target](education/book-02-bash-target/)** - Stream compilation
- **[Book 3: C# Target](education/book-03-csharp-target/)** - Fixed-point approaches
- **[All 13 Books](education/)** - Complete educational series

### For Developers
- **[Architecture](project/docs/ARCHITECTURE.md)** - System design
- **[Source Code](https://github.com/s243a/UnifyWeaver/tree/main/src)** - Browse the codebase

---

## Community

- **Issues & Discussions:** [GitHub Issues](https://github.com/s243a/UnifyWeaver/issues)
- **Source Code:** [github.com/s243a/UnifyWeaver](https://github.com/s243a/UnifyWeaver)

---

## License

- **Code:** MIT OR Apache-2.0
- **Documentation:** CC-BY-4.0

---

<div style="text-align: center; margin-top: 2em; color: #666; font-size: 0.9em;">
  This site is published from the <code>gh-pages</code> branch.<br>
  Source: <a href="https://github.com/s243a/UnifyWeaver">github.com/s243a/UnifyWeaver</a>
</div>
