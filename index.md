---
layout: default
title: UnifyWeaver - Prolog to Multi-Target Code Compiler
description: Transform declarative Prolog programs into efficient streaming scripts for Bash, C#, and more
---

# UnifyWeaver

**Transform declarative Prolog programs into efficient streaming scripts**

UnifyWeaver is a Prolog-to-Code compiler that generates optimized streaming code from declarative logic programs. Write your data processing logic once in Prolog, compile to multiple targets: Bash, C#, PowerShell, or pure Prolog services.

---

## Quick Links

### 📦 Project Documentation
- **[Project README](project/README)** - Overview, installation, and quick start
- **[Extended README](project/docs/EXTENDED_README)** - Comprehensive tutorials and examples
- **[Documentation](project/docs/README)** - Complete technical documentation
- **[GitHub Repository](https://github.com/s243a/UnifyWeaver)** - Source code and issues

### 📚 Educational Materials
- **[Education Project Home](education/README)** - Learning resources overview
- **[Book 1: Core Bash Target](education/book-1-core-bash/README)** - Fundamentals and Bash compilation
- **[Book 2: C# Target](education/book-2-csharp-target/README)** - Multi-target and .NET
- **[Book-Misc: Emerging Features](education/book-misc/README)** - New and experimental features

---

## Key Features

✅ **Multiple Targets** - Compile to Bash, C#, PowerShell, or Prolog services  
✅ **Streaming Processing** - Memory-efficient data pipelines  
✅ **Data Source Plugins** - CSV, JSON, XML, HTTP, AWK, Python integration  
✅ **Advanced Recursion** - Fibonacci, binomial, tree patterns with memoization  
✅ **Parallel Execution** - Automatic partitioning and parallel processing  
✅ **Firewall System** - Security-aware compilation with tool restrictions  
✅ **Template System** - Flexible code generation with fallback strategies  

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
   - Start with [Book 1, Chapter 1](education/book-1-core-bash/01_introduction.md)
   - Try the [Quick Start Guide](project/docs/README.md)

3. **Explore Examples**
   - See [examples/](https://github.com/s243a/UnifyWeaver/tree/main/examples) in the repository
   - Run demos: `swipl examples/data_sources_demo.pl`

---

## Recent Updates

**v0.1 (November 2025)**
- ✅ Perl service infrastructure for Python-free XML processing
- ✅ Multi-call linear recursion (fibonacci, tribonacci)
- ✅ PowerShell compatibility layer
- ✅ C# query runtime with mutual recursion support
- ✅ Enhanced testing infrastructure

---

## Documentation

### For Users
- **[README](project/README.md)** - Project overview
- **[Testing Guide](project/docs/TESTING_GUIDE.md)** - How to run tests
- **[XML Tools Installation](project/docs/XML_PARSING_TOOLS_INSTALLATION.md)** - XML processing setup

### For Learners
- **[Book 1: Core Bash](education/book-1-core-bash/README.md)** - Start here if new to UnifyWeaver
- **[Book 2: C# Target](education/book-2-csharp-target/README.md)** - Multi-target compilation
- **[Book-Misc](education/book-misc/README.md)** - Emerging features (Perl services, Prolog targets)

### For Developers
- **[Contributing Guide](project/docs/CONTRIBUTING.md)** - How to contribute
- **[Architecture Docs](project/docs/)** - Technical deep dives
- **[Source Code](https://github.com/s243a/UnifyWeaver/tree/main/src)** - Browse the codebase

---

## Community

- **Issues & Discussions:** [GitHub Issues](https://github.com/s243a/UnifyWeaver/issues)
- **Source Code:** [github.com/s243a/UnifyWeaver](https://github.com/s243a/UnifyWeaver)
- **Education Materials:** [github.com/s243a/UnifyWeaver_Education](https://github.com/s243a/UnifyWeaver_Education)

---

## License

- **Code:** MIT OR Apache-2.0
- **Documentation:** CC-BY-4.0

---

<div style="text-align: center; margin-top: 2em; color: #666; font-size: 0.9em;">
  This site is published from the <code>gh-pages</code> branch.<br>
  Source repositories:
  <a href="https://github.com/s243a/UnifyWeaver">UnifyWeaver</a> |
  <a href="https://github.com/s243a/UnifyWeaver_Education">UnifyWeaver_Education</a>
</div>
