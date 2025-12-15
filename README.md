# UnifyWeaver

**A Prolog transpiler that turns logic programs into LINQ-style data pipelines.**

One codebase → Bash streams, C# queries, Go binaries, SQL views, and more.

**📚 [Extended Documentation](docs/EXTENDED_README.md)** | **🎓 [Educational Materials](education/README.md)**

---

## Why UnifyWeaver?

Write your data relationships and queries once in Prolog, then compile to the target that fits your environment:

- **Shell scripts** for Unix pipelines and automation
- **Native binaries** for portable, dependency-free deployment
- **SQL views** for database integration
- **.NET assemblies** for enterprise applications
- **Multi-language pipelines** via cross-target glue

UnifyWeaver handles the hard parts—recursion, transitive closures, cycle detection, deduplication—so your generated code is correct and efficient.

---

## Compilation Approaches

UnifyWeaver supports multiple compilation strategies depending on the target and predicate complexity:

| Approach | Description | Targets |
|----------|-------------|---------|
| **Stream/Procedural** | Direct template-based code generation with Unix pipes or LINQ iterators | Bash, Go, Rust, C# Stream, PowerShell |
| **Fixed-Point (Query Engine)** | IR + runtime with semi-naive evaluation for complex recursion | C# Query Runtime |
| **Generator-Based** | Lazy evaluation via Python generators with memoization | Python |
| **Declarative Output** | SQL queries for external database execution | SQL |

## Recursion Pattern Support

Different targets support different recursion patterns. Choose based on your needs:

| Pattern | Bash | C# Query | Go | Rust | Python | SQL | AWK | Prolog |
|---------|:----:|:--------:|:--:|:----:|:------:|:---:|:---:|:------:|
| **Linear Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **Tail Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| **Tree Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **Transitive Closure** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **Mutual Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **Aggregations** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Extended Targets

Additional language targets for native compilation, FFI, and functional programming:

| Pattern | LLVM | WASM | Haskell | VB.NET | F# | Java | Jython | Kotlin |
|---------|:----:|:----:|:-------:|:------:|:--:|:----:|:------:|:------:|
| **Linear Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Tail Recursion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Tree Recursion** | — | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Transitive Closure** | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mutual Recursion** | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Aggregations** | — | — | — | ✅ | ✅ | ✅ | ✅ | ✅ |

**Key:**
- ✅ Full support with optimizations (BFS, loops, memoization, semi-naive)
- — Not supported or limited

## Target Selection Guide

| If you need... | Use |
|----------------|-----|
| Shell scripts for Unix pipelines | **Bash** |
| Standalone binary, no runtime deps | **Go** or **Rust** |
| Complex recursion in .NET apps | **C# Query Runtime** |
| Database views and analytics | **SQL** |
| Python ecosystem integration | **Python** |
| Windows/.NET orchestration | **PowerShell** |
| Lightweight text processing | **AWK** |
| Prolog dialect transpilation | **Prolog** |

---

## Target Features

### Bash Target (v0.2)
Stream-based compilation to Unix shell scripts with pipes and process substitution.
- BFS optimization for transitive closures
- Cycle detection and duplicate prevention
- Template-based code generation
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages

**Docs:** [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### Go Target (v0.6)
Standalone native binaries with no runtime dependencies.
- JSON I/O with schemas and nested path extraction
- Regex matching with capture groups
- Embedded bbolt database storage
- Parallel workers for high-throughput processing
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- **Binding System** — Map predicates to Go stdlib functions

**Docs:** [Go Target Guide](docs/GO_TARGET.md) | [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### Rust Target (v0.2)
Memory-safe native binaries via Cargo.
- Serde JSON integration
- Semantic crawling support
- Full Cargo project scaffolding
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- **Binding System** — Map predicates to Rust stdlib and crates

### C# Target Family (v0.2)
Two compilation modes for different needs:
- **Stream Target** (`csharp_codegen`) — LINQ pipelines for simple predicates
- **Query Runtime** (`csharp_query`) — IR + semi-naive fixpoint for complex recursion

Features: LiteDB integration, mutual recursion via SCC, arithmetic constraints.
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- **Binding System** — Map predicates to .NET APIs and LINQ

**Docs:** [C# Compilation Guide](docs/DOTNET_COMPILATION.md) | [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### Python Target (v0.4)
Generator-based streaming with Python ecosystem integration and multi-runtime support.
- **Pipeline Mode** — Streaming JSONL I/O with typed object output
- **Runtime Selection** — Auto-select CPython, IronPython, PyPy, or Jython based on context
- **Pipeline Chaining** — Connect multiple predicates with `yield from` composition
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- **Cross-Runtime Pipelines** — Stage-based orchestration for mixed Python/C# workflows
- **C# Hosting** — IronPython in-process or CPython subprocess with JSONL glue
- **Binding System** — Map predicates to Python built-ins and libraries
- Native XML via lxml, semantic runtime with SQLite and vector search

**Docs:** [Python Target Guide](docs/PYTHON_TARGET.md) | [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### SQL Target (v0.3)
Compiles predicates to SQL queries for database execution.
- Recursive CTEs for hierarchical data
- Window functions (RANK, ROW_NUMBER, LAG, LEAD)
- All JOIN types, aggregations, subqueries
- Works with SQLite, PostgreSQL, MySQL, SQL Server

**Docs:** [SQL Target Design](SQL_TARGET_DESIGN.md)

### PowerShell Target (v2.6)
Windows automation and .NET orchestration with full object pipeline support.
- Dual-mode: pure PowerShell or Bash-as-a-Service via WSL
- Cross-platform (PowerShell 7+)
- **Binding system**: 68+ bindings for cmdlets, .NET methods, Windows automation
- **Auto-transpilation**: Rules like `sqrt(X, Y)` compile directly to `[Math]::Sqrt($X)`
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- **Object pipeline**: `ValueFromPipeline` parameters, `PSCustomObject` output
- **Advanced joins**: Hash-based and pipelined N-way joins with O(n+m) complexity
- **Firewall security**: Per-predicate mode control (pure/baas/auto)
- Ideal for orchestrating .NET targets (C#, IronPython)

**Docs:** [PowerShell Target Guide](docs/POWERSHELL_TARGET.md) | [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### AWK Target (v0.2)
Lightweight, portable text processing.
- Tail recursion to while loops
- Aggregations (sum, count, max, min, avg)
- Regex matching with capture groups
- **Enhanced Chaining** — Fan-out, merge, conditional routing, and filtering stages
- Runs on any POSIX system

**Docs:** [Enhanced Chaining](docs/ENHANCED_PIPELINE_CHAINING.md)

### Prolog Target
Prolog-to-Prolog transpilation for dialect compatibility.
- SWI-Prolog and GNU Prolog support
- Native binary compilation via gplc
- Executable script generation

---

## Cross-Target Glue System

Compose predicates across multiple languages in unified pipelines:

- **Shell Integration** — TSV/CSV/JSON I/O between AWK, Python, Bash
- **.NET Bridges** — In-process C# ↔ PowerShell ↔ IronPython
- **Python/C# Glue** — IronPython in-process hosting or CPython subprocess with JSONL
- **Pipeline Chaining** — Multi-stage orchestration with automatic runtime grouping
- **Native Orchestration** — Go/Rust compilation with parallel workers
- **Network Communication** — HTTP servers/clients, TCP streaming
- **Service Registry** — Distributed service routing

**Docs:** [Cross-Target Glue Guide](docs/guides/cross-target-glue.md)

---

## Data Sources & ETL

Built-in data source plugins for real-world pipelines:

| Source | Description |
|--------|-------------|
| **CSV/TSV** | Auto-header detection, custom delimiters |
| **JSON** | jq integration for filtering and transformation |
| **HTTP** | REST APIs with caching and custom headers |
| **Python** | Inline scripts, SQLite queries |
| **AWK** | Pattern matching, field extraction |
| **XML/YAML** | Via Python (lxml, PyYAML) |

---

## Control Plane

Security and configuration for production deployments:

- **Firewall** — Multi-service security for external tools
- **Network ACLs** — Host pattern matching and restrictions
- **Import Restrictions** — Python module whitelisting
- **File Access Patterns** — Read/write permission management
- **Preferences** — Layered configuration (global, rule-specific, runtime)

## Installation

Requirements:
- SWI-Prolog 8.0 or higher
- Bash 4.0+ (for associative arrays)

```bash
git clone https://github.com/s243a/UnifyWeaver.git
cd UnifyWeaver
```

## Quick Start

### Test Environment

UnifyWeaver includes a convenient test environment with auto-discovery:

```bash
# Linux/WSL
cd scripts/testing
./init_testing.sh

# Windows (PowerShell)
cd scripts\testing
.\Init-TestEnvironment.ps1
```

Then in the test environment:
```prolog
?- test_all.           % Run all tests
?- test_stream.        % Test stream compilation
?- test_recursive.     % Test basic recursion
?- test_advanced.      % Test advanced recursion patterns
?- test_constraints.   % Test constraint system
```

### Manual Usage

```prolog
?- use_module(unifyweaver(core/recursive_compiler)).
?- test_recursive_compiler.
```

This generates bash scripts in the `output/` directory:
```bash
cd output
bash test_recursive.sh
```

## Usage

### Basic Example

Define your Prolog predicates:
```prolog
% Facts
parent(alice, bob).
parent(bob, charlie).

% Rules
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

Compile to bash:
```prolog
?- compile_recursive(ancestor/2, [], BashCode).
?- write_bash_file('ancestor.sh', BashCode).
```

Use the generated script:
```bash
source parent.sh
source ancestor.sh

ancestor_all alice  # Find all descendants of alice
ancestor_check alice charlie && echo "Yes" || echo "No"  # Check specific relationship
```

**See [Extended Documentation](docs/EXTENDED_README.md) for advanced recursion patterns, data sources, and complete examples.**

## Architecture

UnifyWeaver follows a modular compilation pipeline:

1. **Classification** - Analyzes predicate to determine recursion pattern
2. **Constraint Analysis** - Detects unique/ordering constraints
3. **Pattern Matching** - Tries tail → linear → tree → mutual → basic recursion
4. **Optimization** - Applies pattern-specific optimizations (BFS, loops, memoization)
5. **Target Selection** - Chooses Bash, PowerShell, or C# backend based on requested `target/1` preference
6. **Code Generation / Plan Emission** - Renders shell templates, C# source, or query plans for the selected backend

**See [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [Extended Documentation](docs/EXTENDED_README.md) for detailed architecture.**

## Examples

### Family Tree Queries
```prolog
ancestor(X, Y)    % Transitive closure of parent
descendant(X, Y)  % Reverse of ancestor
sibling(X, Y)     % Same parent, different children
```

### Data Source Integration

```prolog
% CSV/TSV data processing
:- source(csv, users, [csv_file('data/users.csv'), has_header(true)]).

% AWK text processing
:- source(awk, extract_fields, [
    awk_program('$3 > 100 { print $1, $2 }'),
    input_file('data.txt')
]).

% HTTP API integration with caching
:- source(http, github_repos, [
    url('https://api.github.com/users/octocat/repos'),
    cache_duration(3600)
]).

% Python with SQLite
:- source(python, get_users, [
    sqlite_query('SELECT name, age FROM users WHERE active = 1'),
    database('app.db')
]).

% JSON processing with jq
:- source(json, extract_names, [
    jq_filter('.users[] | {name, email} | @tsv'),
    json_file('data.json')
]).
```

**See [Extended Documentation](docs/EXTENDED_README.md) for complete examples.**

### Python Pipeline Example

```prolog
% Compile predicates to a chained Python pipeline
compile_pipeline(
    [parse_user/2, filter_adult/2, format_output/3],
    [runtime(cpython), pipeline_name(user_pipeline)],
    PythonCode
).
```

Generated Python uses efficient generator chaining:
```python
def user_pipeline(input_stream):
    """Chained pipeline: [parse_user, filter_adult, format_output]"""
    yield from format_output(filter_adult(parse_user(input_stream)))
```

For cross-runtime workflows (Python + C#):
```prolog
compile_pipeline(
    [python:extract/1, csharp:validate/1, python:transform/1],
    [pipeline_name(data_processor)],
    Code
).
```

### Current Limitations

- Divide-and-conquer patterns (quicksort, mergesort) not yet supported
- Tree recursion uses list representation only

## Testing

```bash
cd scripts/testing
./init_testing.sh  # or Init-TestEnvironment.ps1 on Windows
```

In SWI-Prolog:
```prolog
?- test_all.          % Run all tests
?- test_stream.       % Stream compilation
?- test_recursive.    % Basic recursion
?- test_advanced.     % Advanced patterns
```

**See [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for adding tests.**

## Documentation

| Resource | Description |
|----------|-------------|
| **[Educational Materials](education/README.md)** | 13-book series covering all targets and patterns |
| [Extended Documentation](docs/EXTENDED_README.md) | Tutorials and advanced examples |
| [Architecture](docs/ARCHITECTURE.md) | System design and compilation pipeline |
| [Enhanced Pipeline Chaining](docs/ENHANCED_PIPELINE_CHAINING.md) | Fan-out, merge, routing, and filter stages |
| [Cross-Target Glue](docs/guides/cross-target-glue.md) | Multi-language pipeline composition |
| [Advanced Recursion](docs/ADVANCED_RECURSION.md) | Recursion patterns deep dive |
| [Testing Guide](docs/TESTING_GUIDE.md) | Testing infrastructure |

## Contributing

Issues and pull requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Key areas:
- Additional recursion patterns (divide-and-conquer, N-ary trees)
- Performance optimizations
- Additional data source plugins
- Native PowerShell code generation (pure mode enhancements)

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Acknowledgments

Developed as an exploration of compiling declarative logic to efficient executable code across multiple target languages—making Prolog's power accessible everywhere from shell scripts to native binaries to database queries.

**Contributors:**
- John William Creighton (@s243a) - Core development
- GPT-5/5.1-Codex (via OpenAI) - Fixed-point architecture, query engine, generator approaches
- Gemini (via gemini-cli) - Constraint awareness features
- Claude (via Claude Code) - Advanced recursion system, test infrastructure, educational materials
