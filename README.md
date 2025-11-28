# UnifyWeaver

A Prolog-to-Bash compiler that transforms declarative logic programs into efficient streaming bash scripts. UnifyWeaver specializes in compiling data relationships and queries into executable bash code with optimized handling of transitive closures and advanced recursion patterns.

**📚 [Extended Documentation](docs/EXTENDED_README.md)** - Comprehensive tutorials, examples, and advanced usage
**🎓 [Educational Materials](https://github.com/s243a/UnifyWeaver_Education)** - Learn UnifyWeaver with hands-on tutorials and examples

## Features

### Core Compilation
- **Stream-based processing** - Memory-efficient compilation using bash pipes and streams
- **BFS optimization** - Transitive closures automatically optimized to breadth-first search
- **Cycle detection** - Proper handling of cyclic graphs without infinite loops
- **Template-based generation** - Clean separation between logic and bash code generation
- **Duplicate prevention** - Efficient tracking ensures each result appears only once
- **Process substitution** - Correct variable scoping in bash loops

### Advanced Recursion
- **Tail recursion optimization** - Converts tail-recursive predicates to iterative bash loops
- **Linear recursion** - Memoized compilation for 1+ independent recursive calls (fibonacci, factorial)
- **Tree recursion** - Structural decomposition with recursive calls on parts (binary tree operations)
- **Mutual recursion** - Handles predicates that call each other cyclically via SCC detection
- **Constraint awareness** - Unique and ordering constraints optimize generated code
- **Pattern detection** - Automatic classification of recursion patterns

### Go Target (v0.1)
- **Standalone Executables** - Compiles Prolog predicates to single-binary Go programs
- **Cross-Platform** - Runs on any platform with Go support, no runtime dependencies
- **Stream Processing** - Efficient stdin/stdout pipeline integration for record processing
- **Match Predicates** - Regex filtering with capture groups for data extraction
- **Multiple Rules** - OR patterns and different body predicates with sequential matching
- **Constraints & Aggregations** - Numeric comparisons (>, <, >=, =<) and sum/count/avg/min/max
- **Smart Compilation** - Selective field assignment and automatic package imports

**See [Go Target Guide](docs/GO_TARGET.md) for details.**

### C# Target Family (v0.1)
- **Query Runtime (`target(csharp_query)`)** - Generates relational plans executed by a shared .NET engine with semi-naive fixpoint evaluation.
- **External Compilation** - Robust `dotnet build` integration with dependency support and file locking prevention.
- **LiteDB Integration** - Built-in support for NoSQL document storage.
- **Mutual recursion support** - Even/odd style dependencies now run entirely inside the C# runtime.
- **Arithmetic & constraints** - LINQ-based pipelines honour `is/2`, comparisons, and deduplication.
- **Streaming codegen (`target(csharp_codegen)`)** - Emit standalone C# projects that mirror Bash streaming templates.

**See [C# Compilation Guide](docs/DOTNET_COMPILATION.md) for details.**

### Data Source Plugin System (v0.1)
- **5 Production-Ready Plugins** - CSV/TSV, AWK, Python, HTTP, JSON data sources
- **Self-Registering Architecture** - Plugin-based system with automatic discovery
- **Template Integration** - Seamless bash code generation with comprehensive error handling
- **Enterprise Security** - Enhanced firewall with multi-service validation
- **Real-World ETL** - Complete pipelines for data transformation and storage
- **SQLite Integration** - Python source with automatic database operations

### PowerShell Target Support (v0.1)
- **Dual-Mode Compilation** - BaaS (Bash-as-a-Service) or pure PowerShell code generation
- **Automatic Mode Detection** - Firewall-aware mode selection based on environment
- **Cross-Platform Support** - Windows, Linux (WSL), macOS PowerShell Core
- **Pure PowerShell** - Native implementations for CSV/JSON without external tools
- **BaaS Mode** - Reuse bash templates via WSL/Git Bash for complex sources (AWK, etc.)
- **Platform Compatibility** - Automatic detection and adaptation to available tools

### Control Plane
- **Enhanced Firewall** - Multi-service security for external tools (python3, curl, wget, jq)
- **Network Access Control** - Host pattern matching and access restrictions
- **Import Restrictions** - Python module whitelisting and validation
- **File Access Patterns** - Read/write permission management
- **Preferences** - Guides implementation choices within policy boundaries
- **Layered Configuration** - Supports global, rule-specific, and runtime overrides

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

### Data Source Integration (v0.1)

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

% XML processing with Python
:- data_source_driver(python).
:- data_source_work_fn(sum_prices).
```

### Complete ETL Pipeline Demo

```bash
cd scripts/testing/test_env5
swipl -g main -t halt examples/pipeline_demo.pl
```

**See [Extended Documentation](docs/EXTENDED_README.md) for complete examples including:**
- Advanced recursion patterns (tail, linear, tree, mutual)
- Graph reachability with cycles
- Recursive computations (factorial, fibonacci)
- Complete ETL pipelines with multiple sources

## What's Supported

### ✅ Recursion Patterns

- **Basic Recursion** - Transitive closures with BFS optimization
- **Tail Recursion** - Converted to iterative loops
- **Linear Recursion** - Memoization for fibonacci, factorial, etc.
- **Tree Recursion** - Structural processing of binary trees
- **Mutual Recursion** - Predicates calling each other with shared memoization

### ✅ Data Sources (v0.1)

- **CSV/TSV** - Auto-header detection, custom delimiters
- **AWK** - Pattern matching, field extraction, text processing pipelines
- **JSON** - jq integration for filtering and transformation
- **HTTP** - REST APIs with caching and custom headers
- **Python** - Inline scripts and SQLite queries
- **XML** - XML processing via Python

### ⚠️ Current Limitations

- Divide-and-conquer patterns (quicksort, mergesort) not yet supported
- Requires Bash 4.0+ for associative arrays
- Tree recursion uses list representation only

**See [Extended Documentation](docs/EXTENDED_README.md) for complete details and troubleshooting.**

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

- **[🎓 Educational Materials](https://github.com/s243a/UnifyWeaver_Education)** - Learn UnifyWeaver with comprehensive tutorials
- **[Extended Documentation](docs/EXTENDED_README.md)** - Comprehensive guide with tutorials and examples
- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Testing infrastructure
- [ADVANCED_RECURSION.md](docs/ADVANCED_RECURSION.md) - Recursion patterns deep dive
- [POWERSHELL_TARGET.md](docs/POWERSHELL_TARGET.md) - PowerShell compilation guide
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [docs/](docs/) - Full documentation index

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

Developed as an exploration of compiling declarative logic to imperative scripts while preserving correctness and efficiency. Special focus on making Prolog's power accessible in bash environments.

**Contributors:**
- John William Creighton (@s243a) - Core development
- Gemini (via gemini-cli) - Constraint awareness features
- Claude (via Claude Code) - Advanced recursion system, test infrastructure
