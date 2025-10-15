# Changelog

All notable changes to UnifyWeaver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2] - 2025-10-14

### Added

#### Data Source Plugin System
- **Complete data source architecture** - Plugin-based system for external data integration
  - 4 production-ready data source plugins: CSV, Python, HTTP, JSON
  - Self-registering plugin architecture following established patterns
  - Template-based bash code generation with comprehensive error handling
  - Configuration validation with detailed error messages

#### New Data Source Plugins

- **CSV/TSV Source** (`src/unifyweaver/sources/csv_source.pl`)
  - Auto-header detection from first row with intelligent column mapping
  - Manual column specification for headerless files
  - Quote handling (strip/preserve/escape) for embedded delimiters
  - TSV support via configurable delimiter
  - Skip lines configuration for comments/metadata

- **Python Embedded Source** (`src/unifyweaver/sources/python_source.pl`)
  - Heredoc `/dev/fd/3` pattern for secure Python code execution
  - SQLite integration with automatic query generation and connection management
  - Inline Python code support with comprehensive error handling
  - External Python file loading and execution
  - Configurable timeout and multiple input modes (stdin, args, none)

- **HTTP Source** (`src/unifyweaver/sources/http_source.pl`)
  - curl and wget support with intelligent tool selection
  - Response caching with configurable TTL and cache management
  - Custom headers, POST data, and query parameter support
  - Cache invalidation and inspection utilities
  - Network timeout management and error handling

- **JSON Source** (`src/unifyweaver/sources/json_source.pl`)
  - jq integration with flexible filter expressions
  - File and stdin input modes for pipeline compatibility
  - Multiple output formats (TSV, JSON, raw, CSV) with @csv/@tsv support
  - Custom filter chaining and composition
  - Flag management (raw, compact, null input)

#### Enhanced Security Framework

- **Multi-Service Firewall** - Extended `src/unifyweaver/core/firewall.pl`
  - Multi-service validation (python3, curl, wget, jq, awk)
  - Network access control with host pattern matching and wildcards
  - Python import restriction system with regex parsing
  - File access patterns for read/write operations
  - Cache directory validation with pattern matching
  - Backward-compatible extensions to existing firewall predicates

- **New Security Policy Terms**
  ```prolog
  - network_access(allowed|denied) - Control HTTP source access
  - network_hosts([pattern1, pattern2, ...]) - Whitelist host patterns  
  - python_modules([module1, module2, ...]) - Restrict Python imports
  - file_read_patterns/file_write_patterns - Control file access
  - cache_dirs([dir1, dir2, ...]) - Restrict cache locations
  ```

#### Comprehensive Testing Suite

- **Unit Tests** - Individual plugin validation
  - `tests/core/test_csv_source.pl` - CSV parsing, headers, TSV support
  - `tests/core/test_python_source.pl` - Heredoc pattern, SQLite, file integration
  - `tests/core/test_firewall_enhanced.pl` - Enhanced security validation

- **Integration Tests** - `tests/core/test_data_sources_integration.pl`
  - Cross-plugin pipeline testing (CSV → Python, HTTP → JSON)
  - Multi-source firewall validation
  - Real-world ETL scenario testing (GitHub API → SQLite)
  - Complete system integration verification

#### Production Examples and Documentation

- **Complete Demo** - `examples/data_sources_demo.pl`
  - Real ETL pipeline: JSONPlaceholder API → JSON parsing → SQLite storage
  - CSV user data processing with auto-header detection
  - Multi-service firewall configuration showcase
  - Interactive and command-line execution modes

- **Implementation Plan** - `docs/DATA_SOURCES_IMPLEMENTATION_PLAN.md`
  - Complete architectural design and implementation timeline
  - Technical specifications and usage examples
  - Real-world use cases and best practices

### Technical Implementation

#### Architecture Features
- **Plugin Registration System** - Self-registering plugins with `:- initialization`
- **Template Integration** - Seamless integration with UnifyWeaver's template system
- **Configuration Validation** - Comprehensive validation with detailed error messages
- **Security-First Design** - Enhanced firewall with deny-by-default approach

#### Code Quality
- **2,000+ lines** of production-ready, thoroughly tested code
- **Comprehensive error handling** throughout all components
- **Proper documentation** with extensive inline comments
- **Follows UnifyWeaver conventions** and coding standards

### Files Added

```
docs/DATA_SOURCES_IMPLEMENTATION_PLAN.md    | 514 +++++++++++++++++
examples/data_sources_demo.pl               | 199 +++++++
src/unifyweaver/sources/csv_source.pl       | 300 ++++++++++
src/unifyweaver/sources/http_source.pl      | 344 +++++++++++
src/unifyweaver/sources/json_source.pl      | 285 ++++++++++
src/unifyweaver/sources/python_source.pl    | 310 ++++++++++
tests/core/test_csv_source.pl               | 108 ++++
tests/core/test_python_source.pl            | 124 ++++
tests/core/test_firewall_enhanced.pl        | 156 +++++
tests/core/test_data_sources_integration.pl | 201 +++++++
```

### Enhanced Existing Files

```
src/unifyweaver/core/firewall.pl            | 250 +++++++++++++++
```

### Compatibility

- **100% backward compatible** - No breaking changes to existing functionality
- **Additive enhancements only** - All existing predicates and behavior preserved
- **Self-contained plugins** - No impact on existing components
- **Safe firewall extensions** - New validation predicates don't affect existing rules

### Contributors
- John William Creighton (@s243a) - Project architecture and design guidance
- Cline (Claude 3.5 Sonnet) - Implementation of complete data source plugin system

## [0.0.1-alpha] - 2025-10-12

### Added
- **Pattern exclusion system** - `forbid_linear_recursion/1` to force alternative compilation strategies
  - Manual forbidding: `forbid_linear_recursion(pred/arity)`
  - Automatic forbidding for ordered constraints (`unordered=false`)
  - Check if forbidden: `is_forbidden_linear_recursion/1`
  - Clear forbid: `clear_linear_recursion_forbid/1`
  - Enables graph recursion with fold helpers for predicates like fibonacci
- **Educational materials workflow** - Support for Chapter 4 tutorial
  - `test_runner_inference.pl` accepts `output_dir(Dir)` option
  - Fixed module imports in `recursive_compiler.pl` for education library alias

### Changed
- **Linear recursion pattern** - Now handles 1+ independent recursive calls (not just single calls)
  - Fibonacci now compiles as linear recursion with aggregation (not tree recursion)
  - Added independence checks: arguments must be pre-computed, no inter-call data flow
  - Added structural pattern detection to distinguish tree recursion (pattern matching) from linear with aggregation (computed scalars)
  - Tests: `fibonacci/2` now uses linear recursion, `tree_sum/2` correctly identified as tree recursion

### Fixed
- **Fibonacci test isolation** - Removed from tree_recursion tests (belongs in linear_recursion)
- **Test runner inference** - Excluded `parse_tree` helper from being tested directly
- **Integration tests** - Fixed module context for test predicates (use `user:` prefix)
- **Pattern detection** - Tree recursion properly distinguished from linear with multiple calls

### Documentation
- Added `RECURSION_PATTERN_THEORY.md` - Comprehensive theory document explaining pattern distinctions
  - Detailed `forbid_linear_recursion` system documentation
  - Graph recursion with fold helper pattern
- Updated `ADVANCED_RECURSION.md` - Reflects new linear recursion behavior with examples
  - Pattern exclusion documentation
- Updated `README.md` - Corrected fibonacci classification and added pattern exclusion feature

## [1.0.0] - 2025-10-11

### Added

#### Core Features
- **Stream-based compilation** - Memory-efficient compilation using bash pipes and streams
- **Template system** - Mustache-style template rendering with file loading and caching
- **Constraint analysis** - Automatic detection of unique and ordering constraints
- **BFS optimization** - Transitive closures automatically optimized to breadth-first search
- **Cycle detection** - Proper handling of cyclic graphs without infinite loops
- **Control plane** - Firewall and preferences system for policy enforcement

#### Advanced Recursion Patterns
- **Tail recursion optimization** - Converts tail-recursive predicates to iterative bash loops
  - Accumulator pattern detection
  - Iterative loop generation
  - Tests: `count_items/3`, `sum_list/3`

- **Linear recursion** - Memoized compilation for single-recursive-call patterns
  - Automatic memoization with associative arrays
  - Pattern detection for exactly one recursive call per clause
  - Tests: `factorial/2`, `length/2`

- **Tree recursion** - Handles multiple recursive calls (2+)
  - List-based tree representation: `[value, [left], [right]]` or `[]`
  - Fibonacci-like pattern recognition
  - Binary tree operations (sum, height, count)
  - Bracket-depth tracking parser for nested structures
  - Tests: `fibonacci/2`, `tree_sum/2`

- **Mutual recursion** - Handles predicates calling each other cyclically
  - SCC (Strongly Connected Components) detection via Tarjan's algorithm
  - Shared memoization tables across predicate groups
  - Call graph analysis
  - Tests: `is_even/1`, `is_odd/1`

#### Module Structure
- `template_system.pl` - Template rendering engine
- `stream_compiler.pl` - Non-recursive predicate compilation
- `recursive_compiler.pl` - Basic recursion analysis and compilation
- `constraint_analyzer.pl` - Constraint detection and optimization
- `firewall.pl` - Policy enforcement for backend usage
- `preferences.pl` - Layered configuration management
- `advanced/` directory with specialized recursion compilers:
  - `advanced_recursive_compiler.pl` - Pattern orchestration
  - `call_graph.pl` - Predicate dependency graphs
  - `scc_detection.pl` - Tarjan's SCC algorithm
  - `pattern_matchers.pl` - Pattern detection utilities
  - `tail_recursion.pl` - Tail recursion compiler
  - `linear_recursion.pl` - Linear recursion compiler
  - `tree_recursion.pl` - Tree recursion compiler
  - `mutual_recursion.pl` - Mutual recursion compiler

#### Testing Infrastructure
- Comprehensive test suite with 28+ tests
- Auto-discovery test system
- Test environment with `init_testing.sh` and PowerShell support
- Individual test predicates: `test_stream`, `test_recursive`, `test_advanced`, `test_constraints`
- Bash test runners for generated scripts

#### Documentation
- README.md with comprehensive examples
- TESTING.md for test environment setup
- ADVANCED_RECURSION.md for recursion pattern details
- PROJECT_STATUS.md for roadmap tracking
- Implementation summaries in `context/` directory

### Fixed
- Module import paths in `stream_compiler.pl` (library → local paths)
- Singleton variable warning in `compile_facts_debug/4`
- Linear pattern matcher now correctly counts recursive calls
- Tree parser handles nested bracket structures properly

### Technical Details

#### Pattern Detection Priority
1. Tail recursion (simplest optimization)
2. Linear recursion (single recursive call)
3. Tree recursion (multiple recursive calls)
4. Mutual recursion (multi-predicate cycles)
5. Basic recursion (fallback with BFS)

#### Code Generation Features
- Associative arrays for O(1) lookups
- Work queues for BFS traversal
- Duplicate detection with process-specific temp files
- Stream functions for composition
- Memoization tables with automatic caching
- Iterative loops for tail recursion

#### Compilation Options
- `unique(true)` - Enables early exit optimization for single-result predicates
- `unordered(true)` - Enables sort-based deduplication
- Runtime options override declarative constraints

### Known Limitations
- Tree recursion uses simple list-based representation only
- Parser has limitations with deeply nested structures (addressed with bracket-depth tracking)
- Memoization disabled by default for tree recursion in v1.0
- Divide-and-conquer patterns not yet supported
- Bash 4.0+ required for associative arrays

### Contributors
- John William Creighton (@s243a) - Core development
- Gemini (via gemini-cli) - Constraint awareness features
- Claude (via Claude Code) - Advanced recursion system, test infrastructure

---

## Release Notes

### v1.0.0 Highlights

This is the initial stable release of UnifyWeaver, featuring:

**Complete Advanced Recursion Support:**
- 4 recursion pattern compilers (tail, linear, tree, mutual)
- Automatic pattern detection and optimization
- Comprehensive test coverage

**Production-Ready Features:**
- Stream-based compilation for memory efficiency
- Template system with file loading
- Constraint-aware code generation
- Policy enforcement via control plane

**Developer Experience:**
- Auto-discovery test environment
- Cross-platform support (Linux, WSL, Windows)
- Extensive documentation
- 28+ passing tests

### Upgrade Notes

This is the first release, so no upgrade path is needed.

### Future Roadmap

**v1.1:** Improved pattern detection and memoization defaults
**v1.2:** Better tree parsing and more complex tree structures
**v1.3:** C# backend with native type support
**v2.0:** Dynamic sources plugin system (AWK, SQL integration)

---

[1.0.0]: https://github.com/s243a/UnifyWeaver/releases/tag/v1.0.0
