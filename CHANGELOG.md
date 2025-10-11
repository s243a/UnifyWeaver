# Changelog

All notable changes to UnifyWeaver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Linear recursion pattern** - Now handles 1+ independent recursive calls (not just single calls)
  - Fibonacci now compiles as linear recursion with aggregation (not tree recursion)
  - Added independence checks: arguments must be pre-computed, no inter-call data flow
  - Added structural pattern detection to distinguish tree recursion (pattern matching) from linear with aggregation (computed scalars)
  - Tests: `fibonacci/2` now uses linear recursion, `tree_sum/2` correctly identified as tree recursion

### Fixed
- **Test runner inference** - Excluded `parse_tree` helper from being tested directly
- **Integration tests** - Fixed module context for test predicates (use `user:` prefix)
- **Pattern detection** - Tree recursion properly distinguished from linear with multiple calls

### Documentation
- Added `RECURSION_PATTERN_THEORY.md` - Comprehensive theory document explaining pattern distinctions
- Updated `ADVANCED_RECURSION.md` - Reflects new linear recursion behavior with examples
- Updated `README.md` - Corrected fibonacci classification

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
