# UnifyWeaver

A Prolog-to-Bash compiler that transforms declarative logic programs into efficient streaming bash scripts. UnifyWeaver specializes in compiling data relationships and queries into executable bash code with optimized handling of transitive closures and advanced recursion patterns.

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
- **Linear recursion** - Memoized compilation for single-recursive-call patterns
- **Mutual recursion** - Handles predicates that call each other cyclically via SCC detection
- **Constraint awareness** - Unique and ordering constraints optimize generated code
- **Pattern detection** - Automatic classification of recursion patterns

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

### Advanced Recursion Examples

#### Tail Recursion (Optimized to Loop)
```prolog
% Count list items with accumulator
count_items([], Acc, Acc).
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).

% Compile with unique constraint
?- compile_advanced_recursive(count_items/3, [unique(true)], BashCode).
```

#### Mutual Recursion (Even/Odd)
```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).

% Compile predicate group together
?- compile_predicate_group([is_even/1, is_odd/1], [unique(true)], BashCode).
```

### Compilation Options

```prolog
% Basic recursion
compile_recursive(Pred/Arity, Options, BashCode)

% Advanced recursion (tail, linear, mutual)
compile_advanced_recursive(Pred/Arity, Options, BashCode)

% Explicit predicate groups (for mutual recursion)
compile_predicate_group([Pred1/Arity1, Pred2/Arity2, ...], Options, BashCode)

% Options:
%   unique(true)     - Only one result expected (enables early exit optimization)
%   unordered(true)  - Results can be in any order (enables optimizations)

% Run full test suite
test_recursive_compiler
```

## Architecture

### Module Structure

**Core Modules:**
- **template_system.pl** - Template rendering engine with mustache-style placeholders
- **stream_compiler.pl** - Handles non-recursive predicates
- **recursive_compiler.pl** - Analyzes basic recursion patterns and generates optimized code
- **constraint_analyzer.pl** - Constraint detection and analysis system

**Advanced Recursion Modules** (`src/unifyweaver/core/advanced/`):
- **advanced_recursive_compiler.pl** - Orchestrator for advanced patterns
- **call_graph.pl** - Build predicate dependency graphs
- **scc_detection.pl** - Tarjan's algorithm for strongly connected components
- **pattern_matchers.pl** - Pattern detection utilities
- **tail_recursion.pl** - Tail recursion → iterative loop compiler
- **linear_recursion.pl** - Linear recursion → memoized compiler
- **mutual_recursion.pl** - Mutual recursion → joint memo table compiler

### Compilation Pipeline

1. **Classification** - Analyzes predicate to determine recursion pattern
2. **Constraint Analysis** - Detects unique/ordering constraints
3. **Pattern Matching** - Tries tail → linear → mutual → basic recursion
4. **Optimization** - Applies pattern-specific optimizations (BFS, loops, memoization)
5. **Template Selection** - Chooses appropriate bash template
6. **Code Generation** - Renders template with predicate-specific values

### Generated Code Features

- Associative arrays for O(1) lookups
- Work queues for BFS traversal
- Duplicate detection
- Process-specific temp files
- Stream functions for composition
- Memoization tables for linear/mutual recursion
- Iterative loops for tail recursion

## Examples

### Family Tree Queries
```prolog
ancestor(X, Y)    % Transitive closure of parent
descendant(X, Y)  % Reverse of ancestor
sibling(X, Y)     % Same parent, different children
```

### Graph Reachability
```prolog
connected(X, Y)   % Direct connection
reachable(X, Y)   % Transitive closure of connected
```

### Recursive Computations
```prolog
factorial(N, Result)  % Linear recursion with memoization
count_items(List, Count)  % Tail recursion optimized to loop
is_even(N), is_odd(N)  % Mutual recursion with shared memo
```

## Recursion Support

### ✅ What Works

**Basic Recursion:**
- Simple self-recursion with base cases (e.g., `ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)`)
- Transitive closure patterns (optimized to BFS)
- Non-recursive predicates that call each other

**Advanced Recursion:**
- **Tail recursion** - Compiled to iterative loops (e.g., `count([], Acc, Acc). count([_|T], A, N) :- A1 is A+1, count(T, A1, N).`)
- **Linear recursion** - Compiled with memoization (e.g., `length([], 0). length([_|T], N) :- length(T, N1), N is N1+1.`)
- **Mutual recursion** - Predicates calling each other in cycles (e.g., `even(N) :- N > 0, N1 is N-1, odd(N1)` with `odd(N) :- N > 0, N1 is N-1, even(N1)`)
- **SCC detection** - Automatic detection of mutually recursive predicate groups
- **Constraint optimization** - Unique constraints enable early exit in generated code

### ⚠️ Current Limitations

**Recursion Patterns:**
- Complex recursive patterns (divide-and-conquer, tree recursion) - not yet supported
- Recursive aggregation with complex accumulation - partial support

### Known Issues

1. **Process ID in Templates**: Some older generated scripts may have `$` instead of `$$` for process IDs, potentially causing temp file collisions in parallel executions. (Recent templates fixed)

2. **SIGPIPE Handling**: When piping output to commands like `head`, you may see "permission denied" errors after the pipe closes. Workaround: use `2>/dev/null` or write to temp file first.

3. **Dependency Ordering**: Related predicates that depend on each other must be sourced in the correct order in test scripts.

4. **Variable Scoping**: The system requires bash 4+ with associative arrays. Earlier bash versions will not work.

### Scope

This compiler focuses on transforming Prolog predicates that represent data relationships and queries into efficient bash scripts. It is not intended for:
- Arithmetic-heavy computations (though basic arithmetic is supported)
- General-purpose Prolog execution
- Complex constraint solving
- Meta-predicates or higher-order logic

## Testing

### Test Environment

The test environment provides automatic test discovery:

```bash
cd scripts/testing
./init_testing.sh  # or Init-TestEnvironment.ps1 on Windows

# In SWI-Prolog:
?- test_all.          % All tests (manual + auto-discovered)
?- test_stream.       % Stream compilation tests
?- test_recursive.    % Basic recursion tests
?- test_advanced.     % Advanced recursion tests
?- test_constraints.  % Constraint system tests
?- test_auto.         % Auto-discovered tests only
```

### Manual Testing

Run individual test suites:
```prolog
?- test_template_system.      % Test template rendering
?- test_stream_compiler.      % Test non-recursive compilation
?- test_recursive_compiler.   % Test recursive predicate compilation
```

Verify generated scripts:
```bash
cd output
bash test.sh           # Test non-recursive predicates
bash test_recursive.sh  # Test recursive predicates
cd advanced
bash test_runner.sh    # Test advanced recursion patterns
```

## Documentation

- [TESTING.md](docs/TESTING.md) - Test environment and adding tests
- [ADVANCED_RECURSION.md](docs/ADVANCED_RECURSION.md) - Advanced recursion patterns
- [PROJECT_STATUS.md](context/PROJECT_STATUS.md) - Current status and roadmap

## Contributing

Issues and pull requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Key areas for contribution:
- Additional recursion patterns (divide-and-conquer, tree recursion)
- Enhanced arithmetic operation support
- Additional graph algorithms
- Performance optimizations
- External data source integration (AWK, SQL)

## Future Enhancements

- **Dynamic sources** - Plugin system for AWK, SQL, and other data sources
- **External optimization** - GNU Parallel and Hadoop Streaming integration
- **Automatic dependency ordering** in generated scripts
- **Enhanced arithmetic** - More complex mathematical operations
- **Parallel execution** support
- **Incremental compilation**

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