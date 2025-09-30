# UnifyWeaver

A Prolog-to-Bash compiler that transforms declarative logic programs into efficient streaming bash scripts. UnifyWeaver specializes in compiling data relationships and queries into executable bash code with optimized handling of transitive closures.

## Features

- **Stream-based processing** - Memory-efficient compilation using bash pipes and streams
- **BFS optimization** - Transitive closures automatically optimized to breadth-first search
- **Cycle detection** - Proper handling of cyclic graphs without infinite loops
- **Template-based generation** - Clean separation between logic and bash code generation
- **Duplicate prevention** - Efficient tracking ensures each result appears only once
- **Process substitution** - Correct variable scoping in bash loops

## Installation

Requirements:
- SWI-Prolog 8.0 or higher
- Bash 4.0+ (for associative arrays)

```bash
git clone https://github.com/s243a/UnifyWeaver.git
cd UnifyWeaver
```

## Quick Start

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

### Compilation Options

```prolog
% Compile individual predicates
compile_recursive(Pred/Arity, Options, BashCode)

% Run full test suite
test_recursive_compiler
```

## Architecture

### Module Structure

- **template_system.pl** - Template rendering engine with mustache-style placeholders
- **stream_compiler.pl** - Handles non-recursive predicates
- **recursive_compiler.pl** - Analyzes recursion patterns and generates optimized code

### Compilation Pipeline

1. **Classification** - Analyzes predicate to determine recursion pattern
2. **Optimization** - Transitive closures converted to BFS
3. **Template Selection** - Chooses appropriate bash template
4. **Code Generation** - Renders template with predicate-specific values

### Generated Code Features

- Associative arrays for O(1) lookups
- Work queues for BFS traversal
- Duplicate detection
- Process-specific temp files
- Stream functions for composition

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

## Current Limitations

### Recursion Support

**What Works:**
- Simple self-recursion with base cases (e.g., `ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)`)
- Transitive closure patterns (optimized to BFS)
- Tail recursion (detected but currently falls back to plain recursion)
- Non-recursive predicates that call each other

**What Doesn't Work:**
- Mutual recursion where predicates call each other in cycles
  - Example: `even(N) :- N > 0, N1 is N-1, odd(N1)` with `odd(N) :- N > 0, N1 is N-1, even(N1)`
  - The system will either fail to compile these or generate incorrect code
- Complex recursive patterns (divide-and-conquer, tree recursion)
- Recursive aggregation (accumulating values through recursive calls)

### Known Issues

1. **Process ID in Templates**: Some generated scripts may have `$` instead of `$$` for process IDs, potentially causing temp file collisions in parallel executions.

2. **SIGPIPE Handling**: When piping output to commands like `head`, you may see "permission denied" errors after the pipe closes. Workaround: use `2>/dev/null` or write to temp file first.

3. **Dependency Ordering**: Related predicates that depend on each other (like `related` depending on `sibling`) must be sourced in the correct order in test scripts.

4. **Variable Scoping**: The system assumes bash 4+ with associative arrays. Earlier bash versions will not work.

### Scope

This compiler focuses on transforming Prolog predicates that represent data relationships and queries into efficient bash scripts. It is not intended for:
- Arithmetic-heavy computations
- Complex constraint solving
- Predicates with side effects
- Meta-predicates or higher-order logic

## Testing

Run the test suite:
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
```

## Contributing

Issues and pull requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Key areas for contribution:
- Mutual recursion support
- Tail recursion optimization
- Arithmetic operation support
- Additional graph algorithms

## Future Enhancements

- Mutual recursion support via strongly connected component detection
- Optimization of tail recursion to loops
- Automatic dependency ordering in generated scripts
- Support for arithmetic operations beyond simple comparisons
- Parallel execution support
- Incremental compilation
- External template file support (currently auto-generated)

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