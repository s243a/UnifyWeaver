# UnifyWeaver Compiler Driver Architecture

**Status:** Design Document
**Version:** 1.0
**Date:** 2025-11-08

## 1. Definition and Purpose

The `compiler_driver.pl` is the central orchestrator or "brain" of the UnifyWeaver transpilation process. It is not a compiler itself, but a manager of the compilation pipeline. Its primary responsibility is to take a target Prolog predicate and guide it through the necessary steps to produce an executable script in a target language (e.g., Bash, PowerShell, C#).

### Entry Points
- `compile/2`: `compile(Predicate, GeneratedScripts)` - Compiles a predicate with default options
- `compile/3`: `compile(Predicate, Options, GeneratedScripts)` - Compiles with custom options (e.g., target language selection)

## 2. Core Workflow

The driver operates via a multi-stage pipeline:

### Step 1: Dependency Analysis
The driver's first task is to understand the predicate's dependencies. It calls the `dependency_analyzer.pl` to build a tree of all other user-defined predicates that the target predicate relies on. For example, compiling `grandparent/2` would identify a dependency on `parent/2`.

**Note on Built-in Predicates:** The dependency analyzer explicitly **excludes** built-in predicates (e.g., `>/2`, `is/2`, `write/1`) from the dependency list. This is by design. Built-ins are handled directly by the backend compilers during code generation, not as separate compilation units. Each backend compiler is responsible for translating built-in operations into the target language's equivalent constructs (e.g., `is/2` becomes bash arithmetic `$((...))` expressions).

### Step 2: Compilation Planning
Based on the dependency tree, the driver creates a compilation plan. It ensures that dependencies are compiled before the predicates that rely on them. This guarantees that when, for example, the `grandparent` bash script is generated, the `parent` bash functions it needs to call are already defined and available.

### Step 3: Predicate Classification
For each predicate in its plan, the driver performs introspection to classify its structure. This is the key to selecting the correct compilation strategy. The primary classifications include:
- **Facts:** A simple collection of data.
- **Non-Recursive Rule:** A rule that does not call itself.
- **Recursive Rule:** A rule that calls itself. This is further sub-classified into patterns like `tail`, `linear`, `tree`, and `mutual` recursion by more advanced components.

### Step 4: Dispatch to Backend Compiler
Based on the classification, the driver dispatches the predicate to the appropriate specialized backend compiler:
- **Facts & Simple Rules** are typically sent to `stream_compiler.pl`, which excels at creating linear, streaming data pipelines in bash.
- **Recursive Predicates** are sent to `recursive_compiler.pl` (or `advanced_recursive_compiler.pl`), which contains the complex logic to translate recursion into optimized bash loops, data structures, and memoization tables.
- **Other Targets:** For targets like C# or PowerShell, it would dispatch to their respective compiler modules (e.g., `csharp_query_target.pl`).

### Step 5: Artifact Assembly
The driver gathers the string of generated code returned by each backend compiler and writes the final, executable scripts to the specified output directory (e.g., `parent.sh`, `grandparent.sh`).

## 3. Architectural Role

The `compiler_driver` acts as a "general contractor." It decouples the high-level orchestration of the compilation process from the low-level, target-specific details of code generation. It reads the blueprint (the Prolog code) and directs the appropriate specialists (the backend compilers) to build the final product.

This modular design allows the UnifyWeaver system to be extended with new recursion patterns, new optimization techniques, or even entirely new target languages simply by adding new backend compilers for the driver to dispatch to.

## 4. Concrete Example

Given these Prolog predicates:
```prolog
parent(alice, bob).
parent(bob, charlie).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

When `compile(grandparent/2, [], Scripts)` is called:

1. **Dependency Analysis**: Identifies `parent/2` as a dependency
2. **Compilation Planning**: Plans to compile `parent/2` first, then `grandparent/2`
3. **Classification**:
   - `parent/2` → Facts (pure data)
   - `grandparent/2` → Non-recursive rule (joins two predicates)
4. **Dispatch**:
   - `parent/2` → `stream_compiler.pl` generates bash script with data
   - `grandparent/2` → `stream_compiler.pl` generates bash pipeline joining results
5. **Assembly**: Returns `['parent.sh', 'grandparent.sh']`

The resulting `grandparent.sh` can be executed as a standalone bash script, piping data through the compilation pipeline.
