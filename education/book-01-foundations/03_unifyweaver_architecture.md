<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: The UnifyWeaver Architecture

Now that we have a grasp of basic Prolog, we can explore how UnifyWeaver translates these concepts into executable code. This chapter provides a high-level overview of the compiler's **principal architecture**.

> **Note on Architecture Variants**: This chapter describes the *principal architecture* used for stream-based targets (Bash, AWK, Go, Rust). Other targets use variant architectures:
> - **Fixed-Point Architecture** (Book 3: C# Target) - Uses iterative fixpoint evaluation for recursive predicates
> - **Query Engine Architecture** (Book 3: C# Target) - Uses IR + runtime library with plan nodes
> - **Generator Architecture** - Uses Python generators for lazy evaluation
>
> The core concepts (classification, constraint analysis, template rendering) apply across all variants, but the execution model differs.

## The Core Concept Revisited

UnifyWeaver treats Prolog as a **declarative specification language** and target languages as **efficient execution targets**. The primary goal is not to replicate all of Prolog's features, but to compile a specific, useful subset of Prolog—data relationships and queries—into optimized code that works well in the target environment.

## The Compilation Pipeline

The transformation from a Prolog predicate to a Bash script follows a clear pipeline:

```
┌──────────────────┐
│ Prolog Predicate │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Classify Pattern │ (recursive_compiler.pl)
└────────┬─────────┘
         │
         ├────────────────────────┐
         │                        │
         ▼                        ▼
┌──────────────────┐      ┌───────────────────────────┐
│ Non-Recursive    │      │ Recursive                 │
│ (stream_compiler)│      │ (advanced_recursive_compiler) │
└────────┬─────────┘      └───────────┬───────────────┘
         │                            │
         │                            ▼
         │                  ┌───────────────────────────┐
         │                  │   Try Advanced Patterns   │
         │                  │  (tail -> linear ->       |
         |                  |  graph -> mutual)         │
         │                  └───────────┬───────────────┘
         │                              │
         └───────────────┬──────────────┘
                         │
                         ▼
┌──────────────────────────────────┐
│ Analyze Constraints & Options    │ (constraint_analyzer.pl)
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ Select & Render Template         │ (template_system.pl)
└────────────────┬─────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │   Bash Script  │
        └────────────────┘
```

1.  **Prolog Predicate:** The process starts with the Prolog predicate you want to compile (e.g., `ancestor/2`).

2.  **Pattern Analysis:** The main `recursive_compiler` first inspects the predicate to classify its pattern (non-recursive, simple recursion, or a candidate for advanced compilation).

3.  **Strategy Selection & Dispatch:**
    *   If the predicate is **not recursive**, it is handed off to the `stream_compiler`.
    *   If the predicate is **recursive**, it is passed to the `advanced_recursive_compiler`.

4.  **Advanced Pattern Matching:** The advanced compiler attempts to match the predicate against its known patterns in order of specificity: tail recursion, then linear recursion, then graph, then mutual recursion (by detecting Strongly Connected Components).

5.  **Constraint Analysis:** The compiler queries the `constraint_analyzer` to fetch any constraints for the predicate (e.g. `unique(true)`).

6.  **Template Rendering:** Based on the analysis, the compiler selects an appropriate Bash code template and uses the `template_system` to generate the final script.

7.  **Bash Script:** The final output is a complete, executable Bash script or function.

## The Core Modules

The UnifyWeaver compiler is built on a set of core Prolog modules.

### 1. `template_system.pl`
This module is a flexible templating engine used to generate the final Bash code. It allows for file-based templates, caching, and a clean separation between the compiler's logic and the Bash implementation details.

### 2. `stream_compiler.pl`
This module handles simple, **non-recursive predicates**, converting them into efficient streaming Unix pipelines.

### 3. `recursive_compiler.pl`
This is the main entry point and **dispatcher**. It performs the initial analysis and decides which specialized compiler to use.

### 4. `constraint_analyzer.pl`
This module manages and analyzes predicate constraints, such as `unique` and `ordered`, which guide the optimization process.

### 5. `advanced_recursive_compiler.pl`
This is the orchestrator for complex recursion. It uses several sub-modules (`pattern_matchers.pl`, `scc_detection.pl`, `tail_recursion.pl`, etc.) to identify and compile advanced patterns into highly optimized Bash code, such as converting tail recursion into iterative loops.

## Structure of the Generated Bash Code

The Bash code generated by UnifyWeaver follows a consistent structure.

### Facts as Associative Arrays

Prolog facts are typically compiled into a `declare -A` statement, which creates an **associative array** (also known as a hash map or dictionary). This allows for O(1), or constant time, lookups.

For a predicate like `parent(alice, bob).`, the generated code would look something like this:

```bash
declare -A parent_data=(
    ["alice:bob"]=1
)

parent() {
    local key="$1:$2"
    # Check if the key exists in the array
    [[ -n "${parent_data[$key]}" ]] && echo "$key"
}
```

### Rules as Functions and Pipelines

Prolog rules are compiled into Bash functions. 

*   **Non-recursive rules** become pipelines of commands that filter, join, and transform data streams.
*   **Recursive rules** (especially transitive closures) become the more complex BFS implementation discussed above, with `while` loops managing the queue and `grep` or similar tools finding the next set of relationships.

By compiling to these optimized structures, UnifyWeaver produces Bash code that is often far more efficient than a naive, direct translation of the Prolog logic would be.

## Next Steps

With an understanding of the architecture, we are now ready to get our hands dirty. In the next chapter, we will write our first UnifyWeaver program, defining a set of facts and rules, compiling them, and executing the resulting Bash script to see it all in action.

---

## Navigation

**←** [Previous: Chapter 2: Prolog Fundamentals for UnifyWeaver](02_prolog_fundamentals) | [📖 Book 1: Foundations](README)
