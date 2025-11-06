<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 1: Core UnifyWeaver & Bash Target

**Learning Path for UnifyWeaver's Core Features and Bash Code Generation**

This book covers UnifyWeaver's fundamental architecture, Prolog-to-Bash compilation, data source plugins, and parallel processing capabilities.

## Prerequisites

- Basic understanding of command-line interfaces
- Familiarity with programming concepts (variables, functions, loops)
- Interest in learning Prolog and/or Bash scripting
- SWI-Prolog 8.0+ installed
- Bash 4.0+ (for associative arrays)

## Learning Path

### Part 1: Foundations (Chapters 1-4)

**1. [Introduction](01_introduction)
- What is UnifyWeaver?
- Why Prolog-to-Bash compilation?
- Use cases and benefits

**2. [Prolog Fundamentals](02_prolog_fundamentals)
- Facts, rules, and queries
- Unification and pattern matching
- Lists and recursion basics

**3. [UnifyWeaver Architecture](03_unifyweaver_architecture)
- Compiler pipeline overview
- Source → Target translation
- Plugin system introduction

**4. [Your First Program](04_your_first_program)
- Writing simple Prolog predicates
- Compiling to Bash
- Running and testing generated scripts

### Part 2: Core Compilation Techniques (Chapters 5-8)

**5. [Stream Compilation](05_stream_compilation)
- Memory-efficient pipeline processing
- Bash pipes and process substitution
- Stream-based query execution

**6. [Advanced Constraints](06_advanced_constraints)
- Unique constraints
- Ordering constraints
- Optimization through constraint awareness

**7. [Variable Scope and Process Substitution](07_variable_scope_and_process_substitution)
- Bash variable scoping challenges
- Process substitution patterns
- Avoiding common pitfalls

**8. [Template System](08_template_system)
- Code generation architecture
- Template syntax and usage
- Creating custom templates

### Part 3: Advanced Recursion (Chapters 9-10)

**9. [Advanced Recursion](09_advanced_recursion)
- Pattern detection and classification
- Tail recursion optimization
- Linear recursion (fibonacci, factorial)
- Tree recursion (binary trees, nested structures)
- Mutual recursion (is_even/is_odd)
- BFS optimization for transitive closures

**10. [Prolog Introspection](10_prolog_introspection)
- Using `clause/2` for dynamic analysis
- Predicate inspection at compile-time
- Meta-programming techniques

### Part 4: Testing & Code Quality (Chapters 11-12)

**11. [Test Runner Inference](11_test_runner_inference)
- Automatic test discovery
- Declarative test specification
- Test result aggregation

**12. [Recursive Compilation](12_recursive_compilation)
- Self-hosting compilation
- Recursive predicate handling
- Advanced compilation strategies

### Part 5: Parallelism & Data Sources (Chapter 13-14)

**13. [Partitioning and Parallel Execution](13_partitioning_and_parallel_execution)
- Data partitioning strategies
- Parallel processing backends
- Performance optimization

**14. [Data Sources and ETL Pipelines](data_sources_pipeline_guide)
- CSV/TSV sources
- JSON sources with jq integration
- HTTP sources for web APIs
- Python sources for complex transformations
- AWK sources for text processing
- XML sources for structured documents (NEW)
- Building complete ETL pipelines

## Appendices

**[A. Recursion Patterns](A_appendix_recursion_patterns)
- Comprehensive recursion pattern reference
- Pattern matching flowchart
- Code examples for each pattern

**[B. SIGPIPE and Streaming Safety](appendix_a_sigpipe_streaming_safety)
- Understanding SIGPIPE errors
- Safe pipeline construction
- Error handling strategies

**[C. Fold Pattern Deep Dive](appendix_b_fold_pattern_deep_dive)
- Fold-based compilation
- Left fold vs right fold
- Advanced fold patterns

## Case Studies

**[Production Pipeline](case_study_production_pipeline)
- Real-world ETL example
- Multi-stage data processing
- Integration patterns

**[Declarative Testing](declarative_testing)
- Test-driven development with UnifyWeaver
- Property-based testing
- Integration testing patterns

## Additional Resources

- Main repository: https://github.com/s243a/UnifyWeaver
- Documentation: `../README.md`
- Examples: `../../examples/`

## What's Next?

After completing Book 1, you'll understand:
- ✅ Core Prolog and UnifyWeaver concepts
- ✅ Bash code generation techniques
- ✅ Data source plugins and ETL pipelines
- ✅ Advanced recursion patterns
- ✅ Parallel processing strategies

**Continue to Book 2** to learn about:
- C# target language compilation
- Multi-target code generation
- Runtime library design
- Cross-platform deployment

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
