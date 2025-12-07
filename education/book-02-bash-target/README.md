<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 2: Bash Target

**Learning Path for UnifyWeaver's Bash Code Generation**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers Prolog-to-Bash compilation, stream processing, data source plugins, and parallel processing capabilities.

> **Prerequisites**: Complete [Book 1: Foundations](../book-01-foundations/README.md) first for architecture and Prolog basics.

## Prerequisites

- [Book 1: Foundations](../book-01-foundations/README.md) completed
- Bash 4.0+ (for associative arrays)
- Basic understanding of Unix pipelines

## Learning Path

### Part 1: Getting Started (Chapter 1)

**[Chapter 1: Your First Program](01_your_first_program.md)**
- Writing simple Prolog predicates
- Compiling to Bash
- Running and testing generated scripts

### Part 2: Core Compilation Techniques (Chapters 2-5)

**[Chapter 2: Stream Compilation](02_stream_compilation.md)**
- Memory-efficient pipeline processing
- Bash pipes and process substitution
- Stream-based query execution

**[Chapter 3: Advanced Constraints](03_advanced_constraints.md)**
- Unique constraints
- Ordering constraints
- Optimization through constraint awareness

**[Chapter 4: Variable Scope and Process Substitution](04_variable_scope_and_process_substitution.md)**
- Bash variable scoping challenges
- Process substitution patterns
- Avoiding common pitfalls

**[Chapter 5: Template System](05_template_system.md)**
- Code generation architecture
- Template syntax and usage
- Creating custom templates

### Part 3: Advanced Recursion (Chapters 6-7)

**[Chapter 6: Advanced Recursion](06_advanced_recursion.md)**
- Pattern detection and classification
- Tail recursion optimization
- Linear recursion (fibonacci, factorial)
- Tree recursion (binary trees, nested structures)
- Mutual recursion (is_even/is_odd)
- BFS optimization for transitive closures

**[Chapter 7: Prolog Introspection](07_prolog_introspection.md)**
- Using `clause/2` for dynamic analysis
- Predicate inspection at compile-time
- Meta-programming techniques

### Part 4: Testing & Code Quality (Chapters 8-9)

**[Chapter 8: Test Runner Inference](08_test_runner_inference.md)**
- Automatic test discovery
- Declarative test specification
- Test result aggregation

**[Chapter 9: Recursive Compilation](09_recursive_compilation.md)**
- Self-hosting compilation
- Recursive predicate handling
- Advanced compilation strategies

### Part 5: Parallelism & Data Sources (Chapters 10-11)

**[Chapter 10: Partitioning and Parallel Execution](10_partitioning_and_parallel_execution.md)**
- Data partitioning strategies
- Parallel processing backends
- Performance optimization

**[Chapter 11: Data Sources and ETL Pipelines](data_sources_pipeline_guide.md)
- CSV/TSV sources
- JSON sources with jq integration
- HTTP sources for web APIs
- Python sources for complex transformations
- AWK sources for text processing
- XML sources for structured documents
- Building complete ETL pipelines

## Appendices

**[A. Recursion Patterns](A_appendix_recursion_patterns.md)**
- Comprehensive recursion pattern reference
- Pattern matching flowchart
- Code examples for each pattern

**[B. SIGPIPE and Streaming Safety](appendix_a_sigpipe_streaming_safety.md)**
- Understanding SIGPIPE errors
- Safe pipeline construction
- Error handling strategies

**[C. Fold Pattern Deep Dive](appendix_b_fold_pattern_deep_dive.md)**
- Fold-based compilation
- Left fold vs right fold
- Advanced fold patterns

## Case Studies

**[Production Pipeline](case_study_production_pipeline.md)**
- Real-world ETL example
- Multi-stage data processing
- Integration patterns

**[Declarative Testing](declarative_testing.md)**
- Test-driven development with UnifyWeaver
- Property-based testing
- Integration testing patterns

## Additional Resources

- Main repository: https://github.com/s243a/UnifyWeaver
- Documentation: `../README.md`
- Examples: `examples/`

## What's Next?

After completing Book 2, you'll understand:
- ✅ Bash code generation techniques
- ✅ Stream compilation and templates
- ✅ Data source plugins and ETL pipelines
- ✅ Advanced recursion patterns
- ✅ Parallel processing strategies

**Continue to Book 3** to learn about:
- C# target language compilation
- Fixed-point approaches
- LINQ-based query execution
- Cross-platform .NET deployment

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
