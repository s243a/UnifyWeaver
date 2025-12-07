# The Prolog Target: Implementation and Usage Guide

**UnifyWeaver Educational Series**

This book provides a comprehensive guide to UnifyWeaver's Prolog target, covering implementation principles, practical usage, and advanced features like dialect selection and fallback mechanisms.

## Table of Contents

### Part 1: Fundamentals

1. [Introduction to the Prolog Target](01_introduction.md)
   - What is the Prolog target?
   - Why transpile Prolog to Prolog?
   - Use cases and benefits

2. [Architecture Overview](02_architecture.md)
   - The transpilation pipeline
   - Core components
   - Module structure

3. [Understanding Prolog Dialects](03_dialects.md)
   - SWI-Prolog vs GNU Prolog
   - Dialect capabilities and constraints
   - When to use each dialect

### Part 2: Using the Prolog Target

4. [Basic Usage](04_basic_usage.md)
   - Your first Prolog transpilation
   - Generating executable scripts
   - Options and configuration

5. [Compilation vs Interpretation](05_compilation_modes.md)
   - Interpreted scripts (SWI-Prolog)
   - Compiled binaries (GNU Prolog)
   - Choosing the right mode
   - Initialization directives

6. [Dependency Management](06_dependencies.md)
   - How dependencies are detected
   - Module imports
   - Partitioner and data source integration

### Part 3: Advanced Features

7. [The Firewall System](07_firewall_integration.md)
   - Security policies for Prolog targets
   - Restricting dialect usage
   - Network and file access controls

8. [Dialect Fallback Mechanisms](08_fallback_mechanisms.md)
   - Preference-based dialect selection
   - Automatic fallback on compilation failure
   - Multi-dialect strategies
   - Error handling and recovery

9. [Validation and Compatibility](09_validation.md)
   - Dialect-specific validation
   - Detecting incompatible features
   - Compatibility warnings

### Part 4: Implementation Deep Dive

10. [Code Generation Pipeline](10_code_generation.md)
    - From AST to Prolog source
    - Shebang and header generation
    - User code transpilation
    - Entry point generation

11. [Dialect-Specific Code](11_dialect_specifics.md)
    - Initialization directives
    - Compile commands
    - Import strategies
    - Header differences

12. [Compilation Infrastructure](12_compilation.md)
    - The compile_script/2 predicate
    - Error detection and reporting
    - Safe compilation with fallback
    - Exit code handling

### Part 5: Practical Applications

13. [Common Patterns](13_patterns.md)
    - Standalone scripts
    - Compiled executables
    - Library integration
    - Testing strategies

14. [Troubleshooting](14_troubleshooting.md)
    - Common errors
    - Debugging generated code
    - Compilation failures
    - Performance considerations

15. [Case Study: Factorial Compilation](15_case_study.md)
    - End-to-end example
    - Code walkthrough
    - Testing and verification

### Appendices

- [Appendix A: Prolog Target API Reference](appendix_a_api_reference.md)
- [Appendix B: Dialect Comparison Table](appendix_b_dialect_comparison.md)
- [Appendix C: Firewall Policy Examples](appendix_c_firewall_examples.md)

---

## How to Use This Book

- **Beginners**: Start with Part 1 and 2 to understand the basics
- **Users**: Focus on Part 2 and 3 for practical usage
- **Implementers**: Part 4 provides deep implementation details
- **Everyone**: Part 5 shows real-world applications

Each chapter includes:
- Conceptual explanations
- Code examples
- Practical exercises
- Common pitfalls

---

**Version**: UnifyWeaver v0.1
**Last Updated**: 2025-11-17

---

## Navigation

[📖 Book 11: Prolog Target](./) | [Next: Chapter 1: Introduction to the Prolog Target →](01_introduction)
