# Book 4: Workflows and Playbooks

**Strategic Guides for AI Agents**

*Part of the [UnifyWeaver Education Series](../README.md)*

Welcome to the comprehensive guide for designing workflows and playbooks that enable AI agents to work effectively with UnifyWeaver!

> **Prerequisites**: Complete [Books 1-3](../README.md) for core concepts. This book helps you plan before exploring more targets.

## What This Book Covers

This book teaches you how to create **strategic guides** for AI agents that go beyond simple scripts:

- **Workflows**: High-level operating principles and strategic frameworks
- **Playbooks**: Specific, actionable plans for concrete tasks
- **Example Libraries**: Reusable patterns and code snippets
- **Tool Integration**: How to leverage UnifyWeaver's capabilities
- **Economic Decision-Making**: Cost, speed, and quality trade-offs

## The Paradigm Shift

### From Scripts to Strategies

Traditional programming: Write scripts that execute fixed sequences of commands.

**Workflow-based AI**: Provide agents with:
- Multiple strategies to achieve goals
- Economic data to make intelligent trade-offs
- Tool libraries for execution
- Heuristic guidance for decision-making

The AI agent becomes an **economic strategist**, not just a script executor.

## Chapters

### [Chapter 1: Introduction - Thinking in Workflows](01_introduction.md)
From scripts to strategies. Learn the distinction between workflows (strategic guides) and playbooks (specific execution plans), and how AI agents use them.

### [Chapter 2: Playbook Format](02_playbook_format.md)
The standardized format for creating playbooks. Covers frontmatter, callout types (`[!output]`, `[!example-record]`), and best practices.

### [Chapter 3: Pipeline Orchestration](03_pipeline_orchestration.md)
Design multi-stage pipelines that chain compilation targets. AWK → Python → Go and other cross-language workflows.

### [Chapter 4: Economic Decision Making](04_economic_decisions.md)
How AI agents choose strategies based on cost, speed, and quality trade-offs. Heuristics and decision frameworks.

### [Chapter 5: Example Libraries](05_example_libraries.md)
Create, organize, and use reusable code patterns. Naming conventions, cross-references, and maintenance.

## Additional Resources

### Example Libraries
[examples_library/](examples_library/)

Reusable code patterns and examples:

- **[compilation_examples.md](examples_library/compilation_examples.md)**: Factorial, ancestor compilation
- **[recursion_examples.md](examples_library/recursion_examples.md)**: Recursive Prolog patterns
- **[testing_examples.md](examples_library/testing_examples.md)**: Test runner generation
- **[parallel_examples.md](examples_library/parallel_examples.md)**: Parallel execution patterns

### Sample Playbooks
[playbooks/](playbooks/)

Complete playbook examples:

- **[factorial_compilation.md](playbooks/factorial_compilation.md)**: Linear recursion example
- **[gnu_prolog_compilation.md](playbooks/gnu_prolog_compilation.md)**: GNU Prolog target

## Quick Start

### 5-Minute Introduction

1. Read [Chapter 1: Introduction](01_introduction.md) (5 min)
2. Skim [Chapter 2: Playbook Format](02_playbook_format.md) (2 min)
3. Look at [Factorial Playbook](playbooks/factorial_compilation.md) (3 min)

### Complete Guide

Work through all chapters (1-2 hours).

## Key Concepts

### Workflows

**Definition**: The overarching strategic guide for an AI agent.

**Contains**:
- High-level goals
- General operating principles
- Types of tools available
- Economic decision criteria
- Heuristic guidance

**Characteristics**:
- Highly reusable
- Defines agent behavior broadly
- Context for many tasks

**Example**: "When compiling Prolog to Bash, analyze the recursion pattern first, then choose between linear recursion, transitive closure, or general recursive compilation based on pattern detection results."

### Playbooks

**Definition**: Specific, detailed project plans for accomplishing particular tasks.

**Contains**:
- Concrete steps
- Specific tools to use
- Conditional logic
- Expected outputs
- Verification steps

**Characteristics**:
- Task-specific
- Detailed and actionable
- Operates within workflow framework

**Example**: "To compile factorial: 1) Create /tmp/factorial.pl with this code, 2) Run compiler_driver with factorial/2, 3) Verify output script exists, 4) Test with inputs 0, 5, 10."

### Example Libraries

**Purpose**: Provide concrete instances of patterns, tools, and solutions.

**Structure**:
- Organized by topic (recursion, compilation, testing)
- Tagged with IDs and metadata
- Cross-referenced between libraries
- Extractable by tools

**Usage**: Playbooks reference examples rather than embedding large code blocks.

## Design Principles

### 1. Separation of Concerns

**Workflows**: Strategic thinking and decision-making
**Playbooks**: Concrete execution plans
**Examples**: Reusable implementation patterns

Don't mix these! Keep strategies separate from tactics.

### 2. Economic Awareness

Every strategy should have:
- **Cost**: Resource usage (time, memory, complexity)
- **Speed**: Execution time
- **Quality**: Correctness, robustness, maintainability

AI agents use this data to make trade-offs.

### 3. Heuristic Guidance

Provide rules of thumb:
- "For lists <100 items, use simple iteration"
- "For recursive depth >10, consider memoization"
- "For production deployments, prefer compiled binaries"

### 4. Composability

Design playbooks and examples to work together:
- Reference examples from libraries
- Chain playbooks for complex workflows
- Build on existing patterns

### 5. Verifiability

Every playbook should include:
- Expected outputs
- Verification steps
- Success criteria
- Error handling

## Callout Reference

### `[!output]` - Expected Output

Specifies what the agent should generate.

```markdown
> [!output]
> language: prolog
> purpose: Define factorial predicate
> format: executable
>
> ```prolog
> factorial(0, 1).
> factorial(N, F) :- N > 0, N1 is N-1, factorial(N1, F1), F is N * F1.
> ```
```

**Guidelines**:
- Keep <10% of playbook content
- Use as template/placeholder
- Include metadata (language, purpose, format)

### `[!example-record]` - Example Reference

Tags reusable examples in libraries.

```markdown
> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion
> child_examples: [unifyweaver.testing.factorial_runner]
```

**Metadata**:
- `id`: Unique identifier (timestamp-based)
- `name`: Namespaced example name
- `pattern`: Classification tag
- `child_examples`: Related examples
- `parent_example`: Parent in hierarchy

## Common Patterns

### Pattern 1: Compile-and-Test Workflow

1. Define Prolog predicate
2. Save to file
3. Compile with compiler_driver
4. Generate test runner
5. Execute tests
6. Verify results

**Example**: [Factorial Complete Workflow](examples_library/compilation_examples.md#factorial-complete-workflow)

### Pattern 2: Data Source Integration

1. Declare data source (`:- source(...)`)
2. Define domain predicate
3. Compile with dynamic source
4. Generate target scripts
5. Verify pipeline

**Example**: [Data Source Playbooks](examples_library/data_source_playbooks.md)

### Pattern 3: Multi-Target Generation

1. Define predicate once
2. Compile to Bash
3. Compile to C#
4. Test both targets
5. Compare results

### Pattern 4: Recursive Pattern Optimization

1. Analyze recursion type
2. Choose optimizer (linear, transitive, general)
3. Compile with appropriate strategy
4. Verify optimization applied

**Example**: [Ancestor Compilation](examples_library/compilation_examples.md#ancestor-compilation)

## Tool Integration

### UnifyWeaver Compiler Driver

**What**: Main compilation orchestrator
**When**: Always use for Prolog→Bash/C# compilation
**How**: `compile(Predicate/Arity, Options, Scripts)`

**Options**:
- `output_dir(Dir)`: Where to write scripts
- `target(bash|csharp)`: Target language
- `optimize(Level)`: Optimization level

### Test Runner Inference

**What**: Automatically generate test suites
**When**: After compilation, for verification
**How**: `generate_test_runner_inferred(Output, Options)`

**Options**:
- `mode(explicit|inferred)`: Test generation mode
- `output_dir(Dir)`: Test script location

### Example Extraction Tool

**What**: Perl script to extract examples from libraries
**When**: Building playbooks dynamically
**How**: `extract_example.pl --id <example-id>`

## Creating New Content

### Adding a New Playbook

1. Start with playbook format template
2. Define goal clearly
3. Provide strategies (if multiple approaches exist)
4. Reference relevant examples from libraries
5. Include expected outputs
6. Add verification steps

### Adding to Example Libraries

1. Choose appropriate library file
2. Use `[!example-record]` callout
3. Assign unique ID (timestamp-based)
4. Use namespaced naming (e.g., `unifyweaver.pattern.name`)
5. Tag with pattern classification
6. Cross-reference related examples

### Expanding Workflows

1. Identify strategic decision points
2. Document alternative approaches
3. Provide economic data (cost/speed/quality)
4. Add heuristics for selection
5. Reference playbooks for each strategy

## Current Status

### Complete ✅
- Chapter 1: Thinking in Workflows
- Playbook format specification
- Core example libraries:
  - Compilation examples (factorial, ancestor)
  - Recursion examples (ancestor)
  - Data source principles

### In Progress 🚧
- Additional playbook examples
- More compilation patterns
- Multi-target workflows
- Partitioning playbooks

### Planned 📝
- Chapter 2: Designing Workflows
- Chapter 3: Economic Decision-Making
- Chapter 4: Tool Orchestration
- Chapter 5: Testing and Verification
- Complete factorial playbook (standalone)
- CSV data source playbook (detailed)
- Parallel execution playbooks
- Firewall policy playbooks

## Best Practices

### For Workflow Designers

1. **Think economically**: Always include cost/speed/quality data
2. **Provide options**: Multiple strategies > single approach
3. **Use heuristics**: Guide decisions with rules of thumb
4. **Stay abstract**: Workflows define principles, not details

### For Playbook Creators

1. **Be specific**: Concrete steps, specific tools
2. **Reference examples**: Don't embed large code blocks
3. **Include verification**: How to know it worked
4. **Handle errors**: What to do when things fail
5. **Keep outputs minimal**: <10% of content in `[!output]` blocks

### For Example Library Maintainers

1. **Tag consistently**: Use pattern classifications
2. **Cross-reference**: Link related examples
3. **Keep examples small**: Focus on one pattern
4. **Provide context**: Explain what and why
5. **Update metadata**: IDs, names, relationships

## Getting Help

- **Concepts**: Read Chapter 1
- **Format**: See playbook_format.md
- **Examples**: Browse examples_library/
- **Issues**: File on GitHub
- **Questions**: Community discussions

## Contributing

Want to contribute?

1. **New playbooks**: Follow format spec, reference examples
2. **Example patterns**: Add to appropriate library file
3. **Documentation**: Improve existing content
4. **Workflows**: Submit strategic guides

All contributions welcome!

## Related Books

This is part of the UnifyWeaver Educational Series:

1. **[Book 1: Foundations](../book-01-foundations/README.md)** - Architecture, Prolog basics
2. **[Book 2: Bash Target](../book-02-bash-target/README.md)** - Stream compilation
3. **[Book 3: C# Target](../book-03-csharp-target/README.md)** - Fixed-point approaches
4. **Book 4: Workflows** (This Book)

**Continue to Book 5 or 6** to explore more targets:
- [Book 5: Python Target](../book-05-python-target/README.md) - Python code generation
- [Book 6: Go Target](../book-06-go-target/README.md) - Native binaries

---

**Version**: 0.2.0
**Last Updated**: 2025-12-07
**Focus**: Strategic AI agent guidance
