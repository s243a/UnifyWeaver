# Workflows and Playbooks - Complete Index

**UnifyWeaver Educational Series**

Welcome to the complete guide for designing AI-driven workflows and executable playbooks!

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Understand the workflow paradigm | [Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md) |
| Learn playbook format | [Playbook Format Specification](playbook_format.md) |
| See working examples | [Playbooks Directory](playbooks/) |
| Find reusable patterns | [Examples Library](examples_library/) |
| Create my first playbook | [Factorial Playbook](playbooks/factorial_compilation.md) |
| Understand the big picture | [README](README.md) |

## Directory Structure

```
book-workflow/
│
├── README.md                          # Start here! Overview and navigation
├── 00_INDEX.md                        # This file - complete directory index
├── playbook_format.md                 # Format specification for playbooks
│
├── ch1_introduction/                  # Chapter 1: Core concepts
│   └── ch1_introduction.md            # Workflows vs scripts, strategic thinking
│
├── playbooks/                         # Concrete executable playbooks
│   ├── README.md                      # Playbooks overview
│   ├── factorial_compilation.md       # ⭐ Beginner: Prolog→Bash compilation
│   └── gnu_prolog_compilation.md      # ⭐ Intermediate: Native binary compilation
│
└── examples_library/                  # Reusable code patterns
    ├── compilation_examples.md        # Factorial, ancestor compilation patterns
    ├── recursion_examples.md          # Common recursive Prolog patterns
    ├── testing_examples.md            # Test runner patterns
    ├── tool_usage_examples.md         # UnifyWeaver tool usage
    ├── log_examples.md                # Logging patterns
    ├── data_source_playbooks.md       # Data source integration principles
    └── partitioning_examples.md       # ⭐ Parallel processing with partitioning
```

## Content by Type

### Conceptual Foundation

**[Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md)**
- From scripts to strategies
- Workflows vs playbooks distinction
- Economic decision-making for AI agents
- Leveraging example libraries

**[README](README.md)**
- Overview of the workflow paradigm
- Book structure and navigation
- Quick start guides (5-min, 30-min, complete)
- Key concepts summary

### Specifications & Standards

**[Playbook Format Specification](playbook_format.md)**
- Complete playbook structure (8 sections)
- Callout types reference (`[!output]`, `[!example-record]`, `[!note]`, etc.)
- Prolog generation guidance
- Complete playbook template
- Best practices for authors and AI agents
- Quality checklist

### Executable Playbooks

**[Playbooks Directory](playbooks/README.md)**

1. **[Factorial Compilation](playbooks/factorial_compilation.md)** ⭐ START HERE
   - **Level**: Beginner
   - **Time**: 5 minutes
   - **Goal**: Compile Prolog factorial to Bash
   - **Teaches**: Basic compilation pipeline, test generation

2. **[GNU Prolog Native Binary](playbooks/gnu_prolog_compilation.md)**
   - **Level**: Intermediate
   - **Time**: 10 minutes
   - **Goal**: Generate and compile standalone executable
   - **Teaches**: Prolog target, dialect differences, binary deployment
   - **Related**: [Book: Prolog Target](../book-prolog-target/README.md)

### Example Libraries

**[Compilation Examples](examples_library/compilation_examples.md)**
- Factorial compilation (linear recursion)
- Complete factorial workflow
- Ancestor compilation (transitive closure)
- Pattern recognition and optimization

**[Recursion Examples](examples_library/recursion_examples.md)**
- Ancestor relationship (transitive closure)
- Common recursive patterns

**[Partitioning Examples](examples_library/partitioning_examples.md)** ⭐ NEW
- Fixed-size partitioning (batch processing)
- Hash-based partitioning (distributed systems)
- Key-based partitioning (grouping/aggregation)
- Parallel execution with backends
- Complete ETL pipeline example

**[Testing Examples](examples_library/testing_examples.md)**
- Test runner generation
- Verification patterns
- Automated testing

**[Tool Usage Examples](examples_library/tool_usage_examples.md)**
- compiler_driver usage
- test_runner_inference
- Example extraction tools

**[Log Examples](examples_library/log_examples.md)**
- Logging patterns
- Debugging techniques

**[Data Source Playbooks](examples_library/data_source_playbooks.md)**
- CSV integration principles
- Dynamic source handling
- Design guidelines

## Learning Paths

### Path 1: Quick Start (30 minutes)

For someone who wants to get productive quickly:

1. [README](README.md) - 5 min
2. [Factorial Playbook](playbooks/factorial_compilation.md) - Execute it! - 10 min
3. [Playbook Format](playbook_format.md) - Skim structure - 5 min
4. [Compilation Examples](examples_library/compilation_examples.md) - Browse - 5 min
5. Try modifying the factorial example - 5 min

**Outcome**: You've executed a playbook and understand the basics.

### Path 2: Playbook Author (2 hours)

For someone who wants to create playbooks:

1. [Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md) - 15 min
2. [Playbook Format Specification](playbook_format.md) - 30 min
3. [Factorial Playbook](playbooks/factorial_compilation.md) - Study structure - 20 min
4. [GNU Prolog Playbook](playbooks/gnu_prolog_compilation.md) - Compare approaches - 20 min
5. [Examples Library](examples_library/) - Browse all - 20 min
6. Create your own playbook using the template - 15 min

**Outcome**: You can write well-formatted, comprehensive playbooks.

### Path 3: Workflow Designer (3 hours)

For someone designing AI agent workflows:

1. [Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md) - 20 min
2. [README](README.md) - Complete read - 15 min
3. All playbooks - Understand strategy vs tactics - 30 min
4. [Playbook Format](playbook_format.md) - Economic data, heuristics - 30 min
5. [Partitioning Examples](examples_library/partitioning_examples.md) - Strategy selection - 30 min
6. Design a multi-strategy workflow - 30 min
7. Implement strategies as playbooks - 25 min

**Outcome**: You can design strategic workflows with multiple options.

### Path 4: Complete Mastery (5 hours)

Read everything in order, execute all examples:

1. **Foundation** (1 hr):
   - README
   - Chapter 1
   - Playbook Format

2. **Playbooks** (2 hrs):
   - Execute factorial playbook
   - Execute GNU Prolog playbook
   - Create variation of each

3. **Examples** (1.5 hrs):
   - Read all example libraries
   - Try examples from each library
   - Extend an example

4. **Creation** (30 min):
   - Design new playbook
   - Write using template
   - Test thoroughly

**Outcome**: You're a playbook and workflow expert.

## By Difficulty Level

### Beginner

Start with these if you're new to UnifyWeaver:

- ⭐ [Factorial Compilation Playbook](playbooks/factorial_compilation.md)
- [Compilation Examples](examples_library/compilation_examples.md) - Factorial section
- [Recursion Examples](examples_library/recursion_examples.md)
- [Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md)

### Intermediate

Once you're comfortable with basics:

- ⭐ [GNU Prolog Compilation Playbook](playbooks/gnu_prolog_compilation.md)
- [Partitioning Examples](examples_library/partitioning_examples.md) - Basic partitioning
- [Data Source Principles](examples_library/data_source_playbooks.md)
- [Testing Examples](examples_library/testing_examples.md)

### Advanced

For experienced users:

- [Partitioning Examples](examples_library/partitioning_examples.md) - Complete ETL
- [Playbook Format](playbook_format.md) - Creating templates
- Design multi-strategy workflows
- Extend example libraries

## By Use Case

### I want to compile Prolog to Bash
→ [Factorial Compilation Playbook](playbooks/factorial_compilation.md)
→ [Compilation Examples](examples_library/compilation_examples.md)

### I want to create standalone executables
→ [GNU Prolog Compilation Playbook](playbooks/gnu_prolog_compilation.md)
→ [Prolog Target Book](../book-prolog-target/README.md)

### I want to process large datasets in parallel
→ [Partitioning Examples](examples_library/partitioning_examples.md)
→ ETL Pipeline example

### I want to integrate external data
→ [Data Source Principles](examples_library/data_source_playbooks.md)
→ CSV compilation examples

### I want to create my own playbook
→ [Playbook Format Specification](playbook_format.md)
→ [Playbook Template](playbook_format.md#complete-playbook-template)

### I want to understand AI agent workflows
→ [Chapter 1: Thinking in Workflows](ch1_introduction/ch1_introduction.md)
→ [README: The Paradigm Shift](README.md#the-paradigm-shift)

## Key Concepts Reference

### Workflows
**Definition**: High-level strategic guides for AI agents
**Location**: [Chapter 1](ch1_introduction/ch1_introduction.md#workflows-vs-playbooks-a-clarification)
**Contains**: Goals, strategies, economic data, heuristics

### Playbooks
**Definition**: Specific execution plans for concrete tasks
**Location**: [Playbook Format](playbook_format.md#what-is-a-playbook)
**Contains**: Steps, commands, verification, troubleshooting

### Economic Data
**Definition**: Cost, speed, quality metrics for decision-making
**Location**: [Playbook Format](playbook_format.md#4-strategies-when-multiple-approaches-exist)
**Usage**: AI agents use this to choose strategies

### Example Libraries
**Definition**: Reusable code patterns and reference implementations
**Location**: [examples_library/](examples_library/)
**Purpose**: Promote reusability, provide concrete instances

### Callouts
**Definition**: Structured information blocks in playbooks
**Location**: [Playbook Format: Callouts](playbook_format.md#callout-types-reference)
**Types**: `[!output]`, `[!example-record]`, `[!note]`, `[!warning]`, `[!tip]`

## Related Books

This book is part of the UnifyWeaver Educational Series:

1. **[Book 1: Foundations](../book-01-foundations/)** - Architecture and Prolog basics
2. **[Book 2: Bash Target](../book-02-bash-target/)** - Bash compilation fundamentals
3. **[Book 3: C# Target](../book-03-csharp-target/)** - .NET integration
4. **Book 4: Workflows and Playbooks** (This Book) - Strategic AI agent guidance
5. **[Book 11: Prolog Target](../book-11-prolog-target/)** - Prolog-to-Prolog transpilation

**Cross-References**:
- GNU Prolog playbook → Prolog Target Book Chapter 15 (Case Study)
- Partitioning examples → Core Bash Book (parallel patterns)
- Compilation examples → Multiple books (target-specific)

## Statistics

**Content Status** (as of 2025-11-17):

| Category | Count | Status |
|----------|-------|--------|
| Chapters | 1 | ✅ Complete |
| Playbooks | 2 | ✅ Published |
| Example Libraries | 7 | ✅ Complete |
| Code Examples | 20+ | ✅ Working |
| Callout Types | 5 | ✅ Documented |

**Upcoming**:
- 6 more playbooks (planned)
- Chapter 2-5 (workflow design, economics, testing)
- Additional example libraries (C#, multi-target)

## Getting Help

- **Questions**: See [README: Getting Help](README.md#getting-help)
- **Format Questions**: [Playbook Format Specification](playbook_format.md)
- **Examples**: [Examples Library](examples_library/)
- **Issues**: File on GitHub
- **Contribute**: See [README: Contributing](README.md#contributing)

## Maintenance

This index is updated when:
- New playbooks added
- New example libraries created
- New chapters published
- Major content revisions

**Last Updated**: 2025-11-17
**Maintainer**: UnifyWeaver Education Team

---

**Happy Learning!** 🚀

Start with the [Factorial Playbook](playbooks/factorial_compilation.md) and work your way through the examples. Remember: workflows are about **strategic thinking**, playbooks are about **reliable execution**, and examples provide **concrete patterns** to build on.

---

## Navigation

[📖 Book 4: Workflows](./) | [Next: Chapter 1: Thinking in Workflows →](01_introduction)
