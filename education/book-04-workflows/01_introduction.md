<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 1: Thinking in Workflows

In traditional programming, we think in **scripts** - fixed sequences of commands. UnifyWeaver lets you create efficient scripts from declarative logic, but the true power of AI-driven development lies in **thinking in workflows**.

## From Scripts to Strategies

A Workflow is a **Strategy Guide** for an intelligent AI agent. Instead of a fixed plan, it provides:

- A clear **goal**
- A **menu of strategies** to achieve that goal
- A **library of tools** required to execute those strategies
- **Economic data** (cost, speed, quality) for each strategy
- **Heuristic guidance** for choosing the best approach

The AI agent becomes an **economic strategist**, not just a script executor.

## Workflows vs Playbooks

Within UnifyWeaver, these terms have distinct meanings:

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

**Example**: "When compiling Prolog to Bash, analyze the recursion pattern first, then choose between linear recursion, transitive closure, or general recursive compilation based on pattern detection."

### Playbooks

**Definition**: Specific, detailed project plans for particular tasks.

**Contains**:
- Concrete steps
- Specific tools to use
- Exact commands
- Expected outputs
- Verification procedures

**Characteristics**:
- Task-specific
- Detailed and actionable
- Operates within workflow framework

**Example**: "To compile factorial: 1) Create /tmp/factorial.pl, 2) Run compiler_driver with factorial/2, 3) Verify output script exists, 4) Test with inputs 0, 5, 10."

## The Strategic Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                      WORKFLOW                           │
│  "How to compile Prolog predicates effectively"         │
│                                                         │
│  Strategies:                                            │
│  - Quick prototyping (cost: low, speed: fast)           │
│  - Production deployment (cost: medium, quality: high)  │
│  - Distributed processing (cost: high, scale: massive)  │
├─────────────────────────────────────────────────────────┤
│                      PLAYBOOKS                          │
│                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Factorial   │ │ Ancestor    │ │ CSV Data    │       │
│  │ Compilation │ │ Compilation │ │ Pipeline    │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
├─────────────────────────────────────────────────────────┤
│                   EXAMPLE LIBRARIES                     │
│                                                         │
│  - compilation_examples.md                              │
│  - recursion_examples.md                                │
│  - testing_examples.md                                  │
└─────────────────────────────────────────────────────────┘
```

## Example Libraries

Both Workflows and Playbooks reference **example libraries** - concrete instances of patterns, tool usage, and solutions.

### Benefits

- **Reusability**: Examples used across many playbooks
- **Maintainability**: Update once, fix everywhere
- **Discoverability**: Browse patterns by category
- **Extraction**: Tools can dynamically retrieve examples

### Structure

```
examples_library/
├── compilation_examples.md    # Factorial, ancestor compilation
├── recursion_examples.md      # Recursive patterns
├── testing_examples.md        # Test runner patterns
├── tool_usage_examples.md     # UnifyWeaver tool usage
├── log_examples.md            # Logging patterns
└── data_source_playbooks.md   # Data source integration
```

### Example Record Format

Examples use the `[!example-record]` callout:

```markdown
> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion
> child_examples: [unifyweaver.testing.factorial_runner]

**Prolog Source:**
\```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
\```
```

## Common Workflow Patterns

### Pattern 1: Compile-and-Test

1. Define Prolog predicate
2. Save to file
3. Compile with compiler_driver
4. Generate test runner
5. Execute tests
6. Verify results

```prolog
% Workflow: compile_and_test
workflow_steps([
    create_prolog_source,
    compile_to_target,
    generate_test_runner,
    run_tests,
    verify_results
]).
```

### Pattern 2: Data Pipeline

1. Declare data source (`:- source(...)`)
2. Define domain predicate
3. Compile with dynamic source
4. Generate target scripts
5. Verify pipeline

```prolog
% Example: CSV data source
:- source(csv, users, [csv_file('users.csv'), has_header(true)]).

active_users(Name, Email) :-
    users(Name, Email, Status),
    Status = "active".
```

### Pattern 3: Multi-Target Generation

1. Define predicate once
2. Compile to Bash
3. Compile to Python
4. Compile to Go
5. Test all targets
6. Compare results

```prolog
% Same predicate, multiple targets
compile_multi_target(Pred, Targets, Results) :-
    maplist(compile_to_target(Pred), Targets, Results).
```

### Pattern 4: Cross-Target Pipeline

1. Design pipeline stages
2. Choose optimal target per stage
3. Generate glue code
4. Assemble pipeline
5. Execute and verify

```prolog
% Example: AWK → Python → Go pipeline
pipeline_stages([
    step(parse, awk),
    step(analyze, python),
    step(aggregate, go)
]).
```

## Economic Decision Making

Every strategy has trade-offs:

| Strategy | Cost | Speed | Quality | When to Use |
|----------|------|-------|---------|-------------|
| Quick Prototype | Low | Fast | Basic | Development, testing |
| Optimized Build | Medium | Slow | High | Production deployment |
| Parallel Execution | High | Very Fast | High | Large data processing |
| Distributed Pipeline | Very High | Scales | Enterprise | Cloud deployment |

### Heuristics

Rules of thumb for strategy selection:

- **For lists <100 items**: Use simple iteration
- **For recursive depth >10**: Consider memoization
- **For data >1GB**: Use parallel execution
- **For production**: Prefer compiled binaries

## Quick Start

### 5-Minute Introduction

1. Read this chapter (5 min)
2. Skim [Chapter 2: Playbook Format](02_playbook_format.md) (2 min)
3. Look at [Factorial Example](examples_library/compilation_examples.md#factorial-compilation) (3 min)

### 30-Minute Tutorial

1. Read this chapter thoroughly (10 min)
2. Study playbook format spec (10 min)
3. Review factorial playbook (5 min)
4. Create a simple playbook for a new task (5 min)

## Design Principles

### 1. Separation of Concerns

- **Workflows**: Strategic thinking
- **Playbooks**: Concrete execution
- **Examples**: Reusable patterns

### 2. Economic Awareness

Every strategy should document:
- **Cost**: Resource usage
- **Speed**: Execution time
- **Quality**: Correctness, robustness

### 3. Composability

Design playbooks and examples to work together:
- Reference examples from libraries
- Chain playbooks for complex workflows
- Build on existing patterns

### 4. Verifiability

Every playbook should include:
- Expected outputs
- Verification steps
- Success criteria
- Error handling

## Summary

Key concepts from this chapter:

- **Workflows** = Strategic guides for AI agents
- **Playbooks** = Specific execution plans
- **Example Libraries** = Reusable code patterns
- **Economic Data** = Cost/speed/quality trade-offs
- **Heuristics** = Rules for strategy selection

The next chapter covers the detailed playbook format for creating actionable execution guides.

---

## Navigation

**←** [Previous: Workflows and Playbooks - Complete Index](00_INDEX) | [📖 Book 4: Workflows](./) | [Next: Chapter 2: Playbook Format →](02_playbook_format)
