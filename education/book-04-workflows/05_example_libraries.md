<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 5: Example Libraries

This chapter covers how to create, organize, and use example libraries - reusable code patterns that power workflows and playbooks.

## What are Example Libraries?

Example libraries are collections of **reusable code patterns** organized by topic:

- **Compilation examples**: Factorial, ancestor, and other compilation patterns
- **Recursion examples**: Common recursive Prolog patterns
- **Testing examples**: Test runner patterns and verification
- **Tool usage examples**: How to use UnifyWeaver tools
- **Data source examples**: Integration with CSV, JSON, XML

## Library Structure

```
examples_library/
├── compilation_examples.md    # Prolog → Target compilation
├── recursion_examples.md      # Recursive patterns
├── testing_examples.md        # Test generation and verification
├── tool_usage_examples.md     # UnifyWeaver tool usage
├── log_examples.md            # Logging and debugging
├── parallel_examples.md       # Parallel execution patterns
├── partitioning_examples.md   # Data partitioning
└── data_source_playbooks.md   # External data integration
```

## Example Record Format

Each example uses the `[!example-record]` callout:

```markdown
> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion
> difficulty: beginner
> child_examples: [unifyweaver.testing.factorial_runner]
> tags: [arithmetic, tail_recursion_candidate]

**Description**: Basic factorial implementation...

**Prolog Source**:
\```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N-1, factorial(N1, F1), F is N * F1.
\```

**Compilation Command**:
\```bash
swipl -q -g "..., compile(factorial/2, [], _), halt"
\```
```

### Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique ID (YYYYMMDD-name format) |
| `name` | Yes | Namespaced name (e.g., `unifyweaver.category.name`) |
| `pattern` | No | Pattern classification |
| `difficulty` | No | beginner/intermediate/advanced |
| `parent_example` | No | Parent example ID |
| `child_examples` | No | List of dependent examples |
| `related` | No | Cross-references to similar examples |
| `tags` | No | Additional classification tags |

## Core Example Libraries

### Compilation Examples

Located at `examples_library/compilation_examples.md`.

**Contains**:
- Factorial compilation (linear recursion)
- Ancestor compilation (transitive closure)
- Complete workflow examples

**Example: Factorial Compilation**

```markdown
> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion

**Prolog Source:**
\```prolog
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
\```

**Compilation:**
\```bash
cd $UNIFYWEAVER_HOME
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    format('Generated: ~w~n', [Scripts]),
    halt"
\```
```

### Recursion Examples

Located at `examples_library/recursion_examples.md`.

**Patterns Covered**:

| Pattern | Example | Optimization |
|---------|---------|--------------|
| Linear | Factorial | Memoization |
| Tail | Sum accumulator | While loop |
| Binary | Fibonacci | Memoization |
| Transitive | Ancestor | BFS/Fixpoint |
| Mutual | Even/Odd | Shared dispatcher |

**Example: Ancestor (Transitive Closure)**

```markdown
> [!example-record]
> id: 20251108-ancestor-pattern
> name: unifyweaver.recursion.ancestor
> pattern: transitive_closure

**Prolog Source:**
\```prolog
% Base facts
parent(alice, bob).
parent(bob, charlie).

% Transitive closure
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
\```

**What UnifyWeaver Does:**
- Recognizes transitive closure pattern
- Generates breadth-first search
- Creates hash-based lookups
```

### Testing Examples

Located at `examples_library/testing_examples.md`.

**Contains**:
- Test runner generation
- Explicit vs inferred tests
- Verification patterns

**Example: Test Runner Generation**

```markdown
> [!example-record]
> id: 20251108-factorial-test
> name: unifyweaver.testing.factorial_runner
> parent_example: unifyweaver.compilation.factorial

**Generate Test Runner:**
\```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"
\```

**Expected Test Output:**
\```
Testing factorial.sh...
Test 1: factorial 0 → 1 PASS
Test 2: factorial 5 → 120 PASS
\```
```

### Parallel Execution Examples

Located at `examples_library/parallel_examples.md`.

**Patterns Covered**:
- MapReduce pattern
- Worker pools
- Fork-join parallelism

**Example: Parallel Processing**

```markdown
> [!example-record]
> id: 20251108-parallel-mapreduce
> name: unifyweaver.parallel.mapreduce
> pattern: mapreduce

**Prolog Definition:**
\```prolog
% Parallel map with bash_fork backend
parallel_process(InputDir, OutputDir) :-
    partition_input(InputDir, 8, Partitions),
    parallel_map(process_partition, Partitions, Results),
    merge_results(Results, OutputDir).
\```

**Generated Bash:**
\```bash
#!/bin/bash
# Process partitions in parallel
for partition in partitions/*; do
    process_partition "$partition" &
done
wait
merge_results output/
\```
```

## Creating New Examples

### Step 1: Choose the Right Library

| Example Type | Library File |
|--------------|--------------|
| Compilation pattern | `compilation_examples.md` |
| Recursive algorithm | `recursion_examples.md` |
| Test generation | `testing_examples.md` |
| Tool usage | `tool_usage_examples.md` |
| Data integration | `data_source_playbooks.md` |
| Parallel processing | `parallel_examples.md` |

### Step 2: Write the Example Record

```markdown
### <Example Title>

> [!example-record]
> id: YYYYMMDD-descriptive-name
> name: unifyweaver.category.specific_name
> pattern: pattern_classification
> difficulty: beginner|intermediate|advanced
> related: [other.example.ids]
> tags: [relevant, tags]

**Description**: What this example demonstrates.

**Prolog Source:**
\```prolog
% Your Prolog code here
\```

**Usage:**
\```bash
# Commands to use this example
\```

**Expected Output:**
\```
# What the user should see
\```

**Notes:**
- Important considerations
- Edge cases
- Performance characteristics
```

### Step 3: Add Cross-References

Link related examples:

```markdown
**Related Examples:**
- [Parent Example](#parent-example) - More general pattern
- [Child Example](#child-example) - Specific application
- [Testing](testing_examples.md#test-section) - How to test this
```

## Referencing Examples from Playbooks

### Inline References

```markdown
For a complete factorial example, see
[Factorial Compilation](examples_library/compilation_examples.md#factorial-compilation).
```

### Importing Patterns

```markdown
## Prolog Source

Use the factorial pattern from the example library:

\```prolog
% From: unifyweaver.compilation.factorial
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N-1, factorial(N1, F1), F is N * F1.
\```
```

### Dynamic Extraction

The example extraction tool can retrieve examples programmatically:

```bash
# Extract example by ID
./extract_example.pl --id 20251108-factorial-compile

# Extract by name
./extract_example.pl --name unifyweaver.compilation.factorial

# Extract all examples with a pattern
./extract_example.pl --pattern linear_recursion
```

## Naming Conventions

### Example IDs

Format: `YYYYMMDD-descriptive-name`

```
20251108-factorial-compile
20251108-ancestor-transitive
20251108-parallel-mapreduce
```

### Namespaced Names

Format: `unifyweaver.category.specific_name`

```
unifyweaver.compilation.factorial
unifyweaver.recursion.ancestor
unifyweaver.testing.factorial_runner
unifyweaver.parallel.mapreduce
```

### Pattern Classifications

Standard pattern tags:

| Pattern | Description |
|---------|-------------|
| `linear_recursion` | Single recursive call |
| `tail_recursion` | Recursive call in tail position |
| `binary_recursion` | Two recursive calls |
| `transitive_closure` | Reachability queries |
| `mutual_recursion` | Predicates calling each other |
| `mapreduce` | Parallel map-reduce |
| `compile_and_test` | Full compilation workflow |

## Example Library Index

Each library should have an index section:

```markdown
## Index

### By Pattern

- **linear_recursion**
  - [Factorial](#factorial-compilation)
  - [Sum](#sum-example)
- **transitive_closure**
  - [Ancestor](#ancestor-compilation)
  - [Reachability](#reachability-example)

### By Difficulty

- **beginner**
  - [Factorial](#factorial-compilation)
- **intermediate**
  - [Ancestor](#ancestor-compilation)
- **advanced**
  - [Parallel Processing](#parallel-example)

### By Tag

- **arithmetic**: Factorial, Sum, Power
- **graph**: Ancestor, Reachability, Path
- **parallel**: MapReduce, Worker Pool
```

## Maintenance Guidelines

### Adding Examples

1. Choose appropriate library file
2. Assign unique ID (check for collisions)
3. Use namespaced name
4. Add pattern classification
5. Include complete, working code
6. Add cross-references
7. Update library index

### Updating Examples

1. Keep the same ID
2. Update version in metadata if significant
3. Update all cross-references
4. Test the updated code
5. Update related playbooks if needed

### Deprecating Examples

```markdown
> [!example-record]
> id: 20251108-old-example
> name: unifyweaver.category.old_example
> deprecated: true
> replacement: unifyweaver.category.new_example

**DEPRECATED**: This example is deprecated. Use
[New Example](#new-example) instead.

**Reason**: Better approach available in newer version.
```

## Best Practices

### For Example Authors

1. **Keep examples focused**: One concept per example
2. **Include complete code**: Examples should work as-is
3. **Document expected output**: Show what success looks like
4. **Add error cases**: Show what can go wrong
5. **Cross-reference liberally**: Connect related examples
6. **Test before publishing**: Verify examples work

### For Playbook Authors

1. **Reference, don't copy**: Link to examples
2. **Provide context**: Explain why an example is relevant
3. **Adapt as needed**: Examples are starting points
4. **Update references**: Keep links current

### For AI Agents

1. **Search by pattern**: Find examples matching the task
2. **Follow cross-references**: Related examples may help
3. **Adapt examples**: Modify for specific needs
4. **Verify before use**: Test examples in current context

## Summary

Example libraries provide:

- **Reusability**: Write once, use everywhere
- **Discoverability**: Browse by pattern, difficulty, tags
- **Maintainability**: Update once, fix everywhere
- **Composability**: Combine examples for complex tasks

Key practices:
1. Organize by topic in dedicated files
2. Use `[!example-record]` callout with metadata
3. Follow naming conventions (IDs, namespaces)
4. Add cross-references between related examples
5. Keep examples focused and complete
6. Reference examples from playbooks instead of copying

This concludes Book 4 on Workflows and Playbooks. Continue to [Book 5: Python Target](../book-05-python-target/) or [Book 6: Go Target](../book-06-go-target/) to explore specific compilation targets.

---

## Navigation

**←** [Previous: Chapter 4: Economic Decision Making](04_economic_decisions) | [📖 Book 4: Workflows](./) | [Next: Book 5: Python Target →](../book-05-python-target/)
