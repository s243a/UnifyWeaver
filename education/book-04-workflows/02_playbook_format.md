<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 2: Playbook Format

This chapter details the standardized format for creating playbooks - detailed execution guides for AI agents within UnifyWeaver workflows.

## What is a Playbook?

A **playbook** is a specific, detailed plan for accomplishing a particular task. Unlike workflows (high-level strategies), playbooks provide:

- **Concrete steps** to execute
- **Specific tools** to use
- **Exact commands** to run
- **Expected outputs** at each stage
- **Verification procedures**
- **Error handling** for common failures

Think of a playbook as a **detailed recipe** that an AI agent can follow reliably.

## Core Structure

A well-designed playbook has these sections:

### 1. Frontmatter (YAML)

```yaml
---
id: playbook-factorial-compile-v1
name: "Factorial Compilation Playbook"
version: 1.0
workflow_id: compile-and-test
pattern: linear_recursion
target: bash
difficulty: beginner
estimated_time: 5_minutes
prerequisites: [swipl, unifyweaver_installed]
---
```

**Fields**:
| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier |
| `name` | Yes | Human-readable title |
| `version` | Yes | Semantic version |
| `workflow_id` | No | Parent workflow |
| `pattern` | No | Pattern classification |
| `target` | No | Output format (bash, python, etc.) |
| `difficulty` | No | beginner/intermediate/advanced |
| `estimated_time` | No | Expected duration |
| `prerequisites` | No | Required tools/setup |

### 2. Goal Statement

Clear, concise statement (1-2 sentences) of what the playbook achieves:

```markdown
## Goal

Compile a factorial predicate from Prolog to optimized Bash,
generate a test runner, and verify results for factorial(0) through factorial(10).
```

**Guidelines**:
- Be specific and measurable
- State the end result clearly
- Include success criteria if not obvious

### 3. Context

Background information, assumptions, and prerequisites:

```markdown
## Context

This playbook demonstrates linear recursion compilation - one of
the most common patterns in UnifyWeaver.

**Assumptions**:
- UnifyWeaver is installed at `$UNIFYWEAVER_HOME`
- SWI-Prolog is available in PATH
- You have write access to `/tmp/`

**Background**:
Linear recursion is automatically optimized by UnifyWeaver's compiler
to use tail recursion and memoization where appropriate.
```

### 4. Strategies (When Multiple Approaches Exist)

```markdown
## Strategies

### Strategy A: Direct Compilation
- **Cost**: Low (single compilation step)
- **Speed**: Fast (<5 seconds)
- **Quality**: Standard
- **When to use**: Quick prototyping

### Strategy B: Optimized with Memoization
- **Cost**: Medium (requires analysis step)
- **Speed**: Slower to compile, faster at runtime
- **Quality**: High (optimized for repeated calls)
- **When to use**: Production deployment

**Heuristic**: Use Strategy A for development, Strategy B for deployment.
```

### 5. Execution Steps

Step-by-step instructions with exact commands:

```markdown
## Execution Steps

### Step 1: Create Prolog Source

Create the factorial predicate in `/tmp/factorial.pl`:

\```bash
cat > /tmp/factorial.pl <<'EOF'
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
EOF
\```

**Verification**:
\```bash
test -f /tmp/factorial.pl && wc -l /tmp/factorial.pl
# Expected: 5 /tmp/factorial.pl
\```
```

**Guidelines**:
- Number steps sequentially
- Provide exact, copy-pasteable commands
- Include verification after critical steps
- Explain what each step does

### 6. Output Specifications

Use the `[!output]` callout for expected outputs:

```markdown
## Expected Outputs

> [!output]
> language: bash
> purpose: Compiled factorial function
> format: executable
> location: education/output/advanced/factorial.sh
>
> \```bash
> #!/bin/bash
> factorial() {
>     local N=$1
>     if [ "$N" -eq 0 ]; then
>         echo 1
>     else
>         local N1=$((N - 1))
>         local F1=$(factorial $N1)
>         echo $((N * F1))
>     fi
> }
> \```
```

**Constraint**: Keep `[!output]` content <10% of total playbook.

### 7. Verification and Testing

```markdown
## Verification

### Test the Compiled Script

\```bash
echo "5" | bash education/output/advanced/factorial.sh
\```

**Expected output**: `120`

### Success Criteria

All of the following must be true:
- `factorial.sh` exists and is executable
- `factorial 0` returns `1`
- `factorial 5` returns `120`
- `factorial 10` returns `3628800`
- Test runner exits with code 0
```

### 8. Error Handling

```markdown
## Troubleshooting

### Error: "No such file: compiler_driver.pl"

**Cause**: `file_search_path` not set correctly

**Solution**:
\```bash
export UNIFYWEAVER_HOME=/path/to/UnifyWeaver
cd $UNIFYWEAVER_HOME
\```

### Error: "Predicate factorial/2 not found"

**Cause**: Prolog file not loaded before compilation

**Solution**: Ensure the load statement appears before `compile(...)`.
```

## Callout Types

### `[!output]` - Expected Output

Specifies what the agent should generate:

```markdown
> [!output]
> language: bash
> purpose: Compiled factorial function
> format: executable
> location: path/to/output.sh
>
> \```bash
> # Template content here
> \```
```

**Mandatory fields**:
- `language`: Programming language or format
- `purpose`: What this output represents

**Optional fields**:
- `format`: Type (executable, data, config)
- `location`: Where to save output
- `expected_size`: small/medium/large

### `[!example-record]` - Library Reference

Tags reusable examples in library files:

```markdown
> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion
> child_examples: [unifyweaver.testing.factorial_runner]
>
> Example content here...
```

**Mandatory fields**:
- `id`: Unique identifier (YYYYMMDD-name format)
- `name`: Namespaced name

**Optional fields**:
- `pattern`: Classification tag
- `parent_example`: Parent in hierarchy
- `child_examples`: Related examples
- `difficulty`: beginner/intermediate/advanced
- `tags`: Additional classifications

### `[!note]` - Additional Information

```markdown
> [!note]
> **Performance Tip**
>
> For factorials >20, consider using memoization.
```

### `[!warning]` - Cautionary Information

```markdown
> [!warning]
> The compilation step modifies files. Ensure you have backups.
```

### `[!tip]` - Best Practices

```markdown
> [!tip]
> Run tests after every compilation to catch regressions.
```

## Referencing Example Libraries

Playbooks should reference examples rather than embedding large code blocks:

```markdown
For a complete factorial example, see
[Factorial Compilation](examples_library/compilation_examples.md#factorial-compilation).
```

**Best practices**:
- Link to specific sections with anchors
- Provide context for relevance
- Use relative paths from playbook location

## Prolog Generation Guidance

When a playbook requires Prolog code generation:

### Predicate Specifications

```markdown
### Predicate: `factorial/2`

**Signature**: `factorial(+N:integer, -F:integer)`

**Semantics**:
- Computes F = N! (factorial of N)
- N must be non-negative integer

**Modes**:
- `factorial(+, -)`: Deterministic
- Input N must be ground
- Output F will be bound to result

**Examples**:
\```prolog
?- factorial(5, F).
F = 120.
\```
```

### Compilation Instructions

```markdown
### Compilation Command

Navigate to `$UNIFYWEAVER_HOME` and run:

\```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [
        output_dir('education/output/advanced'),
        target(bash)
    ], Scripts),
    format('Generated: ~w~n', [Scripts]),
    halt"
\```
```

## Complete Template

```markdown
---
id: playbook-<task>-v1
name: "<Task Name> Playbook"
version: 1.0
pattern: <pattern_type>
target: <output_format>
difficulty: <level>
estimated_time: <X>_minutes
prerequisites: [<tool1>, <tool2>]
---

# <Task Name> Playbook

## Goal

<Clear statement of what this playbook achieves>

## Context

<Background information>

**Assumptions**:
- <Assumption 1>
- <Assumption 2>

## Strategies

### Strategy A: <Approach>
- **Cost**: <Low|Medium|High>
- **Speed**: <Fast|Medium|Slow>
- **Quality**: <Standard|High>
- **When to use**: <Guidance>

**Heuristic**: <Decision guidance>

## Execution Steps

### Step 1: <Action Name>

<Description>

\```bash
<exact command>
\```

**Verification**:
\```bash
<verification command>
# Expected: <output>
\```

## Expected Outputs

> [!output]
> language: <language>
> purpose: <description>
> format: <format_type>
>
> \```<language>
> <template>
> \```

## Verification

### Success Criteria

- <Criterion 1>
- <Criterion 2>

## Troubleshooting

### Error: "<error message>"

**Cause**: <Why this happens>

**Solution**:
\```bash
<how to fix>
\```

## References

- [Example Library Link](examples_library/file.md#section)
```

## Best Practices

### For Playbook Authors

1. **Be explicit**: Exact commands, not vague descriptions
2. **Verify everything**: Check after each critical step
3. **Reference examples**: Link to libraries, don't embed
4. **Show outputs**: Include expected results
5. **Handle errors**: Document common failures
6. **Keep focused**: One playbook = one task
7. **Test thoroughly**: Execute before publishing

### For AI Agents

1. **Follow sequentially**: Execute steps in order
2. **Verify between steps**: Run verification commands
3. **Check criteria**: Confirm all success criteria
4. **Handle errors**: Use troubleshooting section
5. **Report progress**: Log each step

## Quality Checklist

Before publishing a playbook:

- [ ] Frontmatter complete
- [ ] Goal clearly stated (1-2 sentences)
- [ ] Context provides background
- [ ] Strategies include economic data
- [ ] Commands are exact and copy-pasteable
- [ ] Verification after critical steps
- [ ] Expected outputs defined
- [ ] Success criteria specific
- [ ] Common errors documented
- [ ] Example library references included
- [ ] `[!output]` content <10% of total
- [ ] Playbook tested end-to-end

## Summary

A well-structured playbook enables AI agents to:

- **Understand the goal** clearly
- **Choose strategies** based on trade-offs
- **Execute steps** precisely
- **Verify success** at each stage
- **Handle errors** gracefully

The next chapter covers pipeline orchestration for multi-stage workflows.

---

## Navigation

**←** [Previous: Chapter 1: Thinking in Workflows](01_introduction) | [📖 Book 4: Workflows](./) | [Next: Chapter 3: Pipeline Orchestration →](03_pipeline_orchestration)
