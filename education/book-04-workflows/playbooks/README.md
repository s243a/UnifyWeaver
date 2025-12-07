# UnifyWeaver Playbooks

This directory contains concrete,executable playbooks for common UnifyWeaver tasks. Each playbook provides step-by-step instructions that AI agents (or humans!) can follow to accomplish specific goals.

## What are Playbooks?

Playbooks are **detailed execution guides** that provide:
- Concrete, sequential steps
- Exact commands to run
- Expected outputs at each stage
- Verification procedures
- Troubleshooting guidance
- Success criteria

Think of them as **recipes** that ensure consistent, repeatable results.

## Available Playbooks

### Beginner Level

#### [Factorial Compilation](factorial_compilation.md)
**Goal**: Compile a factorial predicate from Prolog to Bash
**Time**: ~5 minutes
**Prerequisites**: SWI-Prolog, UnifyWeaver
**Pattern**: Linear recursion

Perfect introduction to UnifyWeaver's compilation pipeline. Demonstrates:
- Creating Prolog predicates
- Running compiler_driver
- Generating test runners
- Verifying compiled outputs

**When to use**: Learning the basics, testing your UnifyWeaver installation

---

### Intermediate Level

#### [GNU Prolog Native Binary Compilation](gnu_prolog_compilation.md)
**Goal**: Generate and compile a native GNU Prolog executable
**Time**: ~10 minutes
**Prerequisites**: SWI-Prolog, UnifyWeaver, GNU Prolog, gplc
**Pattern**: Prolog-to-Prolog transpilation with native compilation

Demonstrates the Prolog target with compilation to standalone binaries. Shows:
- Dialect-specific code generation
- Correct initialization directives
- Binary vs interpreted mode
- Cross-dialect development workflow

**When to use**: Deploying standalone executables, learning Prolog target

**Related**: [Book: Prolog Target](../../book-prolog-target/README.md)

---

## Coming Soon

### Planned Playbooks

- **CSV Data Source Integration** - Working with external CSV files
- **Parallel Execution with Partitioning** - Using partitioners for batch processing
- **C# Target Compilation** - Generating .NET code
- **Multi-Target Deployment** - Bash + C# + Prolog in one workflow
- **Firewall Policy Configuration** - Security policies for code generation
- **Test-Driven Compilation** - Writing tests before generating code

## How to Use This Directory

### For Learners

1. **Start with factorial_compilation.md** - Foundational concepts
2. **Progress to gnu_prolog_compilation.md** - Advanced features
3. **Try modifying the examples** - Change predicates, targets, options
4. **Create your own playbooks** - Follow the [format specification](../playbook_format.md)

### For AI Agents

1. **Read the playbook sequentially** - Follow steps in order
2. **Execute verification commands** - Confirm each step succeeded
3. **Check success criteria** - Ensure all criteria met
4. **Use troubleshooting section** - Handle errors gracefully
5. **Report progress** - Log what you're doing

### For Instructors

1. **Use as teaching materials** - Step-by-step learning
2. **Assign as exercises** - Students execute playbooks
3. **Create variations** - Modify for different learning objectives
4. **Build on examples** - Reference in your own materials

## Playbook Quality Standards

All playbooks in this directory should:

- ✅ Follow the [playbook format specification](../playbook_format.md)
- ✅ Include complete frontmatter metadata
- ✅ Provide exact, copy-pasteable commands
- ✅ Include verification after critical steps
- ✅ Define clear success criteria
- ✅ Include troubleshooting for common errors
- ✅ Reference example libraries appropriately
- ✅ Be tested end-to-end before publishing
- ✅ Use environment variables ($UNIFYWEAVER_HOME)
- ✅ Document prerequisites clearly

## Creating New Playbooks

Want to contribute a playbook?

1. **Use the template** from [playbook_format.md](../playbook_format.md#complete-playbook-template)
2. **Fill in all sections**:
   - Frontmatter with metadata
   - Clear goal statement
   - Context and assumptions
   - Step-by-step execution
   - Expected outputs
   - Verification procedures
   - Troubleshooting guide
3. **Test thoroughly** - Execute every command
4. **Get feedback** - Have someone else try it
5. **Submit PR** - Add to this directory

## Playbook Naming Convention

Files should be named: `<task>_<subtask>.md`

Examples:
- `factorial_compilation.md` - Compile factorial to Bash
- `gnu_prolog_compilation.md` - Compile to GNU Prolog binary
- `csv_data_source.md` - Integrate CSV data sources
- `parallel_partitioning.md` - Parallel execution with partitioners

IDs in frontmatter: `playbook-<task>-<subtask>-v<version>`

Example: `playbook-factorial-compilation-v1`

## Difficulty Levels

**Beginner**:
- Single target (Bash or Prolog)
- Simple predicates (factorial, fibonacci)
- No external dependencies
- ~5-10 minutes

**Intermediate**:
- Multiple targets or advanced features
- Cross-dialect work
- Data sources or partitioning
- ~10-20 minutes

**Advanced**:
- Complex workflows
- Multiple strategies
- Production deployment scenarios
- Custom extensions
- ~20-45 minutes

## Success Metrics

A good playbook should have:
- **High success rate**: >90% of users complete successfully
- **Clear instructions**: No ambiguity in steps
- **Complete verification**: All outputs verified
- **Good troubleshooting**: Common errors covered
- **Appropriate difficulty**: Matches stated level

## Maintenance

Playbooks should be updated when:
- UnifyWeaver version changes significantly
- Commands or APIs change
- Prerequisites change
- Errors are discovered
- Better approaches are found

Version history should be documented at the end of each playbook.

## Related Resources

- **[Playbook Format](../playbook_format.md)**: Complete specification
- **[Example Libraries](../examples_library/)**: Reusable code patterns
- **[Chapter 1: Thinking in Workflows](../ch1_introduction/)**: Conceptual foundation
- **[Book: Prolog Target](../../book-prolog-target/)**: Deep dive into Prolog compilation

---

**Directory Status**: Active
**Last Updated**: 2025-11-17
**Playbook Count**: 2 (2 published, 6 planned)

Let's build comprehensive, reliable playbooks that make UnifyWeaver accessible to everyone! 🚀
