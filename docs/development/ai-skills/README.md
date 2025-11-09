# AI Skills for UnifyWeaver Development

This directory contains skill definitions and workflow guides designed to help AI assistants work more effectively with the UnifyWeaver codebase.

## LLM Workflow System: Current State & Vision

### Current State (v0.1 - Automated Testing via Playbooks)

The LLM workflow infrastructure is **functional for automated testing**:

✅ **What Works Now:**
- **Playbooks as Test Specifications**: Playbooks (e.g., `examples/prolog_generation_playbook.md`, `examples/mutual_recursion_playbook.md`) serve as executable test specifications
- **Automated Test Generation**: `test_runner_inference.pl` automatically generates tests from compiled scripts
- **End-to-End Workflow**: Complete pipeline from Prolog → Compilation → Testing
- **Robust Test Execution**: Path-independent test runners with dependency resolution
- **Workflow Documentation**: Philosophy, environment setup, and roadmap documents in place

📋 **Current Use Case:**
AI agents can follow playbooks to:
1. Generate Prolog code
2. Compile via `compiler_driver.pl`
3. Auto-generate and execute tests via `test_runner_inference.pl`
4. Verify correctness

### Future Vision (v1.0 - Literate Programming)

🔮 **Not Yet Implemented:**
- **Fully Executable Playbooks**: Playbooks as complete, self-executing literate programs
- **Interactive Notebooks**: Jupyter-style notebooks with embedded Prolog/Bash execution
- **Dynamic Code Generation**: Real-time code synthesis from natural language
- **Advanced Orchestration**: Complex multi-agent workflows with tool composition

📖 **Literate Programming Goal:**
Transform playbooks into executable narratives where documentation, code generation, compilation, and testing are seamlessly integrated into a single, runnable document.

**See Also:**
- `workflow_roadmap.md` - Development roadmap for future features
- `workflow_todo.md` - Specific next steps and enhancements
- `workflow_philosophy.md` - Core principles guiding development

## Directory Structure

- **`*.md`** - Stable, reviewed AI skills ready for use
- **`workflow_*.md`** - Workflow system documentation
- **`README.md`** - This file

**Note:** For educational material and tutorials, see `education/drafts/` in the main project.

## What are AI Skills?

AI skills are structured prompts that provide context, patterns, and workflows for specific tasks. They help AI assistants:

- **Understand project-specific patterns** and conventions
- **Follow consistent workflows** across different sessions
- **Access curated examples** and templates
- **Avoid common pitfalls** specific to the project
- **Maintain best practices** without repetitive explanation

## Using Skills with Different AI Tools

### Claude Code

Claude Code (formerly Claude Dev) supports skills natively through the `.claude/skills/` directory:

1. **Create skill file**: `.claude/skills/skill-name.md`
2. **Invoke skill**: Use the Skill tool or mention the skill by name
3. **Skill format**: Markdown with clear sections for context, instructions, and examples

**Installation for Claude Code:**
```bash
# Copy skills to .claude/skills/ directory
cp docs/development/ai-skills/*.md .claude/skills/
```

### GitHub Copilot

GitHub Copilot can use skills as context when you:

1. **Reference in comments**: Mention the skill file in code comments
2. **Open alongside code**: Keep skill markdown open in another tab
3. **Use in workspace**: Place in `.github/copilot/` directory (if supported)

**Example:**
```prolog
% See docs/development/ai-skills/prolog-stdin-test.md for testing patterns
:- source(json, my_source, [...]).
```

### Cursor IDE

Cursor can leverage skills through:

1. **`.cursorrules` file**: Include skill references in project rules
2. **Context inclusion**: Add skills to Cursor's context via `@docs`
3. **Custom instructions**: Reference skills in per-project instructions

**Example `.cursorrules` entry:**
```
When testing Prolog code, follow patterns in docs/development/ai-skills/prolog-stdin-test.md
```

### ChatGPT / GPT-based Tools

For ChatGPT, Custom GPTs, or GPT-based tools:

1. **Copy skill content** into the conversation or custom instructions
2. **Reference as file**: Upload skill markdown as project context
3. **Link in prompts**: "Follow the workflow in prolog-stdin-test.md"

### Aider

Aider (command-line AI pair programmer) can use skills via:

1. **Add to context**: `aider --read docs/development/ai-skills/skill-name.md`
2. **Reference in `.aider.conf.yml`**: Include in default read files
3. **Mention in messages**: "Use the prolog-stdin-test skill"

**Example `.aider.conf.yml`:**
```yaml
read:
  - docs/development/ai-skills/*.md
```

### Continue.dev

Continue.dev supports skills through:

1. **Context providers**: Configure in `.continue/config.json`
2. **Slash commands**: Create custom commands that reference skills
3. **Manual inclusion**: Add skills to conversation context

**Example `.continue/config.json`:**
```json
{
  "contextProviders": [
    {
      "name": "docs",
      "params": {
        "path": "docs/development/ai-skills"
      }
    }
  ]
}
```

## Available Skills

### prolog-stdin-test.md

Test SWI-Prolog code using stdin without creating temporary files.

**Use cases:**
- Testing UnifyWeaver data sources (CSV, JSON, Python, HTTP)
- Quick Prolog experiments
- Verifying code generation
- Running one-off queries

**Key patterns:**
- `consult(user)` for loading from stdin
- Heredoc templates for different source types
- Test data setup
- Common pitfalls and solutions

## Creating New Skills

When creating a new skill for UnifyWeaver:

1. **Use clear structure**:
   - **Title**: Concise name
   - **Context**: When and why to use this skill
   - **Instructions**: Step-by-step workflow
   - **Examples**: Concrete, runnable examples
   - **Pitfalls**: Common mistakes to avoid
   - **References**: Links to related docs

2. **Include project specifics**:
   - Module paths (e.g., `src/unifyweaver/sources`)
   - Common patterns used in the codebase
   - Testing conventions
   - File organization

3. **Provide templates**:
   - Copy-paste ready code snippets
   - Command-line examples
   - Expected outputs

4. **Cross-reference**:
   - Link to related documentation
   - Reference example files
   - Point to existing code

## Skill File Naming

Use descriptive, hyphenated names:
- `prolog-stdin-test.md` - Testing Prolog via stdin
- `data-source-creation.md` - Creating new data sources
- `constraint-debugging.md` - Debugging constraint systems
- `emoji-compatibility.md` - Handling cross-platform emoji

## Maintenance

Skills should be:
- ✅ **Updated** when workflows change
- ✅ **Validated** with actual code examples
- ✅ **Versioned** alongside code changes
- ✅ **Referenced** in related documentation

## Integration with Project Documentation

Skills complement but don't replace regular documentation:

| Type | Purpose | Audience |
|------|---------|----------|
| **Regular Docs** | Explain concepts, architecture, API | Human developers |
| **AI Skills** | Provide workflows, patterns, context | AI assistants |
| **Code Comments** | Explain specific implementation details | Both |
| **Examples** | Demonstrate usage | Both |

**Best practice**: Skills should reference regular docs for deep dives:
```markdown
See docs/architecture/DATA_SOURCES.md for source type architecture.
```

## Contributing Skills

When you discover a useful workflow or pattern:

1. **Create skill file** - Document the workflow in this directory
2. **Add examples** - Include UnifyWeaver-specific patterns
3. **Sync to Claude** - Run `./scripts/sync_ai_skills.sh` to activate
4. **Update this README** - Add skill description to Available Skills section
5. **Reference in code/docs** - Link from related documentation

**Example workflow:**
```bash
# 1. Create skill
vim docs/development/ai-skills/my-new-skill.md

# 2. Sync to Claude
./scripts/sync_ai_skills.sh

# 3. Test and refine
# ... iterate ...

# 4. Commit when stable
git add docs/development/ai-skills/my-new-skill.md
git commit -m "Add my-new-skill AI skill"
```

**Note:** For educational content or broader tutorials, consider using `education/drafts/` instead.

## Further Reading

- [Claude Code Skills Documentation](https://docs.claude.com/claude-code) (when available)
- [GitHub Copilot Context](https://docs.github.com/copilot)
- [Cursor AI Documentation](https://cursor.sh/docs)
- `docs/development/STDIN_LOADING.md` - Related technical documentation
