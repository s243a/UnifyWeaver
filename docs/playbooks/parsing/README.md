# Playbook Record Parsers

Playbooks reference example records instead of embedding scripts. Use one of the parsers below (in preference order) to extract a record into a temporary file before execution.

## Related Documentation

- **[Example Record Format Specification](../../development/EXAMPLE_RECORD_FORMAT.md)** - Full format specification
- **[Extract Records Tool Specification](../../development/EXTRACT_RECORDS_TOOL_SPECIFICATION.md)** - Tool design and CLI options
- **[Extract Records Skill](../../../skills/skill_extract_records.md)** - AI agent skill for using the extractor

## Available Parsers

| Priority | Tool | Path / Install | Notes |
| --- | --- | --- | --- |
| 1 | Perl Extractor | `scripts/utils/extract_records.pl` | Default choice. Supports `--format`, `--query`, `--file-filter`. |
| 2 | Python Extractor | `docs/playbooks/parsing/extract_records.py` (future) | Placeholder for a dependency-free Python implementation. |
| 3 | `parsc` | [GitHub: parsc](https://github.com/s243a/parsc) | External CLI from @s243a. Use when Perl isn't available. |

## Example Record Format

Example library files use Obsidian-style callouts for human readability and machine parseability.

### Required File Structure

```markdown
---
file_type: UnifyWeaver Example Library
---
# Category Title

## `namespace.category.record_name`

> [!example-record]
> id: namespace.category.record_name
> name: Human Readable Name
> platform: bash

Description text goes here.

\`\`\`bash
#!/bin/bash
# Script content
echo "Hello World"
\`\`\`
```

### Metadata Fields

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | Unique identifier (used for queries) |
| `name` | Yes | Human-readable name |
| `platform` | No | Target platform: `bash`, `prolog`, `powershell`, etc. |

### Record Separator

When extracting multiple records, they are separated by the null character (`\0`) for safe handling of records containing any printable characters.

## Usage Pattern (Perl Extractor)

```bash
# Extract a single record by ID
scripts/utils/extract_records.pl \
  --format content \
  --query "unifyweaver.execution.bash_fork_basic" \
  playbooks/examples_library/bash_parallel_examples.md \
  > tmp/extracted.sh

# Make executable and run
chmod +x tmp/extracted.sh
bash tmp/extracted.sh
```

### Common Options

| Option | Description |
| --- | --- |
| `-f content` | Extract only the code block content (no metadata) |
| `-f full` | Extract the entire record including metadata |
| `-f json` | Output as JSON |
| `-q <pattern>` | Filter records by `id` or `name` (regex supported) |
| `--file-filter all` | Process files without checking frontmatter |

## Verified Playbooks

The following playbooks have been tested with AI agents (Claude Haiku and Gemini 2.5 Pro) and confirmed to work with the Obsidian-style callout format:

| Playbook | Example Library | Haiku Rating | Gemini Rating | Status |
| --- | --- | --- | --- | --- |
| `sqlite_source_playbook.md` | `sqlite_source_examples.md` | 2/10 | 2/10 | Verified |
| `awk_advanced_playbook.md` | `awk_advanced_examples.md` | 4/10 | 2/10 | Verified |
| `sql_window_playbook.md` | `sql_window_examples.md` | 3/10 | 1/10 | Verified |
| `bash_parallel_playbook.md` | `bash_parallel_examples.md` | 3/10 | 1/10 | Verified |

**Rating Scale:** 1-3 = Very clear, deterministic steps | 4-5 = Some interpretation needed | 6-7 = Requires context understanding | 8-10 = Complex reasoning required

*Last tested: 2025-12-09*

## Adding New Parsers

1. Implement the parser (e.g., Python) in this directory.
2. Provide a short README entry describing invocation, dependencies, and compatibility.
3. Update `skills/skill_extract_records.md` to mention the new option/reference.
