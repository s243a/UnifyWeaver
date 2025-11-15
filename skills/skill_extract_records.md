# Skill: Extract Records

## 1. Purpose

This skill enables an agent to parse and extract structured data records from Markdown files that conform to the "UnifyWeaver Example Record Format."

It is a specialized tool that is more robust and semantically aware than using generic tools like `grep` for this specific task.

## 2. When to Use

Use this tool when you are given a path to a Markdown file or directory and your goal is to extract specific examples or data records from it. It is particularly useful when you need to process the records programmatically (e.g., as JSON) or when you need to ensure you are only processing files that adhere to the correct specification.

**Choose this tool over `grep` if:**
*   You need to parse metadata (like `id` or `name`).
*   You need to handle multi-line content blocks reliably.
*   You need the output in a structured format like JSON.

## 3. Tooling Options

Preferred parser order (see [docs/playbooks/parsing/README.md](../docs/playbooks/parsing/README.md) for details):
1. `scripts/utils/extract_records.pl` (Perl, built-in)
2. Python extractor (future, documented in the parser README)
3. External `parsc` CLI (when Perl/Python are unavailable)

The remainder of this skill covers option #1, which ships with the repo.

### 3.1. Tool: `extract_records.pl`

The implementation of this skill is the Perl script `scripts/utils/extract_records.pl`.

#### Command-Line Interface

```
extract_records [OPTIONS] [PATH...]
```

#### Options

| Flag | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-s`, `--separator` | `<char>` | `\0` | The character for separating output records. |
| `-f`, `--format` | `full`\|`content`\|`json` | `full` | The output format for each record. |
| `-q`, `--query` | `<pattern>` | (none) | A regex pattern to filter records by their `name` metadata. |
| `--file-filter` | `<key=value>` | `file_type=UnifyWeaver Example Library` | Filters files based on their YAML frontmatter. Use `all` (or empty) to disable. |
| `-h`, `--help` | | | Display the help message. |

#### Output Formats

- **`full`:** Outputs the entire original Markdown block for each matching record.
- **`content`:** Outputs only the raw content of the record's code block.
- **`json`:** Outputs a stream of JSON objects, one for each record.

## 4. Examples

For concrete usage patterns, an agent should query the example library.

**Example Query:** see the parser README and the playbook example library for concrete invocations tied to current playbooks.
