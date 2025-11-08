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

## 3. Tool: `extract_records.pl`

The implementation of this skill is the Perl script `scripts/utils/extract_records.pl`.

### 3.1. Command-Line Interface

```
extract_records [OPTIONS] [PATH...]
```

### 3.2. Options

| Flag | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-s`, `--separator` | `<char>` | `\0` | The character for separating output records. |
| `-f`, `--format` | `full`\|`content`\|`json` | `full` | The output format for each record. |
| `-q`, `--query` | `<pattern>` | (none) | A regex pattern to filter records by their `name` metadata. |
| `--file-filter` | `<key=value>` | `file_type=UnifyWeaver Example Library` | Filters files based on their YAML frontmatter. Use `"all"` to disable. |
| `-h`, `--help` | | | Display the help message. |

### 3.3. Output Formats

- **`full`:** Outputs the entire original Markdown block for each matching record.
- **`content`:** Outputs only the raw content of the record's code block.
- **`json`:** Outputs a stream of JSON objects, one for each record.

## 4. Full Specification

For the complete specification of the record format that this tool parses, please refer to the document: `docs/development/EXAMPLE_RECORD_FORMAT.md`.
