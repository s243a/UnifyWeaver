# Tool Specification: extract_records

**Status:** Design Document
**Version:** 1.0
**Date:** 2025-10-25

## 1. Purpose

`extract_records` is a stream-oriented command-line tool for parsing Markdown files that conform to the "Example Record Format." It is designed to find and extract structured records from a library of human-readable example files, making them available for programmatic use.

## 2. Language

The tool will be implemented in **Perl** for its powerful and concise text-processing capabilities.

## 3. File Header Specification

To be considered a valid "Example Library," a Markdown file must begin with a YAML frontmatter block.

**Example:**
```yaml
---
file_type: UnifyWeaver Example Library
spec_version: 1.0
---
```

## 4. Input Handling

The tool accepts any number of file or directory paths as arguments.
- If an argument is a file, it is processed.
- If an argument is a directory, the tool recursively searches for all Markdown files (`.md`, `.markdown`).
- If no arguments are provided, it reads from standard input.

## 5. Command-Line Interface

```
extract_records [OPTIONS] [PATH...]
```

| Flag | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-s`, `--separator` | `<char>` | `\0` | The character for separating output records. Supports escape sequences. |
| `-f`, `--format` | `full`\|`content`\|`json` | `full` | The output format for each record. |
| `-q`, `--query` | `<pattern>` | (none) | A regex pattern to filter records by their `name` metadata. |
| `--file-filter` | `<key=value>` | `file_type=UnifyWeaver Example Library` | Filters files based on their YAML frontmatter. Use `"all"` to disable filtering. |
| `-R`, `--recursive` | | | Recursively search directories. This is the default behavior for directory paths. |
| `-h`, `--help` | | | Display a help message. |

## 6. Output Formats

- **`full`:** Outputs the entire original Markdown block for each matching record.
- **`content`:** Outputs only the raw content of the record's code block, stripping all metadata.
- **`json`:** Outputs a stream of JSON objects, one for each record, containing all parsed metadata and content.

## 7. Execution Logic

1.  Build a list of all Markdown files from the input paths.
2.  For each file, read its YAML frontmatter.
3.  Apply the `--file-filter` to discard non-matching files. By default, only processes files identified as `UnifyWeaver Example Library`.
4.  For each valid file, parse it to find all records.
5.  For each record, apply the `--query` filter to its `name` metadata.
6.  For each matching record, print it to standard output in the specified `--format`, followed by the record `--separator`.
