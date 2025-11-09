# Example Record Format Specification

**Status:** Design Document
**Version:** 1.0
**Date:** 2025-10-25

## 1. Overview

This document specifies a standardized format for embedding parsable data records within Markdown files. The primary use case is for creating libraries of examples (e.g., log files, code snippets, configurations) that are both human-readable and machine-parsable.

An AI agent can be provided with tools that understand this format to search, extract, and utilize these records.

## 2. Record Structure

An Example Record consists of three parts: a Header, a Metadata Callout, and a Content Block.

### a. Header

*   **Format:** A standard Markdown header (`#` to `######`).
*   **Purpose:** Provides a human-readable title for the record.
*   **Rule:** The parser should treat the header level as the boundary marker for the record. The record ends when the parser encounters the next header of the same or a higher level (e.g., an `<h3>` record ends at the next `<h3>`, `<h2>`, or `<h1>`).

### b. Metadata Callout

*   **Format:** An Obsidian-style blockquote callout immediately following the header. The callout type must be `[!example-record]`.
*   **Purpose:** Contains the structured, machine-parsable metadata for the record.
*   **Content:** A list of key-value pairs.

### c. Content Block

*   **Format:** Typically a Markdown code block (e.g., ` ```text ... ``` `), but can be any standard Markdown content.
*   **Purpose:** Contains the actual content of the example.
*   **Rule:** This is the content that follows the Metadata Callout and precedes the next boundary header.

## 3. Metadata Fields

The `[!example-record]` callout must contain the following key-value pairs:

### `id` (Required)
*   **Type:** String
*   **Description:** A globally unique identifier for the record.
*   **Recommendation:** Use a date-based format to ensure uniqueness (e.g., `YYYYMMDD-HHMMSS-NNN`).
*   **Example:** `id: 20251025-183500-001`

### `name` (Required)
*   **Type:** String
*   **Description:** A human-readable, namespaced shorthand for referencing the record.
*   **Format:** Uses dot-notation for namespacing (e.g., `namespace.category.item`).
*   **Example:** `name: logs.security.auth_failure`

### Shorthand Referencing

An agent or tool may allow referencing a record by the right-most part of its `name` if that part is unique across the entire library of loaded examples. For instance, if `auth_failure` is unique, it can be used as a valid reference to `logs.security.auth_failure`.

## 4. Example

```markdown
# Log File Examples

---

### Example: Simple Application Log

> [!example-record]
> id: 20251025-183500-001
> name: logs.app.simple_info

`​`​`text
2025-10-25 18:35:01 INFO: Application startup complete.
2025-10-25 18:35:02 INFO: User 'alice' logged in.
`​`​`

---

### Example: Security Authentication Failure

> [!example-record]
> id: 20251025-183600-001
> name: logs.security.auth_failure

`​`​`text
2025-10-25 18:36:15 AUDIT: Authentication failed: incorrect password.
`​`​`
```

## 5. Tooling Expectation

An agent should be provided with a tool (e.g., `extract_records`) that can take a path to a Markdown file as input and return a structured list of the records it contains, parsed according to this specification.
